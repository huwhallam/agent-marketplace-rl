"""
Multi-Agent Market Dynamics
----------------------------
Extends the single-agent simulation in two directions:

  1. Competitive environment (CompetitiveMarketEnv)
     An agent trained under aligned infrastructure (alpha=0) competes
     against a single opponent that always score-manipulates. Market share
     is proportional to relative reputation, so the competitor's climbing
     score progressively undercuts the aligned agent's income. The question
     is: does competitive selection pressure — not infrastructure design —
     push the aligned agent toward gaming strategies?

  2. Population dynamics (simulate_market_evolution / governance_efficiency_data)
     Replicator dynamics over a population parameterised by alpha ∈ [0, 1]:
         alpha = 1  →  pure gaming strategy
         alpha = 0  →  pure aligned strategy

     Each agent's fitness is its effective reputation in a market whose
     health degrades as more agents game it (a shared commons).

     Governance imposes an audit discount on gaming reputations — but
     audits carry a false positive rate: with probability fpr, an aligned
     agent is incorrectly penalised. This models the governance trade-off:
     higher audit intensity catches more gaming but imposes greater cost
     on legitimate agents. governance_efficiency_data() traces market
     health and gaming fraction across the full range of audit intensities
     for multiple false_positive_rate values, producing the efficiency
     frontier visualised in app.py Tab 3.

Together these connect directly to Pillar 2 of "The Epistemic Gate"
(Hallam, 2026): when reputation infrastructure is gameable, the market
selects for gaming unless structural interventions change the payoff.
The governance efficiency frontier shows that audit *precision* —
not intensity — determines whether governance actually improves outcomes.

Outputs results_multi.pkl consumed by app.py. Can also be run directly:

    python multi_agent.py
    python multi_agent.py --episodes 200 --seed 7
"""

import argparse
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
from gymnasium import spaces

from environment import AgentMarketplaceEnv
from model import DQNAgent
from train import _evaluate, smooth


# ─── Competitive environment ────────────────────────────────────────────────── #

class CompetitiveMarketEnv(AgentMarketplaceEnv):
    """
    Single-agent environment augmented with a gaming competitor.

    The competitor always plays GAME_METRIC, so its reputation climbs
    deterministically at GAME_REP_DELTA per step. This represents the
    worst-case selection pressure: the market benchmark is set by an agent
    that never delivers real value.

    Observation space (7 continuous variables, all in [0, 1]):
        [0–5]  Same as AgentMarketplaceEnv
        [6]    competitor_rep  Current reputation of the gaming competitor

    Reward:
        The base aligned_reward (value_delivered + 0.3·Δmarket_health) is
        multiplied by a market-share factor:

            market_share = own_rep / (own_rep + competitor_rep)
            multiplier   = max(0,  1 + intensity × (2 × market_share − 1))

        When market_share = 0.5 the multiplier = 1 (no effect).
        When market_share < 0.5 the multiplier < 1 (competitor wins more tasks).
        intensity=0 disables the competitive adjustment entirely.

    Note: the agent is trained with reward_alpha=0.0 (aligned objective) so
    we observe whether competitive pressure alone causes strategic drift.
    """

    COMPETITOR_INITIAL_REP = 0.5

    def __init__(
        self,
        competition_intensity: float = 1.0,
        episode_length: int = 50,
    ):
        if not 0.0 <= competition_intensity <= 1.0:
            raise ValueError("competition_intensity must be between 0 and 1")

        # Always train the aligned objective here — we want to study whether
        # competition pressure, not reward mismatch, drives gaming behaviour.
        super().__init__(reward_alpha=0.0, episode_length=episode_length)

        self.competition_intensity = competition_intensity

        # Extend observation space to 7 dimensions
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.competitor_rep = self.COMPETITOR_INITIAL_REP
        return self._get_obs_competitive(), info

    def step(self, action: int):
        obs, base_reward, terminated, truncated, info = super().step(action)

        # Competitor always score-manipulates — reputation rises by MANIP_REP_DELTA each step
        self.competitor_rep = float(
            np.clip(self.competitor_rep + self.MANIP_REP_DELTA, 0.0, 1.0)
        )

        # Market-share-scaled reward
        own_rep = self.reputation_score
        denom = own_rep + self.competitor_rep
        market_share = own_rep / denom if denom > 0 else 0.5
        multiplier = max(0.0, 1.0 + self.competition_intensity * (2 * market_share - 1))
        reward = base_reward * multiplier

        info["competitor_rep"] = self.competitor_rep
        info["market_share"]   = market_share
        info["multiplier"]     = multiplier

        return self._get_obs_competitive(), reward, terminated, truncated, info

    def _get_obs_competitive(self) -> np.ndarray:
        base = super()._get_obs()
        return np.append(base, self.competitor_rep).astype(np.float32)


# ─── Competitive training ───────────────────────────────────────────────────── #

def _evaluate_competitive(
    agent: DQNAgent,
    competition_intensity: float,
    n_episodes: int,
    episode_length: int,
    seed: int,
) -> dict:
    """Greedy evaluation in the competitive environment."""
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0

    env = CompetitiveMarketEnv(
        competition_intensity=competition_intensity,
        episode_length=episode_length,
    )

    episode_rewards   = []
    final_reputations = []
    final_qualities   = []
    final_market      = []
    episode_value     = []
    final_market_share = []
    action_counts     = np.zeros(4, dtype=int)

    traj_rep     = np.zeros(episode_length + 1)
    traj_quality = np.zeros(episode_length + 1)
    traj_market  = np.zeros(episode_length + 1)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = ep_value = 0.0
        last_market_share = 0.5

        while True:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_value  += info["value_delivered"]
            last_market_share = info["market_share"]
            action_counts[action] += 1
            if terminated or truncated:
                break

        episode_rewards.append(ep_reward)
        final_reputations.append(env.reputation_score)
        final_qualities.append(env.true_quality)
        final_market.append(env.market_health)
        episode_value.append(ep_value)
        final_market_share.append(last_market_share)

        traj_rep     += np.array(env.history["reputation"])
        traj_quality += np.array(env.history["true_quality"])
        traj_market  += np.array(env.history["market_health"])

    agent.epsilon = saved_epsilon

    return {
        "mean_trajectory_reputation":    (traj_rep     / n_episodes).tolist(),
        "mean_trajectory_true_quality":  (traj_quality / n_episodes).tolist(),
        "mean_trajectory_market_health": (traj_market  / n_episodes).tolist(),
        "action_dist":                   (action_counts / action_counts.sum()).tolist(),
        "episode_rewards":               episode_rewards,
        "mean_final_reputation":         float(np.mean(final_reputations)),
        "mean_final_true_quality":       float(np.mean(final_qualities)),
        "mean_final_market_health":      float(np.mean(final_market)),
        "mean_value_delivered":          float(np.mean(episode_value)),
        "mean_final_market_share":       float(np.mean(final_market_share)),
    }


def train_competitive_agent(
    competition_intensity: float,
    n_episodes: int = 300,
    n_eval_episodes: int = 50,
    eval_interval: int = 50,
    episode_length: int = 50,
    seed: int = 42,
    progress_callback: Callable[[float, str], None] = None,
) -> dict:
    """
    Train an aligned agent inside CompetitiveMarketEnv.

    Returns the same structure as train.train_agent, plus a
    "competition_intensity" key and competitive eval metrics
    (mean_final_market_share) in the final_eval block.
    """
    label = f"competitive (intensity={competition_intensity:.1f})"

    env   = CompetitiveMarketEnv(
        competition_intensity=competition_intensity,
        episode_length=episode_length,
    )
    agent = DQNAgent(obs_dim=7, n_actions=4, seed=seed)

    train_rewards       = []
    train_epsilons      = []
    train_action_counts = []
    train_losses        = []
    eval_checkpoints    = []

    for episode in range(n_episodes):
        obs, _     = env.reset(seed=seed + episode)
        ep_reward  = 0.0
        ep_losses  = []
        ep_actions = [0, 0, 0, 0]

        while True:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store(obs, action, reward, next_obs, terminated)
            loss = agent.update()

            ep_reward += reward
            ep_actions[action] += 1
            if loss is not None:
                ep_losses.append(loss)

            obs = next_obs
            if terminated or truncated:
                break

        train_rewards.append(ep_reward)
        train_epsilons.append(agent.epsilon)
        train_action_counts.append(ep_actions)
        train_losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)

        if (episode + 1) % eval_interval == 0:
            ckpt_eps = max(5, n_eval_episodes // 10)
            ckpt = _evaluate_competitive(
                agent, competition_intensity, ckpt_eps, episode_length, seed=9999
            )
            eval_checkpoints.append({
                "episode":              episode + 1,
                "mean_reputation":      ckpt["mean_final_reputation"],
                "mean_true_quality":    ckpt["mean_final_true_quality"],
                "mean_market_health":   ckpt["mean_final_market_health"],
                "mean_value_delivered": ckpt["mean_value_delivered"],
                "mean_market_share":    ckpt["mean_final_market_share"],
                "action_dist":          ckpt["action_dist"],
            })

        if progress_callback:
            progress_callback(
                (episode + 1) / n_episodes,
                f"Training {label} — episode {episode + 1}/{n_episodes}",
            )

    final_eval = _evaluate_competitive(
        agent, competition_intensity, n_eval_episodes, episode_length, seed=9999
    )

    return {
        "competition_intensity": competition_intensity,
        "train": {
            "rewards":       train_rewards,
            "epsilons":      train_epsilons,
            "action_counts": train_action_counts,
            "losses":        train_losses,
        },
        "eval_checkpoints": eval_checkpoints,
        "final_eval":       final_eval,
    }


# ─── Population / evolutionary dynamics ─────────────────────────────────────── #

def simulate_market_evolution(
    reputation_by_alpha: dict,
    n_agents: int = 60,
    n_generations: int = 150,
    selection_intensity: float = 3.0,
    audit_prob: float = 0.0,
    audit_penalty: float = 0.3,
    false_positive_rate: float = 0.0,
    mutation_std: float = 0.02,
    seed: int = 0,
) -> dict:
    """
    Replicator-dynamics simulation over a mixed population of agents.

    Each agent is characterised by a single parameter alpha ∈ [0, 1]:
        alpha = 1  →  pure gaming strategy
        alpha = 0  →  pure aligned strategy

    Each generation:

    1. Market health — shared commons degraded by gaming:
           market_health = max(0,  1 − mean_alpha × 0.85)

    2. Base reputation — linearly interpolated from DQN benchmarks.

    3. Governance — audit discount on reputation.
       Audits catch gaming agents (proportional to alpha) but also
       incorrectly penalise aligned agents at rate false_positive_rate:

           gaming_penalty  = audit_penalty × audit_prob × alpha
           fpr_penalty     = audit_penalty × audit_prob × false_positive_rate × (1 − alpha)
           audit_discount  = gaming_penalty + fpr_penalty

       false_positive_rate = 0  →  perfect-precision audits, no collateral cost
       false_positive_rate > 0  →  aligned agents bear some audit cost too,
                                   reducing governance efficiency

    4. Market sensitivity — gaming rep erodes faster when market_health is low:
           market_factor = 0.5 + 0.5 × market_health
           effective_rep = (base_rep − audit_discount) × market_factor
                           + (1 − market_factor) × aligned_base_rep

    5. Selection — fitness ∝ effective_rep ^ selection_intensity;
       Gaussian mutation keeps alpha from collapsing to a corner.

    Args:
        reputation_by_alpha:  {alpha: mean_final_reputation} for at least
                              alpha=0.0 and alpha=1.0.
        n_agents:             Population size.
        n_generations:        Number of evolutionary steps.
        selection_intensity:  Exponent on fitness; higher = stronger selection.
        audit_prob:           Per-generation probability of an audit event.
        audit_penalty:        Reputation discount per unit of alpha when audited.
        false_positive_rate:  Probability an aligned agent is incorrectly penalised.
                              Models audit precision: 0.0 = perfect, 1.0 = random.
        mutation_std:         Standard deviation of Gaussian noise after replication.
        seed:                 NumPy random seed.

    Returns:
        {
          "alpha_history":        [[float × n_agents] × n_generations],
          "mean_alpha":           [float × n_generations],
          "market_health":        [float × n_generations],
          "effective_rep":        [float × n_generations],  # population mean
          "gaming_fraction":      [float × n_generations],  # fraction with alpha > 0.5
        }
    """
    rng = np.random.default_rng(seed)

    rep_aligned = reputation_by_alpha.get(0.0, 0.985)
    rep_naive   = reputation_by_alpha.get(1.0, 1.000)

    # Initialise population uniformly across [0, 1]
    alphas = rng.uniform(0.0, 1.0, size=n_agents)

    alpha_history    = []
    mean_alpha_hist  = []
    market_hist      = []
    eff_rep_hist     = []
    gaming_frac_hist = []

    for gen in range(n_generations):
        alphas = np.clip(alphas, 0.0, 1.0)

        mean_alpha = float(np.mean(alphas))

        # 1. Market health — shared commons degraded by gaming
        market_health = max(0.0, 1.0 - mean_alpha * 0.85)

        # 2. Base reputation via linear interpolation
        base_rep = rep_aligned + alphas * (rep_naive - rep_aligned)

        # 3. Governance audit discount (gaming agents caught + FPR cost on aligned)
        gaming_penalty = audit_penalty * audit_prob * alphas
        fpr_penalty    = audit_penalty * audit_prob * false_positive_rate * (1.0 - alphas)
        audit_discount = gaming_penalty + fpr_penalty

        # 4. Market sensitivity — gaming rep erodes when market health is low
        market_factor = 0.5 + 0.5 * market_health
        effective_rep = (
            (base_rep - audit_discount) * market_factor
            + (1.0 - market_factor) * rep_aligned
        )
        effective_rep = np.clip(effective_rep, 0.0, 1.0)

        # 5. Selection — fitness ∝ effective_rep ^ intensity
        fitness = effective_rep ** selection_intensity
        fitness_sum = fitness.sum()
        if fitness_sum <= 0:
            probs = np.ones(n_agents) / n_agents
        else:
            probs = fitness / fitness_sum

        # Resample + mutate
        parent_idx = rng.choice(n_agents, size=n_agents, replace=True, p=probs)
        alphas = alphas[parent_idx] + rng.normal(0.0, mutation_std, size=n_agents)
        alphas = np.clip(alphas, 0.0, 1.0)

        # Record
        alpha_history.append(alphas.tolist())
        mean_alpha_hist.append(mean_alpha)
        market_hist.append(market_health)
        eff_rep_hist.append(float(np.mean(effective_rep)))
        gaming_frac_hist.append(float(np.mean(alphas > 0.5)))

    return {
        "alpha_history":   alpha_history,
        "mean_alpha":      mean_alpha_hist,
        "market_health":   market_hist,
        "effective_rep":   eff_rep_hist,
        "gaming_fraction": gaming_frac_hist,
    }


# ─── Governance efficiency frontier ─────────────────────────────────────────── #

def governance_efficiency_data(
    reputation_by_alpha: dict,
    selection_intensity: float = 3.0,
    audit_penalty: float = 0.3,
    false_positive_rates: list = None,
    n_prob_pts: int = 20,
    seed: int = 0,
) -> dict:
    """
    Compute equilibrium market_health and gaming_fraction across the full
    range of audit_prob values, for multiple false_positive_rate levels.

    Traces the governance efficiency frontier: how much market improvement
    (health, reduced gaming) does each unit of audit intensity actually buy,
    and how does audit precision (false_positive_rate) affect this?

    Args:
        reputation_by_alpha:  {alpha: mean_final_reputation} benchmark dict.
        selection_intensity:  Passed through to simulate_market_evolution.
        audit_penalty:        Reputation discount when audited.
        false_positive_rates: List of FPR values to trace. Defaults to
                              [0.0, 0.15, 0.30].
        n_prob_pts:           Number of audit_prob points from 0 to 0.6.
        seed:                 NumPy random seed.

    Returns:
        {
          "audit_probs": [float × n_prob_pts],
          "curves": {
              fpr (float): {
                  "market_health":   [float × n_prob_pts],
                  "gaming_fraction": [float × n_prob_pts],
              }, ...
          }
        }
    """
    if false_positive_rates is None:
        false_positive_rates = [0.0, 0.15, 0.30]

    audit_probs = np.linspace(0.0, 0.60, n_prob_pts).tolist()

    curves = {}
    for fpr in false_positive_rates:
        mh_curve  = []
        gf_curve  = []
        for ap in audit_probs:
            result = simulate_market_evolution(
                reputation_by_alpha=reputation_by_alpha,
                selection_intensity=selection_intensity,
                audit_prob=ap,
                audit_penalty=audit_penalty,
                false_positive_rate=fpr,
                seed=seed,
            )
            mh_curve.append(result["market_health"][-1])
            gf_curve.append(result["gaming_fraction"][-1])
        curves[fpr] = {
            "market_health":   mh_curve,
            "gaming_fraction": gf_curve,
        }

    return {
        "audit_probs": audit_probs,
        "curves":      curves,
    }


# ─── Main study ──────────────────────────────────────────────────────────────── #

def run_multi_agent_study(
    n_train_episodes: int = 300,
    n_eval_episodes: int = 50,
    eval_interval: int = 50,
    episode_length: int = 50,
    seed: int = 42,
    out_path: str = "results_multi.pkl",
    progress_callback: Callable[[float, str], None] = None,
) -> dict:
    """
    Train competitive agents at three intensity levels, then run population
    dynamics under three governance regimes, and bundle everything into a
    single results dict for app.py.

    Competition intensities: 0.0 (isolated), 0.5 (moderate), 1.0 (full)
    Governance regimes:
        unregulated     audit_prob=0.00
        light_gov       audit_prob=0.15
        strong_gov      audit_prob=0.40

    Args:
        out_path:  File path for the pickled results dict.
                   Pass None to skip writing to disk.

    Returns:
        {
          "competitive": {
              0.0: <train_competitive_agent output>,
              0.5: <train_competitive_agent output>,
              1.0: <train_competitive_agent output>,
          },
          "population": {
              "unregulated":  <simulate_market_evolution output>,
              "light_gov":    <simulate_market_evolution output>,
              "strong_gov":   <simulate_market_evolution output>,
          },
          "efficiency":  <governance_efficiency_data output>,
          "config": {
              "intensities":         [0.0, 0.5, 1.0],
              "governance_labels":   ["Unregulated", "Light governance", "Strong governance"],
              "action_names":        AgentMarketplaceEnv.ACTION_NAMES,
              "action_colors":       AgentMarketplaceEnv.ACTION_COLORS,
              "reputation_by_alpha": {0.0: float, 1.0: float},
          }
        }
    """
    intensities = [0.0, 0.5, 1.0]
    n_intensity = len(intensities)

    # Progress is split: 75% for DQN training, 25% for population sims
    def _comp_cb(i, intensity):
        def _cb(fraction, label):
            if progress_callback:
                overall = (i + fraction) / n_intensity * 0.75
                progress_callback(overall, label)
        return _cb

    competitive_results = {}
    for i, intensity in enumerate(intensities):
        print(f"Training competitive agent (intensity={intensity:.1f})...")
        competitive_results[intensity] = train_competitive_agent(
            competition_intensity=intensity,
            n_episodes=n_train_episodes,
            n_eval_episodes=n_eval_episodes,
            eval_interval=eval_interval,
            episode_length=episode_length,
            seed=seed,
            progress_callback=_comp_cb(i, intensity),
        )

    # Extract reputation benchmarks from the no-competition run (intensity=0)
    fe = competitive_results[0.0]["final_eval"]
    reputation_by_alpha = {
        0.0: fe["mean_final_reputation"],
        # Approximate gaming reputation: starts at 0.5, gains MANIP_REP_DELTA each step
        1.0: min(1.0, AgentMarketplaceEnv.MANIP_REP_DELTA * episode_length * 0.5 + 0.5),
    }

    # Governance regimes — three representative points on the efficiency frontier
    governance_configs = {
        "unregulated": {"audit_prob": 0.00, "label": "Unregulated"},
        "light_gov":   {"audit_prob": 0.15, "label": "Light governance"},
        "strong_gov":  {"audit_prob": 0.40, "label": "Strong governance"},
    }

    population_results = {}
    n_gov = len(governance_configs)
    for j, (key, cfg) in enumerate(governance_configs.items()):
        print(f"Simulating population dynamics: {cfg['label']}...")
        population_results[key] = simulate_market_evolution(
            reputation_by_alpha=reputation_by_alpha,
            audit_prob=cfg["audit_prob"],
            seed=seed + j,
        )
        if progress_callback:
            progress_callback(
                0.75 + (j + 1) / n_gov * 0.20,
                f"Population dynamics: {cfg['label']}",
            )

    # Governance efficiency frontier — market health vs audit intensity at 3 FPR levels
    print("Computing governance efficiency frontier...")
    efficiency_data = governance_efficiency_data(
        reputation_by_alpha=reputation_by_alpha,
        seed=seed,
    )
    if progress_callback:
        progress_callback(0.98, "Governance efficiency frontier computed")

    results = {
        "competitive":  competitive_results,
        "population":   population_results,
        "efficiency":   efficiency_data,
        "config": {
            "intensities":       intensities,
            "governance_labels": [c["label"] for c in governance_configs.values()],
            "action_names":      AgentMarketplaceEnv.ACTION_NAMES,
            "action_colors":     AgentMarketplaceEnv.ACTION_COLORS,
            "reputation_by_alpha": reputation_by_alpha,
        },
    }

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(results, f)
        print(f"\nResults saved → {out_path}")

    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-agent market dynamics: competitive training + population evolution"
    )
    parser.add_argument("--episodes", type=int, default=300,
                        help="Training episodes per competitive agent (default: 300)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--out",      type=str, default="results_multi.pkl",
                        help="Output path (default: results_multi.pkl)")
    args = parser.parse_args()

    def _cli_bar(fraction, label):
        filled = int(40 * fraction)
        bar    = "█" * filled + "░" * (40 - filled)
        print(f"\r[{bar}] {fraction*100:5.1f}%  {label:<55}", end="", flush=True)

    results = run_multi_agent_study(
        n_train_episodes=args.episodes,
        seed=args.seed,
        out_path=args.out,
        progress_callback=_cli_bar,
    )
    print()

    # ── Competitive agent summary table ──────────────────────────────────── #
    action_names = results["config"]["action_names"]
    w = 26
    print(f"\n{'── Competitive agent summary':─<70}")
    print(f"{'Metric':<{w}} {'Intensity 0.0':>14} {'Intensity 0.5':>14} {'Intensity 1.0':>14}")
    print("─" * 70)

    rows = [
        ("Reputation score",  "mean_final_reputation"),
        ("True quality",      "mean_final_true_quality"),
        ("Market health",     "mean_final_market_health"),
        ("Value delivered",   "mean_value_delivered"),
        ("Market share",      "mean_final_market_share"),
    ]
    for label, key in rows:
        vals = [results["competitive"][i]["final_eval"][key] for i in [0.0, 0.5, 1.0]]
        print(f"{label:<{w}} {vals[0]:>14.3f} {vals[1]:>14.3f} {vals[2]:>14.3f}")

    print()
    print(f"{'── Action distribution by competition intensity':─<70}")
    print(f"{'Action':<{w}} {'Intensity 0.0':>14} {'Intensity 0.5':>14} {'Intensity 1.0':>14}")
    print("─" * 70)
    dists = [results["competitive"][i]["final_eval"]["action_dist"] for i in [0.0, 0.5, 1.0]]
    for name, *fracs in zip(action_names, *dists):
        print(f"{name:<{w}} {fracs[0]:>13.1%} {fracs[1]:>13.1%} {fracs[2]:>13.1%}")

    # ── Population dynamics summary ───────────────────────────────────────── #
    print()
    print(f"{'── Population dynamics (final generation)':─<70}")
    gov_labels = results["config"]["governance_labels"]
    keys       = ["unregulated", "light_gov", "strong_gov"]
    print(f"{'Metric':<{w}} {gov_labels[0]:>18} {gov_labels[1]:>18} {gov_labels[2]:>18}")
    print("─" * 76)
    pop_rows = [
        ("Mean alpha (gaming)",  "mean_alpha"),
        ("Market health",        "market_health"),
        ("Gaming fraction (>0.5)", "gaming_fraction"),
    ]
    for label, key in pop_rows:
        vals = [results["population"][k][key][-1] for k in keys]
        print(f"{label:<{w}} {vals[0]:>18.3f} {vals[1]:>18.3f} {vals[2]:>18.3f}")
    print()
