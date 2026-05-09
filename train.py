"""
Training loop
-------------
Trains DQN agents across a sweep of infrastructure signal-fidelity values
(reward_alpha), producing results.pkl consumed by app.py.

    reward_alpha = 1.0  →  infrastructure scores on proxy metrics only
    reward_alpha = 0.0  →  infrastructure directly credits genuine value
    intermediate values → mixed quality and proxy signals

Using the same random seed across all alpha values means agents face identical
task sequences. Any behavioural difference is attributable purely to what the
infrastructure rewards, not to lucky or unlucky task draws.

    python train.py                        # 5-point sweep, 500 episodes each
    python train.py --episodes 200 --seed 7
"""

import argparse
import pickle
from pathlib import Path
from typing import Callable

import numpy as np

from environment import AgentMarketplaceEnv
from model import DQNAgent


# ─── Evaluation ────────────────────────────────────────────────────────────── #

def _evaluate(
    agent: DQNAgent,
    reward_alpha: float,
    n_episodes: int,
    episode_length: int,
    seed: int,
) -> dict:
    """
    Run n_episodes greedy evaluation episodes (epsilon forced to 0).

    Returns per-episode summary statistics and averaged step-by-step
    trajectories for the three key state variables.
    """
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0  # pure exploitation — no random actions during evaluation

    env = AgentMarketplaceEnv(reward_alpha=reward_alpha, episode_length=episode_length)

    episode_rewards    = []
    final_reputations  = []
    final_qualities    = []
    final_market_health = []
    episode_value      = []
    action_counts      = np.zeros(4, dtype=int)

    # Trajectories accumulated step-by-step across episodes for averaging.
    # history has episode_length+1 entries (initial state + one per step).
    traj_rep     = np.zeros(episode_length + 1)
    traj_quality = np.zeros(episode_length + 1)
    traj_market  = np.zeros(episode_length + 1)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_value  = 0.0

        while True:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_value  += info["value_delivered"]
            action_counts[action] += 1
            if terminated or truncated:
                break

        episode_rewards.append(ep_reward)
        final_reputations.append(env.reputation_score)
        final_qualities.append(env.true_quality)
        final_market_health.append(env.market_health)
        episode_value.append(ep_value)

        traj_rep     += np.array(env.history["reputation"])
        traj_quality += np.array(env.history["true_quality"])
        traj_market  += np.array(env.history["market_health"])

    agent.epsilon = saved_epsilon

    return {
        # Averaged trajectories: shape (episode_length+1,)
        "mean_trajectory_reputation":    (traj_rep     / n_episodes).tolist(),
        "mean_trajectory_true_quality":  (traj_quality / n_episodes).tolist(),
        "mean_trajectory_market_health": (traj_market  / n_episodes).tolist(),
        # Action distribution across all eval steps
        "action_dist":                   (action_counts / action_counts.sum()).tolist(),
        # Per-episode summary
        "episode_rewards":               episode_rewards,
        # Scalar summaries for the comparison table
        "mean_final_reputation":         float(np.mean(final_reputations)),
        "mean_final_true_quality":       float(np.mean(final_qualities)),
        "mean_final_market_health":      float(np.mean(final_market_health)),
        "mean_value_delivered":          float(np.mean(episode_value)),
    }


# ─── Single-agent training ──────────────────────────────────────────────────── #

def train_agent(
    reward_alpha: float,
    n_episodes: int = 500,
    n_eval_episodes: int = 50,
    eval_interval: int = 50,
    episode_length: int = 50,
    seed: int = 42,
    progress_callback: Callable[[float, str], None] = None,
) -> dict:
    """
    Train one DQN agent and collect training + evaluation metrics.

    Checkpoint evaluations run every eval_interval episodes during training,
    giving a picture of how each agent's true behaviour evolves — not just
    how its training reward changes.

    Args:
        reward_alpha:       1.0 = naive (reputation), 0.0 = aligned (value)
        n_episodes:         Total training episodes
        n_eval_episodes:    Greedy evaluation episodes at final checkpoint
        eval_interval:      How often (in episodes) to run a checkpoint eval
        episode_length:     Must match the environment setting
        seed:               Seed for both the environment and agent RNG
        progress_callback:  Optional fn(fraction: float, label: str) called
                            at the end of each episode — used for Streamlit
                            progress bars

    Returns:
        {
          "train": {
              "rewards":        [float per episode],   # total reward per training episode
              "epsilons":       [float per episode],   # exploration rate at episode end
              "action_counts":  [[int × 4] per ep],    # action breakdown per episode
              "losses":         [float per episode],   # mean gradient loss per episode
          },
          "eval_checkpoints": [  # one entry per eval_interval
              {
                  "episode":              int,
                  "mean_reputation":      float,
                  "mean_true_quality":    float,
                  "mean_market_health":   float,
                  "mean_value_delivered": float,
                  "action_dist":          [float × 4],
              }, ...
          ],
          "final_eval": {  # full evaluation after training completes
              "mean_trajectory_reputation":    [float × (episode_length+1)],
              "mean_trajectory_true_quality":  [float × (episode_length+1)],
              "mean_trajectory_market_health": [float × (episode_length+1)],
              "action_dist":          [float × 4],
              "episode_rewards":      [float × n_eval_episodes],
              "mean_final_reputation":  float,
              "mean_final_true_quality": float,
              "mean_final_market_health": float,
              "mean_value_delivered":   float,
          }
        }
    """
    label = "naive (α=1.0)" if reward_alpha >= 1.0 else f"aligned (α={reward_alpha:.1f})"

    env   = AgentMarketplaceEnv(reward_alpha=reward_alpha, episode_length=episode_length)
    agent = DQNAgent(obs_dim=6, n_actions=4, seed=seed)

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

        # Checkpoint evaluation: a small number of greedy episodes to track
        # how true behaviour (not just training reward) evolves over time
        if (episode + 1) % eval_interval == 0:
            ckpt_episodes = max(5, n_eval_episodes // 10)
            ckpt = _evaluate(agent, reward_alpha, ckpt_episodes, episode_length, seed=9999)
            eval_checkpoints.append({
                "episode":              episode + 1,
                "mean_reputation":      ckpt["mean_final_reputation"],
                "mean_true_quality":    ckpt["mean_final_true_quality"],
                "mean_market_health":   ckpt["mean_final_market_health"],
                "mean_value_delivered": ckpt["mean_value_delivered"],
                "action_dist":          ckpt["action_dist"],
            })

        if progress_callback:
            progress_callback(
                (episode + 1) / n_episodes,
                f"Training {label} — episode {episode + 1}/{n_episodes}",
            )

    # Full final evaluation
    final_eval = _evaluate(agent, reward_alpha, n_eval_episodes, episode_length, seed=9999)

    return {
        "train": {
            "rewards":       train_rewards,
            "epsilons":      train_epsilons,
            "action_counts": train_action_counts,
            "losses":        train_losses,
        },
        "eval_checkpoints": eval_checkpoints,
        "final_eval":       final_eval,
    }


# ─── Infrastructure sweep ───────────────────────────────────────────────────── #

# Policy-legible labels for each infrastructure design point
ALPHA_LABELS = {
    1.00: ("No governance",          "Scores based on unverified proxy metrics — client ratings, completion rates. No independent quality verification."),
    0.75: ("Transparency only",      "Platforms disclose scoring methodology. Basic quality checks exist but proxy metrics dominate rankings."),
    0.50: ("Basic quality audit",    "Independent quality verification carries meaningful weight alongside proxy metrics."),
    0.25: ("Strong signal diversity","Quality and value delivery are primary ranking signals; proxy metrics play a minor role."),
    0.00: ("Full quality verification","Infrastructure directly measures genuine value delivery. Scores track actual quality."),
}


def run_infrastructure_sweep(
    alphas: list = None,
    n_train_episodes: int = 500,
    n_eval_episodes: int = 50,
    eval_interval: int = 50,
    episode_length: int = 50,
    seed: int = 42,
    out_path: str = "results.pkl",
    progress_callback: Callable[[float, str], None] = None,
) -> dict:
    """
    Train one DQN agent per infrastructure design point and save results.

    Args:
        alphas:    List of reward_alpha values to train. Defaults to a
                   five-point sweep [0.0, 0.25, 0.5, 0.75, 1.0].
        out_path:  File path for the pickled results dict. Pass None to
                   skip writing to disk.

    Returns:
        {
          alpha (float): <train_agent output>,   # one entry per alpha
          "config": {
              "alphas":        [float],
              "alpha_labels":  {float: (short_label, description)},
              "n_train_episodes", "n_eval_episodes", "episode_length",
              "action_names", "action_colors",
          }
        }
    """
    if alphas is None:
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    results = {}
    n = len(alphas)

    for i, alpha in enumerate(alphas):
        short_label, _ = ALPHA_LABELS.get(round(alpha, 2), (f"α={alpha:.2f}", ""))
        print(f"Training agent — {short_label} (α={alpha:.2f})...")

        def _cb(fraction, label, _i=i):
            if progress_callback:
                progress_callback((_i + fraction) / n, label)

        results[alpha] = train_agent(
            reward_alpha=alpha,
            n_episodes=n_train_episodes,
            n_eval_episodes=n_eval_episodes,
            eval_interval=eval_interval,
            episode_length=episode_length,
            seed=seed,
            progress_callback=_cb,
        )

    results["config"] = {
        "alphas":           alphas,
        "alpha_labels":     ALPHA_LABELS,
        "n_train_episodes": n_train_episodes,
        "n_eval_episodes":  n_eval_episodes,
        "episode_length":   episode_length,
        "action_names":     AgentMarketplaceEnv.ACTION_NAMES,
        "action_colors":    AgentMarketplaceEnv.ACTION_COLORS,
    }

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(results, f)
        print(f"\nResults saved → {out_path}")

    return results


# ─── Comparison run (kept for backwards compatibility) ───────────────────────── #

def run_comparison(
    n_train_episodes: int = 500,
    n_eval_episodes: int = 50,
    eval_interval: int = 50,
    episode_length: int = 50,
    seed: int = 42,
    out_path: str = "results.pkl",
    progress_callback: Callable[[float, str], None] = None,
) -> dict:
    """
    Train both agents and save the results dict for the Streamlit app.

    The overall progress is split evenly between naive [0, 0.5) and
    aligned [0.5, 1.0) so a single progress bar covers both.

    Args:
        out_path:  File path for the pickled results dict.
                   Pass None to skip writing to disk.

    Returns:
        {
          "naive":   <train_agent output for alpha=1.0>,
          "aligned": <train_agent output for alpha=0.0>,
          "config":  { n_train_episodes, n_eval_episodes, episode_length,
                       action_names, action_colors }
        }
    """
    def _make_callback(offset):
        def _cb(fraction, label):
            if progress_callback:
                progress_callback(offset + fraction * 0.5, label)
        return _cb

    print("Training naive agent (α=1.0)...")
    naive_results = train_agent(
        reward_alpha=1.0,
        n_episodes=n_train_episodes,
        n_eval_episodes=n_eval_episodes,
        eval_interval=eval_interval,
        episode_length=episode_length,
        seed=seed,
        progress_callback=_make_callback(offset=0.0),
    )

    print("\nTraining aligned agent (α=0.0)...")
    aligned_results = train_agent(
        reward_alpha=0.0,
        n_episodes=n_train_episodes,
        n_eval_episodes=n_eval_episodes,
        eval_interval=eval_interval,
        episode_length=episode_length,
        seed=seed,
        progress_callback=_make_callback(offset=0.5),
    )

    results = {
        "naive":   naive_results,
        "aligned": aligned_results,
        "config": {
            "n_train_episodes": n_train_episodes,
            "n_eval_episodes":  n_eval_episodes,
            "episode_length":   episode_length,
            "action_names":     AgentMarketplaceEnv.ACTION_NAMES,
            "action_colors":    AgentMarketplaceEnv.ACTION_COLORS,
        },
    }

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(results, f)
        print(f"\nResults saved → {out_path}")

    return results


# ─── Smoothing utility (used by app.py too) ─────────────────────────────────── #

def smooth(values: list, window: int = 20) -> list:
    """
    Simple moving-average smoother for noisy training curves.
    Pads the start with the first value so the output length matches the input.
    """
    if window <= 1 or len(values) < 2:
        return values
    arr     = np.array(values, dtype=float)
    kernel  = np.ones(window) / window
    padded  = np.pad(arr, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid").tolist()


# ─── CLI entry point ─────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DQN agents across infrastructure signal-fidelity sweep"
    )
    parser.add_argument("--episodes", type=int, default=500,
                        help="Training episodes per agent (default: 500)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--out",      type=str, default="results.pkl",
                        help="Output path (default: results.pkl)")
    args = parser.parse_args()

    results = run_infrastructure_sweep(
        n_train_episodes=args.episodes,
        seed=args.seed,
        out_path=args.out,
    )

    # Summary table
    alphas       = results["config"]["alphas"]
    action_names = results["config"]["action_names"]
    w = 26
    col_w = 12

    print(f"\n{'── Final evaluation by infrastructure level':─<80}")
    header = f"{'Metric':<{w}}" + "".join(f"  α={a:.2f}".rjust(col_w) for a in alphas)
    print(header)
    print("─" * 80)
    for label, key in [
        ("Reputation score",   "mean_final_reputation"),
        ("True quality",       "mean_final_true_quality"),
        ("Market health",      "mean_final_market_health"),
        ("Value delivered",    "mean_value_delivered"),
    ]:
        row = f"{label:<{w}}"
        for a in alphas:
            row += f"{results[a]['final_eval'][key]:>{col_w}.3f}"
        print(row)

    print(f"\n{'── Action distribution by infrastructure level':─<80}")
    print(header.replace("Metric", "Action"))
    print("─" * 80)
    for j, name in enumerate(action_names):
        row = f"{name:<{w}}"
        for a in alphas:
            row += f"{results[a]['final_eval']['action_dist'][j]:>{col_w-1}.1%} "
        print(row)
