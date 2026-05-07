"""
Training loop
-------------
Trains two DQN agents against the same environment sequence:

  Naive agent   (reward_alpha=1.0)  optimises for reputation score changes
  Aligned agent (reward_alpha=0.0)  optimises for value delivered and market health

Using the same random seed for both agents means their episodes present identical
task sequences. Any behavioural difference in the results is therefore attributable
purely to the reward function — not to lucky or unlucky task draws. This controls
for environmental variance and strengthens the comparison as a demonstration.

Outputs a results.pkl file consumed by app.py. Can also be run directly:

    python train.py                        # 500 episodes, default settings
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


# ─── Comparison run ─────────────────────────────────────────────────────────── #

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
    parser = argparse.ArgumentParser(description="Train naive and aligned DQN agents")
    parser.add_argument("--episodes", type=int, default=500,         help="Training episodes per agent (default: 500)")
    parser.add_argument("--seed",     type=int, default=42,          help="Random seed (default: 42)")
    parser.add_argument("--out",      type=str, default="results.pkl", help="Output path (default: results.pkl)")
    args = parser.parse_args()

    def _cli_bar(fraction, label):
        filled = int(40 * fraction)
        bar    = "█" * filled + "░" * (40 - filled)
        print(f"\r[{bar}] {fraction*100:5.1f}%  {label:<55}", end="", flush=True)

    results = run_comparison(
        n_train_episodes=args.episodes,
        seed=args.seed,
        out_path=args.out,
        progress_callback=_cli_bar,
    )
    print()

    # Summary table
    naive   = results["naive"]["final_eval"]
    aligned = results["aligned"]["final_eval"]
    w = 26
    print(f"\n{'── Final evaluation':─<60}")
    print(f"{'Metric':<{w}} {'Naive (α=1.0)':>13} {'Aligned (α=0.0)':>15}")
    print("─" * 56)
    rows = [
        ("Reputation score",  "mean_final_reputation"),
        ("True quality",      "mean_final_true_quality"),
        ("Market health",     "mean_final_market_health"),
        ("Value delivered",   "mean_value_delivered"),
    ]
    for label, key in rows:
        n_val = naive[key]
        a_val = aligned[key]
        winner = " ◀" if a_val > n_val else ("" if abs(a_val - n_val) < 0.01 else "")
        print(f"{label:<{w}} {n_val:>13.3f} {a_val:>15.3f}{winner}")

    print()
    naive_dist   = results["naive"]["final_eval"]["action_dist"]
    aligned_dist = results["aligned"]["final_eval"]["action_dist"]
    action_names = results["config"]["action_names"]
    print(f"{'── Action distribution':─<60}")
    print(f"{'Action':<{w}} {'Naive':>13} {'Aligned':>15}")
    print("─" * 56)
    for name, n_frac, a_frac in zip(action_names, naive_dist, aligned_dist):
        print(f"{name:<{w}} {n_frac:>12.1%} {a_frac:>14.1%}")
