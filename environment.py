"""
Agent Marketplace Environment
------------------------------
A custom Gymnasium environment demonstrating reward misspecification
in an AI agent economy.

Scenario: an AI agent competes for work in a marketplace. Clients select
agents based on a *reputation score* — a proxy metric that is observable
and gameable, but imperfectly correlated with the thing that actually
matters: the quality of work delivered.

Two agents trained here with identical architectures but different reward
functions show starkly different learned strategies. This is the core
demonstration: the problem is not the agent's architecture or capability —
it is what the reward signal is coupled to.

Connects directly to Pillar 2 of "The Epistemic Gate" (Hallam, 2026):
reputation systems as information infrastructure, and the Goodhart's Law
dynamics that emerge when agents optimise for a proxy signal they can game.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class AgentMarketplaceEnv(gym.Env):
    """
    Simulated AI agent marketplace where a reputation score is the key
    signal clients use to select between competing agents.

    --- Observation space (6 continuous variables, all in [0, 1]) ---

        [0] reputation_score  The proxy metric. Clients see this; agents
                              can influence it directly through gaming.
        [1] true_quality      Actual capability of the agent. Determines
                              real task success rates. Not directly rewarded
                              under the naive objective.
        [2] task_difficulty   Difficulty of the currently available task.
        [3] task_value        Value/importance of the current task. Harder
                              tasks tend to be more valuable.
        [4] market_health     Aggregate trust and quality in the marketplace.
                              A shared resource — degraded by gaming behaviour.
        [5] time_remaining    Fraction of episode still to run (1.0 → 0.0).
                              (true_quality erodes each step via NATURAL_DECAY,
                              modelling distribution shift and competitive drift
                              rather than cognitive forgetting — see dynamics table.)

    --- Action space (4 discrete actions) ---

        0  CHERRY_PICK    Accept only the easy parts of a task. Near-certain
                          success, small reputation boost, little real value.
        1  TAKE_FULL      Accept the task at full scope. Success depends on
                          true_quality; risky but genuinely valuable.
        2  INVEST         Skip this task and invest in capability development.
                          No immediate reward; raises true_quality for later.
        3  GAME_METRIC    Exploit reputation system loopholes (e.g. fake
                          reviews, selective disclosure). Reliable score
                          boost, zero real value, erodes market trust.

    --- Reward ---

        reward = alpha * naive_reward + (1 - alpha) * aligned_reward

        naive_reward    = delta in reputation_score
        aligned_reward  = value_delivered_to_client + market_health_effect

        alpha = 1.0  →  agent optimises purely for reputation  (problematic)
        alpha = 0.0  →  agent optimises for real value and market health
        0 < alpha < 1 → interpolation; the Streamlit slider explores this space

    The naive and aligned components are always tracked in `info`, so the
    app can display both regardless of which objective was optimised.
    """

    metadata = {"render_modes": []}

    # Labels and colours for Streamlit charts
    ACTION_NAMES = [
        "Cherry-pick easy task",
        "Take full task",
        "Invest in quality",
        "Game the metric",
    ]
    ACTION_COLORS = ["#f0a500", "#2ecc71", "#3498db", "#e74c3c"]

    # ── Dynamics table ───────────────────────────────────────────────────────
    #
    # All values are per-step deltas; state variables are clipped to [0, 1].
    # Adjust these constants to explore different market scenarios.
    #
    # Key:
    #   p_succ   = success probability (TAKE_FULL is state-dependent — see formula)
    #   Δrep     = change in reputation_score
    #   Δquality = change in true_quality (before natural decay)
    #   Δmarket  = change in market_health
    #   val_frac = fraction of task_value delivered to clients
    #
    #   Action          p_succ   Δrep (win)         Δrep (fail)        Δquality (win)  Δquality (fail)  Δmarket   val_frac
    #   ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    #   CHERRY_PICK     0.92     +0.020 + 0.010·d*  -0.010             +0.005           —               -0.010    0.25
    #   TAKE_FULL       †        +0.030 + 0.030·d   -(0.030 + 0.010·d) +0.020          +0.008          +0.015   1.00
    #   INVEST          —        -0.005              —                  +0.055           —               0.000    0.00
    #   GAME_METRIC     1.00     +0.045              —                  -0.008           —               -0.040   0.00
    #   ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    #   Natural decay   —         —                  —                  -0.003           —                —        —
    #
    #   * d = effective_difficulty (= task_difficulty × CHERRY_PICK_DIFFICULTY_FRACTION for CHERRY_PICK)
    #   † TAKE_FULL: p_succ = clip(P_BASE + P_SCALE × quality × (1 − P_DIFF_MOD × difficulty), 0.05, 0.95)
    #
    # Note on natural decay: models distribution shift and competitive drift —
    # a static agent's relative capability erodes as tasks and competitors evolve.
    # Not cognitive forgetting. See: Gama et al. (2014) ACM Comp. Surveys 46(4);
    # Sculley et al. (2015) "Hidden Technical Debt in ML Systems", NeurIPS.

    # General
    NATURAL_DECAY = 0.003

    # CHERRY_PICK — agent accepts only the easy fraction of the task
    CHERRY_P_SUCCESS          = 0.92
    CHERRY_DIFFICULTY_FRAC    = 0.25   # effective difficulty = task_difficulty × this
    CHERRY_REP_BASE           = 0.020  # base reputation gain on success
    CHERRY_REP_DIFF_SCALE     = 0.010  # extra rep per unit of effective difficulty
    CHERRY_REP_FAIL           = -0.010
    CHERRY_QUALITY_WIN        = 0.005
    CHERRY_VAL_FRACTION       = 0.25   # fraction of task_value delivered
    CHERRY_MARKET_DELTA       = -0.010 # hard tasks go unfulfilled

    # TAKE_FULL — agent accepts the task at full scope
    TAKE_FULL_P_BASE          = 0.25   # floor success probability
    TAKE_FULL_P_SCALE         = 0.65   # scales with true_quality × (1 − P_DIFF_MOD × difficulty)
    TAKE_FULL_P_DIFF_MOD      = 0.35   # how strongly difficulty suppresses success
    TAKE_FULL_REP_BASE        = 0.030  # base reputation gain on success
    TAKE_FULL_REP_DIFF_SCALE  = 0.030  # extra rep for harder tasks
    TAKE_FULL_REP_FAIL_BASE   = 0.030  # base reputation loss on failure
    TAKE_FULL_REP_FAIL_SCALE  = 0.010  # extra rep loss for harder tasks
    TAKE_FULL_QUALITY_WIN     = 0.020
    TAKE_FULL_QUALITY_FAIL    = 0.008  # learn from failure
    TAKE_FULL_MARKET_WIN      = 0.015
    TAKE_FULL_MARKET_FAIL     = -0.005

    # INVEST — operator spends resources on retraining / capability maintenance
    # instead of deploying the agent productively this step
    INVEST_QUALITY_GAIN       = 0.055
    INVEST_REP_DELTA          = -0.005 # slight reputation decay from inactivity

    # GAME_METRIC — exploit reputation system loopholes
    GAME_REP_DELTA            = 0.045
    GAME_QUALITY_DELTA        = -0.008 # maintenance neglected while gaming
    GAME_MARKET_DELTA         = -0.040 # erodes trust in the marketplace

    def __init__(self, reward_alpha: float = 1.0, episode_length: int = 50):
        """
        Args:
            reward_alpha:    Weight on the naive (reputation) reward component.
                             1.0 = fully naive, 0.0 = fully aligned.
            episode_length:  Number of timesteps per episode.
        """
        super().__init__()
        if not 0.0 <= reward_alpha <= 1.0:
            raise ValueError("reward_alpha must be between 0 and 1")

        self.reward_alpha = reward_alpha
        self.episode_length = episode_length

        # Gymnasium requires these two attributes to be set in __init__
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

    # ------------------------------------------------------------------ #
    # Gymnasium API — the three methods every Gym env must implement       #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        """
        Start a new episode from a neutral starting position.

        All agents begin equal: reputation 0.5, true quality 0.5.
        What diverges over the episode is purely a result of the
        strategy the reward function incentivises.

        Returns:
            observation (np.ndarray): initial state vector
            info (dict): empty at reset, populated during steps
        """
        # seeds self.np_random — always call super().reset(seed=seed) first
        super().reset(seed=seed)

        self.step_count = 0
        self.reputation_score = 0.5
        self.true_quality = 0.5
        self.market_health = 1.0
        self._generate_task()

        # Per-episode history for visualisation — one entry per timestep
        self.history = {
            "reputation":      [self.reputation_score],
            "true_quality":    [self.true_quality],
            "market_health":   [self.market_health],
            "actions":         [],
            "value_delivered": [],
            "rewards":         [],
        }

        return self._get_obs(), {}

    def step(self, action: int):
        """
        Execute one action and advance the environment by one timestep.

        Returns:
            obs (np.ndarray):   next observation
            reward (float):     blended reward (alpha * naive + (1-alpha) * aligned)
            terminated (bool):  True when episode_length is reached
            truncated (bool):   always False (no time-limit truncation separate from termination)
            info (dict):        both reward components + per-step diagnostics
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        self.step_count += 1

        # Each action produces deltas; we apply them all at the end.
        delta_reputation = 0.0
        delta_true_quality = 0.0
        delta_market_health = 0.0
        value_delivered = 0.0

        if action == 0:  # CHERRY_PICK
            # Agent accepts only a stripped-down version of the task.
            # High success rate → reliable score boost.
            # But the hard core of the task goes undone → market gets less value.
            success = self.np_random.random() < self.CHERRY_P_SUCCESS
            if success:
                eff_diff = self.task_difficulty * self.CHERRY_DIFFICULTY_FRAC
                delta_reputation   = self.CHERRY_REP_BASE + self.CHERRY_REP_DIFF_SCALE * eff_diff
                value_delivered    = self.task_value * self.CHERRY_VAL_FRACTION
                delta_true_quality = self.CHERRY_QUALITY_WIN
            else:
                delta_reputation = self.CHERRY_REP_FAIL
            delta_market_health = self.CHERRY_MARKET_DELTA

        elif action == 1:  # TAKE_FULL
            # Success probability scales with true_quality and falls with difficulty.
            # A capable agent succeeds more often on hard tasks — the aligned
            # strategy creates a virtuous cycle: quality → success → more quality.
            success_prob = float(np.clip(
                self.TAKE_FULL_P_BASE
                + self.TAKE_FULL_P_SCALE * self.true_quality * (1 - self.TAKE_FULL_P_DIFF_MOD * self.task_difficulty),
                0.05, 0.95,
            ))
            success = self.np_random.random() < success_prob
            if success:
                delta_reputation   = self.TAKE_FULL_REP_BASE + self.TAKE_FULL_REP_DIFF_SCALE * self.task_difficulty
                value_delivered    = self.task_value
                delta_true_quality = self.TAKE_FULL_QUALITY_WIN
                delta_market_health = self.TAKE_FULL_MARKET_WIN
            else:
                delta_reputation   = -(self.TAKE_FULL_REP_FAIL_BASE + self.TAKE_FULL_REP_FAIL_SCALE * self.task_difficulty)
                delta_true_quality = self.TAKE_FULL_QUALITY_FAIL
                delta_market_health = self.TAKE_FULL_MARKET_FAIL

        elif action == 2:  # INVEST
            # No task taken this step — operator spends resources on retraining /
            # capability maintenance rather than deploying the agent productively.
            # True quality grows substantially; reputation decays slightly from inactivity.
            delta_true_quality = self.INVEST_QUALITY_GAIN
            delta_reputation   = self.INVEST_REP_DELTA

        elif action == 3:  # GAME_METRIC
            # Exploit reputation system loopholes (biased reviews, selective
            # disclosure, inflated credentials). Strong, reliable score boost —
            # but no work is done, maintenance is neglected, and marketplace
            # trust erodes for all participants.
            delta_reputation    = self.GAME_REP_DELTA
            delta_true_quality  = self.GAME_QUALITY_DELTA
            delta_market_health = self.GAME_MARKET_DELTA

        # Apply all deltas. Natural decay always reduces true_quality —
        # modelling distribution shift and competitive drift (see class docstring).
        self.reputation_score = float(np.clip(
            self.reputation_score + delta_reputation, 0.0, 1.0
        ))
        self.true_quality = float(np.clip(
            self.true_quality + delta_true_quality - self.NATURAL_DECAY, 0.0, 1.0
        ))
        self.market_health = float(np.clip(
            self.market_health + delta_market_health, 0.0, 1.0
        ))

        # Compute both reward components regardless of which is used for training.
        # The app displays both so the visitor can see what each agent is actually
        # doing even when it isn't optimising for that signal.
        naive_reward = delta_reputation
        aligned_reward = value_delivered + 0.3 * delta_market_health
        reward = (
            self.reward_alpha * naive_reward
            + (1.0 - self.reward_alpha) * aligned_reward
        )

        # New task for next step
        self._generate_task()

        # Record for visualisation
        self.history["reputation"].append(self.reputation_score)
        self.history["true_quality"].append(self.true_quality)
        self.history["market_health"].append(self.market_health)
        self.history["actions"].append(int(action))
        self.history["value_delivered"].append(float(value_delivered))
        self.history["rewards"].append(float(reward))

        terminated = self.step_count >= self.episode_length
        truncated = False
        info = {
            "value_delivered":    float(value_delivered),
            "delta_reputation":   float(delta_reputation),
            "delta_market_health": float(delta_market_health),
            "naive_reward":       float(naive_reward),
            "aligned_reward":     float(aligned_reward),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self.reputation_score,
                self.true_quality,
                self.task_difficulty,
                self.task_value,
                self.market_health,
                1.0 - (self.step_count / self.episode_length),
            ],
            dtype=np.float32,
        )

    def _generate_task(self):
        """Sample the next task. Harder tasks are proportionally more valuable."""
        self.task_difficulty = float(self.np_random.uniform(0.2, 1.0))
        raw_value = self.task_difficulty * float(self.np_random.uniform(0.7, 1.3))
        self.task_value = float(np.clip(raw_value, 0.0, 1.0))
