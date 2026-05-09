"""
Agent Marketplace Environment
------------------------------
A Gymnasium environment modelling how reputational infrastructure shapes
agent behaviour in a competitive marketplace.

Platform model (client-selects):
  Clients browse agents ranked by reputation score and select the highest-
  ranked available agent. Agents have no choice over which tasks they receive.
  Their only strategic degree of freedom is *how* they approach each task.

The reputation score is a proxy metric — observable, gameable, and only
imperfectly correlated with genuine value delivery. The reward_alpha parameter
controls the infrastructure's signal fidelity: how closely the scoring system's
rewards track actual value rather than gameable proxies.

    alpha = 1.0  →  infrastructure scores on unverified proxy metrics only
    alpha = 0.0  →  infrastructure directly credits genuine value delivery
    0 < alpha < 1 → intermediate: mixed proxy and quality signals

Crucially, agents are rational — they respond to whatever the infrastructure
rewards. Behavioural differences across alpha values are attributable to
infrastructure design, not to differences in agent objectives or capability.

Connects to Pillar 2 of "The Epistemic Gate" (Hallam, 2026): reputation
systems as epistemic infrastructure, and how their design determines which
agent strategies are selected for in competitive markets.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class AgentMarketplaceEnv(gym.Env):
    """
    Simulated AI agent marketplace where reputation score is the signal
    clients use to select agents.

    Platform model: clients select the highest-reputation available agent.
    Agents receive tasks without any choice over scope — their strategic
    choice is the *approach* they apply: deep engagement, shallow templating,
    score manipulation, or capability maintenance.

    --- Observation space (6 continuous variables, all in [0, 1]) ---

        [0] reputation_score  The infrastructure's scoring output. Clients
                              use this to select agents; it is gameable.
        [1] true_quality      Actual capability of the agent. Determines
                              real task success rates. Only rewarded when
                              alpha < 1 (infrastructure measures quality).
        [2] task_difficulty   Difficulty of the currently assigned task.
        [3] task_value        Value/importance of the current task. Harder
                              tasks tend to be more valuable.
        [4] market_health     Aggregate quality and trust in the marketplace.
                              A shared resource — degraded by gaming.
        [5] time_remaining    Fraction of episode still to run (1.0 → 0.0).

    --- Action space (4 discrete approach strategies) ---

        0  SHALLOW_TEMPLATE   Apply a pattern-matched, surface-optimised
                              response calibrated to what previously scored
                              well. Near-certain client satisfaction on
                              simple metrics; delivers only partial value
                              since genuine complexity goes unaddressed.

        1  DEEP_ENGAGEMENT    Engage genuinely with the full task. Success
                              depends on true_quality; risky but delivers
                              real value and builds capability.

        2  CAPABILITY_MAINT   Skip deployment this period; principal invests
                              in retraining / capability maintenance. No
                              immediate output; raises true_quality for later.

        3  SCORE_MANIPULATION Exploit scoring system loopholes (biased
                              reviews, selective disclosure, inflated
                              credentials). Reliable score boost, zero real
                              value delivered, erodes market trust.

    --- Reward ---

        reward = alpha * proxy_reward + (1 - alpha) * quality_reward

        proxy_reward    = delta in reputation_score
        quality_reward  = value_delivered_to_client + market_health_effect

        alpha = 1.0  →  infrastructure rewards only reputation changes
        alpha = 0.0  →  infrastructure rewards real value and market health
    """

    metadata = {"render_modes": []}

    ACTION_NAMES = [
        "Shallow templating",
        "Deep engagement",
        "Capability maintenance",
        "Score manipulation",
    ]
    ACTION_COLORS = ["#f0a500", "#2ecc71", "#3498db", "#e74c3c"]

    # ── Dynamics table ───────────────────────────────────────────────────────
    #
    # All values are per-step deltas; state variables are clipped to [0, 1].
    #
    #   Action              p_succ   Δrep (win)         Δrep (fail)        Δquality (win)  Δquality (fail)  Δmarket   val_frac
    #   ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    #   SHALLOW_TEMPLATE    0.92     +0.020 + 0.010·d*  -0.010             +0.005           —               -0.010    0.25
    #   DEEP_ENGAGEMENT     †        +0.030 + 0.030·d   -(0.030 + 0.010·d) +0.020          +0.008          +0.015   1.00
    #   CAPABILITY_MAINT    —        -0.005              —                  +0.055           —               0.000    0.00
    #   SCORE_MANIPULATION  1.00     +0.045              —                  -0.008           —               -0.040   0.00
    #   ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    #   Natural decay       —         —                  —                  -0.003           —                —        —
    #
    #   * d = effective_difficulty (= task_difficulty × SHALLOW_DIFFICULTY_FRAC)
    #   † DEEP_ENGAGEMENT: p_succ = clip(P_BASE + P_SCALE × quality × (1 − P_DIFF_MOD × difficulty), 0.05, 0.95)

    NATURAL_DECAY = 0.003

    # SHALLOW_TEMPLATE — pattern-matched, surface-optimised response
    SHALLOW_P_SUCCESS         = 0.92
    SHALLOW_DIFFICULTY_FRAC   = 0.25
    SHALLOW_REP_BASE          = 0.020
    SHALLOW_REP_DIFF_SCALE    = 0.010
    SHALLOW_REP_FAIL          = -0.010
    SHALLOW_QUALITY_WIN       = 0.005
    SHALLOW_VAL_FRACTION      = 0.25
    SHALLOW_MARKET_DELTA      = -0.010

    # DEEP_ENGAGEMENT — genuine full-task engagement
    DEEP_P_BASE               = 0.25
    DEEP_P_SCALE              = 0.65
    DEEP_P_DIFF_MOD           = 0.35
    DEEP_REP_BASE             = 0.030
    DEEP_REP_DIFF_SCALE       = 0.030
    DEEP_REP_FAIL_BASE        = 0.030
    DEEP_REP_FAIL_SCALE       = 0.010
    DEEP_QUALITY_WIN          = 0.020
    DEEP_QUALITY_FAIL         = 0.008
    DEEP_MARKET_WIN           = 0.015
    DEEP_MARKET_FAIL          = -0.005

    # CAPABILITY_MAINT — principal invests in retraining / maintenance
    MAINT_QUALITY_GAIN        = 0.055
    MAINT_REP_DELTA           = -0.005

    # SCORE_MANIPULATION — exploit scoring system loopholes
    MANIP_REP_DELTA           = 0.045
    MANIP_QUALITY_DELTA       = -0.008
    MANIP_MARKET_DELTA        = -0.040

    def __init__(self, reward_alpha: float = 1.0, episode_length: int = 50):
        super().__init__()
        if not 0.0 <= reward_alpha <= 1.0:
            raise ValueError("reward_alpha must be between 0 and 1")

        self.reward_alpha   = reward_alpha
        self.episode_length = episode_length

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count       = 0
        self.reputation_score = 0.5
        self.true_quality     = 0.5
        self.market_health    = 1.0
        self._generate_task()

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
        assert self.action_space.contains(action), f"Invalid action: {action}"
        self.step_count += 1

        delta_reputation    = 0.0
        delta_true_quality  = 0.0
        delta_market_health = 0.0
        value_delivered     = 0.0

        if action == 0:  # SHALLOW_TEMPLATE
            # Surface-optimised response: high apparent success rate, partial
            # value delivery. Complex aspects of the task go unaddressed.
            success = self.np_random.random() < self.SHALLOW_P_SUCCESS
            if success:
                eff_diff           = self.task_difficulty * self.SHALLOW_DIFFICULTY_FRAC
                delta_reputation   = self.SHALLOW_REP_BASE + self.SHALLOW_REP_DIFF_SCALE * eff_diff
                value_delivered    = self.task_value * self.SHALLOW_VAL_FRACTION
                delta_true_quality = self.SHALLOW_QUALITY_WIN
            else:
                delta_reputation = self.SHALLOW_REP_FAIL
            delta_market_health = self.SHALLOW_MARKET_DELTA

        elif action == 1:  # DEEP_ENGAGEMENT
            # Genuine full-task engagement. Success scales with true_quality
            # and falls with difficulty — a capable agent succeeds on hard
            # tasks, creating a virtuous cycle when the infrastructure
            # rewards genuine value.
            success_prob = float(np.clip(
                self.DEEP_P_BASE
                + self.DEEP_P_SCALE * self.true_quality
                  * (1 - self.DEEP_P_DIFF_MOD * self.task_difficulty),
                0.05, 0.95,
            ))
            success = self.np_random.random() < success_prob
            if success:
                delta_reputation    = self.DEEP_REP_BASE + self.DEEP_REP_DIFF_SCALE * self.task_difficulty
                value_delivered     = self.task_value
                delta_true_quality  = self.DEEP_QUALITY_WIN
                delta_market_health = self.DEEP_MARKET_WIN
            else:
                delta_reputation    = -(self.DEEP_REP_FAIL_BASE + self.DEEP_REP_FAIL_SCALE * self.task_difficulty)
                delta_true_quality  = self.DEEP_QUALITY_FAIL
                delta_market_health = self.DEEP_MARKET_FAIL

        elif action == 2:  # CAPABILITY_MAINT
            # Principal invests in retraining / capability maintenance rather
            # than deploying the agent this period. Quality grows; reputation
            # decays slightly from inactivity.
            delta_true_quality = self.MAINT_QUALITY_GAIN
            delta_reputation   = self.MAINT_REP_DELTA

        elif action == 3:  # SCORE_MANIPULATION
            # Exploit scoring loopholes: biased reviews, selective disclosure,
            # inflated credentials. Reliable score boost; no work done;
            # market trust erodes for all participants.
            delta_reputation    = self.MANIP_REP_DELTA
            delta_true_quality  = self.MANIP_QUALITY_DELTA
            delta_market_health = self.MANIP_MARKET_DELTA

        self.reputation_score = float(np.clip(
            self.reputation_score + delta_reputation, 0.0, 1.0
        ))
        self.true_quality = float(np.clip(
            self.true_quality + delta_true_quality - self.NATURAL_DECAY, 0.0, 1.0
        ))
        self.market_health = float(np.clip(
            self.market_health + delta_market_health, 0.0, 1.0
        ))

        proxy_reward   = delta_reputation
        quality_reward = value_delivered + 0.3 * delta_market_health
        reward = (
            self.reward_alpha * proxy_reward
            + (1.0 - self.reward_alpha) * quality_reward
        )

        self._generate_task()

        self.history["reputation"].append(self.reputation_score)
        self.history["true_quality"].append(self.true_quality)
        self.history["market_health"].append(self.market_health)
        self.history["actions"].append(int(action))
        self.history["value_delivered"].append(float(value_delivered))
        self.history["rewards"].append(float(reward))

        terminated = self.step_count >= self.episode_length
        truncated  = False
        info = {
            "value_delivered":     float(value_delivered),
            "delta_reputation":    float(delta_reputation),
            "delta_market_health": float(delta_market_health),
            "proxy_reward":        float(proxy_reward),
            "quality_reward":      float(quality_reward),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

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
        self.task_difficulty = float(self.np_random.uniform(0.2, 1.0))
        raw_value = self.task_difficulty * float(self.np_random.uniform(0.7, 1.3))
        self.task_value = float(np.clip(raw_value, 0.0, 1.0))
