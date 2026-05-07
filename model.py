"""
DQN Agent
---------
Deep Q-Network implementation for the Agent Marketplace environment.

The notebook covers vanilla Q-learning. Two additions make DQN stable enough
to actually learn from a continuous state space:

  Experience replay buffer
    Consecutive transitions in an episode are highly correlated — training
    on them sequentially produces biased gradient updates and causes the
    network to "forget" earlier experience. The buffer stores past transitions
    and we sample randomly from it for each update, breaking the temporal
    correlation and allowing every transition to be replayed many times.

  Target network
    The DQN loss target is  y = r + γ · max_a' Q(s', a').  If both sides
    of that equation use the same network, the target shifts every gradient
    step — we are chasing a moving target, which causes oscillation and
    divergence. A second, frozen copy (the target network) provides stable
    targets; it is hard-synced to the online network every TARGET_UPDATE_FREQ
    gradient steps.

Architecture: 6 → 64 → 64 → 4   (two hidden layers, ReLU activations)
The state space is small enough that a modest network converges quickly
on a laptop CPU.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ─── Neural network ────────────────────────────────────────────────────────── #

class DQN(nn.Module):
    """
    Simple feedforward Q-network.

    Input:  state vector  (obs_dim,)
    Output: Q-value for each action  (n_actions,)

    The network learns to predict how much total discounted reward the agent
    will accumulate from state s if it takes action a and then acts optimally.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── Experience replay buffer ──────────────────────────────────────────────── #

class ReplayBuffer:
    """
    Fixed-capacity circular buffer of (s, a, r, s', done) transitions.

    Once full, new transitions overwrite the oldest ones. Random sampling
    breaks temporal correlations and lets each transition train the network
    multiple times — significantly improving data efficiency.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ):
        self.buffer.append((state, int(action), float(reward), next_state, float(terminated)))

    def sample(self, batch_size: int):
        """
        Draw a random minibatch and return stacked tensors.
        Stacking into arrays before converting to tensors is much faster
        than stacking tensors one at a time.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminateds = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(terminateds)),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ─── Agent ─────────────────────────────────────────────────────────────────── #

class DQNAgent:
    """
    DQN agent with experience replay and a target network.

    Wraps two DQN instances (online and target), a replay buffer, and the
    training update. The environment is kept separate — this class only
    knows about (state, action, reward, next_state, terminated) tuples.

    Typical usage in a training loop:

        agent = DQNAgent(obs_dim=6, n_actions=4)
        obs, _ = env.reset()

        while True:
            action          = agent.select_action(obs)
            next_obs, r, terminated, truncated, _ = env.step(action)
            agent.store(obs, action, r, next_obs, terminated)
            agent.update()
            obs = next_obs if not (terminated or truncated) else env.reset()[0]
    """

    # ── Hyperparameters ─────────────────────────────────────────────────────
    #
    #  Parameter             Default   Notes
    #  ────────────────────────────────────────────────────────────────────
    #  HIDDEN_SIZE           64        More than adequate for 6-dim state
    #  BUFFER_CAPACITY       10_000    ~200 complete episodes
    #  BATCH_SIZE            64        Balances gradient variance and speed
    #  GAMMA                 0.95      Moderate discount over 50-step episodes
    #  LR                    1e-3      Adam; fast convergence on small nets
    #  EPSILON_START         1.00      Start with pure exploration
    #  EPSILON_MIN           0.05      Floor: always keep 5% randomness
    #  EPSILON_DECAY         0.995     Per gradient-step multiplicative decay
    #  TARGET_UPDATE_FREQ    100       Hard sync every 100 gradient steps
    #  ────────────────────────────────────────────────────────────────────

    HIDDEN_SIZE         = 64
    BUFFER_CAPACITY     = 10_000
    BATCH_SIZE          = 64
    GAMMA               = 0.95
    LR                  = 1e-3
    EPSILON_START       = 1.0
    EPSILON_MIN         = 0.05
    EPSILON_DECAY       = 0.995
    TARGET_UPDATE_FREQ  = 100

    def __init__(
        self,
        obs_dim: int = 6,
        n_actions: int = 4,
        hidden_size: int = HIDDEN_SIZE,
        gamma: float = GAMMA,
        lr: float = LR,
        epsilon_start: float = EPSILON_START,
        epsilon_min: float = EPSILON_MIN,
        epsilon_decay: float = EPSILON_DECAY,
        buffer_capacity: int = BUFFER_CAPACITY,
        batch_size: int = BATCH_SIZE,
        target_update_freq: int = TARGET_UPDATE_FREQ,
        seed: int = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        self.n_actions          = n_actions
        self.gamma              = gamma
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_min        = epsilon_min
        self.epsilon_decay      = epsilon_decay

        # Online network: trained every step.
        # Target network: frozen copy; synced periodically to stabilise targets.
        self.online_net = DQN(obs_dim, n_actions, hidden_size)
        self.target_net = DQN(obs_dim, n_actions, hidden_size)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)

        self.epsilon    = epsilon_start
        self.steps_done = 0
        self.losses     = []

    # ── Core API ────────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.

        Exploration (probability epsilon):  random action
        Exploitation (probability 1-epsilon): argmax Q(s, a)

        Epsilon decays toward epsilon_min each time update() is called,
        so the agent gradually shifts from exploring to exploiting as it
        accumulates experience.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q = self.online_net(torch.FloatTensor(state).unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ):
        """Add a transition to the replay buffer."""
        self.buffer.push(state, action, reward, next_state, terminated)

    def update(self) -> float | None:
        """
        Sample a minibatch and perform one gradient step on the online network.

        Returns the scalar loss, or None if the buffer hasn't filled a
        full batch yet (the agent collects experience before it starts learning).

        The Bellman target for each transition is:
            y = r + γ · max_a' Q_target(s', a')    if not terminal
            y = r                                    if terminal

        Loss = MSE(Q_online(s, a), y)

        The (1 - terminated) mask zeroes out the bootstrap term for terminal
        transitions — there is no next state to value when the episode ends.
        """
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, terminateds = self.buffer.sample(
            self.batch_size
        )

        # Q-values for the actions the agent actually took: shape (batch,)
        q_pred = (
            self.online_net(states)
            .gather(1, actions.unsqueeze(1))
            .squeeze(1)
        )

        # Bellman targets from the frozen target network: shape (batch,)
        with torch.no_grad():
            q_next_max = self.target_net(next_states).max(dim=1)[0]
            q_target   = rewards + self.gamma * q_next_max * (1.0 - terminateds)

        loss = F.mse_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents the rare large update from destabilising
        # early training when Q-value estimates are far from their true values.
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.steps_done += 1
        self.losses.append(loss.item())

        # Decay epsilon after each gradient step
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Hard sync target ← online every TARGET_UPDATE_FREQ steps
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Return Q-values for all actions in the given state.
        Used by the Streamlit app to visualise what the agent currently values.
        """
        with torch.no_grad():
            return (
                self.online_net(torch.FloatTensor(state).unsqueeze(0))
                .squeeze(0)
                .numpy()
            )

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save(
            {
                "online_net":  self.online_net.state_dict(),
                "target_net":  self.target_net.state_dict(),
                "optimizer":   self.optimizer.state_dict(),
                "epsilon":     self.epsilon,
                "steps_done":  self.steps_done,
                "losses":      self.losses,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, weights_only=True)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon    = ckpt["epsilon"]
        self.steps_done = ckpt["steps_done"]
        self.losses     = ckpt["losses"]
