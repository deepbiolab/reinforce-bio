import copy
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from .nn_models import Actor, Critic
from .replay_buffer import ReplayBuffer


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for i in range(len(x))]
        )
        self.state = x + dx
        return self.state


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(
        self,
        env,
        test_mode=False,
        hidden_size=(400, 300),
        buffer_size=int(1e6),
        batch_size=64,
        gamma=0.99,
        tau=1e-3,
        lr_actor=1e-4,
        lr_critic=1e-3,
        seed=42,
        device=None,
    ):
        """Initialize an Agent object.

        Params
        ======
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            lr (float): learning rate
            gamma (float): discount factor
            tau (float): soft update of target parameters
            update_step (int): how often to update the network
            seed (int): random seed

        """
        self.device = torch.device(device)
        self.seed = random.seed(seed)
        self.test_mode = test_mode

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]

        # Actor Network (w/ Target Network)
        self.actor = Actor(self.state_size, self.action_size, hidden_size, seed).to(
            device
        )
        self.actor_target = Actor(
            self.state_size, self.action_size, hidden_size, seed
        ).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic = Critic(self.state_size, self.action_size, hidden_size, seed).to(
            device
        )
        self.critic_target = Critic(
            self.state_size, self.action_size, hidden_size, seed
        ).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # Replay memory
        self.memory = ReplayBuffer(
            self.action_size, self.buffer_size, self.batch_size, seed, device=device
        )

        # OU Noise process
        self.noise = OUNoise(self.action_size, seed)
        

    def __repr__(self):
        return (
            f"Actor Network Arch: {self.actor}\n"
            f"Critic Network Arch: {self.critic}\n"
            f"State space size: {self.state_size}\n"
            f"Action space size: {self.action_size}\n"
            f"Current Memory size: {len(self.memory)}"
        )

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            critic_loss, actor_loss = self.learn(experiences)
            return critic_loss, actor_loss

    def select_action(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()

        if self.test_mode:
            return np.clip(action, -1, 1)

        self.actor.train()

        # Add Exploration of action selection
        action = action + self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        # get experiences
        states, actions, rewards, next_states, dones = experiences

        # ---------------------- update critic ----------------------#
        # compute target values using target network
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (1 - dones) * self.gamma * Q_targets_next

        # compute curr values using local network
        Q_expected = self.critic(states, actions)

        # compute mean squared loss using td error
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------- update actor ----------------------#
        # compute actor loss using local network
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target network parameters
        self.soft_update()

        return critic_loss, actor_loss

    def soft_update(self):
        """Soft update model parameters.
        θ_target = τ*θ + (1 - τ)*θ_target
        =>
        θ_target = θ_target + τ*(θ - θ_target)
        """
        for target_param, local_param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                target_param.data + self.tau * (local_param.data - target_param.data)
            )
        for target_param, local_param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.copy_(
                target_param.data + self.tau * (local_param.data - target_param.data)
            )

    def save(self, filename):
        """Save model parameters."""
        torch.save(
            {
                "actor_state_dict": self.actor.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
            },
            filename,
        )

    def load(self, filename):
        """Load model parameters."""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
