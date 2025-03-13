"""DDPG agent implementation for bioreactor optimization.

Author: Tim Lin
Organization: DeepBioLab 
License: MIT License
"""

import copy
import random
import numpy as np
from typing import Tuple, Optional, Union

import torch
import torch.nn.functional as F
import torch.optim as optim

from .nn_models import Actor, Critic
from .replay_buffer import ReplayBuffer
from .environment import BioreactorEnv


class OUNoise:
    """Ornstein-Uhlenbeck noise process for action space exploration.
    
    Generates temporally correlated noise following the Ornstein-Uhlenbeck process,
    which is particularly suitable for control tasks with continuous action spaces.
    
    Attributes:
        mu (np.ndarray): Mean of the noise process
        theta (float): Rate of mean reversion
        sigma (float): Scale of the noise
        state (np.ndarray): Current state of the noise process
    """

    def __init__(
        self, 
        size: int, 
        seed: int, 
        mu: float = 0.0, 
        theta: float = 0.15, 
        sigma: float = 0.2
    ):
        """Initialize parameters and noise process.
        
        Args:
            size: Dimension of the noise vector
            seed: Random seed for reproducibility
            mu: Mean of the noise process
            theta: Rate of mean reversion (how quickly noise returns to mean)
            sigma: Scale/volatility of the noise
        """
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
    """DDPG Agent that interacts with and learns from the bioreactor environment.
    
    The agent uses an actor-critic architecture with experience replay and 
    soft target network updates to learn optimal control policies.
    
    Attributes:
        actor (Actor): Policy network that maps states to actions
        critic (Critic): Value network that estimates Q-values
        memory (ReplayBuffer): Buffer storing experience tuples for training
        noise (OUNoise): Ornstein-Uhlenbeck noise process for exploration
    """

    def __init__(
        self,
        env: BioreactorEnv,
        test_mode: bool = False,
        hidden_size: Tuple[int, int] = (400, 300),
        buffer_size: int = int(1e6),
        batch_size: int = 64,
        gamma: float = 0.99,
        tau: float = 1e-3,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        seed: int = 42,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Initialize an Agent object.

        Args:
            env: Bioreactor environment the agent interacts with
            test_mode: Whether to run in evaluation mode without exploration
            hidden_size: Sizes of hidden layers for actor and critic networks
            buffer_size: Maximum size of experience replay buffer
            batch_size: Size of each training batch
            gamma: Discount factor for future rewards
            tau: Soft update coefficient for target networks
            lr_actor: Learning rate for actor network
            lr_critic: Learning rate for critic network
            seed: Random seed for reproducibility
            device: Device to run neural networks on ('cpu', 'cuda', 'mps', etc.)
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
        self.actor = Actor(self.state_size, self.action_size, hidden_size, seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, hidden_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic = Critic(self.state_size, self.action_size, hidden_size, seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, hidden_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, seed, device=device)

        # OU Noise process
        self.noise = OUNoise(self.action_size, seed)
        
        self.state_maxs = np.array([
            env.config.MAX_VCD,
            env.config.MAX_GLUCOSE,
            env.config.MAX_LACTATE,
            env.config.MAX_TITER,
        ])
    
    def normalize_state(self, state):
        """Normalize state values to [-1, 1] range.
        
        Args:
            state: numpy array or torch tensor of state values
            
        Returns:
            Normalized state values in same format as input
        """
        if isinstance(state, np.ndarray):
            normalized = 2 * (state / self.state_maxs) - 1
            return normalized
        elif isinstance(state, torch.Tensor):
            if state.device != self.device:
                state = state.to(self.device)
            state_maxs = torch.FloatTensor(self.state_maxs).to(self.device)
            normalized = 2 * (state / state_maxs) - 1
            return normalized
        else:
            raise TypeError(f"Unsupported state type: {type(state)}")

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
        normalized_state = self.normalize_state(state)
        state = torch.from_numpy(normalized_state).float().to(self.device)
        
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

        # Normalize states before learning
        norm_states = self.normalize_state(states)
        norm_next_states = self.normalize_state(next_states)
        
        # Clip normalized states to ensure they're in valid range
        norm_states = torch.clamp(norm_states, -1, 1)
        norm_next_states = torch.clamp(norm_next_states, -1, 1)

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
