"""
DQN Agent Component
Manages learning, experience replay, and action selection
"""

import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from collections import deque
from .network import QNetwork


class DQNAgent:
    """
    Deep Q-Network Agent
    Learns to play games by maximizing cumulative rewards
    """

    def __init__(self, state_size: int = 9, action_size: int = 3, learning_rate: float = 3e-4):
        """
        Initialize DQN Agent

        Args:
            state_size: Number of state features
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Networks
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        # Experience replay buffer
        self.memory = deque(maxlen=100000)

        # Hyperparameters
        self.gamma = 0.995
        self.batch_size = 128
        self.reward_clip_min = -5.0
        self.reward_clip_max = 5.0
        self.max_grad_norm = 10.0
        self.tau = 0.005
        self.train_frequency = 4
        self.warmup_steps = 5000

        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay_steps = 100000
        self.epsilon = self.epsilon_start
        self.total_steps = 0

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        clipped_reward = float(np.clip(reward, self.reward_clip_min, self.reward_clip_max))
        self.memory.append((state, action, clipped_reward, next_state, done))

    def choose_action(self, state: np.ndarray) -> int:
        """
        Choose action using epsilon-greedy strategy

        Args:
            state: Current game state

        Returns:
            Action index (0, 1, or 2)
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.action_size - 1)
        else:
            # Exploit: use network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(self.device)
                q_values = self.q_network(state_tensor)
                return int(torch.argmax(q_values).item())

    def _soft_update_target_network(self):
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def _update_epsilon(self):
        progress = min(1.0, self.total_steps / float(self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def process_step(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        self.total_steps += 1
        self._update_epsilon()

        if self.total_steps < self.warmup_steps:
            return None
        if self.total_steps % self.train_frequency != 0:
            return None

        return self.train()

    def train(self, batch_size=None):
        """
        Train network using experience replay

        Args:
            batch_size: Number of samples to train on

        Returns:
            Loss value or None if not enough experiences
        """
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.memory) < batch_size:
            return None

        # Sample random batch
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and update
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self._soft_update_target_network()

        return loss.item()

    def update_target_network(self):
        """Synchronize target network with Q-network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Compatibility shim (epsilon now updates per-step linearly)"""
        self._update_epsilon()

    def get_network(self):
        """Get the Q-network"""
        return self.q_network

    def get_hyperparameters(self) -> dict:
        """Get current hyperparameters"""
        return {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "batch_size": self.batch_size,
            "replay_buffer_size": self.memory.maxlen,
            "warmup_steps": self.warmup_steps,
            "train_frequency": self.train_frequency,
            "tau": self.tau,
            "max_grad_norm": self.max_grad_norm,
        }

