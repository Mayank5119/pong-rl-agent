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

    def __init__(self, state_size: int = 9, action_size: int = 3, learning_rate: float = 0.001):
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

        # Q-Networks
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Experience replay buffer
        self.memory = deque(maxlen=10000)

        # Hyperparameters
        self.gamma = 0.99              # Discount factor
        self.epsilon = 1.0             # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

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
                state_tensor = torch.FloatTensor(state)
                q_values = self.q_network(state_tensor)
                return np.argmax(q_values.numpy())

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
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

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
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Synchronize target network with Q-network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Reduce exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "batch_size": self.batch_size,
        }

