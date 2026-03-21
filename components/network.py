"""
Neural Network Component
Defines the Q-Network architecture for DQN
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Deep Q-Network: Maps game states to Q-values for actions

    Architecture:
        Input (9) → Dense(128) → ReLU → Dense(128) → ReLU → Output(3)
    """

    def __init__(self, state_size: int = 9, action_size: int = 3, hidden_size: int = 128):
        """
        Args:
            state_size: Number of state features
            action_size: Number of possible actions
            hidden_size: Number of neurons in hidden layers
        """
        super(QNetwork, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        # Network layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: state → Q-values

        Args:
            state: Game state vector

        Returns:
            Q-values for each action
        """
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

    def save_pretrained(self, path: str):
        """Save model weights"""
        torch.save(self.state_dict(), path)

    @classmethod
    def from_pretrained(cls, path: str):
        """Load model weights"""
        model = cls()
        model.load_state_dict(torch.load(path))
        return model

