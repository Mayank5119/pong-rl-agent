"""
Environment Component
Manages interaction with the Pong game server
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from client import PongEnvSync, PongEnvClient
from models import PongAction


class PongEnvironment:
    """Wrapper for Pong game environment"""

    def __init__(self, server_url: str = "ws://localhost:8000/ws/client"):
        """
        Initialize environment

        Args:
            server_url: WebSocket server URL
        """
        self.server_url = server_url
        self.env = None
        self._ctx = None

    def __enter__(self):
        """Context manager entry"""
        # Store the context manager (_SyncContextManager) separately
        self._ctx = PongEnvSync(PongEnvClient(self.server_url)).sync()
        # __enter__ returns _SyncEnvProxy — used for reset() and step()
        self.env = self._ctx.__enter__()
        return self

    def __exit__(self, *args):
        """Context manager exit"""
        # Call __exit__ on the context manager, not the proxy
        if self._ctx:
            self._ctx.__exit__(*args)

    def reset(self):
        """Reset environment and return initial state"""
        obs = self.env.reset()
        return self.extract_state(obs)

    def step(self, action_idx: int):
        """
        Execute action and return next state, reward, done

        Args:
            action_idx: Action index (0=UP, 1=DOWN, 2=STAY)

        Returns:
            (state, reward, done) tuple
        """
        action_names = ["UP", "DOWN", "STAY"]
        action = PongAction(action=action_names[action_idx])

        obs, reward, done = self.env.step(action)
        state = self.extract_state(obs)

        return state, reward, done

    @staticmethod
    def extract_state(obs) -> np.ndarray:
        """
        Convert observation to state vector

        Args:
            obs: Observation from environment

        Returns:
            Normalized state vector (9 features)
        """
        state = np.array([
            obs.ball_x / 40.0,
            obs.ball_y / 20.0,
            obs.ball_vx / 0.4,
            obs.ball_vy / 0.3,
            obs.player_y / 20.0,
            obs.ai_y / 20.0,
            obs.player_score / 11.0,
            obs.ai_score / 11.0,
            (obs.player_y - obs.ball_y) / 20.0,
            ], dtype=np.float32)

        return state

    def get_observation_dict(self):
        """Get full observation (for visualization, etc.)"""
        pass