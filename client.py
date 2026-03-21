"""
Pong OpenEnv Client - Connects to remote Pong environment server
"""

import asyncio
import json
from typing import Tuple
import websockets
from models import PongAction, PongObservation


class PongEnvClient:
    """
    Client for remote Pong environment.
    Connects via WebSocket to environment server.
    """

    def __init__(self, server_url: str = "ws://localhost:8000/ws/client"):
        """
        Initialize client

        Args:
            server_url: WebSocket URL of the environment server
        """
        self.server_url = server_url
        self.websocket = None
        self.observation = None

    async def connect(self):
        """Connect to environment server"""
        self.websocket = await websockets.connect(self.server_url)

        # Receive initial observation after reset
        message = await self.websocket.recv()
        data = json.loads(message)

        if data["type"] == "reset":
            self.observation = PongObservation(**data["observation"])

        return self.observation

    async def step(self, action: PongAction) -> Tuple[PongObservation, float, bool]:
        """
        Execute one step in the environment

        Args:
            action: PongAction with UP/DOWN/STAY

        Returns:
            (observation, reward, done)
        """
        # Send action
        await self.websocket.send(json.dumps({
            "type": "action",
            "action": action.action
        }))

        # Receive response
        message = await self.websocket.recv()
        data = json.loads(message)

        if data["type"] == "step":
            obs = PongObservation(**data["observation"])
            reward = data["reward"]
            done = data["done"]
            self.observation = obs
            return obs, reward, done

        elif data["type"] == "reset":
            # Game ended, got automatic reset
            obs = PongObservation(**data["observation"])
            self.observation = obs
            return obs, 0, False

        raise ValueError(f"Unknown message type: {data['type']}")

    async def reset(self) -> PongObservation:
        """Reset the environment"""
        await self.websocket.send(json.dumps({"type": "reset"}))

        message = await self.websocket.recv()
        data = json.loads(message)

        obs = PongObservation(**data["observation"])
        self.observation = obs
        return obs

    async def close(self):
        """Close connection to server"""
        if self.websocket:
            await self.websocket.close()


class PongEnvSync:
    """Synchronous wrapper using .sync() pattern"""

    def __init__(self, client: PongEnvClient):
        self.client = client

    async def __aenter__(self):
        await self.client.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()

    async def reset(self) -> PongObservation:
        """Reset the environment"""
        return await self.client.reset()

    async def step(self, action: PongAction) -> Tuple[PongObservation, float, bool]:
        """Execute step"""
        return await self.client.step(action)

    def sync(self):
        """Return a synchronous context manager"""
        return _SyncContextManager(self)


class _SyncContextManager:
    """Helper for .sync() pattern"""

    def __init__(self, env: PongEnvSync):
        self.env = env
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def __enter__(self):
        self.loop.run_until_complete(self.env.__aenter__())
        return _SyncEnvProxy(self.env, self.loop)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.loop.run_until_complete(self.env.__aexit__(exc_type, exc_val, exc_tb))
        self.loop.close()


class _SyncEnvProxy:
    """Synchronous proxy for async environment"""

    def __init__(self, env: PongEnvSync, loop):
        self.env = env
        self.loop = loop

    def reset(self) -> PongObservation:
        """Reset the environment"""
        return self.loop.run_until_complete(self.env.reset())

    def step(self, action: PongAction) -> Tuple[PongObservation, float, bool]:
        """Execute step"""
        return self.loop.run_until_complete(self.env.step(action))


# Example usage
if __name__ == "__main__":
    import random
    import numpy as np

    def train_random_agent(episodes=5, max_steps=500, server_url="ws://localhost:8000/ws/client"):
        """
        Train a random agent on Pong environment using .sync() pattern

        Args:
            episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            server_url: WebSocket server URL
        """
        with PongEnvSync(PongEnvClient(server_url)).sync() as env:
            episode_rewards = []
            episode_lengths = []

            print(f"Starting training: {episodes} episodes, max {max_steps} steps per episode")
            print("=" * 60)

            for episode in range(episodes):
                # Reset environment for new episode
                obs = env.reset()

                episode_reward = 0
                step_count = 0

                print(f"\nEpisode {episode + 1}/{episodes}")
                print("Initial board:")
                print(obs.board)
                print(f"Initial score: Player {obs.player_score} - AI {obs.ai_score}")

                # Run episode
                while step_count < max_steps:
                    # Random action policy
                    action = PongAction(action=random.choice(["UP", "DOWN", "STAY"]))

                    try:
                        obs, reward, done = env.step(action)
                        episode_reward += reward
                        step_count += 1

                        if step_count % 50 == 0 or reward != 0:
                            print(f"  Step {step_count}: action={action.action}, reward={reward:+.0f}, score={obs.player_score}-{obs.ai_score}")

                        if done:
                            break
                    except Exception as e:
                        print(f"Error during step: {e}")
                        break

                episode_rewards.append(episode_reward)
                episode_lengths.append(step_count)

                print(f"Episode finished: steps={step_count}, total_reward={episode_reward:+.0f}, final_score={obs.player_score}-{obs.ai_score}")

            # Training summary
            print("\n" + "=" * 60)
            print("TRAINING SUMMARY")
            print("=" * 60)
            print(f"Episodes: {episodes}")
            print(f"Avg episode length: {np.mean(episode_lengths):.1f} steps")
            print(f"Avg episode reward: {np.mean(episode_rewards):.2f}")
            print(f"Max episode reward: {np.max(episode_rewards):.2f}")
            print(f"Min episode reward: {np.min(episode_rewards):.2f}")
            print(f"Total reward: {np.sum(episode_rewards):.2f}")

            return episode_rewards, episode_lengths

    # Run training
    train_random_agent(episodes=3, max_steps=500)

