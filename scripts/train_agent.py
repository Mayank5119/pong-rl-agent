"""
Train: Run DQN training
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.training import train_dqn_agent


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🎮 PONG RL TRAINING")
    print("=" * 70)
    print()
    print("Make sure Docker server is running:")
    print("  docker run -p 8000:8000 pong-server")
    print()

    # Train agent
    agent, rewards, losses = train_dqn_agent(
        episodes=1000,
        max_steps=2000,
        verbose=True
    )

    print("\n✅ Training complete!")
    print(f"Final agent epsilon: {agent.epsilon:.3f}")
    print(f"Average reward (last 10 episodes): {rewards[-10:].mean():.2f}")

