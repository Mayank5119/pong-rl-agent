"""
Train: Run DQN training
"""

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
        episodes=100,
        max_steps=500,
        verbose=True
    )

    print("\n✅ Training complete!")
    print(f"Final agent epsilon: {agent.epsilon:.3f}")
    print(f"Average reward (last 10 episodes): {rewards[-10:].mean():.2f}")

