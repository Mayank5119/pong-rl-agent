"""
Training Component
Main training loop for DQN agent
"""

import numpy as np
from .agent import DQNAgent
from .environment import PongEnvironment


def train_dqn_agent(
    episodes: int = 100,
    max_steps: int = 500,
    server_url: str = "ws://localhost:8000/ws/client",
    verbose: bool = True
) -> tuple:
    """
    Train DQN agent on Pong environment

    Args:
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        server_url: WebSocket server URL
        verbose: Print progress

    Returns:
        (agent, rewards, losses) tuple
    """

    agent = DQNAgent(state_size=9, action_size=3)
    episode_rewards = []
    episode_losses = []

    if verbose:
        print("=" * 70)
        print("DQN NEURAL NETWORK TRAINING")
        print("=" * 70)
        print(f"Training {episodes} episodes...")
        print()

    with PongEnvironment(server_url) as env:
        for episode in range(episodes):
            state = env.reset()

            episode_reward = 0.0
            episode_loss = 0.0
            step_count = 0

            while step_count < max_steps:
                # Choose action
                action_idx = agent.choose_action(state)

                # Execute step
                next_state, reward, done = env.step(action_idx)

                # Store experience
                agent.remember(state, action_idx, reward, next_state, done)

                # Train
                loss = agent.train()
                if loss is not None:
                    episode_loss += loss

                episode_reward += reward
                step_count += 1
                state = next_state

                if done:
                    break

            # Update target network
            if (episode + 1) % 10 == 0:
                agent.update_target_network()

            # Decay exploration
            agent.decay_epsilon()

            episode_rewards.append(episode_reward)
            episode_losses.append(episode_loss / max(step_count, 1))

            # Progress
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_loss = np.mean(episode_losses[-10:])
                print(f"Episode {episode + 1:3d}/{episodes} | "
                      f"Reward: {episode_reward:+.1f} | "
                      f"Avg Reward: {avg_reward:+.2f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Epsilon: {agent.epsilon:.3f}")

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"Total episodes:           {episodes}")
        print(f"Avg reward (all):         {np.mean(episode_rewards):.2f}")
        print(f"Avg reward (last 10):     {np.mean(episode_rewards[-10:]):.2f}")
        print(f"Best episode:             {np.max(episode_rewards):.2f}")
        print(f"Worst episode:            {np.min(episode_rewards):.2f}")
        print(f"Total cumulative reward:  {np.sum(episode_rewards):.2f}")
        print(f"Final epsilon:            {agent.epsilon:.3f}")
        print("=" * 70)

    return agent, np.array(episode_rewards), np.array(episode_losses)

