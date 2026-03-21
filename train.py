"""
RL Training Script for Pong Environment

This script demonstrates proper reinforcement learning training patterns:
- Multiple episodes (episodes loop)
- Environment reset between episodes
- Per-step and per-episode metrics
- Training summary statistics
"""

import random
import numpy as np
from client import PongEnvSync, PongEnvClient
from models import PongAction


def train_random_agent(episodes=10, max_steps=500, server_url="ws://localhost:8000/ws/client"):
    """
    Train a random agent on Pong environment following standard RL loop:

    for episode in range(episodes):
        obs = reset()
        done = False
        while not done:
            action = policy(obs)
            obs, reward, done = step(action)
            # update policy

    Args:
        episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        server_url: WebSocket server URL

    Returns:
        episode_rewards: List of total rewards per episode
        episode_lengths: List of steps taken per episode
    """
    with PongEnvSync(PongEnvClient(server_url)).sync() as env:
        episode_rewards = []
        episode_lengths = []

        print(f"Starting training: {episodes} episodes, max {max_steps} steps/episode")
        print("=" * 70)

        for episode in range(episodes):
            # RESET: Start new episode
            obs = env.reset()

            episode_reward = 0.0
            step_count = 0
            done = False

            print(f"\n[Episode {episode + 1}/{episodes}]")
            print(f"Initial state: P{obs.player_score}-A{obs.ai_score}, Ball @ ({obs.ball_x:.1f}, {obs.ball_y:.1f})")

            # RUN EPISODE: Step loop
            while step_count < max_steps and not done:
                # POLICY: Random action (for demo; replace with learned policy)
                action = PongAction(action=random.choice(["UP", "DOWN", "STAY"]))

                # STEP: Execute action
                obs, reward, done = env.step(action)
                episode_reward += reward
                step_count += 1

                # Log significant events
                if reward != 0:
                    print(f"  Step {step_count:3d}: {action.action:4s} → reward={reward:+.0f}, score={obs.player_score}-{obs.ai_score}")

                if step_count % 100 == 0:
                    print(f"  Step {step_count:3d}: Ongoing... score={obs.player_score}-{obs.ai_score}")

                if done:
                    print(f"  EPISODE DONE: Game ended!")
                    break

            # EPISODE STATS
            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)

            print(f"  Summary: {step_count} steps, reward={episode_reward:+.2f}, final score P{obs.player_score}-A{obs.ai_score}")

        # TRAINING SUMMARY
        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"Total episodes:      {episodes}")
        print(f"Avg steps/episode:   {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"Avg reward/episode:  {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Best episode:        {np.max(episode_rewards):.2f} reward")
        print(f"Worst episode:       {np.min(episode_rewards):.2f} reward")
        print(f"Total reward:        {np.sum(episode_rewards):.2f}")
        print("=" * 70)

        return np.array(episode_rewards), np.array(episode_lengths)


def train_simple_policy(episodes=10, max_steps=500, server_url="ws://localhost:8000/ws/client"):
    """
    Train a simple hard-coded policy that follows the ball.

    This shows how to replace the random policy with actual decision logic.
    """
    with PongEnvSync(PongEnvClient(server_url)).sync() as env:
        episode_rewards = []
        episode_lengths = []

        print(f"Starting training: {episodes} episodes, max {max_steps} steps/episode")
        print("Policy: Simple ball-following (move paddle towards ball)")
        print("=" * 70)

        for episode in range(episodes):
            obs = env.reset()

            episode_reward = 0.0
            step_count = 0
            done = False

            print(f"\n[Episode {episode + 1}/{episodes}]")

            while step_count < max_steps and not done:
                # POLICY: Follow the ball
                paddle_center = obs.player_y + 2  # Paddle height / 2

                if obs.ball_y < paddle_center - 1:
                    action = PongAction(action="UP")
                elif obs.ball_y > paddle_center + 1:
                    action = PongAction(action="DOWN")
                else:
                    action = PongAction(action="STAY")

                obs, reward, done = env.step(action)
                episode_reward += reward
                step_count += 1

                if reward != 0:
                    print(f"  Step {step_count:3d}: {action.action:4s} → reward={reward:+.0f}, score={obs.player_score}-{obs.ai_score}")

                if step_count % 100 == 0:
                    print(f"  Step {step_count:3d}: Ongoing... score={obs.player_score}-{obs.ai_score}")

                if done:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(step_count)

            print(f"  Summary: {step_count} steps, reward={episode_reward:+.2f}, final score P{obs.player_score}-A{obs.ai_score}")

        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"Total episodes:      {episodes}")
        print(f"Avg steps/episode:   {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"Avg reward/episode:  {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Best episode:        {np.max(episode_rewards):.2f} reward")
        print(f"Worst episode:       {np.min(episode_rewards):.2f} reward")
        print(f"Total reward:        {np.sum(episode_rewards):.2f}")
        print("=" * 70)

        return np.array(episode_rewards), np.array(episode_lengths)


if __name__ == "__main__":
    # Train random agent
    print("\n" + "#" * 70)
    print("# TRAINING: RANDOM POLICY")
    print("#" * 70)
    rewards_random, lengths_random = train_random_agent(episodes=3, max_steps=300)

    # Train simple policy
    print("\n\n" + "#" * 70)
    print("# TRAINING: SIMPLE BALL-FOLLOWING POLICY")
    print("#" * 70)
    rewards_simple, lengths_simple = train_simple_policy(episodes=3, max_steps=300)

    # Compare policies
    print("\n\n" + "#" * 70)
    print("# POLICY COMPARISON")
    print("#" * 70)
    print(f"Random policy avg reward:      {np.mean(rewards_random):.2f}")
    print(f"Ball-following policy reward:  {np.mean(rewards_simple):.2f}")
    print(f"Improvement:                   {np.mean(rewards_simple) - np.mean(rewards_random):+.2f}")

