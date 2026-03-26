"""
Training Component
Main training loop for DQN agent
"""

import os
import random
import numpy as np
import torch
from .agent import DQNAgent
from .environment import PongEnvironment


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _moving_average(values: np.ndarray, window: int = 50) -> float:
    if len(values) == 0:
        return float("-inf")
    if len(values) < window:
        return float(np.mean(values))
    return float(np.mean(values[-window:]))


def _train_single_seed(
    seed: int,
    episodes: int,
    max_steps: int,
    server_url: str,
    verbose: bool,
) -> tuple:
    _set_seed(seed)
    agent = DQNAgent(state_size=9, action_size=3)
    episode_rewards = []
    episode_losses = []

    if verbose:
        print("=" * 70)
        print(f"DQN NEURAL NETWORK TRAINING (seed={seed})")
        print("=" * 70)
        print(f"Training {episodes} episodes...")
        print()

    with PongEnvironment(server_url) as env:
        for episode in range(episodes):
            state = env.reset()

            episode_reward = 0.0
            episode_loss = 0.0
            train_updates = 0
            step_count = 0

            while step_count < max_steps:
                action_idx = agent.choose_action(state)
                next_state, reward, done = env.step(action_idx)

                loss = agent.process_step(state, action_idx, reward, next_state, done)
                if loss is not None:
                    episode_loss += loss
                    train_updates += 1

                episode_reward += reward
                step_count += 1
                state = next_state

                if done:
                    break

            episode_rewards.append(episode_reward)
            episode_losses.append(episode_loss / max(train_updates, 1))

            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_loss = np.mean(episode_losses[-10:])
                print(
                    f"Episode {episode + 1:4d}/{episodes} | "
                    f"Reward: {episode_reward:+.1f} | "
                    f"Avg Reward: {avg_reward:+.2f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Epsilon: {agent.epsilon:.3f} | "
                    f"Steps: {agent.total_steps}"
                )

    rewards_array = np.array(episode_rewards)
    losses_array = np.array(episode_losses)

    if verbose:
        print("\n" + "=" * 70)
        print(f"TRAINING SUMMARY (seed={seed})")
        print("=" * 70)
        print(f"Total episodes:           {episodes}")
        print(f"Avg reward (all):         {np.mean(rewards_array):.2f}")
        print(f"Avg reward (last 10):     {np.mean(rewards_array[-10:]):.2f}")
        print(f"Best episode:             {np.max(rewards_array):.2f}")
        print(f"Worst episode:            {np.min(rewards_array):.2f}")
        print(f"Total cumulative reward:  {np.sum(rewards_array):.2f}")
        print(f"Final epsilon:            {agent.epsilon:.3f}")
        print("=" * 70)

    return agent, rewards_array, losses_array


def train_dqn_agent(
    episodes: int = 1000,
    max_steps: int = 2000,
    server_url: str = "ws://localhost:8000/ws/client",
    verbose: bool = True,
    seeds: tuple[int, ...] = (0, 1, 2),
    checkpoint_path: str = "checkpoints/best_dqn.pt",
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

    best_agent = None
    best_rewards = None
    best_losses = None
    best_seed = None
    best_score = float("-inf")

    for seed in seeds:
        agent, rewards, losses = _train_single_seed(
            seed=seed,
            episodes=episodes,
            max_steps=max_steps,
            server_url=server_url,
            verbose=verbose,
        )

        score = _moving_average(rewards, window=50)
        if score > best_score:
            best_score = score
            best_seed = seed
            best_agent = agent
            best_rewards = rewards
            best_losses = losses

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "seed": best_seed,
            "moving_avg_reward_50": best_score,
            "state_dict": best_agent.get_network().state_dict(),
            "hyperparameters": best_agent.get_hyperparameters(),
        },
        checkpoint_path,
    )

    if verbose:
        print("\n" + "=" * 70)
        print("BEST MODEL SELECTED")
        print("=" * 70)
        print(f"Best seed:               {best_seed}")
        print(f"Best moving avg (50):    {best_score:.2f}")
        print(f"Checkpoint:              {checkpoint_path}")
        print("=" * 70)

    return best_agent, best_rewards, best_losses

