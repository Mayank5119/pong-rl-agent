"""
Neural Network Training for Pong RL Agent
Using Deep Q-Network (DQN) approach
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from client import PongEnvSync, PongEnvClient
from models import PongAction


class QNetwork(nn.Module):
    """Deep Q-Network: learns to map game states to action values"""

    def __init__(self, state_size=9, action_size=3, hidden_sizes=(128, 128)):
        """
        Args:
            state_size: Size of compact feature input (9)
            action_size: Number of actions (UP=0, DOWN=1, STAY=2)
            hidden_sizes: Size of hidden layers
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], action_size)
        self.relu = nn.ReLU()

    def forward(self, state):
        """Forward pass through network"""
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # Q-values for each action


class DQNAgent:
    """Deep Q-Network Agent for Pong"""

    def __init__(self, state_size=9, action_size=3, learning_rate=3e-4):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Networks (current and target)
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

    def choose_action(self, state):
        """
        Epsilon-greedy action selection:
        - With probability epsilon: choose random action (exploration)
        - With probability 1-epsilon: choose best action from network (exploitation)
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
        """Train network using experience replay"""
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.memory) < batch_size:
            return  # Not enough experiences yet

        # Sample random batch from memory
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

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

        # Compute loss and update weights
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self._soft_update_target_network()

        return loss.item()

    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Compatibility shim (epsilon now updates per-step linearly)"""
        self._update_epsilon()


def extract_state(obs):
    """Convert observation to state vector for network input"""
    return np.array([
        obs.ball_x / 40.0,
        obs.ball_y / 40.0,
        obs.ball_vx / 0.4,
        obs.ball_vy / 0.3,
        obs.player_y / 40.0,
        obs.ai_y / 40.0,
        (obs.player_y - obs.ball_y) / 40.0,
        (obs.ai_y - obs.ball_y) / 40.0,
        (obs.player_score - obs.ai_score) / 7.0,
    ], dtype=np.float32)


def train_dqn_agent(episodes=1000, max_steps=2000, server_url="ws://localhost:8000/ws/client"):
    """
    Train DQN agent using real neural network learning

    Args:
        episodes: Number of training episodes
        max_steps: Max steps per episode
        server_url: WebSocket server URL
    """
    agent = DQNAgent(state_size=9, action_size=3)

    print("=" * 70)
    print("DQN NEURAL NETWORK TRAINING")
    print("=" * 70)
    print(f"Training {episodes} episodes with neural network learning...")
    print()

    episode_rewards = []
    episode_losses = []

    with PongEnvSync(PongEnvClient(server_url)).sync() as env:
        for episode in range(episodes):
            obs = env.reset()
            state = extract_state(obs)

            episode_reward = 0.0
            episode_loss = 0.0
            step_count = 0

            while step_count < max_steps:
                # Agent chooses action (exploration vs exploitation)
                action_idx = agent.choose_action(state)
                action_name = ["UP", "DOWN", "STAY"][action_idx]

                # Execute action in environment
                obs, reward, done = env.step(PongAction(action=action_name))
                next_state = extract_state(obs)

                loss = agent.process_step(state, action_idx, reward, next_state, done)
                if loss is not None:
                    episode_loss += loss

                episode_reward += reward
                step_count += 1
                state = next_state

                if done:
                    break

            episode_rewards.append(episode_reward)
            episode_losses.append(episode_loss / max(step_count, 1))

            # Progress update
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_loss = np.mean(episode_losses[-10:])
                print(f"Episode {episode + 1:3d}/{episodes} | "
                      f"Reward: {episode_reward:+.1f} | "
                      f"Avg Reward: {avg_reward:+.2f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Epsilon: {agent.epsilon:.3f}")

    # Training summary
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


if __name__ == "__main__":
    # Train the DQN agent
    agent, rewards, losses = train_dqn_agent(episodes=1000, max_steps=2000)

    print("\n✅ Training complete! Agent saved in memory.")
    print("You can now use this agent to play games or continue training.")

