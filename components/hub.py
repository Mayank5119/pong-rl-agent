"""
Hugging Face Hub Component
Model management, versioning, and sharing
"""

import os
import json
from pathlib import Path
import torch
from huggingface_hub import HfApi, hf_hub_download


class HFHubManager:
    """Manage model versions on Hugging Face Hub"""

    def __init__(self, repo_name: str, hf_token: str = None):
        """
        Initialize Hub Manager

        Args:
            repo_name: "username/repo-name"
            hf_token: HF token (or use HF_TOKEN env var)
        """
        self.repo_name = repo_name
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.api = HfApi()

    def save_model(self, agent, version: str, description: str = "") -> Path:
        """
        Save model locally

        Args:
            agent: DQN agent
            version: Version string (e.g., "v1")
            description: Model description

        Returns:
            Path to saved model directory
        """
        save_dir = Path(f"models/{version}")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save network weights
        torch.save(
            agent.q_network.state_dict(),
            save_dir / "q_network.pt"
        )

        # Save target network
        torch.save(
            agent.target_network.state_dict(),
            save_dir / "target_network.pt"
        )

        # Save hyperparameters
        config = {
            **agent.get_hyperparameters(),
            "description": description,
        }

        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"✅ Model saved to {save_dir}")
        return save_dir

    def push_to_hub(self, local_dir: Path, commit_message: str = "Upload model"):
        """
        Push model to Hugging Face Hub

        Args:
            local_dir: Local directory containing model
            commit_message: Commit message
        """
        if not self.hf_token:
            print("⚠️  HF_TOKEN not found. Set it to push to Hub.")
            print("Get token from: https://huggingface.co/settings/tokens")
            return

        try:
            self.api.upload_folder(
                folder_path=str(local_dir),
                repo_id=self.repo_name,
                token=self.hf_token,
                commit_message=commit_message,
                repo_type="model",
            )
            print(f"✅ Model pushed to https://huggingface.co/{self.repo_name}")
        except Exception as e:
            print(f"❌ Failed to push: {e}")

    def load_model_from_hub(self, agent_class):
        """
        Download and load model from Hub

        Args:
            agent_class: DQNAgent class

        Returns:
            Loaded agent or None
        """
        try:
            # Download weights
            q_network_path = hf_hub_download(
                repo_id=self.repo_name,
                filename="q_network.pt",
                token=self.hf_token,
            )

            # Download config
            config_path = hf_hub_download(
                repo_id=self.repo_name,
                filename="config.json",
                token=self.hf_token,
            )

            with open(config_path) as f:
                config = json.load(f)

            # Create agent
            agent = agent_class(
                state_size=config["state_size"],
                action_size=config["action_size"],
                learning_rate=config["learning_rate"],
            )

            # Load weights
            agent.q_network.load_state_dict(torch.load(q_network_path))

            print(f"✅ Model loaded from {self.repo_name}")
            return agent
        except Exception as e:
            print(f"❌ Failed to load: {e}")
            return None

    def create_model_card(self, local_dir: Path, description: str, tags: list = None):
        """
        Create a model card for documentation

        Args:
            local_dir: Model directory
            description: Model description
            tags: List of tags (e.g., ["rl", "pong", "dqn"])
        """
        tags = tags or ["rl", "pong", "dqn"]

        model_card = f"""---
tags: {tags}
---

# Pong DQN Agent

## Model Description

{description}

## Intended Use

This model is a DQN agent trained to play Pong. It can be used for:
- Playing Pong games
- RL benchmarking
- Learning reinforcement learning

## How to Use

```python
from components.agent import DQNAgent
from components.hub import HFHubManager

manager = HFHubManager("username/pong-dqn")
agent = manager.load_model_from_hub(DQNAgent)
```

## Training Details

- Algorithm: Deep Q-Network (DQN)
- Environment: Pong
- Framework: PyTorch

## License

MIT
"""

        with open(local_dir / "README.md", "w") as f:
            f.write(model_card)

        print(f"✅ Model card created at {local_dir / 'README.md'}")

