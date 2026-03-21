"""
Hugging Face Hub Integration for Pong DQN Agent
Learn: Model versioning, sharing, and deployment
"""

import os
from pathlib import Path
import torch
from huggingface_hub import HfApi, hf_hub_download
from train_dqn import QNetwork, DQNAgent


class HFHubManager:
    """Manage model versions on Hugging Face Hub"""

    def __init__(self, repo_name: str, hf_token: str = None):
        """
        Args:
            repo_name: "username/repo-name"
            hf_token: Your HF token (or use HF_TOKEN env var)
        """
        self.repo_name = repo_name
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.api = HfApi()

    def save_model(self, agent: DQNAgent, version: str, description: str = ""):
        """Save model locally"""
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
            "state_size": agent.state_size,
            "action_size": agent.action_size,
            "learning_rate": agent.learning_rate,
            "gamma": agent.gamma,
            "epsilon": agent.epsilon,
            "epsilon_min": agent.epsilon_min,
            "epsilon_decay": agent.epsilon_decay,
            "batch_size": agent.batch_size,
            "description": description,
        }

        # Save config
        import json
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"✅ Model saved to {save_dir}")
        return save_dir

    def push_to_hub(self, local_dir: Path, commit_message: str = "Upload model"):
        """Push model to Hugging Face Hub"""
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

    def load_model_from_hub(self):
        """Download and load model from Hub"""
        try:
            # Download weights
            q_network_path = hf_hub_download(
                repo_id=self.repo_name,
                filename="q_network.pt",
                token=self.hf_token,
            )

            # Load config
            import json
            config_path = hf_hub_download(
                repo_id=self.repo_name,
                filename="config.json",
                token=self.hf_token,
            )

            with open(config_path) as f:
                config = json.load(f)

            # Create agent
            agent = DQNAgent(
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


def example_usage():
    """Example: Save and push model to Hub"""

    # 1. Train agent (simplified)
    print("1️⃣  Training agent...")
    agent = DQNAgent(state_size=9, action_size=3)
    # ... training loop would go here ...
    print("   Agent trained!")

    # 2. Save locally
    print("\n2️⃣  Saving to local filesystem...")
    manager = HFHubManager(repo_name="your-username/pong-dqn-v1")
    save_dir = manager.save_model(
        agent,
        version="v1",
        description="Initial DQN agent trained on Pong"
    )

    # 3. Push to Hub (requires HF token)
    print("\n3️⃣  Pushing to Hugging Face Hub...")
    manager.push_to_hub(
        save_dir,
        commit_message="Initial DQN model v1"
    )

    # 4. Load from Hub
    print("\n4️⃣  Loading model from Hub...")
    loaded_agent = manager.load_model_from_hub()

    print("\n✅ Complete workflow demonstrated!")


if __name__ == "__main__":
    print("=" * 70)
    print("HUGGING FACE HUB INTEGRATION EXAMPLE")
    print("=" * 70)
    print()

    # This is just a demo - in real use:
    # 1. Train your agent with train_dqn.py
    # 2. Save it: manager.save_model(agent, "v1")
    # 3. Push it: manager.push_to_hub(save_dir)

    print("To use this module:")
    print()
    print("1. Set HF_TOKEN environment variable:")
    print("   export HF_TOKEN=hf_xxxxx")
    print()
    print("2. Create HF Hub repo: https://huggingface.co/new")
    print()
    print("3. Use in your code:")
    print("   from hf_hub_integration import HFHubManager")
    print("   manager = HFHubManager('username/repo-name')")
    print("   manager.save_model(agent, 'v1')")
    print("   manager.push_to_hub(save_dir)")
    print()
    print("=" * 70)

