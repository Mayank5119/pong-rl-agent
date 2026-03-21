"""
Hub: Manage models on Hugging Face Hub
"""

from pathlib import Path
from components.agent import DQNAgent
from components.hub import HFHubManager


def save_and_push_model(repo_name: str, version: str = "v1", push: bool = False):
    """
    Save and optionally push model to Hub

    Args:
        repo_name: "username/repo-name"
        version: Version string
        push: Whether to push to Hub
    """
    print("\n" + "=" * 70)
    print("🤗 HUGGING FACE HUB INTEGRATION")
    print("=" * 70)
    print()

    # Create agent (in real usage, load trained agent)
    agent = DQNAgent(state_size=9, action_size=3)

    # Create manager
    manager = HFHubManager(repo_name)

    # Save locally
    print(f"📁 Saving model locally...")
    save_dir = manager.save_model(
        agent,
        version,
        description="DQN agent trained on Pong"
    )

    # Create model card
    print(f"📝 Creating model card...")
    manager.create_model_card(
        save_dir,
        description="A DQN agent trained to play Pong using reinforcement learning",
        tags=["rl", "pong", "dqn", "pytorch"]
    )

    # Push to Hub (optional)
    if push:
        print(f"☁️  Pushing to Hub...")
        manager.push_to_hub(save_dir, commit_message=f"Upload {version}")
    else:
        print(f"\n💡 To push to Hub, set HF_TOKEN environment variable:")
        print(f"   export HF_TOKEN=hf_xxxxx")
        print(f"   python scripts/hub_agent.py --push")

    print(f"\n✅ Done!")
    print(f"Model saved at: {save_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage Pong agent on HF Hub")
    parser.add_argument(
        "--repo",
        default="username/pong-dqn",
        help="Repository name (username/repo-name)"
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="Version string"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push to Hub (requires HF_TOKEN)"
    )

    args = parser.parse_args()

    save_and_push_model(args.repo, args.version, args.push)

