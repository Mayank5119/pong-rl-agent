"""
Gradio Dashboard — Pong RL Agent
Deploys the DQN agent as an interactive web app.
Run locally or push to Hugging Face Spaces.

Usage:
    python gradio_app.py
    python gradio_app.py --share
    python gradio_app.py --server-url ws://myhost:8000/ws/client
"""

import argparse
import os
import time
import numpy as np
import torch
from PIL import Image, ImageDraw

import gradio as gr

# ---------------------------------------------------------------------------
# Optional runtime imports — fail gracefully so the UI always starts
# ---------------------------------------------------------------------------
try:
    from train_dqn import DQNAgent, extract_state
    HAS_AGENT = True
except ImportError:
    HAS_AGENT = False

try:
    from client import PongEnvSync, PongEnvClient
    from models import PongAction
    HAS_CLIENT = True
except ImportError:
    HAS_CLIENT = False


# ---------------------------------------------------------------------------
# Visualiser
# ---------------------------------------------------------------------------

class PongGameVisualizer:
    """Renders a PongObservation as a PIL image."""

    @staticmethod
    def render(obs, width: int = 40, height: int = 40, scale: int = 10) -> Image.Image:
        """
        Convert a game observation into a pixel-art style image.

        Args:
            obs:    Observation object with ball/paddle/score fields.
            width:  Logical game width (pixels before scaling).
            height: Logical game height.
            scale:  Pixel scale factor.

        Returns:
            PIL.Image.Image
        """
        W, H = width * scale, height * scale
        img = Image.new("RGB", (W, H), (15, 15, 30))
        draw = ImageDraw.Draw(img)

        # ── Centre dashed line ──────────────────────────────────────────────
        dash, gap = 12, 8
        for y in range(0, H, dash + gap):
            draw.rectangle(
                [(W // 2 - 2, y), (W // 2 + 2, min(y + dash, H))],
                fill=(60, 60, 80),
            )

        # ── Scores ──────────────────────────────────────────────────────────
        draw.text((W // 4 - 6, 12), str(obs.player_score), fill=(0, 210, 255))
        draw.text((3 * W // 4 - 6, 12), str(obs.ai_score), fill=(255, 80, 80))

        paddle_h = 3 * scale
        paddle_w = max(2, scale // 2)

        player_x = 0
        ai_x = max(0, width - 1)

        # ── Player paddle (left, cyan) ───────────────────────────────────────
        py = int(obs.player_y * scale)
        px = int(player_x * scale)
        draw.rectangle([(px, py), (px + paddle_w, py + paddle_h)], fill=(0, 210, 255))

        # ── AI paddle (right, red) ───────────────────────────────────────────
        ay = int(obs.ai_y * scale)
        ax = int(ai_x * scale)
        draw.rectangle(
            [(ax - paddle_w, ay), (ax, ay + paddle_h)], fill=(255, 80, 80)
        )

        # ── Ball (white with soft halo) ──────────────────────────────────────
        bx = int(np.clip(obs.ball_x, 0, width - 1) * scale)
        by = int(np.clip(obs.ball_y, 0, height - 1) * scale)
        r = max(2, scale // 3)
        draw.ellipse([(bx - r - 2, by - r - 2), (bx + r + 2, by + r + 2)], fill=(80, 80, 100))
        draw.ellipse([(bx - r, by - r), (bx + r, by + r)], fill=(255, 255, 255))

        # ── Labels ──────────────────────────────────────────────────────────
        draw.text((8, H - 20), "YOU", fill=(0, 210, 255))
        draw.text((W - 38, H - 20), "AI", fill=(255, 80, 80))

        return img


# ---------------------------------------------------------------------------
# Mock observation (used when game server is unavailable)
# ---------------------------------------------------------------------------

class _MockObs:
    """Fake observation for demo mode — no server needed."""

    def __init__(self):
        self.ball_x = float(np.random.uniform(5, 35))
        self.ball_y = float(np.random.uniform(2, 18))
        self.ball_vx = float(np.random.choice([-0.3, 0.3]))
        self.ball_vy = float(np.random.uniform(-0.2, 0.2))
        self.player_y = float(np.random.uniform(2, 14))
        self.ai_y = float(np.random.uniform(2, 14))
        self.player_score = int(np.random.randint(0, 7))
        self.ai_score = int(np.random.randint(0, 7))


# ---------------------------------------------------------------------------
# Core play function — single definition, no duplication
# ---------------------------------------------------------------------------

def play_game_realtime(
        num_steps: int,
        server_url: str = "ws://localhost:8000/ws/client",
) -> tuple:
    """
    Run the DQN agent for *num_steps* in the Pong environment.

    Falls back to demo mode automatically if the server is unreachable or
    if training dependencies (train_dqn, models) are not installed.

    Args:
        num_steps:  How many game steps to simulate.
        server_url: WebSocket URL of the Pong game server.

    Yields:
        (frame: PIL.Image, stats_markdown: str)
    """
    if not HAS_AGENT or not HAS_CLIENT:
        for step in range(min(num_steps, 120)):
            obs = _MockObs()
            frame = PongGameVisualizer.render(obs)
            stats = (
                "### Demo mode (real-time preview)\n"
                "_`train_dqn` or `models` not found — streaming random frames._\n\n"
                f"Frame: **{step + 1}** / {num_steps}  \n"
                f"Score: YOU **{obs.player_score}** – AI **{obs.ai_score}**"
            )
            yield frame, stats
            time.sleep(0.03)
        return

    def _load_agent_for_inference() -> tuple:
        agent = DQNAgent(state_size=9, action_size=3)
        checkpoint_path = os.path.join("checkpoints", "best_dqn.pt")

        if not os.path.exists(checkpoint_path):
            agent.epsilon = 0.0
            return agent, "⚠️ No checkpoint found (`checkpoints/best_dqn.pt`) — using untrained policy."

        checkpoint = torch.load(checkpoint_path, map_location=agent.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        agent.q_network.load_state_dict(state_dict)
        agent.target_network.load_state_dict(agent.q_network.state_dict())
        agent.q_network.eval()
        agent.target_network.eval()
        agent.epsilon = 0.0

        best_seed = checkpoint.get("seed", "?") if isinstance(checkpoint, dict) else "?"
        best_score = checkpoint.get("moving_avg_reward_50", None) if isinstance(checkpoint, dict) else None
        if isinstance(best_score, (int, float)):
            status = f"✅ Loaded trained checkpoint (seed={best_seed}, moving_avg_50={best_score:.2f})"
        else:
            status = f"✅ Loaded trained checkpoint (seed={best_seed})"
        return agent, status

    try:
        agent, model_status = _load_agent_for_inference()

        with PongEnvSync(PongEnvClient(server_url)).sync() as env:
            obs = env.reset()
            total_reward = 0.0
            steps_played = 0

            for step in range(num_steps):
                state = extract_state(obs)
                action_idx = agent.choose_action(state)
                action_name = ["UP", "DOWN", "STAY"][action_idx]
                obs, reward, done = env.step(PongAction(action=action_name))
                total_reward += reward
                steps_played = step + 1

                frame = PongGameVisualizer.render(obs)
                live_stats = (
                    "### Live game\n"
                    f"| | |\n|---|---|\n"
                    f"| Model | {model_status} |\n"
                    f"| Step | {steps_played} / {num_steps} |\n"
                    f"| Reward so far | {total_reward:.1f} |\n"
                    f"| Score | YOU **{obs.player_score}** – AI **{obs.ai_score}** |\n"
                    f"| Exploration (ε) | {agent.epsilon:.3f} |"
                )
                yield frame, live_stats
                time.sleep(0.03)

                if done:
                    break

        final_frame = PongGameVisualizer.render(obs)
        final_stats = (
            "### Game results\n"
            f"| | |\n|---|---|\n"
            f"| Model | {model_status} |\n"
            f"| Steps played | {steps_played} |\n"
            f"| Total reward | {total_reward:.1f} |\n"
            f"| Score | YOU **{obs.player_score}** – AI **{obs.ai_score}** |\n"
            f"| Exploration (ε) | {agent.epsilon:.3f} |"
        )
        yield final_frame, final_stats
        return

    except Exception as exc:
        obs = _MockObs()
        frame = PongGameVisualizer.render(obs)
        stats = (
            "### Demo mode (server unavailable)\n"
            f"```\n{str(exc)[:120]}\n```\n\n"
            "Start the server with:\n"
            "```\ndocker run -p 8000:8000 pong-server\n```\n\n"
            f"Showing a random frame. Score: YOU **{obs.player_score}** – AI **{obs.ai_score}**"
        )
        yield frame, stats
        return


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def create_dashboard(server_url: str = "ws://localhost:8000/ws/client") -> gr.Blocks:
    """
    Build and return the full Gradio dashboard.

    Args:
        server_url: WebSocket URL passed through to play_game().

    Returns:
        gr.Blocks instance (call .launch() to start).
    """

    def play_with_server(num_steps: int):
        yield from play_game_realtime(num_steps=num_steps, server_url=server_url)

    with gr.Blocks(
            title="Pong RL Agent",
            theme=gr.themes.Soft(primary_hue="cyan"),
    ) as demo:

        gr.Markdown(
            "# Pong RL Agent — interactive dashboard\n"
            "> DQN agent trained with PyTorch · deployed with Gradio"
        )

        with gr.Tabs():

            # ── Tab 1 · Play ─────────────────────────────────────────────────
            with gr.Tab("Play game"):
                gr.Markdown(
                    "Adjust the slider, then click **Play** to watch the agent in real time."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        num_steps_slider = gr.Slider(
                            minimum=10,
                            maximum=2000,
                            value=100,
                            step=10,
                            label="Steps to simulate",
                            info="More steps = longer game",
                        )
                        play_btn = gr.Button("Play", variant="primary", size="lg")
                        result_md = gr.Markdown("_Click Play to start._")

                    with gr.Column(scale=2):
                        # FIX: game_frame declared before the click handler
                        game_frame = gr.Image(
                            label="Game view  (cyan = you, red = AI)",
                            height=420,
                        )

                # Stream frames in real time while the episode is being simulated
                play_btn.click(
                    fn=play_with_server,
                    inputs=[num_steps_slider],
                    outputs=[game_frame, result_md],
                )

                gr.Examples(
                    examples=[[50], [150], [500], [1000], [1500], [2000]],
                    inputs=[num_steps_slider],
                    label="Presets",
                )

            # ── Tab 2 · Architecture ─────────────────────────────────────────
            with gr.Tab("Model info"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
## Network architecture

| Layer | Type | Size |
|-------|------|------|
| Input | Feature vector | 9 features |
| FC 1 | Linear + ReLU | 128 neurons |
| FC 2 | Linear + ReLU | 128 neurons |
| Output | Q-values | 3 actions |

**Input representation**

- Ball position: `ball_x`, `ball_y`
- Ball velocity: `ball_vx`, `ball_vy`
- Paddle positions: `player_y`, `ai_y`
- Relative distances: `player_y - ball_y`, `ai_y - ball_y`
- Score context: `player_score - ai_score`

**Actions (3):** `UP` · `DOWN` · `STAY`
""")
                    with gr.Column():
                        gr.Markdown("""
## Training hyperparameters

| Parameter | Value |
|-----------|-------|
| Algorithm | DQN |
| Optimiser | Adam |
| Learning rate | 0.0003 |
| Discount γ | 0.995 |
| Replay buffer | 100 000 |
| Batch size | 128 |
| ε start | 1.0 |
| ε min | 0.05 |
| ε schedule | Linear over 100k steps |
| Warmup | 5 000 steps |
| Train frequency | Every 4 steps |
| Target update | Soft, τ = 0.005 |
| Gradient clip | 10.0 |

**Reward shaping**

- +1 for hitting the ball
- +2 for scoring a point
- −1 for missing the ball
- Reward clipped to [-5, 5] for stability
""")

            # ── Tab 3 · How it works ─────────────────────────────────────────
            with gr.Tab("How it works"):
                gr.Markdown("""
## The DQN training loop

```
Observe state s (ball + paddle positions)
        ↓
Q-Network → Q(s, UP)  Q(s, DOWN)  Q(s, STAY)
        ↓
ε-greedy policy → choose action a
        ↓
Execute a in Pong → get reward r, next state s′
        ↓
Store (s, a, r, s′, done) in replay buffer
        ↓
Sample random mini-batch → compute TD target
        ↓
TD target = r  +  γ · max Q_target(s′, a′)
        ↓
Minimise MSE(Q_network(s,a), TD target)
        ↓
Every 10 episodes → copy Q_network → target network
        ↓
Repeat ↑
```

**ε-greedy exploration** — with probability ε the agent picks a random
action (explore). Otherwise it picks `argmax Q(s, a)` (exploit). ε decays
from 1.0 down to 0.01 over training so the agent starts curious and becomes
increasingly confident.

**Experience replay** — storing `(s, a, r, s′)` tuples and sampling random
batches breaks the correlation between consecutive updates, which otherwise
destabilises training.

**Target network** — a frozen copy of Q_network provides stable regression
targets. Without it the targets shift every step, making training diverge.

**Bellman target**

```
Q*(s, a) = r  +  γ · max_a′ Q*(s′, a′)
```
""")

            # ── Tab 4 · Hub & deploy ─────────────────────────────────────────
            with gr.Tab("Hub & deploy"):
                gr.Markdown("""
## Lean setup note

This workspace is trimmed to the local training + dashboard flow:

1. Run Docker server
2. Run `python scripts/train_agent.py`
3. Run `python gradio_app.py`

Hugging Face Hub helper scripts/components were removed from this setup.
""")

            # ── Tab 5 · Quick start ──────────────────────────────────────────
            with gr.Tab("Quick start"):
                gr.Markdown("""
## Running locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build and start the game server
docker build -t pong-server .
docker run -p 8000:8000 pong-server

# 3. Train the agent
python scripts/train_agent.py

# 4. Launch this dashboard
python gradio_app.py
```

## Project layout

```
pong-rl/
├── components/
│   ├── agent.py          ← DQN agent (learning, replay, action)
│   ├── network.py        ← QNetwork (PyTorch nn.Module)
│   ├── environment.py    ← PongEnvironment wrapper
│   ├── training.py       ← train_dqn_agent() loop
│   └── __init__.py
├── server/
│   ├── app.py            ← FastAPI + WebSocket server
│   └── pong_environment.py
├── gradio_app.py         ← this file (standalone dashboard)
├── scripts/train_agent.py ← run training
├── client.py             ← WebSocket client
├── models.py
└── requirements.txt
```
""")

    demo.queue()
    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pong RL Agent — Gradio dashboard")
    parser.add_argument(
        "--server-url",
        default="ws://localhost:8000/ws/client",
        help="WebSocket URL of the Pong game server",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Generate a public Gradio share link",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Local port (default: 7860)",
    )
    args = parser.parse_args()

    print()
    print("=" * 68)
    print("  Pong RL Agent · Gradio Dashboard")
    print("=" * 68)
    print(f"  Game server : {args.server_url}")
    print(f"  Dashboard   : http://localhost:{args.port}")
    if args.share:
        print("  Sharing     : public link will be printed below")
    print()
    print("  Tip: if the Docker server is not running, demo mode activates")
    print("       automatically so the UI is always usable.")
    print()

    # FIX: use create_dashboard() (the richer multi-tab version)
    #      not create_gradio_interface() which had duplicate play_game bugs
    demo = create_dashboard(server_url=args.server_url)
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )