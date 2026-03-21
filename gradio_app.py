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
import numpy as np
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
    def render(obs, width: int = 400, height: int = 200, scale: int = 4) -> Image.Image:
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

        paddle_h = 4 * scale
        paddle_w = 2 * scale

        # ── Player paddle (left, cyan) ───────────────────────────────────────
        py = int(obs.player_y * scale)
        draw.rectangle([(4, py), (4 + paddle_w, py + paddle_h)], fill=(0, 210, 255))

        # ── AI paddle (right, red) ───────────────────────────────────────────
        ay = int(obs.ai_y * scale)
        draw.rectangle(
            [(W - 4 - paddle_w, ay), (W - 4, ay + paddle_h)], fill=(255, 80, 80)
        )

        # ── Ball (white with soft halo) ──────────────────────────────────────
        bx = int(obs.ball_x * scale)
        by = int(obs.ball_y * scale)
        r = 3 * scale
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

def play_game(
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

    Returns:
        (frame: PIL.Image, stats_markdown: str)
    """
    if not HAS_AGENT or not HAS_CLIENT:
        obs = _MockObs()
        frame = PongGameVisualizer.render(obs)
        return frame, (
            "### Demo mode\n"
            "_`train_dqn` or `models` not found — showing a random frame._\n\n"
            f"Score: YOU **{obs.player_score}** – AI **{obs.ai_score}**"
        )

    try:
        agent = DQNAgent(state_size=9, action_size=3)

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
                if done:
                    break

        frame = PongGameVisualizer.render(obs)
        stats = (
            "### Game results\n"
            f"| | |\n|---|---|\n"
            f"| Steps played | {steps_played} |\n"
            f"| Total reward | {total_reward:.1f} |\n"
            f"| Score | YOU **{obs.player_score}** – AI **{obs.ai_score}** |\n"
            f"| Exploration (ε) | {agent.epsilon:.3f} |"
        )
        return frame, stats

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
        return frame, stats


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
                    "Adjust the slider, then click **Play** to watch the agent."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        num_steps_slider = gr.Slider(
                            minimum=10,
                            maximum=500,
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

                # FIX: outputs list now matches the 2-tuple returned by play_game
                play_btn.click(
                    fn=lambda steps: play_game(steps, server_url),
                    inputs=[num_steps_slider],
                    outputs=[game_frame, result_md],
                )

                gr.Examples(
                    examples=[[50], [150], [300], [500]],
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
| Input | State vector | 9 features |
| FC 1 | Linear + ReLU | 128 neurons |
| FC 2 | Linear + ReLU | 128 neurons |
| Output | Q-values | 3 actions |

**Input features (9)**

1. Ball X position (normalised)
2. Ball Y position (normalised)
3. Ball X velocity
4. Ball Y velocity
5. Player paddle Y
6. AI paddle Y
7. Player score
8. AI score
9. Paddle-to-ball Y distance

**Actions (3):** `UP` · `DOWN` · `STAY`
""")
                    with gr.Column():
                        gr.Markdown("""
## Training hyperparameters

| Parameter | Value |
|-----------|-------|
| Algorithm | DQN |
| Optimiser | Adam |
| Learning rate | 0.001 |
| Discount γ | 0.99 |
| Replay buffer | 10 000 |
| Batch size | 32 |
| ε start | 1.0 |
| ε min | 0.01 |
| ε decay | 0.995 |
| Target net sync | Every 10 eps |

**Reward shaping**

- +1 for hitting the ball
- +2 for scoring a point
- −1 for missing the ball
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
## Sharing with Hugging Face Hub

**Upload a trained model**

```python
from components.hub import HFHubManager

manager = HFHubManager("your-username/pong-dqn")
save_dir = manager.save_model(agent, version="v1",
                              description="DQN after 100 episodes")
manager.create_model_card(save_dir, description="...")
manager.push_to_hub(save_dir, commit_message="Upload v1")
```

**Load it back**

```python
agent = manager.load_model_from_hub(DQNAgent)
```

**Deploy on HF Spaces**

1. Create a Space at huggingface.co/new-space and select **Gradio**.
2. Upload this file as `app.py`.
3. Add your `requirements.txt`.
4. Set `HF_TOKEN` as a repository secret.

**Push from the command line**

```bash
export HF_TOKEN=hf_xxxxxxxxxx
python hub_agent.py --repo your-username/pong-dqn --push
```
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
python train_agent.py

# 4. Launch this dashboard
python gradio_app.py

# 5. (Optional) push your model to the Hub
export HF_TOKEN=hf_xxxxx
python hub_agent.py --repo your-name/pong-dqn --push
```

## Project layout

```
pong-rl/
├── components/
│   ├── agent.py          ← DQN agent (learning, replay, action)
│   ├── network.py        ← QNetwork (PyTorch nn.Module)
│   ├── environment.py    ← PongEnvironment wrapper
│   ├── training.py       ← train_dqn_agent() loop
│   ├── hub.py            ← HFHubManager
│   └── app.py            ← Gradio component (used by deploy_app.py)
├── server/
│   ├── app.py            ← FastAPI + WebSocket server
│   └── pong_environment.py
├── gradio_app.py         ← this file (standalone dashboard)
├── train_agent.py        ← run training
├── hub_agent.py          ← hub operations
├── client.py             ← WebSocket client
└── Dockerfile
```
""")

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