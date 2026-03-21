"""
Deployment Component
Gradio interface for model demonstration and deployment.
Part of the pong-rl components package.
"""

import numpy as np
from PIL import Image, ImageDraw

import gradio as gr

from .agent import DQNAgent
from .environment import PongEnvironment


# ---------------------------------------------------------------------------
# Visualiser
# ---------------------------------------------------------------------------

class GameVisualizer:
    """Renders a Pong game observation as a PIL image."""

    @staticmethod
    def render_observation(
            obs,
            width: int = 400,
            height: int = 200,
            scale: int = 4,
    ) -> Image.Image:
        """
        Convert a game observation into a pixel-art style image.

        Args:
            obs:    Observation object (ball_x, ball_y, player_y, ai_y, scores).
            width:  Logical board width before scaling.
            height: Logical board height before scaling.
            scale:  Pixel scale factor.

        Returns:
            PIL.Image.Image ready to pass to gr.Image.
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

        # ── Score display ────────────────────────────────────────────────────
        draw.text((W // 4 - 6, 12), str(obs.player_score), fill=(0, 210, 255))
        draw.text((3 * W // 4 - 6, 12), str(obs.ai_score), fill=(255, 80, 80))

        paddle_h = 4 * scale
        paddle_w = 2 * scale

        # ── Player paddle (left, cyan) ───────────────────────────────────────
        py = int(obs.player_y * scale)
        draw.rectangle(
            [(4, py), (4 + paddle_w, py + paddle_h)],
            fill=(0, 210, 255),
        )

        # ── AI paddle (right, red) ───────────────────────────────────────────
        ay = int(obs.ai_y * scale)
        draw.rectangle(
            [(W - 4 - paddle_w, ay), (W - 4, ay + paddle_h)],
            fill=(255, 80, 80),
        )

        # ── Ball ─────────────────────────────────────────────────────────────
        bx = int(obs.ball_x * scale)
        by = int(obs.ball_y * scale)
        r = 3 * scale
        draw.ellipse(
            [(bx - r - 2, by - r - 2), (bx + r + 2, by + r + 2)],
            fill=(80, 80, 100),
        )
        draw.ellipse(
            [(bx - r, by - r), (bx + r, by + r)],
            fill=(255, 255, 255),
        )

        # ── Labels ──────────────────────────────────────────────────────────
        draw.text((8, H - 20), "YOU", fill=(0, 210, 255))
        draw.text((W - 38, H - 20), "AI", fill=(255, 80, 80))

        return img


# ---------------------------------------------------------------------------
# Play function
# ---------------------------------------------------------------------------

def _play_game(
        agent: DQNAgent,
        num_steps: int,
        server_url: str,
) -> tuple:
    """
    Run the agent for *num_steps* in the Pong environment.

    Args:
        agent:      DQNAgent instance.
        num_steps:  Maximum steps to simulate.
        server_url: WebSocket URL of the game server.

    Returns:
        (frame: PIL.Image | None, stats_markdown: str)
    """
    try:
        with PongEnvironment(server_url) as env:
            state = env.reset()
            total_reward = 0.0
            step_count = 0

            # Keep the last raw observation so we can render it
            last_obs = env.last_obs if hasattr(env, "last_obs") else None

            for step in range(num_steps):
                action_idx = agent.choose_action(state)
                next_state, reward, done = env.step(action_idx)

                total_reward += reward
                step_count = step + 1
                state = next_state

                if done:
                    break

            # Render the final frame if env exposes the raw observation
            frame = None
            if hasattr(env, "last_obs") and env.last_obs is not None:
                frame = GameVisualizer.render_observation(env.last_obs)

        stats = (
            "### Game results\n"
            f"| | |\n|---|---|\n"
            f"| Steps played | {step_count} |\n"
            f"| Total reward | {total_reward:.1f} |\n"
            f"| Exploration ε | {agent.epsilon:.3f} |"
        )
        return frame, stats

    except Exception as exc:
        return None, (
            "### Server unavailable\n"
            f"```\n{str(exc)[:200]}\n```\n\n"
            "Start the server with:\n"
            "```\ndocker run -p 8000:8000 pong-server\n```"
        )


# ---------------------------------------------------------------------------
# Dashboard factory
# ---------------------------------------------------------------------------

def create_gradio_interface(
        server_url: str = "ws://localhost:8000/ws/client",
) -> gr.Blocks:
    """
    Build and return the full Gradio dashboard.

    Args:
        server_url: WebSocket URL passed through to the environment.

    Returns:
        gr.Blocks instance — call .launch() to start.
    """
    agent = DQNAgent(state_size=9, action_size=3)

    with gr.Blocks(
            title="Pong RL Agent",
            theme=gr.themes.Soft(primary_hue="cyan"),
    ) as demo:

        gr.Markdown(
            "# Pong RL Agent — dashboard\n"
            "> DQN · PyTorch · Gradio"
        )

        with gr.Tabs():

            # ── Tab 1 · Play ─────────────────────────────────────────────────
            with gr.Tab("Play game"):
                gr.Markdown(
                    "Adjust the slider then click **Play** to watch the agent."
                )

                with gr.Row():
                    # ── Left column: controls ────────────────────────────────
                    with gr.Column(scale=1):
                        num_steps = gr.Slider(
                            minimum=10,
                            maximum=500,
                            value=100,
                            step=10,
                            label="Steps to simulate",
                            info="More steps = longer game clip",
                        )
                        play_btn = gr.Button("Play", variant="primary", size="lg")
                        stats = gr.Markdown("_Click Play to start._")

                    # ── Right column: game frame ──────────────────────────────
                    # game_frame MUST be declared before play_btn.click so the
                    # variable is in scope when the outputs list is built.
                    with gr.Column(scale=2):
                        game_frame = gr.Image(
                            label="Game view  (cyan = you · red = AI)",
                            height=420,
                        )

                # Both outputs are now resolved — game_frame and stats are
                # defined above in the same scope as this .click() call.
                play_btn.click(
                    fn=lambda steps: _play_game(agent, steps, server_url),
                    inputs=[num_steps],
                    outputs=[game_frame, stats],   # ← no more unresolved name
                )

                gr.Examples(
                    examples=[[50], [150], [300], [500]],
                    inputs=[num_steps],
                    label="Presets",
                )

            # ── Tab 2 · Model info ────────────────────────────────────────────
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

1. Ball X (normalised)
2. Ball Y (normalised)
3. Ball Vx
4. Ball Vy
5. Player paddle Y
6. AI paddle Y
7. Player score
8. AI score
9. Paddle-to-ball Y distance

**Actions:** `UP` · `DOWN` · `STAY`
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
| Target sync | Every 10 eps |

**Reward shaping**
- +1 hit the ball
- +2 score a point
- −1 miss the ball
""")

            # ── Tab 3 · How it works ─────────────────────────────────────────
            with gr.Tab("How it works"):
                gr.Markdown("""
## The DQN loop

```
Observe state s  →  Q-Network  →  Q(s, UP)  Q(s, DOWN)  Q(s, STAY)
        ↓
ε-greedy policy  →  choose action a
        ↓
Execute a  →  reward r, next state s′
        ↓
Store (s, a, r, s′, done) in replay buffer
        ↓
Sample random mini-batch  →  compute TD target
        →  r + γ · max Q_target(s′, a′)
        ↓
Minimise MSELoss(Q_network(s,a), TD target)
        ↓
Every 10 episodes: copy Q_network → target network
        ↓  repeat ↑
```

**Key concepts**

- **ε-greedy** — explore randomly at first, exploit learned Q-values later
- **Replay buffer** — breaks correlation between consecutive updates
- **Target network** — frozen copy provides stable regression targets
- **Bellman target** — Q*(s,a) = r + γ · max Q*(s′,a′)
""")

            # ── Tab 4 · About ─────────────────────────────────────────────────
            with gr.Tab("About"):
                gr.Markdown("""
## Tech stack

| | |
|---|---|
| PyTorch | Neural network & training |
| Hugging Face Hub | Model versioning & sharing |
| Gradio | This web interface |
| FastAPI | WebSocket game server |
| Docker | Containerised deployment |

## Links
- [PyTorch docs](https://pytorch.org/docs/)
- [HF Hub docs](https://huggingface.co/docs/hub)
- [TRL library](https://github.com/huggingface/trl)
- [DQN paper](https://arxiv.org/abs/1312.5602)
""")

    return demo


# ---------------------------------------------------------------------------
# Entry point (when run directly, not imported as a component)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True)