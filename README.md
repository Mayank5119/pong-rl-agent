# Pong RL (Lean Setup)

Minimal local setup for training and visualizing a DQN Pong agent.

## Run Flow

1. Start Docker server
2. Train agent
3. Launch Gradio UI

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Build and run game server

```bash
docker build -f server/Dockerfile -t pong-server .
docker run -p 8000:8000 pong-server
```

### 3) Train the agent

```bash
python scripts/train_agent.py
```

### 4) Launch dashboard

```bash
python gradio_app.py
```

Open: http://localhost:7860

## Current Game Config

- Board: 40 x 40
- Paddle: 1 x 3
- Ball: 1 x 1

## Current Training Config

- Input: compact feature vector (9 values)
- Network: 9 → 128 → 128 → 3 (ReLU)
- Optimizer: Adam, learning rate 0.0003
- Loss: SmoothL1Loss (Huber)
- Gamma: 0.995
- Replay buffer: 100000
- Batch size: 128
- Reward clipping: [-5, 5]
- Warmup: 5000 steps (random/exploratory collection before updates)
- Train frequency: every 4 environment steps
- Target update: soft update, tau = 0.005
- Gradient clipping: 10.0
- Epsilon: linear schedule 1.0 → 0.05 over 100000 steps
- Default training length: 1000 episodes, 2000 max steps/episode
- Seeds: (0, 1, 2), best checkpoint selected by moving average reward

### Selected Input Features (9)

1. `ball_x / 40.0`
2. `ball_y / 40.0`
3. `ball_vx / 0.4`
4. `ball_vy / 0.3`
5. `player_y / 40.0`
6. `ai_y / 40.0`
7. `(player_y - ball_y) / 40.0`
8. `(ai_y - ball_y) / 40.0`
9. `(player_score - ai_score) / 7.0`

## Where to Edit Configs

- Core training settings: `components/agent.py`
- Network architecture: `components/network.py`
- Training loop and seed/checkpoint logic: `components/training.py`
- Entrypoint defaults: `scripts/train_agent.py`

## Project Structure

```text
pong_env/
├── client.py
├── gradio_app.py
├── models.py
├── README.md
├── requirements.txt
├── run_docker.bat
├── train_dqn.py
├── components/
│   ├── __init__.py
│   ├── agent.py
│   ├── environment.py
│   ├── network.py
│   └── training.py
├── scripts/
│   ├── __init__.py
│   └── train_agent.py
└── server/
    ├── app.py
    ├── Dockerfile
    └── pong_environment.py
```

## Troubleshooting

### Server connection errors

Make sure the Docker container is running:

```bash
docker run -p 8000:8000 pong-server
```

### Gradio UI issues

```bash
pip install --upgrade gradio
python gradio_app.py
```
