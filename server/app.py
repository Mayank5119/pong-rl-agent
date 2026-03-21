"""
FastAPI server for Pong OpenEnv
Serves WebSocket endpoint for clients to connect to
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import json
import sys
import os

# Add parent directory to path to import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PongAction, PongObservation
from pong_environment import PongGame


app = FastAPI(title="Pong OpenEnv Server")

# Active sessions
sessions = {}


@app.get("/")
async def root():
    """Serve home page"""
    return {"message": "Pong OpenEnv Server", "status": "running"}


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for game communication"""
    await websocket.accept()

    # Create game instance for this client
    game = PongGame()
    step_count = 0
    max_steps_per_episode = 1000  # Timeout after this many steps
    sessions[client_id] = game

    try:
        # Send initial observation
        game.reset()
        step_count = 0
        obs = _get_observation(game)
        await websocket.send_text(json.dumps({
            "type": "reset",
            "observation": obs.dict()
        }))

        # Main loop
        while True:
            # Receive action from client
            message = await websocket.receive_text()
            data = json.loads(message)

            if data["type"] == "action":
                # Execute step
                action = data["action"]
                state, reward, done = game.step(action)
                step_count += 1

                # Check for timeout (prevent infinite games)
                if step_count >= max_steps_per_episode:
                    done = True
                    info = "Max steps reached"
                else:
                    info = ""

                obs = _get_observation(game, done=done, info=info)

                await websocket.send_text(json.dumps({
                    "type": "step",
                    "observation": obs.dict(),
                    "reward": reward,
                    "done": done
                }))

                if done:
                    # Auto-reset on game end
                    game.reset()
                    step_count = 0
                    obs = _get_observation(game)
                    await websocket.send_text(json.dumps({
                        "type": "reset",
                        "observation": obs.dict()
                    }))

            elif data["type"] == "reset":
                # Manual reset
                game.reset()
                step_count = 0
                obs = _get_observation(game)
                await websocket.send_text(json.dumps({
                    "type": "reset",
                    "observation": obs.dict()
                }))

    except WebSocketDisconnect:
        if client_id in sessions:
            del sessions[client_id]
    except Exception as e:
        print(f"Error: {e}")
        if client_id in sessions:
            del sessions[client_id]


def _get_observation(game: PongGame, done: bool = False, info: str = "") -> PongObservation:
    """Convert game state to observation"""
    return PongObservation(
        board=game.render_board(),
        player_score=game.player_score,
        ai_score=game.ai_score,
        ball_x=game.ball_x,
        ball_y=game.ball_y,
        ball_vx=game.ball_vx,
        ball_vy=game.ball_vy,
        player_y=int(game.player_y),
        ai_y=int(game.ai_y),
        done=done,
        info=info
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

