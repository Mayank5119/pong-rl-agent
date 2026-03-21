"""
Pydantic models for Pong OpenEnv
Shared between client and server
"""

from pydantic import BaseModel, Field
from typing import Literal


class PongAction(BaseModel):
    """Action: what the agent/player sends"""
    action: Literal["UP", "DOWN", "STAY"] = Field(
        description="Player paddle movement"
    )


class PongObservation(BaseModel):
    """Observation: what the environment returns"""
    board: str = Field(description="ASCII board state (20x40)")
    player_score: int
    ai_score: int
    ball_x: float
    ball_y: float
    ball_vx: float
    ball_vy: float
    player_y: int
    ai_y: int
    done: bool = Field(default=False, description="Episode termination flag")
    info: str = Field(default="", description="Termination reason")

