"""
Pong Game Environment - Pure game logic (no dependencies except numpy)
"""

import numpy as np
from typing import Tuple, Dict
import math


class PongGame:
    """Core Pong game logic"""

    BOARD_WIDTH = 40
    BOARD_HEIGHT = 40
    PADDLE_HEIGHT = 3
    BALL_SIZE = 1

    BALL_SPEED = 0.3
    AI_SPEED = 0.4
    PADDLE_SPEED = 2

    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset game state"""
        self.ball_x = self.BOARD_WIDTH / 2
        self.ball_y = self.BOARD_HEIGHT / 2

        angle = np.random.uniform(-45, 45) * (math.pi / 180)
        direction = 1 if np.random.random() > 0.5 else -1
        self.ball_vx = direction * self.BALL_SPEED * math.cos(angle)
        self.ball_vy = self.BALL_SPEED * math.sin(angle)

        self.player_y = (self.BOARD_HEIGHT - self.PADDLE_HEIGHT) // 2
        self.ai_y = (self.BOARD_HEIGHT - self.PADDLE_HEIGHT) // 2

        self.player_score = 0
        self.ai_score = 0

        return self._get_state()

    def step(self, player_action: str) -> Tuple[Dict, int, bool]:
        """Execute one game step"""
        reward = 0

        # Update player paddle
        if player_action == "UP":
            self.player_y = max(0, self.player_y - self.PADDLE_SPEED)
        elif player_action == "DOWN":
            self.player_y = min(self.BOARD_HEIGHT - self.PADDLE_HEIGHT,
                               self.player_y + self.PADDLE_SPEED)

        # Update AI paddle
        ai_center = self.ai_y + self.PADDLE_HEIGHT / 2
        if ai_center < self.ball_y - 1:
            self.ai_y = min(self.BOARD_HEIGHT - self.PADDLE_HEIGHT,
                           self.ai_y + self.AI_SPEED)
        elif ai_center > self.ball_y + 1:
            self.ai_y = max(0, self.ai_y - self.AI_SPEED)

        # Update ball position
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Ball collision with top/bottom walls
        if self.ball_y < 0:
            self.ball_y = -self.ball_y
            self.ball_vy = -self.ball_vy
        elif self.ball_y > self.BOARD_HEIGHT - 1:
            self.ball_y = 2 * (self.BOARD_HEIGHT - 1) - self.ball_y
            self.ball_vy = -self.ball_vy

        # Ball collision with player paddle (left side)
        if (self.ball_x < 2 and
            self.player_y <= self.ball_y <= self.player_y + self.PADDLE_HEIGHT):
            self.ball_x = 2 - (self.ball_x - 2)
            self.ball_vx = -self.ball_vx * 1.05
            hit_pos = (self.ball_y - self.player_y) / self.PADDLE_HEIGHT
            self.ball_vy += (hit_pos - 0.5) * 0.3
            reward += 0.2

        # Ball collision with AI paddle (right side)
        if (self.ball_x > self.BOARD_WIDTH - 3 and
            self.ai_y <= self.ball_y <= self.ai_y + self.PADDLE_HEIGHT):
            self.ball_x = 2 * (self.BOARD_WIDTH - 3) - self.ball_x
            self.ball_vx = -self.ball_vx * 1.05
            hit_pos = (self.ball_y - self.ai_y) / self.PADDLE_HEIGHT
            self.ball_vy += (hit_pos - 0.5) * 0.3

        # Ball out of bounds - left side (AI scores)
        if self.ball_x < 0:
            self.ai_score += 1
            reward -= 3
            self._reset_ball()

        # Ball out of bounds - right side (player scores)
        if self.ball_x > self.BOARD_WIDTH:
            self.player_score += 1
            reward += 5
            self._reset_ball()

        done = self.player_score >= 7 or self.ai_score >= 7

        return self._get_state(), reward, done

    def _reset_ball(self):
        """Reset ball to center"""
        self.ball_x = self.BOARD_WIDTH / 2
        self.ball_y = self.BOARD_HEIGHT / 2

        angle = np.random.uniform(-45, 45) * (math.pi / 180)
        direction = 1 if np.random.random() > 0.5 else -1
        self.ball_vx = direction * self.BALL_SPEED * math.cos(angle)
        self.ball_vy = self.BALL_SPEED * math.sin(angle)

    def _get_state(self) -> Tuple:
        return (
            self.ball_x, self.ball_y, self.ball_vx, self.ball_vy,
            self.player_y, self.ai_y, self.player_score, self.ai_score
        )

    def render_board(self) -> str:
        """Render game board as ASCII"""
        board = [['.' for _ in range(self.BOARD_WIDTH)] for _ in range(self.BOARD_HEIGHT)]

        # Draw player paddle (left side)
        for y in range(int(self.player_y), int(self.player_y) + self.PADDLE_HEIGHT):
            if 0 <= y < self.BOARD_HEIGHT:
                board[y][0] = '|'

        # Draw AI paddle (right side)
        for y in range(int(self.ai_y), int(self.ai_y) + self.PADDLE_HEIGHT):
            if 0 <= y < self.BOARD_HEIGHT:
                board[y][self.BOARD_WIDTH - 1] = '|'

        # Draw center line
        for y in range(self.BOARD_HEIGHT):
            if y % 2 == 0:
                board[y][self.BOARD_WIDTH // 2] = ':'

        # Draw ball
        ball_x_int = int(round(self.ball_x))
        ball_y_int = int(round(self.ball_y))
        if 0 <= ball_x_int < self.BOARD_WIDTH and 0 <= ball_y_int < self.BOARD_HEIGHT:
            board[ball_y_int][ball_x_int] = 'O'

        result = ""
        for row in board:
            result += ''.join(row) + '\n'

        return result.strip()
