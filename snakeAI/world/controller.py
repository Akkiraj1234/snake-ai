"""AI and UI handles over one thread-safe snake simulation."""
from __future__ import annotations

import asyncio
from threading import RLock
from typing import Any

from snakeAI.utils import (Action, Cell, Direction, OBJECTS, InteractionResult,
                           WORLD_W, WORLD_H, OBSTACLE_COUNT, FOOD_COUNT, INITIAL_BODY_LENGTH)
from .enviroment import Enviroment, Snake
from .objects import obj_map

_VECTORS = {Direction.UP: (-1, 0), Direction.DOWN: (1, 0), Direction.LEFT: (0, -1), Direction.RIGHT: (0, 1)}
_LEFT = {Direction.UP: Direction.LEFT, Direction.LEFT: Direction.DOWN, Direction.DOWN: Direction.RIGHT, Direction.RIGHT: Direction.UP}
_RIGHT = {value: key for key, value in _LEFT.items()}


class Controller:
    """The rule engine: actions, object interaction, movement, and rewards."""

    def __init__(self, env: Enviroment) -> None:
        self.env, self.snake, self._lock = env, Snake(), RLock()

    def new_world(self, seed: int | None = None) -> None:
        with self._lock:
            self.env.genrate_new_world(seed)
            self.snake.reset()

    def _target(self, direction: int) -> tuple[int, int]:
        y, x = self.env.snake_head  # type: ignore[misc]
        dy, dx = _VECTORS[direction]
        return y + dy, x + dx

    def _move(self, position: tuple[int, int], grow: bool) -> None:
        self.env.snake.appendleft(position)
        self.env.snake_head = position
        self.env.world[position[0]][position[1]] = Cell.SNAKE
        if not grow:
            tail = self.env.snake.pop()
            if tail != position:
                self.env.world[tail[0]][tail[1]] = Cell.GRASS

    def act(self, action: int) -> tuple[float, bool, dict[str, Any]]:
        """Apply one relative action and return ``(reward, done, info)``."""
        with self._lock:
            if not self.env.snake or self.snake.is_dead:
                return 0.0, True, {"reason": "dead_or_uninitialized"}
            if action not in (Action.IDLE, Action.EAT, Action.FORWARD, Action.LEFT, Action.RIGHT):
                raise ValueError(f"unknown action: {action!r}")
            if action == Action.IDLE:
                reward = self.snake.tick()
                return reward, self.snake.is_dead, {"action": action, "moved": False}

            direction = self.env.snake_direction
            if action == Action.LEFT:
                direction = _LEFT[direction]
            elif action == Action.RIGHT:
                direction = _RIGHT[direction]
            y, x = self._target(direction)
            target_cell = Cell.WALL if not (0 <= y < self.env.world_h and 0 <= x < self.env.world_w) else self.env.world[y][x]
            object_config = OBJECTS[target_cell]
            object_handler = obj_map.get(target_cell)
            interaction_action = Action.EAT if action == Action.EAT else Action.FORWARD
            result = object_handler(interaction_action, self.snake.get_state()) if object_handler else InteractionResult()
            reward = result.reward + self.snake.apply(result.health_change, result.hunger_change)
            blocked = object_config.blocks_movement
            ate_food = action == Action.EAT and object_config.consumable
            if not blocked:
                self._move((y, x), grow=ate_food and object_config.grows_snake)
                self.env.snake_direction = direction
            reward += self.snake.tick()
            return reward, self.snake.is_dead, {
                "action": action, "target": (y, x), "target_cell": target_cell,
                "moved": not blocked, "ate_food": ate_food, "direction": self.env.snake_direction,
            }

    async def act_async(self, action: int) -> tuple[float, bool, dict[str, Any]]:
        return await asyncio.to_thread(self.act, action)

    def _cell_or_wall(self, y: int, x: int) -> int:
        return self.env.world[y][x] if 0 <= y < self.env.world_h and 0 <= x < self.env.world_w else Cell.WALL

    def get_frame(self, width: int | None = None, height: int | None = None) -> dict[str, Any]:
        """Copy a full grid or a head-centred AI view; callers cannot mutate state."""
        with self._lock:
            if width is None or height is None:
                grid = [row[:] for row in self.env.world]
            else:
                if width <= 0 or height <= 0:
                    raise ValueError("frame dimensions must be positive")
                hy, hx = self.env.snake_head  # type: ignore[misc]
                grid = [[self._cell_or_wall(hy + y - height // 2, hx + x - width // 2)
                         for x in range(width)] for y in range(height)]
            return {"world": grid, "snake": list(self.env.snake), "state": self.snake.get_state(), "direction": self.env.snake_direction}


class AIController:
    """AI-facing commands and local observations."""
    def __init__(self, controller: Controller) -> None: self._controller = controller
    def act(self, action: int): return self._controller.act(action)
    async def act_async(self, action: int): return await self._controller.act_async(action)
    def get_frame(self, width: int = 10, height: int = 10): return self._controller.get_frame(width, height)
    def new_world(self, seed: int | None = None): self._controller.new_world(seed)


class UIController:
    """Read-only rendering handle; rendering cannot mutate the simulation."""
    def __init__(self, controller: Controller) -> None: self._controller = controller
    def get_frame(self): return self._controller.get_frame()
    async def get_frame_async(self): return await asyncio.to_thread(self.get_frame)


def build_world(world_w: int = WORLD_W, world_h: int = WORLD_H, obsital_c: int = OBSTACLE_COUNT,
                food_c: int = FOOD_COUNT, initial_body_l: int = INITIAL_BODY_LENGTH,
                maze: bool = False, seed: int | None = None) -> tuple[UIController, AIController]:
    """Create one shared simulation and return ``(ui_controller, ai_controller)``."""
    core = Controller(Enviroment(world_w, world_h, obsital_c, food_c, initial_body_l, maze, seed))
    core.new_world(seed)
    return UIController(core), AIController(core)
