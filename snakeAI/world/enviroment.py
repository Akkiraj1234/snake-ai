"""The in-memory snake world, with no AI or rendering concerns."""
from __future__ import annotations

from collections import deque
import random

from snakeAI.utils import (
    Cell, Direction, FOOD_RARITY, FOOD_COUNT, INITIAL_BODY_LENGTH,
    MAX_HEALTH, MAX_HUNGER, FULL_HEALTH_REWARD, FULL_HUNGER_REWARD,
    HEALTH_GAIN_REWARD, STARVATION_DAMAGE, STARVATION_REWARD, DEATH_REWARD,
    OBSTACLE_COUNT, WORLD_H, WORLD_W, State,
)

Position = tuple[int, int]


class Snake:
    """Survival state only. Coordinates and movement belong to the world/controller."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.health = float(MAX_HEALTH)
        self.hunger = float(MAX_HUNGER)

    @property
    def is_dead(self) -> bool:
        return self.health <= 0

    def get_state(self) -> State:
        return State(self.health, self.hunger, self.is_dead)

    def apply(self, health_change: float = 0, hunger_change: float = 0) -> float:
        """Apply one object's effects and return all resulting survival reward."""
        old_health, old_hunger = self.health, self.hunger
        self.health = max(0.0, min(float(MAX_HEALTH), self.health + health_change))
        self.hunger = max(0.0, min(float(MAX_HUNGER), self.hunger + hunger_change))
        reward = max(0.0, self.health - old_health) * HEALTH_GAIN_REWARD
        if old_health < MAX_HEALTH and self.health == MAX_HEALTH:
            reward += FULL_HEALTH_REWARD
        if old_hunger < MAX_HUNGER and self.hunger == MAX_HUNGER:
            reward += FULL_HUNGER_REWARD
        return reward

    def tick(self) -> float:
        """Advance a turn and return starvation/death reward."""
        old_health = self.health
        if self.hunger > 0:
            self.hunger = max(0.0, self.hunger - 1)
        else:
            self.health = max(0.0, self.health - STARVATION_DAMAGE)
        reward = STARVATION_REWARD if self.health < old_health else 0.0
        if old_health > 0 and self.is_dead:
            reward += DEATH_REWARD
        return reward


class Enviroment:  # Original public spelling retained for compatibility.
    """Mutable grid, snake coordinates, and deterministic generation only."""

    def __init__(self, world_w: int = WORLD_W, world_h: int = WORLD_H,
                 obsital_c: int = OBSTACLE_COUNT, food_c: int = FOOD_COUNT,
                 initial_body_l: int = INITIAL_BODY_LENGTH, maze: bool = False,
                 seed: int | None = None) -> None:
        if world_w < 3 or world_h < 3:
            raise ValueError("world dimensions must be at least 3x3")
        if initial_body_l < 1 or initial_body_l > max(world_w - 2, world_h - 2):
            raise ValueError("initial_body_l does not fit inside the world")
        self.world_w, self.world_h = world_w, world_h
        self.obsital_c, self.food_c = max(0, obsital_c), max(0, food_c)
        self.initial_body_l, self.maze = initial_body_l, maze
        self.rng = random.Random(seed)
        self.world: list[list[int]] = []
        self.snake: deque[Position] = deque()
        self.snake_head: Position | None = None
        self.snake_direction = Direction.RIGHT

    def build_flat(self) -> None:
        self.world = [[Cell.GRASS for _ in range(self.world_w)] for _ in range(self.world_h)]

    def create_border(self) -> None:
        for x in range(self.world_w):
            self.world[0][x] = self.world[-1][x] = Cell.WALL
        for y in range(self.world_h):
            self.world[y][0] = self.world[y][-1] = Cell.WALL

    def _free_cells(self) -> list[Position]:
        return [(y, x) for y in range(1, self.world_h - 1) for x in range(1, self.world_w - 1)
                if self.world[y][x] == Cell.GRASS]

    def genrate_snake(self) -> None:
        """Place a straight snake completely inside the border, headed right/down."""
        if self.initial_body_l <= self.world_w - 2:
            y, head_x = self.world_h // 2, (self.world_w - 2 + self.initial_body_l) // 2
            positions = [(y, head_x - i) for i in range(self.initial_body_l)]
            self.snake_direction = Direction.RIGHT
        else:
            head_y, x = (self.world_h - 2 + self.initial_body_l) // 2, self.world_w // 2
            positions = [(head_y - i, x) for i in range(self.initial_body_l)]
            self.snake_direction = Direction.DOWN
        self.snake, self.snake_head = deque(positions), positions[0]
        for y, x in positions:
            self.world[y][x] = Cell.SNAKE

    def genrate_opsiticals(self) -> None:
        free = self._free_cells()
        if self.maze:
            head_y, head_x = self.snake_head  # type: ignore[misc]
            for y, x in free:
                if y % 2 == 0 and y != head_y and x != head_x:
                    self.world[y][x] = Cell.WALL
            return
        for y, x in self.rng.sample(free, min(self.obsital_c, len(free))):
            self.world[y][x] = Cell.OBSTACLE

    def genrate_food(self) -> None:
        free = self._free_cells()
        foods, weights = list(FOOD_RARITY), list(FOOD_RARITY.values())
        for y, x in self.rng.sample(free, min(self.food_c, len(free))):
            self.world[y][x] = self.rng.choices(foods, weights=weights, k=1)[0]

    def erase_world(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)
        self.world, self.snake, self.snake_head = [], deque(), None
        self.snake_direction = Direction.RIGHT

    def genrate_new_world(self, seed: int | None = None) -> None:
        self.erase_world(seed)
        self.build_flat()
        self.create_border()
        self.genrate_snake()
        self.genrate_opsiticals()
        self.genrate_food()


Environment = Enviroment


def build_world(*args, **kwargs):
    """Compatibility import; new code should import this from ``controller``."""
    from .controller import build_world as _build_world
    return _build_world(*args, **kwargs)
