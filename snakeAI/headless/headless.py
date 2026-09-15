from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..var import (
    Cell,
    Direction,
    Action,
    WORLD_W,
    WORLD_H,
    OBSTACLE_COUNT,
    INITIAL_BODY_LENGTH,
    FOOD,
    MAX_HEALTH,
    MAX_HUNGER,
)
from ..objects import State


Position = tuple[int, int]


@dataclass(slots=True)
class EnvironmentConfig:
    width: int = WORLD_W
    height: int = WORLD_H

    main_ground: Cell = Cell.GRASS
    other_grounds: tuple[Cell, ...] = ()

    border: Cell = Cell.WALL

    obstacle_count: int = OBSTACLE_COUNT

    food: dict = field(
        default_factory=lambda: FOOD
    )

    initial_body_length: int = INITIAL_BODY_LENGTH


class Environment:
    """
    Headless game environment.

    This class contains the world and its rules.
    It does not know anything about Pygame or the neural network.
    """

    def __init__(
        self,
        config: EnvironmentConfig | None = None,
    ) -> None:
        self.config = config or EnvironmentConfig()

        self.width = self.config.width
        self.height = self.config.height

        self.world: list[list[Cell]] = []

        self.snake: list[Position] = []

        self.direction = Direction.RIGHT

        self.state = State(
            health=MAX_HEALTH,
            hunger=MAX_HUNGER,
            direction=self.direction,
        )

        self.food: dict[Position, Cell] = {}

        self.reset()

    # ---------------------------------------------------------
    # WORLD
    # ---------------------------------------------------------

    def reset(self) -> None:
        """Create a completely new environment."""

        self.state.health = MAX_HEALTH
        self.state.hunger = MAX_HUNGER
        self.state._dead = False

        self.direction = Direction.RIGHT
        self.state.direction = self.direction

        self._create_world()
        self._create_snake()
        self._spawn_obstacles()
        self._spawn_food()

    def _create_world(self) -> None:
        """Create the base terrain."""

        self.world = [
            [
                self.config.main_ground
                for _ in range(self.width)
            ]
            for _ in range(self.height)
        ]

    def _create_snake(self) -> None:
        """Create the initial snake."""

        cx = self.width // 2
        cy = self.height // 2

        self.snake = [
            (cx - i, cy)
            for i in range(self.config.initial_body_length)
        ]

    # ---------------------------------------------------------
    # OBJECT PLACEMENT
    # ---------------------------------------------------------

    def _spawn_obstacles(self) -> None:
        """Spawn obstacles without placing them on the snake."""

        occupied = set(self.snake)

        candidates = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in occupied
        ]

        random.shuffle(candidates)

        for position in candidates[:self.config.obstacle_count]:
            x, y = position
            self.world[y][x] = Cell.OBSTACLE

    def _spawn_food(self) -> None:
        """Spawn food according to the configured food rules."""

        self.food.clear()

        occupied = set(self.snake)

        for y in range(self.height):
            for x in range(self.width):
                if self.world[y][x] != Cell.GRASS:
                    continue

                if (x, y) in occupied:
                    continue

                # Food spawning can be expanded later.
                pass

    # ---------------------------------------------------------
    # MOVEMENT
    # ---------------------------------------------------------

    def step(self, action: Action) -> None:
        """
        Execute one simulation step.

        The neural network will eventually call this indirectly.
        """

        if self.state.is_dead:
            return

        self._apply_action(action)
        self._move_snake()
        self._update_survival()

    def _apply_action(self, action: Action) -> None:

        if action == Action.LEFT:
            self.direction = self._turn_left(self.direction)

        elif action == Action.RIGHT:
            self.direction = self._turn_right(self.direction)

        elif action == Action.FORWARD:
            pass

        elif action == Action.IDLE:
            pass

        self.state.direction = self.direction

    def _move_snake(self) -> None:

        head_x, head_y = self.snake[0]

        dx, dy = self._direction_vector(self.direction)

        new_head = (
            head_x + dx,
            head_y + dy,
        )

        self.snake.insert(0, new_head)
        self.snake.pop()

    # ---------------------------------------------------------
    # RULES
    # ---------------------------------------------------------

    def _update_survival(self) -> None:

        self.state.hunger -= 1

        if self.state.hunger <= 0:
            self.state.hunger = 0
            self.state.health -= 1

        if self.state.health <= 0:
            self.state.health = 0
            self.state._dead = True

    # ---------------------------------------------------------
    # DIRECTION
    # ---------------------------------------------------------

    @staticmethod
    def _direction_vector(
        direction: Direction,
    ) -> tuple[int, int]:

        return {
            Direction.UP: (0, -1),
            Direction.DOWN: (0, 1),
            Direction.LEFT: (-1, 0),
            Direction.RIGHT: (1, 0),
        }[direction]

    @staticmethod
    def _turn_left(
        direction: Direction,
    ) -> Direction:

        return {
            Direction.UP: Direction.LEFT,
            Direction.LEFT: Direction.DOWN,
            Direction.DOWN: Direction.RIGHT,
            Direction.RIGHT: Direction.UP,
        }[direction]

    @staticmethod
    def _turn_right(
        direction: Direction,
    ) -> Direction:

        return {
            Direction.UP: Direction.RIGHT,
            Direction.RIGHT: Direction.DOWN,
            Direction.DOWN: Direction.LEFT,
            Direction.LEFT: Direction.UP,
        }[direction]

    # ---------------------------------------------------------
    # WORLD ACCESS
    # ---------------------------------------------------------

    def get_cell(self, position: Position) -> Cell:

        x, y = position

        if x < 0 or x >= self.width:
            return self.config.border

        if y < 0 or y >= self.height:
            return self.config.border

        if position in self.snake:
            return Cell.SNAKE

        if position in self.food:
            return self.food[position]

        return self.world[y][x]

    def get_view(
        self,
        width: int,
        height: int,
    ) -> list[list[Cell]]:

        head_x, head_y = self.snake[0]

        start_x = head_x - width // 2
        start_y = head_y - height // 2

        return [
            [
                self.get_cell(
                    (start_x + x, start_y + y)
                )
                for x in range(width)
            ]
            for y in range(height)
        ]