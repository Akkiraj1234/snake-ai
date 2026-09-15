from __future__ import annotations
from collections import deque
import random

from snakeAI.utils import (
    SURVIVAL,
    WORLD,
    FOOD_RARITY,
    Cell,
    Direction, 
    State,
)

Position = tuple[int, int]


class Snake:
    """
    Manage the snake's survival state.

    The snake tracks only survival-related state such as health and hunger.
    Position, movement, and interactions with the world are handled by the
    world/controller.
    """
    def __init__(self) -> None:
        """
        Initialize the snake with its default survival state.
        """
        self.reset()

    def reset(self) -> None:
        """
        Restore the snake's health and hunger to their maximum values.
        """
        self.health = float(SURVIVAL.max_health)
        self.hunger = float(SURVIVAL.max_hunger)

    @property
    def is_dead(self) -> bool:
        """
        Return whether the snake has no remaining health.
        """
        return self.health <= 0

    def get_state(self) -> State:
        """
        Return the snake's current survival state.

        The returned state contains the current health, hunger, and death
        status used by the agent as part of its observation.
        """
        return State(
            self.health, 
            self.hunger, 
            self.is_dead
        )

    def apply(self, health_change: float = 0, hunger_change: float = 0) -> float:
        """
        Apply health and hunger changes and return the resulting reward.

        Changes are clamped to their valid ranges. Rewards are granted for
        health recovery and for reaching maximum health or hunger.
        """
        old_health, old_hunger = self.health, self.hunger
        self.health = max(0.0, min(float(SURVIVAL.max_health), self.health + health_change))
        self.hunger = max(0.0, min(float(SURVIVAL.max_hunger), self.hunger + hunger_change))
        
        # rewardig for health gain
        reward = max(0.0, self.health - old_health) * SURVIVAL.health_gain_reward
        
        if old_health < SURVIVAL.max_health and self.health == SURVIVAL.max_health:
            reward += SURVIVAL.full_health_reward
            
        if old_hunger < SURVIVAL.max_hunger and self.hunger == SURVIVAL.max_hunger:
            reward += SURVIVAL.full_hunger_reward
        
        return reward

    def tick(self) -> float:
        """
        Advance survival by one turn and return the resulting reward.

        Hunger decreases while food is available. Once hunger reaches zero,
        the snake instead takes starvation damage. A death reward is applied
        when the snake transitions from alive to dead.
        """
        old_health = self.health
        
        if self.hunger > 0:
            self.hunger = max(0.0, self.hunger - SURVIVAL.tick_health_decrease)
            
        else:
            self.health = max(0.0, self.health - SURVIVAL.starvation_damage)
        
        reward = SURVIVAL.starvation_reward if self.health < old_health else 0.0
        
        if old_health > 0 and self.is_dead:
            reward += SURVIVAL.death_reward
        
        return reward

class Environment:
    """
    Manage the mutable game world and the snake's spatial state.

    The environment owns the world grid, snake coordinates, movement
    direction, and deterministic world generation. Survival state such as
    health and hunger is handled separately by :class:`Snake`.

    World generation is driven by a local random generator. A fixed seed
    therefore produces the same generated world, making obstacle placement
    and other randomized elements reproducible.
    """

    def __init__(
        self,
        world_w: int = WORLD.width,
        world_h: int = WORLD.height,
        food_c: int = WORLD.food_count,
        obstacle_c: int = WORLD.obstacle_count,
        initial_body_l: int = WORLD.initial_body_length,
        maze: bool = False,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the environment with the given world configuration.

        Args:
            world_w: Width of the world grid.
            world_h: Height of the world grid.
            food_c: Number of food items to generate.
            obstacle_c: Number of obstacles to generate in normal mode.
            initial_body_l: Initial length of the snake.
            maze: Whether to generate a maze as the world layout.
            seed: Optional seed used for deterministic generation.

        Raises:
            ValueError: If the world dimensions are smaller than 10x10 or
                the initial snake cannot fit inside the world.
        """
        if world_w < 10 or world_h < 10:
            raise ValueError("world dimensions must be at least 10x10")

        if initial_body_l < 1 or initial_body_l > max(world_w - 2, world_h - 2):
            raise ValueError("initial_body_l does not fit inside the world")

        self.world_w = world_w
        self.world_h = world_h
        self.obstacle_c = max(0, obstacle_c)
        self.food_c = max(1, food_c)
        self.initial_body_l = initial_body_l

        self.maze = maze
        self.rng = random.Random(seed)
        self.world: list[list[int]] = []

        self.snake: deque[Position] = deque()
        self.snake_head: Position | None = None
        self.snake_direction = Direction.RIGHT

    def build_flat(self) -> None:
        """
        Create a new open world filled with grass cells.

        Replaces the current world grid using the configured world
        dimensions.
        """
        self.world = [
            [Cell.GRASS for _ in range(self.world_w)]
            for _ in range(self.world_h)
        ]

    def create_border(self) -> None:
        """
        Surround the world with a one-cell-wide wall border.

        The border is applied to all four edges of the existing world grid.
        """
        for x in range(self.world_w):
            self.world[0][x] = self.world[-1][x] = Cell.WALL

        for y in range(self.world_h):
            self.world[y][0] = self.world[y][-1] = Cell.WALL
    
    def _free_cells(self) -> list[Position]:
            """
            Return all unoccupied grass cells inside the world border.
    
            Border cells are excluded because they are reserved for walls.
            """
            return [
                (y, x)
                for y in range(1, self.world_h - 1)
                for x in range(1, self.world_w - 1)
                if self.world[y][x] == Cell.GRASS
            ]

    def generate_maze(self) -> None:
        """
        Generate a connected maze inside the existing world.

        Interior cells are converted to walls and passages are carved using
        randomized depth-first search. The outer world border remains intact.
        """
        for y in range(1, self.world_h - 1):
            for x in range(1, self.world_w - 1):
                self.world[y][x] = Cell.WALL

        start = (1, 1)
        self.world[start[0]][start[1]] = Cell.GRASS
        stack = [start]

        while stack:
            y, x = stack[-1]
            neighbours = []

            for dy, dx in ((-2, 0),(2, 0),(0, -2),(0, 2)): 
                ny, nx = y + dy, x + dx
                
                if (
                    1 <= ny < self.world_h - 1
                    and 1 <= nx < self.world_w - 1
                    and self.world[ny][nx] == Cell.WALL
                ):
                    neighbours.append((ny, nx))

            if not neighbours:
                stack.pop()
                continue

            ny, nx = self.rng.choice(neighbours)

            self.world[(y + ny) // 2][(x + nx) // 2] = Cell.GRASS
            self.world[ny][nx] = Cell.GRASS
            stack.append((ny, nx))
    
    def generate_snake(self) -> None:
        """
        Place the initial snake entirely inside the world border.

        The snake is placed horizontally when its initial length fits within
        the world width; otherwise it is placed vertically. The head is placed
        at the leading end and the initial direction is set accordingly.
        """
        if self.initial_body_l <= self.world_w - 2:
            y = self.world_h // 2
            head_x = (self.world_w - 2 + self.initial_body_l) // 2
            
            positions = [
                (y, head_x - i)
                for i in range(self.initial_body_l)
            ]
            self.snake_direction = Direction.RIGHT

        else:
            head_y = (self.world_h - 2 + self.initial_body_l) // 2
            x = self.world_w // 2

            positions = [
                (head_y - i, x)
                for i in range(self.initial_body_l)
            ]

            self.snake_direction = Direction.DOWN

        self.snake = deque(positions)
        self.snake_head = positions[0]

        for y, x in positions:
            self.world[y][x] = Cell.SNAKE

    def generate_obstacles(self) -> None:
        """
        Generate seeded obstacle noise across free cells.

        A fixed random seed produces the same obstacle layout, allowing the
        seed to act as a reproducible noise source for world generation.
        """
        if self.maze:
            return self.generate_maze()
        
        free = self._free_cells()

        for y, x in self.rng.sample(
            free,
            min(self.obstacle_c, len(free)),
            
        ):
            self.world[y][x] = Cell.OBSTACLE

    def generate_food(self) -> None:
        """
        Place food on randomly selected free cells.

        Food types are selected according to their configured rarity weights.
        """
        free = self._free_cells()
        foods = list(FOOD_RARITY)
        weights = list(FOOD_RARITY.values())

        for y, x in self.rng.sample(
            free,
            min(self.food_c, len(free)),
        ):
            self.world[y][x] = self.rng.choices(
                foods,
                weights=weights,
                k=1,
            )[0]

    def erase_world(self, seed: int | None = None) -> None:
        """
        Clear the current world and reset all spatial state.

        Args:
            seed: Optional new seed for subsequent world generation.
        """
        if seed is not None:
            self.rng.seed(seed)

        self.world = []
        self.snake = deque()
        self.snake_head = None
        self.snake_direction = Direction.RIGHT

    def generate_new_world(self, seed: int | None = None) -> None:
        """
        Generate a fresh world using the configured generation mode.

        The world is built from a flat grass grid and surrounded by a border.
        In maze mode, a maze is generated as the obstacle layout. Otherwise,
        seeded obstacle noise is generated independently. The snake and food
        are then placed on the remaining free cells.

        Args:
            seed: Optional new seed used for deterministic generation.
        """
        self.erase_world(seed)
        self.build_flat()
        self.create_border()
        self.generate_obstacles()
        self.generate_snake()
        self.generate_food()