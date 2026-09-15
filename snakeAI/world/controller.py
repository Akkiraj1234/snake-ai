from __future__ import annotations
from threading import RLock
import asyncio

from snakeAI.utils import (
    Action, 
    Cell,
    Direction, 
    OBJECTS, 
    InteractionResult,
    lock,
    RewardInfo,
    WORLD,
)

from .enviroment import Environment, Snake
from .objects import obj_map


_VECTORS = {
    Direction.UP: (-1, 0), 
    Direction.DOWN: (1, 0), 
    Direction.LEFT: (0, -1), 
    Direction.RIGHT: (0, 1)
}
_LEFT = {
    Direction.UP: Direction.LEFT, 
    Direction.LEFT: Direction.DOWN, 
    Direction.DOWN: Direction.RIGHT, 
    Direction.RIGHT: Direction.UP
}
_RIGHT = {
    Direction.LEFT: Direction.UP,
    Direction.DOWN: Direction.LEFT,
    Direction.RIGHT: Direction.DOWN,
    Direction.UP: Direction.RIGHT
}

# need 2 controller
# 1. async safe ui controller get frame, state and all and ai decision too
# 2. ai cotroller give frame in chunk and ui reprsentive wa
#   - allow crating new world
#   - send next move allow moving snake
#   - get ai data chunk and snake data
#   - get reward data and all



class Controller:
    """
    Coordinate game state, actions, interactions, movement, and rewards.

    The controller connects the environment and snake, applying game rules
    and managing state changes that result from agent actions.
    """
    def __init__(self, env: Environment) -> None:
        """
        Initialize a controller for the given environment.

        Args:
            env: Environment instance used to manage the world and its layout.
        """
        self.env = env
        self.snake = Snake()
        self._lock = RLock()
        self._ai_data = {}
        self._age = 0
        self._last_action = Action.IDLE
        self._last_reward = 0.0
        self._last_message = "waiting"
    
    def _target(self, direction: int) -> tuple[int, int]:
        """
        Return the cell directly ahead of the snake in the given direction.

        The returned position is calculated relative to the snake's current head
        and does not modify the environment.
        """
        y, x = self.env.snake_head  # type: ignore[misc]
        dy, dx = _VECTORS[direction]
        return y + dy, x + dx

    def _move(self, position: tuple[int, int], grow: bool) -> None:
        """
        Move the snake's head to the given position.

        The new head is added to the front of the snake and the corresponding
        world cell is marked as occupied. When ``grow`` is false, the tail is
        removed to keep the snake's length unchanged; otherwise the tail remains
        and the snake grows by one cell.
        """
        # moving snake removing back and adding front
        self.env.snake.appendleft(position)
        self.env.snake_head = position
        self.env.world[position[0]][position[1]] = Cell.SNAKE
        
        # if grow then remove back else no
        if grow: return
        
        tail = self.env.snake.pop()
        
        if tail != position:
            self.env.world[tail[0]][tail[1]] = Cell.GRASS
    
    def _cell_or_wall(self, y: int, x: int) -> int:
        """
        Return the cell at the given coordinates.

        Coordinates outside the world boundaries are treated as wall cells.
        """
        return (
            self.env.world[y][x]
            if 0 <= y < self.env.world_h
            and 0 <= x < self.env.world_w
            else Cell.WALL
        )

    @lock
    def new_world(self, seed: int | None = None) -> None:
        """
        Generate a new world and reset the snake's survival state.

        The environment is regenerated using the optional seed, after which
        the snake is restored to its initial health and hunger.

        Args:
            seed: Optional seed used to generate a deterministic world.
        """
        self.env.generate_new_world(seed)
        self.snake.reset()
        self._age = 0
        self._last_action = Action.IDLE
        self._last_reward = 0.0
        self._last_message = "new world"
    
    @lock
    def get_frame(self, width: int | None = None, height: int | None = None) -> list[list[int]]:
        """
        Return a read-only copy of the requested world view.

        When ``width`` and ``height`` are omitted, a copy of the complete world
        grid is returned. When both are provided, a view centered on the snake's
        head is returned with the requested dimensions.

        Cells outside the world boundaries are represented as wall cells.
        The returned grid is independent of the environment and can therefore
        be modified by the caller without affecting the world state.

        Args:
            width: Width of the head-centered view. Must be provided together
                with ``height``.
            height: Height of the head-centered view. Must be provided together
                with ``width``.

        Returns:
            A two-dimensional grid containing the requested world view.
        """
        if width is None or height is None:
            return [row[:] for row in self.env.world]

        hy, hx = self.env.snake_head   # type: ignore[misc]
        get_cell = lambda x, y : self._cell_or_wall(
            hy + y - height // 2, hx + x - width // 2
        )

        return [
            [get_cell(x, y) for x in range(width)]
            for y in range(height)
        ]

    @lock
    def act(self, action: int) -> RewardInfo:
        """
        Apply one action and advance the environment by one turn.

        The action is resolved relative to the snake's current direction, and the
        cell in front of the snake is evaluated through its configured interaction
        handler. Any resulting health, hunger, and reward changes are applied
        before movement is performed. Survival effects are then advanced for the
        turn, and the outcome is returned as a reward, termination state, and
        concise action message.

        Args:
            action: Action to perform during the turn.

        Raises:
            ValueError: If ``action`` is not a recognized action.

        Returns:
            A tuple containing the reward earned during the turn, whether the
            snake is dead, and a concise message describing the action outcome.
        """
        if action not in Action._data.values():
            raise ValueError(f"unknown action: {action!r}")

        if not self.env.snake or self.snake.is_dead:
            return 0.0, True, "dead_or_uninitialized"

        if action == Action.IDLE:
            reward = self.snake.tick()
            message = "snake did not move"
            self._record_turn(action, reward, message)
            return reward, self.snake.is_dead, message

        # Resolve the direction from the current heading.
        direction = self.env.snake_direction

        if action == Action.LEFT:
            direction = _LEFT[direction]

        elif action == Action.RIGHT:
            direction = _RIGHT[direction]

        # Forward movement is represented by the resolved direction.
        target = self._target(direction)
        y, x = target

        target_cell = self._cell_or_wall(y, x)
        cell_name = next(
            name for name, value in Cell._data.items() if value == target_cell
        )
        
        # Interaction handlers apply rewards/stat changes. Configuration still
        # owns spatial rules such as blocking movement and growing the snake.
        object_config = getattr(OBJECTS, cell_name)
        object_handler = obj_map.get(target_cell)

        result = (
            object_handler(
                action,
                self.snake.get_state(),
            )
            if object_handler
            else InteractionResult()
        )

        reward = result.reward

        # Apply any health/hunger changes caused by the interaction.
        reward += self.snake.apply(
            result.health_change,
            result.hunger_change,
        )

        blocked = object_config.blocks_movement
        
        ate_food = (
            action == Action.EAT
            and object_config.consumable
        )

        if not blocked:
            self._move(
                target,
                grow=ate_food and object_config.grows_snake,
            )
            self.env.snake_direction = direction

        # Survival effects are applied once per turn after the action.
        reward += self.snake.tick()

        cell_name = cell_name.lower()
        if action == Action.EAT:
            message = f"ate {cell_name}" if ate_food else f"tried to eat {cell_name}"
        elif blocked:
            message = f"hit {cell_name}"
        else:
            movement = {
                Action.FORWARD: "moved forward",
                Action.LEFT: "turned left and moved",
                Action.RIGHT: "turned right and moved",
            }[action]
            message = f"{movement} onto {cell_name}"

        self._record_turn(action, reward, message)
        return reward, self.snake.is_dead, message

    def _record_turn(self, action: int, reward: float, message: str) -> None:
        """Store the small status snapshot used by read-only renderers."""
        self._age += 1
        self._last_action = action
        self._last_reward = reward
        self._last_message = message

    async def act_async(self, action: int) -> RewardInfo:
        """
        Apply an action asynchronously and return its turn outcome.

        The action is executed in a worker thread through :func:`asyncio.to_thread`,
        allowing the synchronous controller logic to run without blocking the
        event loop.

        Args:
            action: Action to perform during the turn.

        Returns:
            A tuple containing the reward earned during the turn, whether the
            snake is dead, and a concise message describing the action outcome.

        Raises:
            ValueError: If ``action`` is not a recognized action.
        """
        return await asyncio.to_thread(self.act, action)

    def save_data(self, ai_result: dict[str, object]) -> None:
        """Store optional trainer metrics for presentation-only consumers."""
        self._ai_data = ai_result


class AIController:
    """
    AI-facing commands and local observations.
    """
    def __init__(self, controller: Controller) -> None: 
        self._controller = controller
    
    def act(self, action: int): 
        return self._controller.act(action)
    
    async def act_async(self, action: int): 
        return await self._controller.act_async(action)
    
    def get_frame(self, width: int = 10, height: int = 10): 
        return self._controller.get_frame(width, height)
    
    def new_world(self, seed: int | None = None): 
        self._controller.new_world(seed)
        
    def send_ai_result(self, data: dict[str, object]) -> None:
        """Publish optional Q-learning metrics for the terminal UI."""
        self._controller.save_data(data)


class UIController:
    """
    Read-only rendering handle; rendering cannot mutate the simulation.
    """
    def __init__(self, controller: Controller) -> None: 
        self._controller = controller
        
    def get_frame_data(self):
        """Return the grid and the compact status data needed by a UI."""
        state = self._controller.snake.get_state()
        return self._controller.get_frame(), {
            **self._controller._ai_data,
            "age": self._controller._age,
            "health": state.health,
            "hunger": state.hunger,
            "action": self._controller._last_action,
            "reward": self._controller._last_reward,
            "message": self._controller._last_message,
        }
    
    async def get_frame_async(self): 
        return await asyncio.to_thread(self._controller.get_frame)


def build_world(
    world_w: int = WORLD.width, 
    world_h: int = WORLD.height,
    obstacle_c: int = WORLD.obstacle_count, 
    food_c: int = WORLD.food_count,
    initial_body_l: int = WORLD.initial_body_length,
    maze: bool = False, seed: int | None = None
    
) -> tuple[UIController, AIController]:
    """
    Create one shared simulation and return ``(ui_controller, ai_controller)``.
    """
    core = Controller(Environment(
        world_w=world_w,
        world_h=world_h,
        obstacle_c=obstacle_c,
        food_c=food_c,
        initial_body_l=initial_body_l,
        maze=maze,
        seed=seed,
    ))
    
    core.new_world(seed)
    return UIController(core), AIController(core)
