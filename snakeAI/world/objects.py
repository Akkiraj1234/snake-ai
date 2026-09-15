from __future__ import annotations

from snakeAI.utils import (
    Cell,
    Action,
    EAT_NON_FOOD,
    OBJECTS,
    InteractionResult,
    State,
)


def Apple(action: int, state: State) -> InteractionResult:
    """
    Resolve an interaction with an apple.

    The apple can only be consumed when the ``EAT`` action is performed.
    Its configured health and hunger effects are applied only when the
    corresponding stat is not already full.

    Args:
        action: Action performed by the snake.
        state: Current snake state used to determine whether health or hunger
            can receive the apple's effects.

    Returns:
        The reward and stat changes produced by the interaction.
    """
    if action != Action.EAT:
        return InteractionResult()

    config = OBJECTS[Cell.APPLE]

    return InteractionResult(
        reward = config.reward,
        health_change = (0 if state.is_health_full else config.health_change),
        hunger_change = (0 if state.is_hunger_full else config.hunger_change),
    )


def Banana(action: int, state: State) -> InteractionResult:
    """
    Resolve an interaction with a banana.

    The banana can only be consumed when the ``EAT`` action is performed.
    Its configured health and hunger effects are applied only when the
    corresponding stat is not already full.

    Args:
        action: Action performed by the snake.
        state: Current snake state used to determine whether health or hunger
            can receive the banana's effects.

    Returns:
        The reward and stat changes produced by the interaction.
    """
    if action != Action.EAT:
        return InteractionResult()

    config = OBJECTS[Cell.BANANA]

    return InteractionResult(
        reward = config.reward,
        health_change = (0 if state.is_health_full else config.health_change ),
        hunger_change = (0 if state.is_hunger_full else config.hunger_change ),
    )


def Frog(action: int, state: State) -> InteractionResult:
    """
    Resolve an interaction with a frog.

    The frog can only be consumed when the ``EAT`` action is performed.
    Its configured health and hunger effects are applied only when the
    corresponding stat is not already full.

    Args:
        action: Action performed by the snake.
        state: Current snake state used to determine whether health or hunger
            can receive the frog's effects.

    Returns:
        The reward and stat changes produced by the interaction.
    """
    if action != Action.EAT:
        return InteractionResult()

    config = OBJECTS[Cell.FROG]

    return InteractionResult(
        reward = config.reward,
        health_change = (0 if state.is_health_full else config.health_change),
        hunger_change = (0 if state.is_hunger_full else config.hunger_change),
    )


def Wall(action: int, state: State) -> InteractionResult:
    """
    Resolve an interaction with a wall.

    Contact with a wall applies its configured reward and health change.
    Attempting to eat the wall additionally applies the configured
    non-food eating penalty.

    Args:
        action: Action performed by the snake.
        state: Current snake state.

    Returns:
        The reward and health change produced by the interaction.
    """
    config = OBJECTS[Cell.WALL]

    reward = config.reward
    health = config.health_change

    if action == Action.EAT:
        reward += EAT_NON_FOOD.reward
        health += EAT_NON_FOOD.health_change

    return InteractionResult(
        reward = reward,
        health_change = health,
    )


def Obstacle(action: int, state: State) -> InteractionResult:
    """
    Resolve an interaction with an obstacle.

    Contact with an obstacle applies its configured reward and health change.
    Attempting to eat the obstacle additionally applies the configured
    non-food eating penalty.

    Args:
        action: Action performed by the snake.
        state: Current snake state.

    Returns:
        The reward and health change produced by the interaction.
    """
    config = OBJECTS[Cell.OBSTACLE]

    reward = config.reward
    health = config.health_change

    if action == Action.EAT:
        reward += EAT_NON_FOOD.reward
        health += EAT_NON_FOOD.health_change

    return InteractionResult(
        reward = reward,
        health_change = health,
    )


def Water(action: int, state: State) -> InteractionResult:
    """
    Resolve an interaction with water.

    Water applies its configured reward, health change, and hunger change
    regardless of the action performed.

    Args:
        action: Action performed by the snake.
        state: Current snake state.

    Returns:
        The reward and stat changes produced by the interaction.
    """
    config = OBJECTS[Cell.WATER]

    return InteractionResult(
        reward = config.reward,
        health_change = config.health_change,
        hunger_change = config.hunger_change,
    )


def Self(action: int, state: State) -> InteractionResult:
    """
    Resolve an interaction with the snake's own body.

    Contact with the snake's body applies its configured reward and health
    change. If the snake attempts to eat itself, the configured hunger change
    is also applied.

    Args:
        action: Action performed by the snake.
        state: Current snake state.

    Returns:
        The reward and stat changes produced by the interaction.
    """
    config = OBJECTS[Cell.SNAKE]

    hunger = (
        config.hunger_change
        if action == Action.EAT
        else 0
    )

    return InteractionResult(
        reward = config.reward,
        health_change = config.health_change,
        hunger_change = hunger,
    )


def Grass(action: int, state: State) -> InteractionResult:
    """
    Resolve an interaction with grass.

    Grass has no gameplay effect and therefore produces an empty interaction
    result.

    Args:
        action: Action performed by the snake.
        state: Current snake state.

    Returns:
        An empty ``InteractionResult`` with no reward or stat changes.
    """
    return InteractionResult()


obj_map = {
    Cell.GRASS: Grass,
    Cell.WALL: Wall,
    Cell.SNAKE: Self,
    Cell.WATER: Water,
    Cell.APPLE: Apple,
    Cell.BANANA: Banana,
    Cell.FROG: Frog,
    Cell.OBSTACLE: Obstacle,
}