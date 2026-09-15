"""
Shared values loaded from ``config.toml``.

This module only loads configuration and exposes simple shared data types.
Game rules stay in objects.py and controller.py.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias
from dataclasses import dataclass
from functools import wraps
from threading import RLock
from pathlib import Path
import tomllib
import sys
import os


class ENUM:
    """
    Provides attribute-based access to named and nested values.

    An ``ENUM`` wraps a mapping and exposes its keys as attributes, allowing
    values to be accessed using dot notation instead of dictionary indexing.
    Nested mappings can be represented by nested ``ENUM`` instances, enabling
    hierarchical access such as ``CONFIG.survival.max_health``.

    Unlike :class:`enum.Enum`, this implementation returns the underlying
    values directly rather than wrapping them in enum members. This is useful
    for values such as ``Cell.APPLE`` and ``Action.EAT``, where the AI and
    simulation operate directly on integer values. It avoids requiring
    ``.value`` at every use site and keeps the representation lightweight.

    Missing attributes are treated as undefined values. A configuration
    warning is printed with the full attribute path, and ``0`` is returned
    as the fallback value.

    Args:
        data: Mapping containing the values exposed by this instance.
        name: Dot-separated path identifying this instance, used in warnings
            for missing values.
    """
    def __init__(self, data: dict[str, Any], name: str = "") -> None:
        self._data = data
        self._name = name

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        
        except KeyError:
            print(f"[CONFIG] Missing: '{self._name}.{name}' using default: 0")
            return 0


def _build_enum(data: dict[str, Any], prefix: str = "") -> ENUM:
    """
    Recursively convert a mapping into nested ``ENUM`` instances.

    Scalar values are preserved as-is, while nested mappings are recursively
    converted into ``ENUM`` instances. The accumulated path is passed to
    each nested instance so missing-value warnings can identify the complete
    configuration path.

    Args:
        data: Mapping to convert into an ``ENUM`` hierarchy.
        prefix: Dot-separated path of the current mapping within the
            configuration hierarchy.

    Returns:
        An ``ENUM`` containing the converted mapping and its nested values.
    """
    result = {}

    for name, value in data.items():
        path = f"{prefix}.{name}"

        if isinstance(value, dict):
            result[name] = _build_enum(value, path)
        else:
            result[name] = value

    return ENUM(result, prefix)


def _load_config(prefix: str = "config") -> ENUM:
    """
    Load and convert the application configuration from ``config.toml``.

    The TOML file is parsed using :mod:`tomllib` and recursively converted
    into nested ``ENUM`` instances, providing attribute-based access to the
    configuration.

    Args:
        prefix: Root name used when reporting missing configuration values.

    Returns:
        The loaded configuration as a nested ``ENUM`` hierarchy.
    """
    config_path = Path(__file__).parent.with_name("config.toml")
    
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)
    
    return _build_enum(raw_config, prefix)

 
# =========================================================
# Var name
# =========================================================
CONFIG: ENUM = _load_config()

Cell = ENUM({
    "GRASS": 0,
    "WALL": 1,
    "SNAKE": 2,
    "WATER": 3,
    "APPLE": 4,
    "BANANA": 5,
    "FROG": 6,
    "OBSTACLE": 7
})

Action = ENUM({
    "IDLE": 0,
    "EAT": 1,
    "FORWARD": 2,
    "LEFT": 3,
    "RIGHT": 4
})

Direction = ENUM({
    "UP": 0,
    "DOWN": 1,
    "LEFT": 2,
    "RIGHT": 3
})

SURVIVAL = CONFIG.survival
WORLD = CONFIG.world
AI = CONFIG.ai
RUNTIME = CONFIG.runtime
OBJECTS = CONFIG.objects

EAT_NON_FOOD = CONFIG.eat_non_food

FOOD_RARITY = {
    getattr(Cell, name): values.spawn_weight
    for name, values in CONFIG.objects._data.items()
    if values.spawn_weight > 0
}

# =========================================================
# data class
# =========================================================

@dataclass(slots=True)
class State:
    health: float = SURVIVAL.max_health
    hunger: float = SURVIVAL.max_hunger
    _dead: bool = False

    @property
    def is_hunger_full(self) -> bool:
        return self.hunger >= SURVIVAL.max_hunger

    @property
    def is_health_full(self) -> bool:
        return self.health >= SURVIVAL.max_health

    @property
    def is_dead(self) -> bool:
        return self._dead


@dataclass(slots=True)
class InteractionResult:
    reward: float = 0.0
    health_change: float = 0.0
    hunger_change: float = 0.0


RewardInfo: TypeAlias = tuple[float, bool, str]

# ==================================================
# utils
# ==================================================

def lock(func: Callable) -> Callable:
    """
    Protect an instance method with the instance's ``_lock``.

    The decorated method executes while ``self._lock`` is held and releases
    the lock automatically when the method returns or raises an exception.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
        with self._lock:
            return func(self, *args, **kwargs)

    return wrapper


def lockfunc(lockinstance: RLock) -> Callable:
    """
    Protect a function with the given lock.

    Returns a decorator that executes the decorated function while
    ``lockinstance`` is held and releases it automatically when the function
    returns or raises an exception.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with lockinstance:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def supports_emoji() -> bool:
    """
    Heuristically determine whether the current terminal can likely render
    Unicode emoji.

    This is not a guarantee. The result is based on the output encoding and
    common terminal environment indicators known to support Unicode output.

    Returns:
        ``True`` when the terminal is likely to support emoji, otherwise
        ``False``.
    """
    encoding = (sys.stdout.encoding or "").lower()

    if "utf-8" not in encoding:
        return False

    if "WT_SESSION" in os.environ:
        return True

    if os.environ.get("TERM_PROGRAM") in {
        "vscode",
        "Apple_Terminal",
        "iTerm.app",
    }:
        return True

    term = os.environ.get("TERM", "").lower()

    if term in {
        "xterm-256color",
        "screen-256color",
    }:
        return True

    return True