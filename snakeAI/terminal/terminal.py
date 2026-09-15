from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from typing import Any, TextIO

from blessed import Terminal

from snakeAI.utils import Cell, supports_emoji
from snakeAI.world import UIController


_EMOJI_CELLS = {
    Cell.GRASS: "🍀",
    Cell.WALL: "🧱",
    Cell.SNAKE: "🐍",
    Cell.WATER: "🟦",
    Cell.APPLE: "🍎",
    Cell.BANANA: "🍌",
    Cell.FROG: "🐸",
    Cell.OBSTACLE: "🪨",
}
_ASCII_CELLS = {
    Cell.GRASS: " ",
    Cell.WALL: "#",
    Cell.SNAKE: "O",
    Cell.WATER: "~",
    Cell.APPLE: "A",
    Cell.BANANA: "B",
    Cell.FROG: "F",
    Cell.OBSTACLE: "@",
}

_ACTION_NAMES = {0: "IDLE", 1: "EAT", 2: "FORWARD", 3: "LEFT", 4: "RIGHT"}


def _crop_world(world: list[list[int]], width: int, height: int) -> list[list[int]]:
    """Center a size-limited map view on the snake's head."""
    head_y, head_x = next(
        (
            (y, x)
            for y, row in enumerate(world)
            for x, cell in enumerate(row)
            if cell == Cell.SNAKE
        ),
        (len(world) // 2, len(world[0]) // 2),
    )
    start_y = max(0, min(head_y - height // 2, len(world) - height))
    start_x = max(0, min(head_x - width // 2, len(world[0]) - width))
    return [row[start_x:start_x + width] for row in world[start_y:start_y + height]]


def _tile(cell: int, cells: Mapping[int, str]) -> str:
    """Use two terminal columns per cell, including emoji cells."""
    return f"{cells.get(cell, '?')} "


def _value(data: Mapping[str, Any], *names: str, default: Any = "—") -> Any:
    return next((data[name] for name in names if name in data), default)


def _format_q_values(data: Mapping[str, Any]) -> list[str]:
    values = _value(data, "q_values", "q", default={})
    if isinstance(values, Mapping):
        pairs = values.items()
    elif isinstance(values, (list, tuple)):
        pairs = enumerate(values)
    else:
        return ["Q values: —"]

    lines = ["Q values"]
    for action, value in list(pairs)[:5]:
        try:
            rendered = f"{float(value):7.3f}"
        except (TypeError, ValueError):
            rendered = str(value)
        lines.append(f"  {_ACTION_NAMES.get(action, str(action)):<8} {rendered}")
    return lines


def _panel_lines(data: Mapping[str, Any], emoji: bool) -> list[str]:
    health = float(_value(data, "health", default=0.0))
    hunger = float(_value(data, "hunger", default=0.0))
    action = _value(data, "action")
    exploration = _value(data, "exploration", "epsilon")
    if isinstance(exploration, (float, int)):
        exploration = f"{float(exploration) * 100:.1f}%"

    heart = "♥" if emoji else "HP"
    return [
        "AI STATUS",
        f"Age:          {_value(data, 'age')}",
        f"{heart} Health:   {health:5.1f}",
        f"Hunger:       {hunger:5.1f}",
        f"Choice:       {_ACTION_NAMES.get(action, str(action))}",
        f"Reward:       {float(_value(data, 'reward', default=0.0)):7.2f}",
        f"Exploration:  {exploration}",
        f"Known states: {_value(data, 'known_states', 'states')}",
        f"Explored:     {_value(data, 'explored', 'updates')}",
        *_format_q_values(data),
    ]


def _draw(
    term: Terminal,
    world: list[list[int]],
    data: Mapping[str, Any],
    emoji: bool,
) -> str:
    """Build one size-aware Blessed frame without writing to the terminal."""
    if term.width < 24 or term.height < 8:
        return "Terminal is too small. Resize to at least 24x8."

    panel_width = 30 if term.width >= 84 else 0
    map_columns = max(10, (term.width - panel_width - (7 if panel_width else 4)) // 2)
    map_rows = max(5, term.height - 5)
    visible = _crop_world(
        world,
        min(map_columns, len(world[0])),
        min(map_rows, len(world)),
    )
    map_width = len(visible[0]) * 2
    map_lines = [f"┌{'─' * map_width}┐"]
    cells = _EMOJI_CELLS if emoji else _ASCII_CELLS
    map_lines.extend(
        f"│{''.join(_tile(cell, cells) for cell in row)}│" for row in visible
    )
    map_lines.append(f"└{'─' * map_width}┘")

    status = _panel_lines(data, emoji)
    lines = [term.bold_white(" SNAKE AI ")]
    if panel_width:
        content_rows = max(len(map_lines), len(status))
        for index in range(content_rows):
            left = map_lines[index] if index < len(map_lines) else " " * (map_width + 2)
            right = status[index] if index < len(status) else ""
            lines.append(f"{left}   {right[:panel_width]}")
    else:
        lines.extend(map_lines)
        lines.extend(status[:max(0, term.height - len(lines))])

    lines.append(f"Last: {_value(data, 'message', default='')}"[:term.width])
    return "\n".join(line[:term.width] for line in lines[:term.height])


async def render(
    controller: UIController,
    fps: float = 30.0,
    *,
    emoji: bool | None = None,
    stream: TextIO | None = None,
) -> None:
    """Render a black-background Blessed UI at a delta-time-capped FPS.

    Blessed writes are intentionally synchronous: each frame is small, and
    awaiting the frame interval keeps the asyncio event loop responsive.
    This function only reads from ``controller``; another task advances the
    simulation.
    """
    if fps <= 0:
        raise ValueError("fps must be greater than 0")

    output = stream or sys.stdout
    term = Terminal(stream=output)
    emoji = supports_emoji() if emoji is None else emoji
    frame_time = 1.0 / fps
    loop = asyncio.get_running_loop()
    previous_time = loop.time()
    elapsed = frame_time  # Render the first frame immediately.

    try:
        with term.fullscreen(), term.hidden_cursor():
            while True:
                now = loop.time()
                elapsed += now - previous_time
                previous_time = now

                if elapsed < frame_time:
                    await asyncio.sleep(frame_time - elapsed)
                    continue

                # Do not render a backlog when the terminal is briefly slow.
                elapsed %= frame_time
                world, data = controller.get_frame_data()
                output.write(term.home + term.on_black + term.white + term.clear)
                output.write(_draw(term, world, data, emoji))
                output.flush()
    finally:
        output.write(term.normal)
        output.flush()
