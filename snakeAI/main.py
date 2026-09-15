"""Command-line entry point for the Snake AI simulation."""
from __future__ import annotations

import argparse
import asyncio
import inspect
import random
import sys
from dataclasses import dataclass

from .parser import parse_args
from .utils import AI, RUNTIME, Action
from .world import AIController, UIController, build_world
from .headless import report_headless
from .terminal import render
from .AI import run_demo_agent



@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    """Resolved runtime settings shared by the simulation and terminal UI."""

    step_delay: float
    fps: float
    emoji: bool
    headless: bool
    update_interval: float
    save_on_exit: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> RuntimeOptions:
        step_delay = RUNTIME.start_delay / args.speed
        step_delay = min(RUNTIME.max_delay, max(RUNTIME.min_delay, step_delay))
        return cls(
            step_delay=step_delay,
            fps=args.fps,
            emoji=args.emoji,
            headless=args.headless,
            update_interval=args.update_interval,
            save_on_exit=args.save_on_exit,
        )


async def save_active_trainer(controller: AIController) -> bool:
    """Call a future trainer's optional ``save`` hook without assuming one."""
    save = getattr(controller, "save", None)
    if not callable(save):
        return False

    result = save()
    if inspect.isawaitable(result):
        await result
    return True


async def cancel_tasks(tasks: list[asyncio.Task[object]]) -> None:
    """Cancel and collect background tasks so no task is left unretrieved."""
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def run(args: argparse.Namespace) -> None:
    """Build the world, launch its tasks, and guarantee orderly shutdown."""
    options = RuntimeOptions.from_args(args)
    ui_controller, ai_controller = build_world()
    tasks: list[asyncio.Task[object]] = [
        asyncio.create_task(run_demo_agent(ai_controller, options), name="snake-agent"),
    ]

    if options.headless:
        if options.update_interval > 0:
            tasks.append(asyncio.create_task(
                report_headless(ui_controller, options.update_interval),
                name="headless-status",
            ))
    else:
        tasks.append(asyncio.create_task(
            render(ui_controller, fps=options.fps, emoji=options.emoji),
            name="terminal-renderer",
        ))

    try:
        await asyncio.gather(*tasks)
    finally:
        await cancel_tasks(tasks)
        if options.save_on_exit and await save_active_trainer(ai_controller):
            print("Saved trainer state.", file=sys.stderr)


async def main(argv: list[str] | None = None) -> None:
    """Parse CLI options and run the selected terminal or headless mode."""
    await run(parse_args(argv))


def cli() -> int:
    """Run the application and convert normal interruption/errors to exit codes."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSnake AI stopped.", file=sys.stderr)
        return 0
    except Exception as error:
        print(f"Snake AI failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
