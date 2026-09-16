"""Command-line entry point for the Snake AI simulation."""
from __future__ import annotations

import asyncio
import inspect
import sys
from dataclasses import dataclass
from typing import Any, Callable

from .world import AIController, UIController, build_world
from .parser import parse_args, build_ctx
from .utils import ENUM

from .headless import render_headless
from .terminal import render_tui
from .application import render_gui
from .AI import run_agent




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


class App():
    def __init__(self, ctx: ENUM) -> None:
        self.ctx = ctx
        self.ui_c: UIController | None = None
        self.ai_cr: AIController | None = None
    
    def build(self) -> None:
        contoller = build_world()
        self.ui_c = contoller[0]
        self.ai_c = contoller[1]
    
    def get_render(self) ->  Callable:
        if self.ctx.headless:
            return render_headless
        
        elif self.ctx.tui:
            return render_tui
        
        else: 
            return render_gui
        
    async def run(self) -> None:
        tasks: list[asyncio.Task[object]] = [
            asyncio.create_task(
                run_agent(self.ai_c, self.ctx),
                name="snake-agent"
            ),
            asyncio.create_task(
                self.get_render()(
                    self.ui_c, self.ctx
                ),
                name = "render_agent"
            )
        ]
        
        try:
            await asyncio.gather(*tasks)
        finally:
            await cancel_tasks(tasks)
            if self.ctx.save_on_exit and await save_active_trainer(self.ai_controller):
                print("Saved trainer state.", file=sys.stderr)
        
        print("\nSnake AI stopped.", file=sys.stderr)
    
    def cleanup(self) -> None:
        pass


def main() -> int:
    """
    Run the application and return an appropriate process exit code.
    """
    ctx = build_ctx(parse_args())
    app = App(ctx)
    
    try:
        asyncio.run(app.run())

    except KeyboardInterrupt:
        return 130

    except Exception as error:
        print("Snake AI failed because of an unexpected error.", file=sys.stderr)
        if ctx.dev: print(error, file=sys.stderr)
        return 1

    finally:
        app.cleanup()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
