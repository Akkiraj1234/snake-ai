from __future__ import annotations
from collections.abc import Callable
import asyncio
import sys

from .world import AIController, UIController, build_world
from .parser import parse_args, build_ctx
from .utils import ENUM

from .headless import render_headless
from .terminal import render_tui
from .application import render_gui
from .AI import run_agent



async def cancel_tasks(tasks: list[asyncio.Task[object]]) -> None:
    """
    Cancel unfinished tasks and wait for all tasks to complete.

    Cancellation is requested for every unfinished task, then all tasks
    are awaited so their results or exceptions are retrieved before
    returning.
    """
    for task in tasks:
        if not task.done():
            task.cancel()
    
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


class App:
    """ 
    Coordinate application controllers, background tasks, and rendering. 
    
    The application lifecycle consists of building the controllers, 
    running the AI and renderer concurrently, and cleaning up all 
    resources after execution finishes or fails. 
    """
    def __init__(self, ctx: ENUM) -> None:
        """
        Initialize the application with the parsed execution context.
        """
        self.ctx = ctx
        self.ui_c: UIController | None = None
        self.ai_cr: AIController | None = None
        
    def build(self) -> None:
        """
        Build and initialize the application controllers. 
        
        Creates the UI and AI controllers required by the renderer and agent tasks. 
        """
        contoller = build_world()
        self.ui_c = contoller[0]
        self.ai_c = contoller[1]

    def get_render(self) -> Callable:
        """
        Return the renderer selected by the application context. 
        
        Returns the headless, terminal, or GUI renderer according to the 
        configured execution mode. 
        """
        if self.ctx.headless:
            return render_headless

        if self.ctx.tui:
            return render_tui

        return render_gui
        """
    Run the application and return an appropriate process exit code.
    """
    async def run(self) -> None:
        """
        Run the AI agent and renderer concurrently. 
        
        Execution continues until either task completes. Any remaining 
        tasks are then cancelled and awaited before returning or 
        propagating an exception.
        """
        if self.ui_c is None or self.ai_c is None:
            raise RuntimeError("App.build() must be called before App.run()")

        tasks = {
            asyncio.create_task(
                run_agent(self.ai_c, self.ctx),
                name="snake-agent",
            ),
            asyncio.create_task(
                self.get_render()(self.ui_c, self.ctx),
                name="renderer",
            ),
        }

        try:
            done, _ = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            
            for task in done:
                task.result()

        finally:
            await cancel_tasks(list(tasks))
    
    def cleanup(self) -> None:
        if self.ui_c is not None:
            self.ui_c.cleanup()
            self.ui_c = None
        
        if self.ai_c is not None:
            self.ai_c.cleanup()
            self.ai_c = None



def main() -> int:
    """ 
    Initialize and run the application. Builds the application, 
    runs its asynchronous event loop, handles normal interruption and 
    unexpected failures, and always releases application resources b
    efore returning the process exit code. 
    """
    ctx = build_ctx(parse_args())
    app = App(ctx)

    try:
        app.build()
        asyncio.run(app.run())

    except KeyboardInterrupt:
        return 130

    except Exception as error:
        print(
            "Snake AI failed because of an unexpected error.", 
            file=sys.stderr
        )

        if ctx.dev:
            print(error, file=sys.stderr)

        return 1

    finally:
        app.cleanup()

    print("\nSnake AI stopped.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
