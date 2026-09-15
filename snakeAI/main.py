from __future__ import annotations
import asyncio

from .world.controller import build_world
from .parser import parse_args



async def main() -> None:
    """
    Start the Snake AI application using the requested configuration.

    By default, the application trains the AI using the graphical UI
    at normal simulation speed and 60 FPS.

    ``--train`` enables training, while ``--view`` runs the simulation
    purely for observation without training.
    """
    args = parse_args()
    world_controller, ai_controller = build_world()

    # TODO: Pass configuration to the application runner.
    #
    # await run(
    #     world_controller=world_controller,
    #     ai_controller=ai_controller,
    #     train=args.train,
    #     headless=args.headless,
    #     tui=args.tui,
    #     speed=args.speed,
    #     fps=args.fps,
    #     update_interval=args.update_interval,
    # )


if __name__ == "__main__":
    asyncio.run(main())