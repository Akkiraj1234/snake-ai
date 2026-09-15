from __future__ import annotations
import asyncio

from .world import build_world
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
    ui_c, ai_c = build_world()
    
    import time
    
    while True:
        ai_c.new_world()
        
        world, result = ui_c.get_frame_data()
        
        for i in world:
            print(i)
        
        time.sleep(0.2)
        print("\033[2J\033[H", end="")

if __name__ == "__main__":
    asyncio.run(main())