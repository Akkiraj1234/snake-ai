from snakeAI.world import UIController
import asyncio


async def render_headless(ui: UIController, interval: float) -> None:
    """Periodically print the same compact state shown by the terminal UI."""
    while True:
        await asyncio.sleep(interval)
        _, data = ui.get_frame_data()
        print(
            "age={age} health={health:.1f} hunger={hunger:.1f} "
            "action={action} reward={reward:.2f} last={message}".format(**data),
            flush=True,
        )