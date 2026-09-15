from snakeAI.world import AIController
from snakeAI.utils import AI, Action
import random
import asyncio


async def run_demo_agent(controller: AIController, options) -> None:
    """Temporary agent loop until the training agent is connected here.

    It honours the configured simulation delay and publishes enough data for
    the terminal UI to remain useful during development.
    """
    actions = tuple(Action._data.values())

    while True:
        controller.new_world()
        controller.send_ai_result({
            "exploration": "demo",
            "known_states": 0,
            "updates": 0,
            "q_values": {},
        })

        for _ in range(AI.max_age):
            action = random.choice(actions)
            _, dead, _ = await controller.act_async(action)
            if dead:
                break
            await asyncio.sleep(options.step_delay)
            
