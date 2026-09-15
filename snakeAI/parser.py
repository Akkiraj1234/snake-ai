import argparse
from collections.abc import Sequence
from textwrap import dedent

from snakeAI.utils import RUNTIME


HELP_EPILOG = dedent(
    """
    modes:
        snake-ai
            Train the AI using the default settings.

        snake-ai --train
            Explicitly start training.

        snake-ai --view
            Watch the AI using the current trained model without training.

    examples:
        Train faster:
            snake-ai --train --speed 5

        Train very fast without rendering:
            snake-ai --train --headless --speed 100

        Train headlessly with progress updates every second:
            snake-ai --train --headless --update-interval 1

        Watch the trained AI:
            snake-ai --view

        Watch at 30 FPS:
            snake-ai --view --fps 30

        Watch in the terminal:
            snake-ai --view --tui

        Train in the terminal without emoji:
            snake-ai --train --tui --no-emoji

    notes:
        --speed controls how quickly the simulation runs.
        --fps controls how often the UI is refreshed.
        --update-interval controls progress output in headless mode.
    """
).strip()


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """
    Help formatter that preserves examples while showing defaults.
    """
    pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the Snake AI application.

    The application supports training and viewing modes. Training is the
    default when no mode is explicitly selected.

    Args:
        argv: Optional sequence of arguments to parse. When omitted, arguments
            are read from ``sys.argv``.

    Returns:
        Parsed command-line arguments as an :class:`argparse.Namespace`.

    Raises:
        SystemExit: If invalid arguments are supplied or ``--help`` is used.
    """
    parser = argparse.ArgumentParser(
        prog="snake-ai",
        description=(
            "Train and visualize a Snake AI in a configurable simulation."
        ),
        epilog=HELP_EPILOG,
        formatter_class=HelpFormatter,
    )
    
    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--train",
        action="store_true",
        help="Train the Snake AI.",
    )

    mode.add_argument(
        "--view",
        action="store_true",
        help="Run the trained AI without updating its training state.",
    )
    
    presentation = parser.add_argument_group("presentation")

    presentation.add_argument(
        "--headless",
        action="store_true",
        help="Run training without rendering a live UI.",
    )

    presentation.add_argument(
        "--tui",
        action="store_true",
        help="Use the terminal UI instead of the graphical UI.",
    )

    presentation.add_argument(
        "--fps",
        type=float,
        default=RUNTIME.render_fps,
        metavar="FPS",
        help=(
            "UI refresh rate in frames per second. "
            "This controls rendering only, not simulation speed."
        ),
    )

    presentation.add_argument(
        "--no-emoji",
        action="store_false",
        dest="emoji",
        default=RUNTIME.use_emoji,
        help="Use ASCII tiles instead of emoji in the terminal UI.",
    )
    
    simulation = parser.add_argument_group("simulation")

    simulation.add_argument(
        "--speed",
        type=float,
        default=1.0,
        metavar="MULTIPLIER",
        help=(
            "Simulation speed multiplier. "
            "1.0 is normal speed; values below 1.0 slow the simulation "
            "and values above 1.0 speed it up."
        ),
    )
    
    output = parser.add_argument_group("output")

    output.add_argument(
        "--update-interval",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help=(
            "Interval between progress updates in headless mode. "
            "Set to 0 to disable periodic progress output."
        ),
    )

    output.add_argument(
        "--save-on-exit",
        action=argparse.BooleanOptionalAction,
        default=RUNTIME.save_on_exit,
        help="Save the active trainer state when the application exits.",
    )

    args = parser.parse_args(argv)
    
    if not args.train and not args.view:
        args.train = True

    if args.speed <= 0:
        parser.error("--speed must be greater than 0")

    if args.fps <= 0:
        parser.error("--fps must be greater than 0")

    if args.update_interval < 0:
        parser.error("--update-interval cannot be negative")

    if args.headless and args.tui:
        parser.error("--headless and --tui cannot be used together")

    if args.headless and args.view:
        parser.error("--headless can only be used in training mode")

    return args