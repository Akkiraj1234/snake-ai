from collections.abc import Sequence
from textwrap import dedent
from pathlib import Path
import argparse

from snakeAI.utils import RUNTIME, ENUM



HELP_EPILOG = dedent(
    """
    examples:
        Train the AI:
            snake-ai

        Train faster:
            snake-ai --train --speed 5

        Train without a UI:
            snake-ai --train --headless

        Train very fast without a UI:
            snake-ai --train --headless --speed 100

        Train with headless progress updates:
            snake-ai --train --headless --update-interval 1

        Watch the trained AI:
            snake-ai --view

        Watch at 30 FPS:
            snake-ai --view --fps 30

        Watch in the terminal:
            snake-ai --view --tui

        Use a specific model:
            snake-ai --view --model data/snake.pkl

        Show development errors:
            snake-ai --dev
    """
).strip()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the Snake AI application.

    Training is used by default when neither ``--train`` nor ``--view`` is
    specified.

    Args:
        argv: Optional sequence of arguments to parse. When omitted,
            arguments are read from ``sys.argv``.

    Returns:
        Parsed command-line arguments.

    Raises:
        SystemExit: If invalid arguments are supplied or ``--help`` is used.
    """
    parser = argparse.ArgumentParser(
        prog = "snake-ai",
        description = "Train and visualize a Snake AI.",
        epilog = HELP_EPILOG,
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--train",
        action = "store_true",
        help = "Train the Snake AI.",
    )

    mode.add_argument(
        "--view",
        action = "store_true",
        help = "Watch the trained AI without training it.",
    )

    parser.add_argument(
        "--headless",
        action = "store_true",
        help = "Run without a live UI.",
    )

    parser.add_argument(
        "--tui",
        action = "store_true",
        help = "Use the terminal UI instead of the graphical UI.",
    )

    parser.add_argument(
        "--only-ansi",
        action = "store_true",
        help = "Use ANSI terminal rendering only.",
    )

    parser.add_argument(
        "--fps",
        type = float,
        default = RUNTIME.render_fps,
        metavar = "FPS",
        help = "UI refresh rate in frames per second.",
    )

    parser.add_argument(
        "--speed",
        type = float,
        default = 1.0,
        metavar = "MULTIPLIER",
        help = "Simulation speed multiplier.",
    )

    parser.add_argument(
        "--update-interval",
        type = float,
        default = 0.5,
        metavar = "SECONDS",
        help = "Interval between progress updates in headless mode.",
    )

    parser.add_argument(
        "--save-on-exit",
        action = argparse.BooleanOptionalAction,
        default = RUNTIME.save_on_exit,
        help = "Save the trainer state when the application exits.",
    )

    parser.add_argument(
        "--model",
        type = Path,
        default = RUNTIME.model_path,
        metavar = "PATH",
        help = "Path to the model weights file.",
    )

    parser.add_argument(
        "--dev",
        action = "store_true",
        help = "Enable development-oriented error reporting.",
    )
 
    args = parser.parse_args(argv)
    return args


def build_ctx(args: argparse.Namespace) -> ENUM:
    """
    Validate parsed arguments and build the application runtime context.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Runtime configuration used by the application.

    Raises:
        ValueError: If an argument is invalid or incompatible with another
            option.
    """
    if not args.train and not args.view:
        args.train = True

    if args.speed <= 0:
        raise ValueError("--speed must be greater than 0")

    if args.fps <= 0:
        raise ValueError("--fps must be greater than 0")

    if args.update_interval < 0:
        raise ValueError("--update-interval cannot be negative")

    if args.headless and args.tui:
        raise ValueError("--headless and --tui cannot be used together")

    if args.headless and args.view:
        raise ValueError("--headless can only be used with training")

    if args.only_ansi and not args.tui:
        raise ValueError("--only-ansi requires --tui")
    
    step_delay = RUNTIME.start_delay / args.speed
    step_delay = min(RUNTIME.max_delay, max(RUNTIME.min_delay, step_delay))
    
    return ENUM({
        "train": args.train,
        "view": args.view,
        "headless": args.headless,
        "tui": args.tui,
        "fps": args.fps,
        "only_ansi": args.only_ansi,
        "speed": args.speed,
        "step_delay": step_delay,
        "update_interval": args.update_interval,
        "save_on_exit": args.save_on_exit,
        "model": args.model,
        "dev": args.dev,
    })