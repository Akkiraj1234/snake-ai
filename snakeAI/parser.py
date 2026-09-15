import argparse


help_command_text = """
    Examples:

    Train the AI with the default settings:
        snake-ai

    Train at 5x simulation speed:
        snake-ai --train --speed 5

    Watch an already-trained AI:
        snake-ai --view

    Watch at 30 FPS:
        snake-ai --view --fps 30

    Train without a UI:
        snake-ai --train --headless

    Train extremely fast without a UI:
        snake-ai --train --headless --speed 100

    Train headlessly and print progress every second:
        snake-ai --train --headless --update-interval 1
    """


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the Snake AI application.

    The application has two execution modes:

    ``--train``
        Train the Snake AI.

    ``--view``
        Run the simulation using an existing/trained AI without
        updating its training state.

    By default, the application runs in training mode.

    The simulation speed is controlled independently from presentation.
    ``--speed`` changes how quickly the simulation progresses, while
    ``--fps`` controls how frequently the UI is refreshed.

    In headless mode, ``--update-interval`` controls how frequently
    progress information is printed.

    Returns:
        argparse.Namespace:
            Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        prog = "snake-ai",
        description = "Train and visualize a Snake AI.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        epilog = help_command_text
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
        help = "Watch the simulation without training the AI.",
    )
    
    parser.add_argument(
        "--headless",
        action = "store_true",
        help = "Run without a live UI.",
    )

    parser.add_argument(
        "--tui",
        action = "store_true",
        help = "Use a terminal UI instead of the graphical UI.",
    )

    parser.add_argument(
        "--fps",
        type = float,
        default = 60.0,
        metavar = "FPS",
        help = (
            "UI refresh rate in frames per second. "
            "This does not control simulation speed."
        ),
    )

    parser.add_argument(
        "--speed",
        type=float,
        default = 1.0,
        metavar = "MULTIPLIER",
        help = (
            "Simulation speed multiplier. "
            "1.0 is normal speed; values below 1.0 are slower, "
            "and values above 1.0 are faster. "
            "Typical values: 0.1 (very slow), 0.5 (slow), "
            "1.0 (normal), 2.0 (fast), 5.0 (very fast), "
            "10.0+ (extremely fast)."
        ),
    )

    parser.add_argument(
        "--update-interval",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help=(
            "How often progress is printed in headless mode. "
            "Set to 0 to disable periodic output."
        ),
    )

    args = parser.parse_args()

    # Default to training mode.
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
        parser.error("--view cannot be used with --headless")

    return args