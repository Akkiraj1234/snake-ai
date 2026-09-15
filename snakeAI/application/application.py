import tkinter as tk

from .utils import Theme
from .world import World


APP_CAPTION = "Snake Game"

PADDING = 10
FOOTER_HEIGHT = 30

MAIN_WIDTH_RATIO = 0.70

MIN_WINDOW_WIDTH = 900
MIN_WINDOW_HEIGHT = 600


THEME_DATA = Theme({
    "screen_color": "#1e1e1e",
    "main_panel": "#2d2d2d",
    "panel_color": "#323232",
})


class App:
    def __init__(self, full_screen: bool = False):
        self.running: bool = True
        self.full_screen: bool = full_screen

        self.root: tk.Tk | None = None

        self.main_frame: tk.Frame | None = None
        self.panel_frame: tk.Frame | None = None
        self.footer_frame: tk.Frame | None = None

        self.world_canvas: tk.Canvas | None = None
        self.world: World | None = None

    def initialize_app(self) -> None:
        """Initialize Tkinter and configure the application."""

        self.root = tk.Tk()

        self.root.title(APP_CAPTION)

        self.root.minsize(
            MIN_WINDOW_WIDTH,
            MIN_WINDOW_HEIGHT,
        )

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        width = int(screen_width * 0.8)
        height = int(screen_height * 0.8)

        if self.full_screen:
            self.root.attributes("-fullscreen", True)
        else:
            self.root.geometry(f"{width}x{height}")

        self.setup_layout()

        # Create the world AFTER the canvas exists.
        if self.world_canvas is not None:
            self.world = World(self.world_canvas)

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.on_exit,
        )

    def setup_layout(self) -> None:
        """Create the application layout."""

        if self.root is None:
            return

        self.root.configure(
            background=THEME_DATA.screen_color,
        )

        content = tk.Frame(
            self.root,
            background=THEME_DATA.screen_color,
        )

        content.pack(
            fill=tk.BOTH,
            expand=True,
            padx=PADDING,
            pady=(PADDING, 0),
        )

        content.grid_columnconfigure(
            0,
            weight=7,
        )

        content.grid_columnconfigure(
            1,
            weight=3,
        )

        content.grid_rowconfigure(
            0,
            weight=1,
        )

        # Main game frame.
        self.main_frame = tk.Frame(
            content,
            background=THEME_DATA.main_panel,
        )

        self.main_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, PADDING),
        )

        # Canvas where World is rendered.
        self.world_canvas = tk.Canvas(
            self.main_frame,
            background=THEME_DATA.main_panel,
            highlightthickness=0,
            borderwidth=0,
        )

        self.world_canvas.pack(
            fill=tk.BOTH,
            expand=True,
        )

        # Side panel.
        self.panel_frame = tk.Frame(
            content,
            background=THEME_DATA.panel_color,
        )

        self.panel_frame.grid(
            row=0,
            column=1,
            sticky="nsew",
        )

        # Footer.
        self.footer_frame = tk.Frame(
            self.root,
            height=FOOTER_HEIGHT,
            background=THEME_DATA.panel_color,
        )

        self.footer_frame.pack(
            fill=tk.X,
            padx=PADDING,
            pady=PADDING,
        )

        self.footer_frame.pack_propagate(False)

    def draw_screen(self) -> None:
        """Draw/update the game world."""

        if self.world is not None:
            self.world.draw()

    def on_exit(self) -> None:
        """Close the application safely."""

        root = self.root
        self.root = None

        if root is not None:
            root.destroy()

    def mainloop(self) -> None:
        """Start the Tkinter event loop."""

        if self.root is None:
            self.initialize_app()

        if self.root is not None:
            self.root.mainloop()

    def __enter__(self) -> "App":
        self.initialize_app()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.on_exit()