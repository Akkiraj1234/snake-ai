"""Small tile-map renderer used by :mod:`application`.

The world always contains exactly 22 columns × 15 rows.

Tkinter Canvas is used for rendering. The logical world size never changes;
only the visual size of each cell changes when the canvas is resized.
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk

from PIL import Image, ImageTk


ASSET_DIR = Path(__file__).resolve().parent.parent.parent / "assets/walls"

WORLD_COLUMNS = 22
WORLD_ROWS = 15


class World:
    """
    Draw a grass field with a one-tile stone border.

    The world dimensions are always 22 × 15.
    The canvas automatically determines the size of each cell.
    """

    def __init__(
        self,
        canvas: tk.Canvas,
        tile_size: int = 32,
        seed: int = 7,
    ) -> None:
        self.canvas = canvas
        self.tile_size = tile_size
        self.seed = seed

        self._light_grass = self._load_tile(
            ASSET_DIR / "lightgrass.png"
        )
        self._dark_grass = self._load_tile(
            ASSET_DIR / "darkgrass.png"
        )
        self._wall = self._load_tile(
            ASSET_DIR / "wall.png"
        )

        # Keep references to PhotoImage objects.
        # Tkinter images disappear if Python has no reference to them.
        self._scaled_tiles: dict[
            tuple[str, int, int],
            ImageTk.PhotoImage,
        ] = {}

        self._resize_job: str | None = None

        # Redraw the world whenever the canvas changes size.
        self.canvas.bind("<Configure>", self._on_resize)

    def _load_tile(self, path: Path) -> Image.Image:
        """Load a source tile using Pillow."""

        if not path.exists():
            raise FileNotFoundError(
                f"World tile not found: {path}"
            )

        return Image.open(path).convert("RGBA")

    def _scaled_tile(
        self,
        name: str,
        tile: Image.Image,
        width: int,
        height: int,
    ) -> ImageTk.PhotoImage:
        """
        Return a cached Tkinter image sized for one grid cell.
        """

        key = (name, width, height)

        if key not in self._scaled_tiles:
            resized = tile.resize(
                (width, height),
                Image.Resampling.NEAREST,
            )

            self._scaled_tiles[key] = ImageTk.PhotoImage(
                resized
            )

        return self._scaled_tiles[key]

    def _on_resize(self, event: tk.Event) -> None:
        """Redraw the world when the canvas is resized."""

        # Avoid unnecessary repeated redraws while the user is dragging
        # the window.
        if self._resize_job is not None:
            self.canvas.after_cancel(self._resize_job)

        self._resize_job = self.canvas.after(
            10,
            self.draw,
        )

    def draw(self) -> None:
        """
        Draw exactly 22 × 15 cells.

        The logical grid never changes. Only the visual cell dimensions
        change according to the current canvas size.
        """

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        if width <= 1 or height <= 1:
            return

        columns = WORLD_COLUMNS
        rows = WORLD_ROWS

        # Calculate exact pixel boundaries.
        # Using round() ensures the complete canvas is covered without
        # accumulating integer-division errors.
        x_edges = [
            round(index * width / columns)
            for index in range(columns + 1)
        ]

        y_edges = [
            round(index * height / rows)
            for index in range(rows + 1)
        ]

        self.canvas.delete("world")

        for row in range(rows):
            for column in range(columns):
                x = x_edges[column]
                right = x_edges[column + 1]

                y = y_edges[row]
                bottom = y_edges[row + 1]

                cell_width = right - x
                cell_height = bottom - y

                if cell_width <= 0 or cell_height <= 0:
                    continue

                is_border = (
                    column == 0
                    or column == columns - 1
                    or row == 0
                    or row == rows - 1
                )

                if is_border:
                    name = "wall"
                    tile = self._wall

                elif (column + row) % 2 == 0:
                    name = "light_grass"
                    tile = self._light_grass

                else:
                    name = "dark_grass"
                    tile = self._dark_grass

                image = self._scaled_tile(
                    name,
                    tile,
                    cell_width,
                    cell_height,
                )

                self.canvas.create_image(
                    x,
                    y,
                    image=image,
                    anchor=tk.NW,
                    tags="world",
                )

        self._resize_job = None

    def cell_to_pixel(
        self,
        column: int,
        row: int,
    ) -> tuple[int, int, int, int]:
        """
        Convert a logical world cell to its current pixel rectangle.

        Returns:
            (x, y, width, height)
        """

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        x1 = round(column * width / WORLD_COLUMNS)
        x2 = round((column + 1) * width / WORLD_COLUMNS)

        y1 = round(row * height / WORLD_ROWS)
        y2 = round((row + 1) * height / WORLD_ROWS)

        return (
            x1,
            y1,
            x2 - x1,
            y2 - y1,
        )
