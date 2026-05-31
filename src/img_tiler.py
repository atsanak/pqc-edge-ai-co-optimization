from typing import Optional, List, Tuple
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class SimpleImageTiler:
    """
    Minimal image tiler with:
      - load(image_path)
      - split()                       -> cuts image into rows x cols tiles
      - get_tile(r, c)                -> PIL.Image for that tile
      - show_grid()                   -> original image with grid + dimensions
      - show_tile(r, c)               -> display a single tile + dimensions
      - show_all_tiles()              -> display all tiles + per-tile dimensions
      - save_tile(r, c, out_path)     -> save a single tile
      - get_dimensions()              -> return (image_size, tile_sizes matrix)
      - print_dimensions()            -> print sizes to console
    """

    def __init__(self, rows: int = 4, cols: int = 4):
        self.rows = rows
        self.cols = cols
        self.image: Optional[Image.Image] = None
        self.xs: Optional[np.ndarray] = None  # vertical boundaries
        self.ys: Optional[np.ndarray] = None  # horizontal boundaries
        self.tiles: List[Image.Image] = []    # cached tiles

    # ---- setup ----
    def load_from_path(self, image_path: str) -> "SimpleImageTiler":
        self.image = Image.open(image_path).convert("RGB")
        self._compute_grid()
        self.tiles.clear()
        return self
    
    def load(self, img: Image.Image | str | Path) -> "SimpleImageTiler":
        if isinstance(img, (str, Path)):
            self.image = Image.open(img).convert("RGB")
        else:
            self.image = img.convert("RGB")
        self._compute_grid()
        self.tiles.clear()
        return self

    def _require_image(self):
        if self.image is None:
            raise ValueError("No image loaded. Call .load(...) first.")

    def _compute_grid(self):
        self._require_image()
        W, H = self.image.size
        self.xs = np.linspace(0, W, self.cols + 1, dtype=int)
        self.ys = np.linspace(0, H, self.rows + 1, dtype=int)

    # ---- core ----
    def split(self) -> List[Image.Image]:
        """Cut the image into tiles and cache them."""
        self._require_image()
        if self.xs is None or self.ys is None:
            self._compute_grid()

        self.tiles = []
        for r in range(self.rows):
            for c in range(self.cols):
                left, right  = self.xs[c], self.xs[c + 1]
                upper, lower = self.ys[r], self.ys[r + 1]
                self.tiles.append(self.image.crop((left, upper, right, lower)))
        return self.tiles

    def _idx(self, r: int, c: int) -> int:
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise IndexError(f"(r, c) out of range: ({r}, {c})")
        return r * self.cols + c

    def get_tile(self, r: int, c: int) -> Image.Image:
        """Return the tile at (row r, col c)."""
        if not self.tiles:
            self.split()
        return self.tiles[self._idx(r, c)]

    # ---- dimensions ----
    def get_dimensions(self) -> Tuple[Tuple[int, int], List[List[Tuple[int, int]]]]:
        """
        Returns:
          image_size: (W, H)
          tile_sizes: rows x cols list of (w, h)
        """
        self._require_image()
        if not self.tiles:
            self.split()
        W, H = self.image.size  # type: ignore
        tile_sizes: List[List[Tuple[int, int]]] = []
        for r in range(self.rows):
            row_sizes = []
            for c in range(self.cols):
                w, h = self.get_tile(r, c).size
                row_sizes.append((w, h))
            tile_sizes.append(row_sizes)
        return (W, H), tile_sizes

    def print_dimensions(self):
        (W, H), tile_sizes = self.get_dimensions()
        print(f"Input image size: {W} x {H} px")
        for r in range(self.rows):
            for c in range(self.cols):
                w, h = tile_sizes[r][c]
                print(f"Tile (r{r}, c{c}): {w} x {h} px")

    # ---- visualize ----
    def show_grid(self, linewidth: float = 0.8, title: str = "Original with grid"):
        """Display the original image with the tiling grid overlay + dimensions."""
        self._require_image()
        if self.xs is None or self.ys is None:
            self._compute_grid()

        W, H = self.image.size  # type: ignore
        fig, ax = plt.subplots(figsize=(6, 6 * H / max(W, 1)))
        ax.imshow(self.image)
        for x in self.xs: ax.axvline(x, linewidth=linewidth)
        for y in self.ys: ax.axhline(y, linewidth=linewidth)
        ax.set_title(f"{title}  |  image: {W}×{H}px  |  grid: {self.rows}×{self.cols}")
        ax.text(0.01, 0.99, f"{W}×{H}px", transform=ax.transAxes,
                va="top", ha="left", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    def show_tile(self, r: int, c: int, title_prefix: str = "Tile"):
        """Display a single tile with its pixel dimensions."""
        tile = self.get_tile(r, c)
        w, h = tile.size
        plt.figure()
        plt.imshow(tile)
        plt.title(f"{title_prefix} (r{r}, c{c}) — {w}×{h}px")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def show_all_tiles(self, figsize_scale: float = 2.2, show_indices: bool = True):
        """Display all tiles in a rows×cols grid with per-tile dimensions."""
        if not self.tiles:
            self.split()

        # Build a sizes matrix once
        _, tile_sizes = self.get_dimensions()

        fig, axes = plt.subplots(self.rows, self.cols,
                                 figsize=(self.cols * figsize_scale, self.rows * figsize_scale))
        axes = np.atleast_2d(axes)

        for r in range(self.rows):
            for c in range(self.cols):
                ax = axes[r, c]
                tile = self.get_tile(r, c)
                w, h = tile_sizes[r][c]
                ax.imshow(tile)
                if show_indices:
                    ax.set_title(f"r{r} c{c} — {w}×{h}px", fontsize=9)
                ax.axis("off")

        plt.tight_layout()
        plt.show()

    # ---- save ----
    def save_tile(self, r: int, c: int, out_path: str):
        tile = self.get_tile(r, c)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        tile.save(out_path)


# --------- Demo: retrieve, display, save, and show dimensions ----------
if __name__ == "__main__":
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_20fps", "ezgif-frame-001.png")
    rows, cols = 4, 4
    r, c = 1, 2  # choose a tile to display/save

    tiler = SimpleImageTiler(rows=rows, cols=cols).load(image_path)

    # Create tiles
    tiler.split()

    # Show original with grid and dimensions overlay
    tiler.show_grid()

    # Show all tiles at once (with per-tile dimensions in titles)
    tiler.show_all_tiles()

    # Show a specific tile (with dimensions)
    tiler.show_tile(r, c)

    # Save that specific tile
    out_path = f"out/tile_r{r}_c{c}.png"
    tiler.save_tile(r, c, out_path)
    print("Saved:", out_path)

    # Print dimensions to console (image + each tile)
    tiler.print_dimensions()
