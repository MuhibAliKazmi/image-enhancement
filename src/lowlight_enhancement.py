"""
Classical low-light image enhancement: a comparative analysis.

Compares three non-learning enhancement techniques on a grayscale low-light
image:

    1. Linear brightness scaling  - add a fixed constant with cv2.add (saturating)
    2. Global histogram equalisation - cv2.equalizeHist (whole-image CDF mapping)
    3. CLAHE - cv2.createCLAHE (contrast limited, tile-local equalisation)

The script produces:
    * a 4-panel visual comparison (Original | +Beta | Hist Eq | CLAHE)
    * a histogram comparison showing the "conceptual histogram shift"
    * printed contrast statistics (standard deviation) for all four images

If ``data/original`` contains no readable image, a synthetic dark image is
generated with NumPy so the analysis can still be run immediately.

Usage
-----
    python src/lowlight_enhancement.py                  # darkest dataset image
    python src/lowlight_enhancement.py path/to/img.jpg  # a specific image
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "original"
FIGURE_DIR = PROJECT_ROOT / "analysis" / "figures"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Constant added by Method 1. The dataset images have a mean luminance of
# roughly 7-27 (out of 255), so +60 lifts the shadow region into a visible
# range without pushing most of the histogram to pure white.
BETA = 60

# CLAHE parameters: clipLimit caps the slope of the local CDF (and therefore
# the amount of noise amplification); the tile grid is the local neighbourhood.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

METHOD_NAMES = (
    "Original",
    f"Linear Brightness (+{BETA})",
    "Global Histogram Equalization",
    f"CLAHE (clip={CLAHE_CLIP_LIMIT}, tile={CLAHE_TILE_GRID[0]}x{CLAHE_TILE_GRID[1]})",
)

# Backends that cannot open a window; used to decide whether to call plt.show().
NON_INTERACTIVE_BACKENDS = ("agg", "pdf", "ps", "svg", "cairo", "template")


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #

def display_path(path: Path) -> str:
    """Shorten a path for printing when it lives inside the project.

    Images outside the project root (e.g. ~/Pictures/night.jpg) are shown in
    full, since ``relative_to`` would raise ValueError for them.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def find_input_image(explicit_path: str | None) -> Path | None:
    """Return the image to analyse, or None if none can be found.

    An explicitly supplied path always wins. Otherwise the darkest image in the
    dataset is chosen, since that is the most interesting case to enhance.
    """
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        return path

    candidates = sorted(
        p for p in DATA_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    ) if DATA_DIR.is_dir() else []

    if not candidates:
        return None

    # Rank by mean grayscale luminance and pick the darkest readable image.
    darkest_path, darkest_mean = None, np.inf
    for path in candidates:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        mean = float(image.mean())
        if mean < darkest_mean:
            darkest_path, darkest_mean = path, mean
    return darkest_path


def make_synthetic_dark_image(width: int = 640, height: int = 427) -> np.ndarray:
    """Build a deterministic, underexposed test image with NumPy.

    It mimics a night scene: a near-black background with a faint gradient,
    a few dim "light sources", and mild sensor noise. All intensities stay in
    the lower quarter of the 8-bit range so the enhancement methods have
    something meaningful to recover.
    """
    rng = np.random.default_rng(seed=42)

    # Base illumination: a weak vertical gradient around intensity 20-45.
    gradient = np.linspace(20, 45, height, dtype=np.float32)[:, None]
    image = np.repeat(gradient, width, axis=1)

    # Dim Gaussian light blobs of varying size and brightness.
    blobs = [
        (int(width * 0.30), int(height * 0.35), 3, 26),
        (int(width * 0.68), int(height * 0.60), 4, 18),
        (int(width * 0.50), int(height * 0.78), 2, 12),
        (int(width * 0.82), int(height * 0.25), 2, 10),
    ]
    yy, xx = np.mgrid[0:height, 0:width]
    for cx, cy, sigma, peak in blobs:
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        image += peak * np.exp(-dist_sq / (2.0 * sigma ** 2 * 100))

    # Fine structure (so contrast-limited enhancement has detail to work with).
    x = np.linspace(0, 6 * np.pi, width, dtype=np.float32)
    y = np.linspace(0, 4 * np.pi, height, dtype=np.float32)
    image += 8.0 * np.outer(np.sin(y), np.cos(x))

    # Mild Gaussian read noise, then clamp to the valid 8-bit range.
    image += rng.normal(loc=0.0, scale=3.0, size=image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Enhancement methods (all operate on a uint8 grayscale image)
# --------------------------------------------------------------------------- #

def linear_brightness(gray: np.ndarray, beta: int = BETA) -> np.ndarray:
    """Method 1: add a constant to every pixel.

    ``cv2.add`` saturates at 255 instead of wrapping around, so bright pixels
    are clipped rather than turned black. The caller reports the clipped share.
    """
    return cv2.add(gray, beta)


def global_histogram_equalization(gray: np.ndarray) -> np.ndarray:
    """Method 2: remap intensities through the global CDF (OpenCV's version).

    This spreads the occupied part of the histogram across the full 0-255
    range. Because the mapping is global, enhancement is applied uniformly and
    noise in flat regions is amplified together with real detail.
    """
    return cv2.equalizeHist(gray)


def clahe(gray: np.ndarray,
          clip_limit: float = CLAHE_CLIP_LIMIT,
          tile_grid: tuple[int, int] = CLAHE_TILE_GRID) -> np.ndarray:
    """Method 3: Contrast Limited Adaptive Histogram Equalization.

    Equalisation is computed per tile (here 8x8 tiles) and the CDF slope is
    clipped at ``clip_limit``, so local contrast improves without the extreme
    noise amplification of global equalisation. Tile borders are blended
    bilinearly by OpenCV.
    """
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(gray)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def contrast_stats(images: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Per-image summary statistics; standard deviation is the contrast metric.

    For an 8-bit image a higher std means pixel values are spread further from
    their mean, i.e. the image has more perceived contrast.
    """
    return {
        "Mean": np.array([img.mean() for img in images]),
        "Std Dev (contrast)": np.array([img.std() for img in images]),
        "Min": np.array([img.min() for img in images]),
        "Max": np.array([img.max() for img in images]),
        "Clipped pixels (%)": np.array(
            [float(np.mean((img == 0) | (img == 255)) * 100.0) for img in images]
        ),
    }


def print_metrics(stats: dict[str, np.ndarray]) -> None:
    """Print the comparison table, highlighting the contrast change vs original."""
    base_std = stats["Std Dev (contrast)"][0]
    header = f"{'Method':<44}{'Mean':>8}{'Std Dev':>10}{'vs Orig':>10}{'Min':>6}{'Max':>6}{'Clip %':>8}"
    print("\n" + header)
    print("-" * len(header))
    for i, name in enumerate(METHOD_NAMES):
        mean = stats["Mean"][i]
        std = stats["Std Dev (contrast)"][i]
        delta = (std - base_std) / base_std * 100.0 if base_std else 0.0
        delta_text = "  (base)" if i == 0 else f"{delta:+.1f}%"
        print(
            f"{name:<44}{mean:8.2f}{std:10.2f}{delta_text:>10}"
            f"{stats['Min'][i]:6d}{stats['Max'][i]:6d}{stats['Clipped pixels (%)'][i]:8.2f}"
        )
    print("-" * len(header))
    print("Note: 'Clip %' counts pixels pinned at 0 or 255 (information lost).")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def plot_comparison(images: list[np.ndarray], source_label: str) -> plt.Figure:
    """Figure 1: the original and three enhanced versions, side by side."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle(f"Low-Light Enhancement Comparison\n{source_label}", fontsize=13)

    for ax, image, title in zip(axes.flat, images, METHOD_NAMES):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{title}\nmean={image.mean():.1f}, std={image.std():.1f}",
                     fontsize=10)
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_histograms(images: list[np.ndarray]) -> plt.Figure:
    """Figure 2: histograms of all four images.

    The four small panels give a per-image view, while the wide bottom panel
    overlays every histogram so the shift from dark (left) to bright (right)
    intensities is directly visible.
    """
    colors = ("#444444", "#1f77b4", "#d62728", "#2ca02c")
    fig = plt.figure(figsize=(13, 9))
    fig.suptitle("Histogram Comparison: Conceptual Shift of Dark Pixels to the Bright Side",
                 fontsize=13)
    grid = fig.add_gridspec(2, 4, height_ratios=(1.0, 1.15), hspace=0.35, wspace=0.25)

    # --- Row 1: one histogram per method ----------------------------------- #
    for i, (image, title, color) in enumerate(zip(images, METHOD_NAMES, colors)):
        ax = fig.add_subplot(grid[0, i])
        ax.hist(image.ravel(), bins=256, range=(0, 256), color=color, alpha=0.85)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, 255)
        ax.set_xlabel("Pixel intensity")
        ax.set_ylabel("Pixel count" if i == 0 else "")
        ax.grid(alpha=0.25)

    # --- Row 2: all four overlaid ------------------------------------------ #
    ax_all = fig.add_subplot(grid[1, :])
    for image, title, color in zip(images, METHOD_NAMES, colors):
        ax_all.hist(image.ravel(), bins=256, range=(0, 256), histtype="step",
                    linewidth=1.5, color=color, label=f"{title} (mean {image.mean():.1f})")
    ax_all.set_xlim(0, 255)
    ax_all.set_title("Overlaid histograms (density view)", fontsize=11)
    ax_all.set_xlabel("Pixel intensity")
    ax_all.set_ylabel("Pixel count")
    ax_all.legend(fontsize=8, framealpha=0.9)
    ax_all.grid(alpha=0.25)

    return fig


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str]) -> int:
    explicit = argv[1] if len(argv) > 1 else None
    path = find_input_image(explicit)

    # --- 1. Load and preprocess ------------------------------------------- #
    if path is None:
        print(f"No image found in {DATA_DIR}. Generating a synthetic dark image.\n")
        gray = make_synthetic_dark_image()
        source_label = "Source: synthetic NumPy image (no files in data/original)"
        stem = "synthetic_dark"
        # Persist it so the same test input can be reused later.
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        synthetic_path = DATA_DIR / f"{stem}.png"
        cv2.imwrite(str(synthetic_path), gray)
        print(f"Saved synthetic input to {display_path(synthetic_path)}\n")
    else:
        # IMREAD_GRAYSCALE gives the single-channel image the assignment requires.
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"ERROR: could not decode image: {path}", file=sys.stderr)
            return 1
        source_label = (
            f"Source: {display_path(path)} ({gray.shape[1]}x{gray.shape[0]})"
        )
        stem = path.stem
        print(f"Loaded {display_path(path)} "
              f"({gray.shape[1]}x{gray.shape[0]}, mean={gray.mean():.1f})")

    # --- 2. Apply the three enhancement methods --------------------------- #
    linear = linear_brightness(gray, BETA)
    equalized = global_histogram_equalization(gray)
    clahe_img = clahe(gray)

    images = [gray, linear, equalized, clahe_img]

    # --- 3. Quantitative comparison (std dev = contrast) ------------------ #
    stats = contrast_stats(images)
    print_metrics(stats)

    # --- 4. Figures ------------------------------------------------------- #
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig1 = plot_comparison(images, source_label)
    comparison_path = FIGURE_DIR / f"{stem}_comparison.png"
    fig1.savefig(comparison_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visual comparison  -> {display_path(comparison_path)}")

    fig2 = plot_histograms(images)
    histogram_path = FIGURE_DIR / f"{stem}_histograms.png"
    fig2.savefig(histogram_path, dpi=150, bbox_inches="tight")
    print(f"Saved histogram comparison -> {display_path(histogram_path)}")

    # Show the figures interactively, but only when the backend can display
    # windows (e.g. not Agg / a headless server).
    if plt.get_backend().lower() not in NON_INTERACTIVE_BACKENDS:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
