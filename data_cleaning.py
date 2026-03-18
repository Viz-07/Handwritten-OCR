"""
data_cleaning.py
=================================================
Implements the preprocessing pipeline for historical Spanish OCR:
  1. Grayscale conversion
  2. Sauvola local binarization (handles faded ink & stained parchment)

All public functions accept a PIL Image and return a PIL Image so the
rest of the pipeline stays format-agnostic.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PIL import Image
from skimage.filters import threshold_sauvola

# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _pil_to_gray_array(img: Image.Image) -> np.ndarray:
    """Convert any PIL Image to a uint8 grayscale NumPy array."""
    return np.array(img.convert("L"), dtype=np.uint8)


def _gray_array_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a 2-D uint8 NumPy array back to a grayscale PIL Image."""
    return Image.fromarray(arr.astype(np.uint8), mode="L")

import cv2
from skimage.transform import rotate
from skimage.feature import canny
from scipy.ndimage import label

def clahe_enhance(img: Image.Image, clip_limit: float = 2.0, tile_size: int = 8) -> Image.Image:
    """CLAHE contrast enhancement — critical for faded iron gall ink."""
    img = np.array(img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(img)
    return Image.fromarray(enhanced)

def morphological_cleanup(img: Image.Image, min_component_size: int = 5) -> Image.Image:
    """Remove isolated noise pixels after binarization."""
    arr = np.array(img)
    ink = (arr == 0).astype(np.uint8)
    labeled, n = label(ink)
    sizes = np.bincount(labeled.ravel())
    mask = sizes[labeled] >= min_component_size
    cleaned = np.where(mask & ink, 0, 255).astype(np.uint8)
    return Image.fromarray(cleaned)

def correct_page_tilt(img: Image.Image, angle: float) -> Image.Image:
    """
    Correct scanner-level page tilt (e.g. book placed slightly crooked).
    This is a global rigid rotation — safe for handwriting since it doesn't
    alter relative letter positions, only the page orientation.
    Only call this if your scans are visibly tilted; skip otherwise.

    Parameters
    ----------
    angle : float
        Degrees counter-clockwise. Typical range: -5.0 to +5.0.
        Measure visually from one of your scans before hardcoding.
    """
    return img.rotate(angle, expand=True, fillcolor=(255, 255, 255))

logger = logging.getLogger(__name__)
    
# ─────────────────────────────────────────────
# Step 1 — Grayscale
# ─────────────────────────────────────────────

def to_grayscale(img: Image.Image) -> Image.Image:
    """
    Convert image to grayscale.

    Parameters
    ----------
    img : PIL.Image.Image
        Source image (any mode).

    Returns
    -------
    PIL.Image.Image
        Grayscale image (mode='L').
    """
    return img.convert("L")


# ─────────────────────────────────────────────
# Step 2 — Sauvola Binarization
# ─────────────────────────────────────────────

# Maximum number of pixels (w × h) fed to Sauvola in one go.
# Sauvola builds a float64 integral image — at this cap (~4 000 × 3 000)
# that costs ≈ 96 MB, safely within a typical 8 GB machine.
# The result is always upscaled back to the original resolution with
# nearest-neighbour interpolation so downstream coordinates stay valid.
_SAUVOLA_MAX_PIXELS: int = 12_000_000   # ~4 000 × 3 000


def sauvola_binarize(
    img: Image.Image,
    window_size: int = 51,
    k: float = 0.2,
    r: Optional[float] = None,
    max_pixels: int = _SAUVOLA_MAX_PIXELS,
) -> Image.Image:
    """
    Apply Sauvola local thresholding to produce a clean binary image.

    Unlike Otsu's global threshold, Sauvola adapts to local variations in
    illumination and ink density — essential for parchment-backed prints.

    Threshold formula per pixel p:
        T(p) = mean(p) * [1 + k * (std(p)/r - 1)]
    where r is the dynamic range of std (default: 128 for uint8).

    Memory safety
    -------------
    Sauvola internally allocates a float64 integral image the same size as
    the input.  For very large scans (e.g. 350 DPI spreads) this can exceed
    2 GB.  If the image has more pixels than `max_pixels`, it is downscaled
    before thresholding and the binary result is upscaled back to the
    original size with nearest-neighbour interpolation so that pixel
    coordinates remain consistent for downstream stages.

    Note on k direction: since std/r < 1 for most smooth document regions,
    a *lower* k keeps the threshold closer to the local mean, catching more
    faded ink pixels. Recommended starting range for 16th-century prints:
    k=0.15–0.20.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image (any mode; converted to grayscale internally).
    window_size : int
        Side length of the local neighbourhood window. Must be odd.
        Larger windows → more tolerance for broad illumination gradients.
        Recommended range for 200 DPI scans: 31–71.
    k : float
        Sensitivity parameter. See note above. Default: 0.2.
    r : float, optional
        Max value of std. Defaults to 128 for 8-bit images.
    max_pixels : int
        Hard cap on total pixel count before Sauvola runs.
        Default: 12 000 000 (~4 000 × 3 000).

    Returns
    -------
    PIL.Image.Image
        Binary image (mode='L'): 255 = background, 0 = ink.
        Always returned at the same (w, h) as the input.
    """
    if window_size % 2 == 0:
        window_size += 1
        logger.warning("Sauvola: window_size adjusted to %d (must be odd)", window_size)

    original_size = img.size          # (w, h) — restored after binarization
    w, h          = original_size
    total_px      = w * h

    # ── Downscale if needed ────────────────────────────────────────────────
    working_img = img
    if total_px > max_pixels:
        scale       = (max_pixels / total_px) ** 0.5
        new_w       = max(1, int(w * scale))
        new_h       = max(1, int(h * scale))
        working_img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.info(
            "Sauvola: image downscaled from %dx%d (%.1f MP) to %dx%d (%.1f MP) "
            "to stay within memory cap of %.0f MP.",
            w, h, total_px / 1e6,
            new_w, new_h, (new_w * new_h) / 1e6,
            max_pixels / 1e6,
        )
        # Adjust window_size proportionally; keep it odd and at least 3
        window_size = max(3, int(window_size * scale))
        if window_size % 2 == 0:
            window_size += 1

    gray      = _pil_to_gray_array(working_img)
    thresh_map = threshold_sauvola(gray, window_size=window_size, k=k, r=r)
    binary     = np.where(gray <= thresh_map, 0, 255).astype(np.uint8)

    result = _gray_array_to_pil(binary)

    # ── Restore original resolution ────────────────────────────────────────
    if result.size != original_size:
        result = result.resize(original_size, Image.NEAREST)
        logger.debug("Sauvola: binary result upscaled back to %dx%d.", *original_size)

    logger.debug(
        "Sauvola: window=%d  k=%.2f  ink_coverage=%.2f%%",
        window_size, k,
        100.0 * np.sum(binary == 0) / binary.size,
    )
    return result


# ─────────────────────────────────────────────
# Public pipeline entry point
# ─────────────────────────────────────────────
    
def preprocess(
    img: Image.Image,
    *,
    tilt_angle: float = 0.0,       # set to 0.0 to skip rotation
    sauvola_window: int = 51,
    sauvola_k: float = 0.2,
) -> Image.Image:
    """
    Full Stage-1 preprocessing pipeline.

    Steps applied in order:
      1. Grayscale conversion
      2. Sauvola local binarization

    Parameters
    ----------
    img : PIL.Image.Image
        Raw scan page (any mode).
    sauvola_window : int
        Sauvola neighbourhood window size (must be odd, default 51).
    sauvola_k : float
        Sauvola sensitivity (default 0.2).

    Returns
    -------
    PIL.Image.Image
        Cleaned, binarized page image (mode='L').
    """
    if tilt_angle != 0.0:
        logger.info("Preprocessing: step 0 — page tilt correction (%.1f°)", tilt_angle)
        img = correct_page_tilt(img, tilt_angle)

    logger.info("Preprocessing: step 1/3 — grayscale")
    img = to_grayscale(img)

    logger.info("Preprocessing: step 2/3 — CLAHE enhancement")
    img = clahe_enhance(img)

    logger.info("Preprocessing: step 3/3 — Sauvola binarization + cleanup")
    img = sauvola_binarize(img, window_size=sauvola_window, k=sauvola_k)
    img = morphological_cleanup(img, min_component_size=3)

    return img