"""
pdf_convert.py — PDF Ingestion + Stage 1 Preprocessing
=======================================================
Reads every .pdf inside `pdfs/`, renders each page at 200 DPI,
detects and splits double-page spreads, then applies the Stage-1
preprocessing pipeline (grayscale → Sauvola binarization).

Why 200 DPI?
  350 DPI on large-format historical spreads produces images exceeding
  19 000 × 14 000 pixels.  Sauvola's float64 integral image alone would
  require 2+ GB.  200 DPI keeps single pages comfortably under 4 000 × 6 000
  (~100 MB for Sauvola) while remaining above the ~150 DPI minimum for
  reliable OCR character recognition.

Output folders
--------------
raw_pages/   — original RGB scans at 200 DPI, one PNG per logical page
bw_pages/    — preprocessed black-and-white binary PNGs (same filenames)
"""

import logging
import sys
from pathlib import Path

import fitz                      # PyMuPDF
import numpy as np
from PIL import Image

from data_cleaning import preprocess

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
RENDER_DPI = 200          # reduced from 350 — prevents OOM on large spreads

# ── Paths ─────────────────────────────────────────────────────────────────────
PDF_FOLDER = Path("pdfs")
RAW_DIR    = Path("raw_pages")
BW_DIR     = Path("bw_pages")

RAW_DIR.mkdir(exist_ok=True)
BW_DIR.mkdir(exist_ok=True)

# ── Double-page detection ─────────────────────────────────────────────────────

def is_double_page(img: Image.Image,
                   threshold_ratio: float = 1.1,
                   gutter_threshold: float = 0.95) -> bool:
    """
    Return True if `img` looks like a two-page spread.

    Two heuristics (either is sufficient):
    1. Width/height ratio exceeds `threshold_ratio`  (landscape-ish pages).
    2. The vertical centre strip is predominantly white — indicative of a
       gutter between two facing pages.
    """
    w, h = img.size
    if w / h > threshold_ratio:
        return True

    gray  = np.array(img.convert("L"))
    strip = gray[:, w // 2 - 10: w // 2 + 10]
    return float(np.mean(strip > 240)) > gutter_threshold


def split_double_page(img: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Crop a double-page spread into left and right halves."""
    w, h = img.size
    return img.crop((0, 0, w // 2, h)), img.crop((w // 2, 0, w, h))


# ── Per-page save ─────────────────────────────────────────────────────────────

def save_page(img: Image.Image, page_idx: int) -> None:
    """
    Save raw scan to raw_pages/ and preprocessed binary to bw_pages/.

    Parameters
    ----------
    img : PIL.Image.Image
        Original rendered page (RGB).
    page_idx : int
        Zero-based page counter used to build the filename.
    """
    filename = f"page_{page_idx:03d}.png"
    raw_path = RAW_DIR / filename
    bw_path  = BW_DIR  / filename

    img.save(raw_path, "PNG")

    logger.info("  → binarising  %s  (%d × %d px)", filename, img.width, img.height)
    bw = preprocess(img)
    bw.save(bw_path, "PNG")

    logger.info("  ✓  raw → %s   bw → %s", raw_path, bw_path)


# ── Main loop ─────────────────────────────────────────────────────────────────

def convert_pdfs() -> None:
    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDFs found in '%s'. Nothing to do.", PDF_FOLDER)
        return

    page_counter = 0

    for pdf_path in pdf_files:
        logger.info("Opening: %s", pdf_path.name)
        doc = fitz.open(pdf_path)

        for page in doc:
            pix  = page.get_pixmap(dpi=RENDER_DPI)
            mode = "RGB" if pix.n < 4 else "RGBA"
            img  = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

            if is_double_page(img):
                logger.info("  page %d: double spread — splitting", page.number)
                left, right = split_double_page(img)
                save_page(left,  page_counter); page_counter += 1
                save_page(right, page_counter); page_counter += 1
            else:
                save_page(img, page_counter); page_counter += 1

        doc.close()

    logger.info(
        "Done. %d page(s) →  raw: '%s'   bw: '%s'",
        page_counter, RAW_DIR, BW_DIR,
    )


if __name__ == "__main__":
    convert_pdfs()