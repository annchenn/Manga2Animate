"""
Composes N per-panel images into a labeled grid (numbered caption "1".."N" below
each cell) -- the flux2 backend's equivalent of the single 3x2 sheet the gemini
backend gets back from the image model in one piece. Default cell_size matches
FLUX2_PANEL_WIDTH/HEIGHT (768x640); generate_storyboard.generate_flux2() is the
only caller.
"""
import math

from PIL import Image, ImageDraw, ImageFont

COLUMNS = 3
CELL_SIZE = (768, 640)  # (width, height)
LABEL_HEIGHT = 40
PADDING = 8
BG_COLOR = (255, 255, 255)
LABEL_COLOR = (0, 0, 0)


def _fit_into(img: Image.Image, box: tuple[int, int]) -> Image.Image:
    """Scale img to fit fully inside box (no cropping); letterbox-center on white."""
    img = img.copy()
    img.thumbnail(box, Image.LANCZOS)
    canvas = Image.new("RGB", box, BG_COLOR)
    x = (box[0] - img.width) // 2
    y = (box[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def compose_grid(
    panels: list[Image.Image],
    columns: int = COLUMNS,
    cell_size: tuple[int, int] = CELL_SIZE,
) -> Image.Image:
    n = len(panels)
    rows = math.ceil(n / columns)
    cell_w, cell_h = cell_size
    total_w = columns * cell_w + (columns + 1) * PADDING
    total_h = rows * (cell_h + LABEL_HEIGHT) + (rows + 1) * PADDING

    canvas = Image.new("RGB", (total_w, total_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    font = _load_font(28)

    for idx, panel in enumerate(panels):
        row, col = divmod(idx, columns)
        x = PADDING + col * (cell_w + PADDING)
        y = PADDING + row * (cell_h + LABEL_HEIGHT + PADDING)

        fitted = _fit_into(panel, (cell_w, cell_h))
        canvas.paste(fitted, (x, y))

        label = str(idx + 1)
        bbox = draw.textbbox((0, 0), label, font=font)
        label_w = bbox[2] - bbox[0]
        label_x = x + (cell_w - label_w) // 2
        label_y = y + cell_h + (LABEL_HEIGHT - (bbox[3] - bbox[1])) // 2
        draw.text((label_x, label_y), label, fill=LABEL_COLOR, font=font)

    return canvas
