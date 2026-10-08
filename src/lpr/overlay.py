"""Draw text (incl. Thai) on an OpenCV image using PIL."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_font(font_path: Path | str, size: int = 28):
    try:
        return ImageFont.truetype(str(font_path), size)
    except OSError:
        return ImageFont.load_default()


def draw_lines_top_right(image_bgr, lines, font, pad=12):
    """Translucent box with one text line per entry, in the top-right corner."""
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(pil)
    sizes = [draw.textbbox((0, 0), t, font=font) for t in lines]
    widths = [b[2] - b[0] for b in sizes]
    heights = [b[3] - b[1] for b in sizes]
    box_w = max(widths, default=0) + pad * 2
    box_h = sum(heights) + pad * (len(lines) + 1)
    x0, y0 = max(0, pil.width - box_w - 20), 20

    layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=14, fill=(0, 0, 0, 160))
    pil = Image.alpha_composite(pil, layer)
    draw = ImageDraw.Draw(pil)
    y = y0 + pad
    for text, h in zip(lines, heights):
        draw.text((x0 + pad, y), text, font=font, fill=(255, 255, 255, 255))
        y += h + pad
    return cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
