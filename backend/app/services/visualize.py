"""Draw predicted entity boxes onto the receipt image.

Ported from invoice_extraction_v3.ipynb `visualize()`. Returns a PIL image with
per-class colored boxes and a legend, plus a base64 data-URI helper so the API
can embed the annotated image directly in the JSON response.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.core.config import settings

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
    "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000",
    "#ffd8b1", "#000075", "#808080", "#ff4500", "#2e8b57",
]


def annotate(image: Image.Image, words: list[dict]) -> Image.Image:
    """Return a copy of `image` with colored bboxes per predicted entity class.

    `words` items are {text, label, box (normalized 0-1000), confidence}.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    label_color: dict[str, str] = {}
    for w in words:
        lbl = w["label"]
        if lbl == "O":
            continue
        if lbl not in label_color:
            label_color[lbl] = PALETTE[len(label_color) % len(PALETTE)]
        c = label_color[lbl]

        b = w["box"]
        x1, y1 = b[0] * W / 1000, b[1] * H / 1000
        x2, y2 = b[2] * W / 1000, b[3] * H / 1000
        draw.rectangle([x1, y1, x2, y2], outline=c, width=2)

    # Legend in the top-left corner
    y_off = 5
    for lbl, c in label_color.items():
        draw.rectangle([5, y_off, 18, y_off + 12], fill=c)
        draw.text((22, y_off - 2), lbl, fill="black", font=font)
        y_off += 16
    return img


def save_annotated(image: Image.Image, stem: str | None = None) -> tuple[str, str]:
    """Save the annotated image to the backend output dir.

    Returns (absolute_path, url) where url is the static-serving path.
    """
    name = f"{stem + '_' if stem else ''}{uuid.uuid4().hex}.png"
    out_dir = Path(settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    image.save(path, format="PNG")
    url = f"{settings.output_url_path}/{name}"
    return str(path), url
