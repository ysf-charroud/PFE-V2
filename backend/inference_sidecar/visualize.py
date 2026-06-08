import hashlib
import os
import uuid
from PIL import Image, ImageDraw

_color_cache: dict[str, tuple] = {}


def _label_color(label: str) -> tuple:
    if label not in _color_cache:
        h = int(hashlib.md5(label.encode()).hexdigest()[:6], 16)
        _color_cache[label] = ((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF)
    return _color_cache[label]


def annotate(image: Image.Image, words: list) -> Image.Image:
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = image.size

    for word in words:
        label = word["label"]
        if label == "O":
            continue
        b = word["box"]
        x0, y0, x1, y1 = b[0] * w / 1000, b[1] * h / 1000, b[2] * w / 1000, b[3] * h / 1000
        r, g, b_ = _label_color(label)
        draw.rectangle([x0, y0, x1, y1], outline=(r, g, b_, 255), width=2, fill=(r, g, b_, 50))

    return Image.alpha_composite(img, overlay).convert("RGB")


def save_image(image: Image.Image, output_dir: str, stem: str = None) -> str:
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{stem}_" if stem else ""
    filename = f"{prefix}{uuid.uuid4().hex[:10]}.png"
    image.save(os.path.join(output_dir, filename))
    return filename
