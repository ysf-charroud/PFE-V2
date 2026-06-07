"""Image -> structured fields inference, ported from invoice_extraction_v3.ipynb.

Pipeline: PaddleOCR (text + boxes) -> LayoutLMv3Processor -> token classification
-> aggregate subwords to words (first subword wins) -> group words by label.
"""
from __future__ import annotations

import re
import statistics

import numpy as np
import torch
from PIL import Image, ImageOps

from app.core.config import settings
from app.services.model_loader import bundle

# A token that is purely digits/separators (a price/amount) vs. a name word.
_NUMERIC_RE = re.compile(r"^[\d.,]+$")

# Labels treated as the item's name, price, and quantity when building rows.
_NAME_LABELS = {"menu.nm", "menu.sub_nm"}
_PRICE_LABELS = {"menu.price", "menu.unitprice"}
_QTY_LABELS = {"menu.cnt", "menu.sub_cnt"}


def _normalize_box(
    x1: float, y1: float, x2: float, y2: float, W: int, H: int
) -> list[int]:
    """Scale a pixel box to LayoutLMv3's 0-1000 space, clamped and monotonic."""
    box = [
        max(0, min(1000, int(x1 * 1000 / W))),
        max(0, min(1000, int(y1 * 1000 / H))),
        max(0, min(1000, int(x2 * 1000 / W))),
        max(0, min(1000, int(y2 * 1000 / H))),
    ]
    # Processor errors on non-monotonic boxes; force x2>=x1, y2>=y1.
    if box[2] < box[0]:
        box[2] = box[0]
    if box[3] < box[1]:
        box[3] = box[1]
    return box


def _enhance_for_ocr(image: Image.Image) -> Image.Image:
    """Boost legibility of faint thermal receipts before OCR.

    Grayscale + autocontrast lifts washed-out blue thermal print; the result is
    only fed to the OCR reader, not the LayoutLMv3 processor (which keeps the
    original image), so box coordinates stay aligned to the source pixels.
    """
    g = ImageOps.grayscale(image)
    g = ImageOps.autocontrast(g, cutoff=2)
    return g.convert("RGB")


def _fix_label(text: str, label: str) -> str:
    """Cheap post-fixes for common per-token model slips inside the menu block."""
    if label in _NAME_LABELS and _NUMERIC_RE.match(text) and any(c.isdigit() for c in text):
        # A bare amount mislabeled as a name -> it's a price.
        return "menu.price"
    if label in _PRICE_LABELS and text.isalpha():
        # An all-letters token mislabeled as a price -> it's a name.
        return "menu.nm"
    return label


def _group_rows(items: list[dict], row_tol: float) -> list[list[dict]]:
    """Cluster word dicts (with 'yc', 'x1') into rows by vertical proximity.

    Words whose vertical centers fall within `row_tol` of the current row join
    it; each row is returned left-to-right. This is what lets us pair an item
    name with the price sitting on the same baseline.
    """
    rows: list[list[dict]] = []
    current: list[dict] = []
    for w in sorted(items, key=lambda d: d["yc"]):
        if current and w["yc"] - current[-1]["yc"] > row_tol:
            rows.append(sorted(current, key=lambda d: d["x1"]))
            current = []
        current.append(w)
    if current:
        rows.append(sorted(current, key=lambda d: d["x1"]))
    return rows


def _build_structure(word_list: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """Turn flat labeled words into ordered menu line-items + a totals summary.

    line_items pair each menu name with the quantity/price on its row; rows that
    carry a name but no price are treated as modifiers of the previous item
    (e.g. "MEDIUM WELL" under "WAGYU RIBEYE"). summary maps each sub_total.*/
    total.* label to its numeric value(s).
    """
    if not word_list:
        return [], {}

    enriched = [
        {**w, "yc": (w["box"][1] + w["box"][3]) / 2, "x1": w["box"][0]}
        for w in word_list
    ]
    heights = [w["box"][3] - w["box"][1] for w in word_list]
    row_tol = max(8.0, statistics.median(heights) * 0.6)

    # --- Menu line-items ---
    menu = [w for w in enriched if w["label"].startswith("menu.")]
    line_items: list[dict] = []
    for row in _group_rows(menu, row_tol):
        names = [w["text"] for w in row if w["label"] in _NAME_LABELS]
        prices = [w["text"] for w in row if w["label"] in _PRICE_LABELS]
        qty = next((w["text"] for w in row if w["label"] in _QTY_LABELS), None)
        if not names and not prices:
            continue
        if prices or not line_items:
            line_items.append(
                {
                    "name": " ".join(names),
                    "qty": qty,
                    "price": " ".join(prices) or None,
                }
            )
        elif names:  # continuation/modifier row (name, no price)
            line_items[-1]["name"] = f"{line_items[-1]['name']} {' '.join(names)}".strip()
    line_items = [li for li in line_items if li["name"] or li["price"]]

    # --- Totals summary (keep only the numeric amount per label, dropping the
    # descriptive label words like "SUBTOTAL"/"PB1"/"DUE") ---
    summary: dict[str, list[str]] = {}
    for w in enriched:
        lbl = w["label"]
        if (lbl.startswith("sub_total.") or lbl.startswith("total.")) and _NUMERIC_RE.match(
            w["text"]
        ):
            summary.setdefault(lbl, []).append(w["text"])
    return line_items, {k: " ".join(v) for k, v in summary.items()}


def predict(image: Image.Image) -> dict:
    """Run the full pipeline on a PIL image.

    Returns a dict with:
        fields:     {label: " ".join(words)}  (flat, kept for backward compat)
        line_items: [{name, qty, price}, ...]  (menu rows reconstructed by layout)
        summary:    {sub_total.*/total.* label: numeric value}
        words:      [{text, label, box, confidence}, ...]  (label 'O' excluded)
        num_words:  number of OCR words kept
    """
    if not bundle.is_loaded:
        bundle.load()

    model = bundle.model
    processor = bundle.processor
    ocr_reader = bundle.ocr_reader
    device = bundle.device
    id2label = model.config.id2label

    if image.mode != "RGB":
        image = image.convert("RGB")
    W, H = image.size

    # 1. OCR -> words, normalized boxes, confidences
    #
    # PaddleOCR returns *text regions*, often merging several words/columns into a
    # single coarse box (e.g. "NASI GORENG 25,000"). The model was fine-tuned on
    # CORD's word-level tokens (one tight box per word), so we split each region
    # into whitespace tokens, allocating each token a slice of the region by
    # character count. We interpolate that slice *along the detected quad's top
    # and bottom edges*, so on a tilted/perspective photo each token gets a tight
    # box that follows the text baseline instead of one inflated axis-aligned box
    # spanning the whole line.
    #
    # OCR runs on a contrast-enhanced (and, for small images, upscaled) copy to
    # read faint thermal print; quad coordinates are divided back by `scale` so
    # boxes map onto the original pixels. The LayoutLMv3 processor still receives
    # the untouched `image`.
    scale = 1.0
    long_side = max(W, H)
    if settings.ocr_upscale_to and long_side < settings.ocr_upscale_to:
        scale = min(2.0, settings.ocr_upscale_to / long_side)
    ocr_img = _enhance_for_ocr(image)
    if scale != 1.0:
        ocr_img = ocr_img.resize((round(W * scale), round(H * scale)))

    # PaddleOCR .ocr() returns one list per image; result[0] is this image's
    # detections (or None when nothing is found). Each detection is
    # [quad, (text, prob)] with quad four [x, y] points (TL, TR, BR, BL).
    result = ocr_reader.ocr(np.array(ocr_img), cls=True)
    detections = result[0] if result else None
    words: list[str] = []
    boxes: list[list[int]] = []
    confidences: list[float] = []
    for quad, (text, prob) in detections or []:
        if prob < settings.ocr_min_confidence or not text.strip():
            continue
        tokens = text.split()
        if not tokens:
            continue

        tl, tr, br, bl = ((p[0] / scale, p[1] / scale) for p in quad)
        tl, tr, br, bl = list(tl), list(tr), list(br), list(bl)
        # total "character columns" = chars + one space between each token pair
        total_chars = sum(len(t) for t in tokens) + (len(tokens) - 1)
        pos = 0
        for tok in tokens:
            f0 = pos / total_chars if total_chars else 0.0
            f1 = (pos + len(tok)) / total_chars if total_chars else 1.0
            pos += len(tok) + 1  # advance past this token and one space

            # Interpolate top edge (tl->tr) and bottom edge (bl->br) at f0..f1,
            # then take the axis-aligned bound of those four points.
            corners_x = [
                tl[0] + (tr[0] - tl[0]) * f0, tl[0] + (tr[0] - tl[0]) * f1,
                bl[0] + (br[0] - bl[0]) * f0, bl[0] + (br[0] - bl[0]) * f1,
            ]
            corners_y = [
                tl[1] + (tr[1] - tl[1]) * f0, tl[1] + (tr[1] - tl[1]) * f1,
                bl[1] + (br[1] - bl[1]) * f0, bl[1] + (br[1] - bl[1]) * f1,
            ]
            box = _normalize_box(
                min(corners_x), min(corners_y), max(corners_x), max(corners_y), W, H
            )
            words.append(tok)
            boxes.append(box)
            confidences.append(float(prob))

    if not words:
        return {"fields": {}, "line_items": [], "summary": {}, "words": [], "num_words": 0}

    # 2. Process
    enc = processor(
        image,
        words,
        boxes=boxes,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    word_ids = enc.word_ids(0)  # capture BEFORE moving tensors to device

    enc_dev = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc_dev).logits.squeeze(0)
    preds = logits.argmax(-1).tolist()

    # 3. Aggregate token -> word predictions (first subword wins)
    word_predictions: dict[int, int] = {}
    for tok_idx, w_id in enumerate(word_ids):
        if w_id is None:
            continue
        if w_id not in word_predictions:
            word_predictions[w_id] = preds[tok_idx]

    # 4. Group words by label + build per-word list
    grouped: dict[str, list[str]] = {}
    word_list: list[dict] = []
    for w_idx in sorted(word_predictions):
        if w_idx >= len(words):
            continue
        lbl = id2label[word_predictions[w_idx]]
        if lbl == "O":
            continue
        lbl = _fix_label(words[w_idx], lbl)  # repair obvious per-token slips
        grouped.setdefault(lbl, []).append(words[w_idx])
        word_list.append(
            {
                "text": words[w_idx],
                "label": lbl,
                "box": boxes[w_idx],
                "confidence": confidences[w_idx],
            }
        )

    # 5. Reconstruct layout: menu line-items + totals summary from box positions
    line_items, summary = _build_structure(word_list)

    fields = {k: " ".join(v) for k, v in grouped.items()}
    return {
        "fields": fields,
        "line_items": line_items,
        "summary": summary,
        "words": word_list,
        "num_words": len(words),
    }
