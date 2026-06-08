from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
from PIL import Image

from model_loader import bundle


def _normalize_box(pts, width, height):
    x0 = min(p[0] for p in pts)
    y0 = min(p[1] for p in pts)
    x1 = max(p[0] for p in pts)
    y1 = max(p[1] for p in pts)
    x0 = max(0, min(int(x0 * 1000 / width),  1000))
    y0 = max(0, min(int(y0 * 1000 / height), 1000))
    x1 = max(x0, min(int(x1 * 1000 / width),  1000))
    y1 = max(y0, min(int(y1 * 1000 / height), 1000))
    return [x0, y0, x1, y1]


def _split_phrase(text, box):
    """EasyOCR returns phrase-level boxes. Split into per-word boxes proportionally
    by character count — CORD training data used per-word boxes, so inference must too."""
    toks = text.split()
    if len(toks) <= 1:
        return [(text, box)]
    x1, y1, x2, y2 = box
    span = max(1, x2 - x1)
    total = sum(len(t) for t in toks) + (len(toks) - 1)  # chars + spaces
    out, pos = [], 0
    for t in toks:
        wx1 = int(x1 + span * pos / total)
        pos += len(t)
        wx2 = int(x1 + span * pos / total)
        pos += 1  # account for space
        out.append((t, [wx1, y1, max(wx1, wx2), y2]))
    return out


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Detect dominant near-horizontal line angle and rotate to correct skew."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80,
        minLineLength=gray.shape[1] // 5, maxLineGap=20,
    )
    if lines is None or len(lines) < 3:
        return gray
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(a) < 45:
                angles.append(a)
    if not angles:
        return gray
    angle = float(np.median(angles))
    if abs(angle) < 0.5:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _preprocess_for_ocr(image: Image.Image) -> np.ndarray:
    """OCR preprocessing pipeline — only affects EasyOCR input, not LayoutLMv3.

    1. Grayscale
    2. CLAHE  — local contrast boost (helps Canny for deskew)
    3. Deskew — Hough-line skew correction for angled phone photos

    Deliberately stops at grayscale: EasyOCR's CRAFT detector is a neural net that
    does its own internal preprocessing — binarization destroys the contrast gradients
    it relies on for text region detection.
    """
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return _deskew(enhanced)


def _upscale_if_needed(image: Image.Image) -> Image.Image:
    target = bundle.ocr_upscale_to
    if target <= 0:
        return image
    w, h = image.size
    if max(w, h) < target:
        scale = min(target / max(w, h), 8.0)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


def predict(image: Image.Image) -> dict:
    model = bundle.model
    processor = bundle.processor
    ocr = bundle.ocr
    id2label = model.config.id2label

    image = image.convert("RGB")
    ocr_image = _upscale_if_needed(image)
    width, height = image.size
    ocr_w, ocr_h = ocr_image.size
    scale_x = width / ocr_w
    scale_y = height / ocr_h

    # --- Step 1: EasyOCR ---
    # Returns: [(quad, text, confidence), ...]
    # quad = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    ocr_result = ocr.readtext(
        _preprocess_for_ocr(ocr_image),
        text_threshold=0.4,   # lower = detect fainter text
        low_text=0.3,         # lower = detect smaller/lighter chars
        link_threshold=0.4,   # keep default — lowering this merges rows into one region
        min_size=10,
    )

    words, boxes = [], []
    for (quad, text, conf) in ocr_result:
        if conf < bundle.ocr_min_confidence or not text.strip():
            continue
        scaled = [[p[0] * scale_x, p[1] * scale_y] for p in quad]
        norm_box = _normalize_box(scaled, width, height)
        for wtext, wbox in _split_phrase(text.strip(), norm_box):
            words.append(wtext)
            boxes.append(wbox)

    if not words:
        return {"fields": {}, "line_items": [], "summary": {}, "words": []}

    # --- Step 2: LayoutLMv3 encoding ---
    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    word_ids = encoding.word_ids(0)  # MUST capture before moving to device
    encoding = {k: v.to(bundle.device) for k, v in encoding.items()}

    # --- Step 3: Inference ---
    with torch.no_grad():
        logits = model(**encoding).logits

    predictions = logits.argmax(-1).squeeze().tolist()
    if isinstance(predictions, int):
        predictions = [predictions]

    # --- Step 4: Majority vote subword → word ---
    votes = defaultdict(list)
    for tok_idx, w_id in enumerate(word_ids):
        if w_id is not None:
            votes[w_id].append(predictions[tok_idx])
    word_preds = {
        w_id: Counter(v).most_common(1)[0][0]
        for w_id, v in votes.items()
    }

    word_list = [
        {"text": w, "box": b, "label": id2label[word_preds.get(i, 0)]}
        for i, (w, b) in enumerate(zip(words, boxes))
    ]

    # --- Step 5a: Flat fields dict (all non-O labels, for convenience) ---
    flat = defaultdict(list)
    for item in word_list:
        if item["label"] != "O":
            flat[item["label"]].append(item["text"])
    fields = {k: " ".join(v) for k, v in flat.items()}

    # --- Step 5b: Spatial row clustering ---
    # Sort labeled words by y-centre then x so we process top-to-bottom, left-to-right
    labeled_idxs = [i for i, item in enumerate(word_list) if item["label"] != "O"]
    labeled_idxs.sort(key=lambda i: (
        (boxes[i][1] + boxes[i][3]) / 2,
        boxes[i][0],
    ))

    rows, cur, last_yc = [], [], None
    for i in labeled_idxs:
        yc = (boxes[i][1] + boxes[i][3]) / 2
        h = max(1, boxes[i][3] - boxes[i][1])
        # Words within 60% of one line-height of the previous centre → same row
        if last_yc is None or abs(yc - last_yc) <= max(8, h * 0.6):
            cur.append(i)
        else:
            rows.append(cur)
            cur = [i]
        last_yc = yc
    if cur:
        rows.append(cur)

    line_items, summary = [], {}
    for row in rows:
        row_fields = defaultdict(list)
        for i in sorted(row, key=lambda j: boxes[j][0]):  # left-to-right within row
            row_fields[word_list[i]["label"]].append(word_list[i]["text"])
        row_fields = {k: " ".join(v) for k, v in row_fields.items()}
        if any(k.startswith("menu") for k in row_fields):
            line_items.append(row_fields)
        else:
            for k, v in row_fields.items():
                summary[k] = (summary[k] + " " + v).strip() if k in summary else v

    return {
        "fields": fields,
        "line_items": line_items,
        "summary": summary,
        "words": word_list,
    }
