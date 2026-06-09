import re
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


# Two concatenated amounts with thousand separators, e.g. "110,00070,000".
_DOUBLE_AMOUNT = re.compile(r"^(\d{1,3}[.,]\d{3})(\d{1,3}[.,]\d{3,})$")
# Letter↔digit transition, e.g. "RE216000" → "RE" + "216000".
_ALNUM_SPLIT = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")


def _atomize(text: str):
    """Split a phrase into atomic tokens so column-mashed OCR can be salvaged.

    Splits on:
      1. whitespace (default)
      2. letter↔digit transitions  ("RE216000" → "RE", "216000")
      3. two concatenated amounts   ("110,00070,000" → "110,000", "70,000")
      4. long pure-digit runs       ("111000222000" → "111000", "222000")
    """
    out = []
    for tok in text.split():
        for part in _ALNUM_SPLIT.split(tok):
            if not part:
                continue
            m = _DOUBLE_AMOUNT.match(part)
            if m:
                out.append(m.group(1))
                out.append(m.group(2))
                continue
            if part.isdigit() and len(part) >= 10:
                mid = len(part) // 2
                out.append(part[:mid])
                out.append(part[mid:])
                continue
            out.append(part)
    return out


def _split_phrase(text, box):
    """PaddleOCR returns phrase-level boxes. Split into per-word/per-amount boxes
    proportionally by character count — CORD training data used per-word boxes."""
    toks = _atomize(text)
    if len(toks) <= 1:
        return [(text, box)]
    x1, y1, x2, y2 = box
    span = max(1, x2 - x1)
    total = sum(len(t) for t in toks) + (len(toks) - 1)
    out, pos = [], 0
    for t in toks:
        wx1 = int(x1 + span * pos / total)
        pos += len(t)
        wx2 = int(x1 + span * pos / total)
        pos += 1
        out.append((t, [wx1, y1, max(wx1, wx2), y2]))
    return out


def _deskew(image: Image.Image) -> Image.Image:
    """Detect dominant near-horizontal line angle via Hough lines and rotate to
    correct. No-op for images that are already straight (< 0.5°) or have too
    few line features to estimate confidently."""
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80,
        minLineLength=gray.shape[1] // 5, maxLineGap=20,
    )
    if lines is None or len(lines) < 3:
        return image
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(a) < 45:
                angles.append(a)
    if not angles:
        return image
    angle = float(np.median(angles))
    if abs(angle) < 0.5:
        return image
    return image.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def _upscale_if_needed(image: Image.Image):
    """Receipts shot at <2000px long side hurt PaddleOCR detection. Lanczos
    upscale so the long side hits OCR_UPSCALE_TO."""
    target = bundle.ocr_upscale_to
    if target <= 0:
        return image
    w, h = image.size
    long_side = max(w, h)
    if long_side >= target:
        return image
    scale = min(target / long_side, 4.0)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def predict(image: Image.Image) -> dict:
    model = bundle.model
    processor = bundle.processor
    ocr = bundle.ocr
    id2label = model.config.id2label

    image = image.convert("RGB")
    # Deskew BEFORE everything else so OCR boxes and LayoutLMv3 visual features
    # share the same coordinate frame.
    image = _deskew(image)

    ocr_image = _upscale_if_needed(image)
    ocr_w, ocr_h = ocr_image.size
    img_arr = np.array(ocr_image)

    # --- Step 1: PaddleOCR ---
    ocr_result = ocr.ocr(img_arr, cls=True)
    lines = ocr_result[0] if ocr_result and ocr_result[0] else []

    words, boxes = [], []
    for entry in lines:
        quad, (text, conf) = entry[0], entry[1]
        if conf < bundle.ocr_min_confidence or not text.strip():
            continue
        norm_box = _normalize_box(quad, ocr_w, ocr_h)
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

    # Softmax over the label dimension → per-token probability distribution.
    # probs[t, label_id] = model's confidence that token t has that label.
    probs = logits.softmax(-1).squeeze(0).cpu().numpy()  # [seq_len, num_labels]
    predictions = logits.argmax(-1).squeeze().tolist()
    if isinstance(predictions, int):
        predictions = [predictions]

    # --- Step 4: Majority vote subword → word (+ confidence) ---
    # Group subword-token indices by the word they belong to.
    tok_groups = defaultdict(list)
    for tok_idx, w_id in enumerate(word_ids):
        if w_id is not None:
            tok_groups[w_id].append(tok_idx)

    word_preds, word_confs = {}, {}
    for w_id, tok_idxs in tok_groups.items():
        win = Counter(predictions[t] for t in tok_idxs).most_common(1)[0][0]
        word_preds[w_id] = win
        # Confidence = mean probability assigned to the winning label across
        # this word's subword tokens.
        word_confs[w_id] = float(np.mean([probs[t, win] for t in tok_idxs]))

    word_list = [
        {
            "text": w,
            "box": b,
            "label": id2label[word_preds.get(i, 0)],
            "conf": round(word_confs.get(i, 0.0), 4),
        }
        for i, (w, b) in enumerate(zip(words, boxes))
    ]

    # --- Step 5a: Flat fields dict ---
    flat = defaultdict(list)
    for item in word_list:
        if item["label"] != "O":
            flat[item["label"]].append(item["text"])
    fields = {k: " ".join(v) for k, v in flat.items()}

    # --- Step 5b: Spatial row clustering ---
    labeled_idxs = [i for i, item in enumerate(word_list) if item["label"] != "O"]
    labeled_idxs.sort(key=lambda i: (
        (boxes[i][1] + boxes[i][3]) / 2,
        boxes[i][0],
    ))

    rows, cur, last_yc = [], [], None
    for i in labeled_idxs:
        yc = (boxes[i][1] + boxes[i][3]) / 2
        h = max(1, boxes[i][3] - boxes[i][1])
        if last_yc is None or abs(yc - last_yc) <= max(5, h * 0.5):
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
        for i in sorted(row, key=lambda j: boxes[j][0]):
            row_fields[word_list[i]["label"]].append(word_list[i]["text"])
        row_fields = {k: " ".join(v) for k, v in row_fields.items()}
        if any(k.startswith("menu") for k in row_fields):
            line_items.append(row_fields)
        else:
            for k, cell in row_fields.items():
                if k in summary:
                    summary[k] = {
                        "text": (summary[k]["text"] + " " + cell["text"]).strip(),
                        "conf": (summary[k]["conf"] + cell["conf"]) / 2,
                    }
                else:
                    summary[k] = cell

    return {
        "fields": fields,
        "line_items": line_items,
        "summary": summary,
        "words": word_list,
    }
