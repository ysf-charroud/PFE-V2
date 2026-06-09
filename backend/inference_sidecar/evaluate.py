"""Evaluate the fine-tuned LayoutLMv3 on the CORD v2 test split.

Reproduces the notebook's token-level evaluation (masking -100 positions) and
writes per-label precision / recall / F1 + overall metrics to metrics.json,
which the sidecar serves at GET /metrics for the frontend Models dashboard.

Run (from backend/inference_sidecar, with the venv):
    MODEL_DIR=/abs/path/to/cord_layoutlmv3 .venv/bin/python evaluate.py
"""
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get(
    "MODEL_DIR", os.path.join(_HERE, "..", "..", "cord_layoutlmv3")
)
OUT_PATH = os.environ.get("METRICS_PATH", os.path.join(_HERE, "metrics.json"))
MAX_SAMPLES = int(os.environ.get("EVAL_MAX_SAMPLES", "0"))  # 0 = all test samples


def parse_cord(ground_truth, img_w, img_h):
    """Parse CORD ground_truth JSON into (words, normalized_bboxes, label_strs).

    Mirrors the notebook: each word carries a 'quad' (4 corners) and inherits
    the 'category' label of its parent valid_line.
    """
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    words, boxes, labels = [], [], []
    for line in gt.get("valid_line", []):
        category = line.get("category", "O")
        for w in line.get("words", []):
            text = w.get("text", "").strip()
            quad = w.get("quad", {})
            if not text or not quad:
                continue
            xs = [quad.get(f"x{i}", 0) for i in (1, 2, 3, 4)]
            ys = [quad.get(f"y{i}", 0) for i in (1, 2, 3, 4)]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            box = [
                max(0, min(1000, int(x1 * 1000 / img_w))),
                max(0, min(1000, int(y1 * 1000 / img_h))),
                max(0, min(1000, int(x2 * 1000 / img_w))),
                max(0, min(1000, int(y2 * 1000 / img_h))),
            ]
            if box[2] < box[0]:
                box[2] = box[0]
            if box[3] < box[1]:
                box[3] = box[1]
            words.append(text)
            boxes.append(box)
            labels.append(category)
    return words, boxes, labels


def main():
    device = torch.device("cpu")
    print(f"Loading model from {MODEL_DIR} ...")
    processor = LayoutLMv3Processor.from_pretrained(MODEL_DIR, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR).to(device).eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    print("Loading CORD v2 test split ...")
    test = load_dataset("naver-clova-ix/cord-v2", split="test")
    n = len(test) if MAX_SAMPLES <= 0 else min(MAX_SAMPLES, len(test))
    print(f"Evaluating on {n} test receipts ...")

    all_preds, all_labels = [], []
    t0 = time.time()
    for i in range(n):
        ex = test[i]
        image = ex["image"].convert("RGB")
        w, h = image.size
        words, boxes, lbl_strs = parse_cord(ex["ground_truth"], w, h)
        if not words:
            continue
        label_ids = [label2id.get(l, 0) for l in lbl_strs]

        enc = processor(
            images=image,
            text=words,
            boxes=boxes,
            word_labels=label_ids,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = enc.pop("labels")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        preds = logits.argmax(-1).squeeze(0).cpu().numpy()
        labels = labels.squeeze(0).numpy()

        mask = labels != -100  # drop special tokens & subword continuations
        all_preds.append(preds[mask])
        all_labels.append(labels[mask])

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{n}  ({time.time() - t0:.0f}s)")

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    # Per-label precision / recall / F1 / support across all label ids.
    label_ids = list(range(num_labels))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0
    )
    per_label = []
    for idx in label_ids:
        if support[idx] == 0:
            continue  # class absent from the test set
        per_label.append(
            {
                "label": id2label[idx],
                "precision": round(float(p[idx]), 4),
                "recall": round(float(r[idx]), 4),
                "f1": round(float(f1[idx]), 4),
                "support": int(support[idx]),
            }
        )
    per_label.sort(key=lambda x: x["support"], reverse=True)

    # Overall metrics. Macro weights every class equally (reflects rare fields);
    # micro is token-level accuracy; weighted accounts for class frequency.
    def avg(kind):
        pp, rr, ff, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=label_ids, average=kind, zero_division=0
        )
        return {"precision": round(float(pp), 4), "recall": round(float(rr), 4), "f1": round(float(ff), 4)}

    accuracy = float((y_pred == y_true).mean())
    metrics = {
        "model_dir": os.path.abspath(MODEL_DIR),
        "dataset": "naver-clova-ix/cord-v2",
        "split": "test",
        "num_receipts": n,
        "num_tokens": int(len(y_true)),
        "num_labels": num_labels,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": {
            "accuracy": round(accuracy, 4),
            "macro": avg("macro"),
            "micro": avg("micro"),
            "weighted": avg("weighted"),
        },
        "per_label": per_label,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone in {time.time() - t0:.0f}s. Wrote {OUT_PATH}")
    print(
        f"  accuracy={accuracy:.4f}  macro_f1={metrics['overall']['macro']['f1']:.4f}  "
        f"weighted_f1={metrics['overall']['weighted']['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
