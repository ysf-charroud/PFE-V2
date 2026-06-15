"""Donut inference sidecar — FastAPI app.

Run with:
    uvicorn main:app --port 8001
or:
    python -m uvicorn main:app --port 8001

The Express layer (``backend/src/services/inference.js``) polls ``/health``
on startup and forwards uploads to ``/infer``. The response shape matches
the contract the LayoutLMv3 sidecar used, so the Express side did not need
to change:

    {
      "receipt": <normalised receipt dict>,
      "raw":     <raw CORD-shape parse — for debugging>,
      "num_words": 0,            # no OCR step; kept for backwards compat
      "annotated_filename": null,
      "processing_ms": <float>
    }
"""

from __future__ import annotations

import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from inference import predict
from model_loader import bundle
from normalize import normalize

METRICS_PATH = os.environ.get("METRICS_PATH", os.path.join(_HERE, "metrics.json"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Heavy model load runs once at startup, off the request path.
    bundle.load()
    yield


app = FastAPI(
    title="Donut Inference Sidecar",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": bundle.is_loaded,
        "device": bundle.device_str,
        "model_dir": bundle.resolved_model_dir,
    }


def _report_data_path() -> Path:
    """Look for report_data.json next to the model dir (notebook's output location)."""
    if bundle.resolved_model_dir:
        # resolved dir might be .../donut-cord-finetuned/final/ — check parent too
        candidate = Path(bundle.resolved_model_dir) / "report_data.json"
        if candidate.is_file():
            return candidate
        parent_candidate = Path(bundle.resolved_model_dir).parent / "report_data.json"
        if parent_candidate.is_file():
            return parent_candidate
    return Path(
        os.environ.get(
            "METRICS_PATH",
            str(Path(__file__).resolve().parent.parent.parent
                / "model" / "donut-cord-finetuned" / "report_data.json"),
        )
    )


def _build_metrics_from_report(report: dict) -> dict:
    """Reshape the notebook's report_data.json into the schema the frontend
    Metrics type expects (see frontend/src/lib/api.ts).

    The notebook records:
      report["test_metrics"]:
        exact_match_accuracy, field_micro_precision/recall/f1, field_macro_f1,
        mean_normalized_TED, TED_accuracy, avg_latency_ms, p50/p95_latency_ms,
        test_samples
      report["metrics"]["per_field"] (validation split):
        { top-level-field: {precision, recall, f1, support} }
    """
    test = report.get("test_metrics") or {}
    val = report.get("metrics") or {}
    per_field = val.get("per_field") or {}

    accuracy = test.get("exact_match_accuracy", 0.0)
    micro_p = test.get("field_micro_precision", 0.0)
    micro_r = test.get("field_micro_recall", 0.0)
    micro_f1 = test.get("field_micro_f1", 0.0)
    macro_f1 = test.get("field_macro_f1", 0.0)

    per_label = [
        {
            "label": name,
            "precision": stats.get("precision", 0.0),
            "recall":    stats.get("recall", 0.0),
            "f1":        stats.get("f1", 0.0),
            "support":   stats.get("support", 0),
        }
        for name, stats in per_field.items()
    ]
    per_label.sort(key=lambda r: r["support"], reverse=True)

    # Support-weighted averages from the per-label table (mirrors sklearn).
    total_support = sum(r["support"] for r in per_label) or 1
    weighted_p = sum(r["precision"] * r["support"] for r in per_label) / total_support
    weighted_r = sum(r["recall"]    * r["support"] for r in per_label) / total_support
    weighted_f = sum(r["f1"]        * r["support"] for r in per_label) / total_support

    # Macro precision/recall not in test_metrics — approximate from val per-label
    if per_label:
        macro_p = sum(r["precision"] for r in per_label) / len(per_label)
        macro_r = sum(r["recall"]    for r in per_label) / len(per_label)
    else:
        macro_p = macro_r = macro_f1

    return {
        "available": True,
        "model_dir": bundle.resolved_model_dir or "",
        "dataset": "naver-clova-ix/cord-v2",
        "split": "test (overall) + validation (per-label)",
        "num_receipts": test.get("test_samples"),
        "num_tokens": None,        # not applicable to Donut (no token classification)
        "num_labels": len(per_label),
        "generated_at": report.get("generated_at"),
        "overall": {
            "accuracy": accuracy,
            "macro":    {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
            "micro":    {"precision": micro_p, "recall": micro_r, "f1": micro_f1},
            "weighted": {"precision": weighted_p, "recall": weighted_r, "f1": weighted_f},
        },
        # Normalized Tree-Edit-Distance — the Donut paper's structure-aware metric.
        "ted_accuracy": test.get("TED_accuracy"),
        "mean_normalized_ted": test.get("mean_normalized_TED"),
        "per_label": per_label,
    }


@app.get("/metrics")
def metrics() -> dict:
    """Evaluation metrics produced by the training notebook (CORD test split).

    Looks for ``report_data.json`` in the model directory; reshapes its
    contents into the structure the frontend ``Metrics`` type expects.
    Returns ``{available: false, detail: ...}`` when no report file is found,
    so the frontend can show a clean "metrics not available" state.
    """
    path = _report_data_path()
    if not path.is_file():
        return {
            "available": False,
            "detail": f"No report_data.json found at {path}. Run the evaluation cells in donut_cord_finetune.ipynb to generate it.",
        }
    try:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "detail": f"Could not read {path}: {exc}"}
    return _build_metrics_from_report(report)


@app.get("/metrics")
def metrics():
    """Evaluation metrics produced by evaluate.py (CORD test split)."""
    if not os.path.exists(METRICS_PATH):
        return {"available": False, "detail": "Run evaluate.py to generate metrics."}
    with open(METRICS_PATH) as f:
        return {"available": True, **json.load(f)}


@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    # ``annotate`` is accepted for backwards compatibility with the Express
    # layer; Donut produces no annotations so the value is ignored.
    do_annotate: bool = Query(True, alias="annotate"),  # noqa: ARG001
) -> dict:
    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Empty file.")

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Cannot read image.")

    t0 = time.perf_counter()
    try:
        raw, conf = predict(image)
    except Exception as exc:  # noqa: BLE001 — surface any failure as 500
        raise HTTPException(500, f"Inference failed: {exc}")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    receipt = normalize(raw, conf)

    return {
        "receipt": receipt,
        "raw": raw,                 # raw CORD-shape parse, handy for debugging
        "num_words": 0,             # no OCR step; legacy field kept for compat
        "annotated_filename": None, # Donut produces no annotations
        "processing_ms": round(elapsed_ms, 1),
    }
