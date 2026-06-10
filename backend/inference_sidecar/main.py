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
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from inference import predict
from model_loader import bundle
from normalize import normalize


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
        raw = predict(image)
    except Exception as exc:  # noqa: BLE001 — surface any failure as 500
        raise HTTPException(500, f"Inference failed: {exc}")
    elapsed_ms = (time.perf_counter() - t0) * 1000

    receipt = normalize(raw)

    return {
        "receipt": receipt,
        "raw": raw,                 # raw CORD-shape parse, handy for debugging
        "num_words": 0,             # no OCR step; legacy field kept for compat
        "annotated_filename": None, # Donut produces no annotations
        "processing_ms": round(elapsed_ms, 1),
    }
