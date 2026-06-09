"""Inference sidecar — FastAPI app.

Run with:
    uvicorn main:app --port 8001
or:
    python -m uvicorn main:app --port 8001
"""
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError

from inference import predict
from model_loader import bundle
from normalize import normalize
from visualize import annotate, save_image

_HERE = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(_HERE, "..", "storage", "annotated"),
)

METRICS_PATH = os.environ.get("METRICS_PATH", os.path.join(_HERE, "metrics.json"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle.load()
    yield


app = FastAPI(title="Inference Sidecar", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bundle.is_loaded, "device": bundle.device_str}


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
    do_annotate: bool = Query(True, alias="annotate"),
):
    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Empty file.")

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Cannot read image.")

    try:
        result = predict(image)
    except Exception as exc:
        raise HTTPException(500, f"Inference failed: {exc}")

    annotated_filename = None
    if do_annotate and result["words"]:
        stem = Path(file.filename).stem if file.filename else None
        annotated_filename = save_image(annotate(image, result["words"]), OUTPUT_DIR, stem)

    return {
        "receipt": normalize(result),
        "num_words": len(result["words"]),
        "annotated_filename": annotated_filename,
    }
