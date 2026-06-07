"""Extraction API routes."""
from __future__ import annotations

import io
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.schemas.extraction import ExtractionResponse
from app.services.inference import predict
from app.services.visualize import annotate, save_annotated

router = APIRouter(tags=["extraction"])


@router.post("/extract", response_model=ExtractionResponse)
async def extract(
    file: UploadFile = File(...),
    annotate_image: bool = Query(
        True,
        alias="annotate",
        description="Include the annotated image (base64 data URI) in the response.",
    ),
) -> ExtractionResponse:
    """Receive an invoice/receipt image and return the extracted fields.

    When annotate=true (default) the response also includes `annotated_image`,
    a base64 data URI of the receipt with colored entity boxes drawn on it.
    """
    # --- Validate content type ---
    if file.content_type not in settings.allowed_image_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Allowed: {', '.join(settings.allowed_image_types)}"
            ),
        )

    # --- Read + size check ---
    contents = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.max_upload_mb} MB limit.",
        )
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file.",
        )

    # --- Decode image ---
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not read the uploaded file as an image.",
        )

    # --- Inference ---
    start = time.perf_counter()
    try:
        result = predict(image)
    except Exception as exc:  # noqa: BLE001 - surface as 500 with context
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {exc}",
        )
    elapsed_ms = (time.perf_counter() - start) * 1000

    annotated_path = None
    annotated_url = None
    if annotate_image and result["words"]:
        stem = Path(file.filename).stem if file.filename else None
        annotated_img = annotate(image, result["words"])
        annotated_path, annotated_url = save_annotated(annotated_img, stem=stem)

    return ExtractionResponse(
        filename=file.filename,
        fields=result["fields"],
        line_items=result["line_items"],
        summary=result["summary"],
        words=result["words"],
        num_words=result["num_words"],
        processing_ms=round(elapsed_ms, 1),
        annotated_image_path=annotated_path,
        annotated_image_url=annotated_url,
    )
