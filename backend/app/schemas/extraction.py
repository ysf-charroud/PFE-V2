"""Pydantic models for the extraction endpoint."""
from pydantic import BaseModel, Field


class WordPrediction(BaseModel):
    """A single OCR word with its predicted entity label and normalized box."""
    text: str
    label: str
    box: list[int] = Field(..., description="Normalized [x1, y1, x2, y2] in 0-1000 space")
    confidence: float | None = Field(None, description="OCR confidence (0-1)")


class LineItem(BaseModel):
    """One menu row reconstructed from box layout: a name with its qty/price."""
    name: str = ""
    qty: str | None = None
    price: str | None = None


class ExtractionResponse(BaseModel):
    """Response returned by POST /api/extract."""
    filename: str | None = None
    fields: dict[str, str] = Field(
        default_factory=dict,
        description="Extracted entities grouped by label, e.g. {'total.total_price': '75,000'}",
    )
    line_items: list[LineItem] = Field(
        default_factory=list,
        description="Menu items reconstructed by row layout: each pairs a name with its qty/price.",
    )
    summary: dict[str, str] = Field(
        default_factory=dict,
        description="Totals block (sub_total.*/total.*) mapped to their numeric values.",
    )
    words: list[WordPrediction] = Field(
        default_factory=list,
        description="Per-word OCR text, predicted label and box (label 'O' words excluded)",
    )
    num_words: int = 0
    processing_ms: float | None = None
    annotated_image_path: str | None = Field(
        None,
        description="Absolute path to the annotated image saved on the backend. "
        "Null when annotate=false.",
    )
    annotated_image_url: str | None = Field(
        None,
        description="URL to fetch the annotated image (served as a static file). "
        "Null when annotate=false.",
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    device: str
