"""Application configuration.

Values can be overridden via environment variables (or a .env file).
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> backend/  -> project root
BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API ---
    app_name: str = "Invoice Extraction API"
    app_version: str = "0.1.0"
    api_prefix: str = "/api"

    # CORS — the React dev server origins allowed to call this API
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # --- Model ---
    # Path to the fine-tuned LayoutLMv3 model + processor directory.
    model_dir: str = str(PROJECT_ROOT / "model" / "cord_layoutlmv3")
    # Force CPU even if CUDA is available (handy for local dev).
    force_cpu: bool = False
    # OCR language for PaddleOCR (single ISO code, e.g. "en", "fr", "ch").
    ocr_lang: str = "en"
    # Minimum OCR confidence to keep a detected word.
    ocr_min_confidence: float = 0.5
    # Upscale images whose long side is below this (px) before OCR, to read
    # small/faint text. Capped at 2x. Set 0 to disable upscaling.
    ocr_upscale_to: int = 1600

    # --- Uploads ---
    max_upload_mb: int = 10
    allowed_image_types: list[str] = ["image/jpeg", "image/png", "image/webp"]

    # --- Output storage ---
    # Where annotated images are written on disk.
    output_dir: str = str(BACKEND_DIR / "storage" / "annotated")
    # URL path under which saved images are served (StaticFiles mount).
    output_url_path: str = "/files/annotated"


settings = Settings()

# Ensure the output directory exists at import time.
Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
