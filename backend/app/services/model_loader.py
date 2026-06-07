"""Loads the LayoutLMv3 model, processor and PaddleOCR reader once, at startup.

These are expensive to construct (model weights ~500 MB, PaddleOCR detector/
recognizer downloads), so we build them a single time and reuse across requests.
"""
from __future__ import annotations

import threading

import torch

from app.core.config import settings


class ModelBundle:
    """Holds the model, processor, OCR reader and target device."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.ocr_reader = None
        self.device: torch.device | None = None
        self._loaded = False
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device_str(self) -> str:
        return str(self.device) if self.device is not None else "uninitialized"

    def load(self) -> None:
        """Load all components. Idempotent and thread-safe."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return

            # Imported here so module import stays cheap and import errors surface
            # at startup rather than at request time.
            from paddleocr import PaddleOCR
            from transformers import (
                LayoutLMv3ForTokenClassification,
                LayoutLMv3Processor,
            )

            use_cuda = torch.cuda.is_available() and not settings.force_cpu
            self.device = torch.device("cuda" if use_cuda else "cpu")

            self.processor = LayoutLMv3Processor.from_pretrained(
                settings.model_dir,
                apply_ocr=False,  # OCR is supplied by PaddleOCR, never re-derived
            )
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                settings.model_dir
            )
            self.model.to(self.device).eval()

            # use_angle_cls handles rotated text; show_log=False keeps PaddleOCR's
            # banner/progress output (which uses block chars) off Windows' cp1252
            # stdout. use_gpu needs paddlepaddle-gpu installed to take effect.
            # enable_mkldnn=False avoids paddlepaddle 3.x's oneDNN fused_conv2d
            # crash on Windows CPU ("OneDnnContext does not have the input Filter").
            self.ocr_reader = PaddleOCR(
                use_angle_cls=True,
                lang=settings.ocr_lang,
                use_gpu=use_cuda,
                enable_mkldnn=False,
                show_log=False,
            )

            self._loaded = True


# Module-level singleton shared across the app.
bundle = ModelBundle()
