"""Donut inference: PIL image → token sequence → parsed CORD dict.

The model + processor do all the heavy lifting. This module just orchestrates
``generate()``, strips the wrapping task tokens, and hands the sequence to
``DonutProcessor.token2json``.

No OCR step, no row clustering, no spatial reasoning — that's the whole point
of switching to Donut.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from PIL import Image

from model_loader import bundle


# Strips the leading task tag (e.g. ``<s_cord-v2>``) from the decoded sequence.
_LEADING_TAG = re.compile(r"^<[^>]*>")


def predict(image: Image.Image) -> dict[str, Any]:
    """Run Donut on one image and return the parsed CORD-shape dict.

    Always returns a ``dict``. On any decoding failure, returns ``{}`` — the
    caller (the FastAPI endpoint) then surfaces an empty receipt rather than
    a 500, which keeps the frontend recoverable.
    """
    if not bundle.is_loaded or bundle.model is None or bundle.processor is None:
        raise RuntimeError("Model bundle not loaded.")

    if image.mode != "RGB":
        image = image.convert("RGB")

    processor = bundle.processor
    model = bundle.model

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(bundle.device)

    with torch.inference_mode():
        out = model.generate(
            pixel_values,
            decoder_input_ids=bundle.decoder_start_ids,
            max_length=bundle.max_length,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

    seq = processor.batch_decode(out.sequences)[0]
    seq = (
        seq.replace(processor.tokenizer.eos_token, "")
        .replace(processor.tokenizer.pad_token, "")
    )
    seq = _LEADING_TAG.sub("", seq, count=1).strip()

    try:
        parsed = processor.token2json(seq)
    except Exception:
        # token2json raises on malformed sequences. Better to return an empty
        # dict than a 500 — the frontend will show "no fields detected".
        return {}

    return parsed if isinstance(parsed, dict) else {}
