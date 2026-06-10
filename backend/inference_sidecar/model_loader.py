"""Load the fine-tuned Donut (``VisionEncoderDecoderModel``) + its processor.

Exposes a module-level singleton ``bundle`` that the rest of the sidecar uses.
The bundle is created empty at import time and populated by ``bundle.load()``
during FastAPI's lifespan hook so heavy work doesn't run during module import.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

_HERE = Path(__file__).resolve().parent

# Default points at <repo>/model/donut-cord-finetuned. The training notebook
# saves into a `final/` subdirectory of OUTPUT_DIR, so we also accept that.
_DEFAULT_MODEL_DIR = _HERE.parent.parent / "model" / "donut-cord-finetuned"

MODEL_DIR = os.environ.get("MODEL_DIR", str(_DEFAULT_MODEL_DIR))
FORCE_CPU = os.environ.get("FORCE_CPU", "false").lower() == "true"
TASK_TOKEN = os.environ.get("TASK_TOKEN", "<s_cord-v2>")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "768"))
USE_SLOW_TOKENIZER = os.environ.get("USE_SLOW_TOKENIZER", "true").lower() == "true"


def _resolve_model_dir(base: str) -> Path:
    """Pick the right path for ``from_pretrained``.

    If ``<base>/config.json`` exists, use ``base`` directly.
    Else if ``<base>/final/config.json`` exists, use ``<base>/final``
    (the notebook's default save location).
    Else return ``base`` unchanged and let the loader raise its own error.
    """
    p = Path(base)
    if (p / "config.json").is_file():
        return p
    final = p / "final"
    if (final / "config.json").is_file():
        return final
    return p


class ModelBundle:
    """Holds the loaded processor + model + ancillary state."""

    def __init__(self) -> None:
        self.processor: DonutProcessor | None = None
        self.model: VisionEncoderDecoderModel | None = None
        self.device: torch.device | None = None
        self.is_loaded: bool = False
        self.task_token: str = TASK_TOKEN
        self.max_length: int = MAX_LENGTH
        # Pre-tokenised <s_cord-v2> tensor, ready to feed to generate().
        self.decoder_start_ids: torch.Tensor | None = None
        # The directory we actually loaded from (useful for /health debugging).
        self.resolved_model_dir: str = ""

    def load(self) -> None:
        resolved = _resolve_model_dir(MODEL_DIR)
        self.resolved_model_dir = str(resolved)
        if not (resolved / "config.json").is_file():
            raise FileNotFoundError(
                f"No fine-tuned Donut model found at {resolved} — "
                f"checked for config.json directly and in a final/ subdir. "
                f"Train the model with donut_cord_finetune.ipynb and place "
                f"the result under model/donut-cord-finetuned/."
            )

        use_gpu = not FORCE_CPU and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_gpu else "cpu")
        print(f"Loading Donut from {resolved} on {self.device} ...")

        # use_fast=False forces the slow XLMRoberta tokenizer, avoiding the
        # slow->fast conversion that needs protobuf — that conversion breaks
        # on Lightning AI's `cloudspace` and some slim Docker base images.
        self.processor = DonutProcessor.from_pretrained(
            str(resolved), use_fast=not USE_SLOW_TOKENIZER
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(str(resolved))
        self.model.to(self.device)
        self.model.eval()

        # Pre-encode the task-start token so the inference path stays hot.
        tok = self.processor.tokenizer
        self.decoder_start_ids = tok(
            self.task_token, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)

        # Sanity-check: the task token should resolve to a single id, not unk.
        task_id = tok.convert_tokens_to_ids(self.task_token)
        if task_id == tok.unk_token_id:
            print(
                f"WARNING: task token {self.task_token!r} resolved to <unk>. "
                f"Generation will likely produce gibberish — verify TASK_TOKEN "
                f"matches what the model was trained with."
            )

        self.is_loaded = True
        print(f"Donut ready on {self.device}.")

    @property
    def device_str(self) -> str:
        return str(self.device) if self.device is not None else "unknown"


bundle = ModelBundle()
