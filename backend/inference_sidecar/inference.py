"""Donut inference: PIL image → token sequence → parsed CORD dict + confidence.

The model + processor do the heavy lifting. This module orchestrates
``generate()`` (capturing per-token scores), strips the wrapping task tokens,
hands the sequence to ``DonutProcessor.token2json`` for the values, and walks
the same token stream to derive a per-field **confidence** from the decoder's
softmax probabilities.

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

# Matches a CORD structural tag: <s_NAME> (open) or </s_NAME> (close).
_TAG_RE = re.compile(r"^</?s_(.+)>$")

# Top-level CORD groups. Everything else <s_x> is a leaf field.
_GROUPS = {"menu", "sub_total", "total", "void_menu"}


def _field_confidences(gen_ids, probs, tokenizer):
    """Walk the generated token stream and aggregate decoder probabilities into
    a per-field confidence, mirroring token2json's structure.

    Returns ``(groups, overall)`` where::

        groups = {
            "menu":  [ {leaf_key: mean_prob, ...}, ...],  # one dict per line item
            "total": [ {leaf_key: mean_prob, ...} ],
            ...
        }

    ``leaf_key`` is the raw CORD field name (``nm``, ``price``, ``total_price``…).
    ``overall`` is the mean confidence across every detected field.
    """
    toks = tokenizer.convert_ids_to_tokens(gen_ids)
    skip = {tokenizer.eos_token, tokenizer.pad_token, tokenizer.bos_token, bundle.task_token}

    groups: dict[str, list] = {}
    cur_group = None
    items: list = []          # item dicts for the current group
    cur: dict = {}            # current item: leaf_key -> [probs]
    field = None              # currently open leaf field
    buf: list = []            # content-token probs for the open field

    def commit_field():
        if field is not None and buf:
            cur.setdefault(field, []).extend(buf)

    for tok, p in zip(toks, probs):
        if tok in skip:
            continue
        if tok == "<sep/>":  # list-item separator within a group
            commit_field()
            field, buf = None, []
            if cur:
                items.append(cur)
                cur = {}
            continue
        m = _TAG_RE.match(tok)
        if m:
            name, opening = m.group(1), not tok.startswith("</")
            if name in _GROUPS:
                commit_field()
                if cur:
                    items.append(cur)
                if cur_group is not None:
                    groups[cur_group] = items
                # An open starts a new group; a close ends the current one.
                cur_group = name if opening else None
                items, cur, field, buf = [], {}, None, []
            else:  # leaf field open/close
                commit_field()
                field, buf = (name, []) if opening else (None, [])
        elif field is not None:  # content token inside an open field
            buf.append(p)

    commit_field()
    if cur:
        items.append(cur)
    if cur_group is not None:
        groups[cur_group] = items

    mean_groups = {
        g: [{k: sum(v) / len(v) for k, v in it.items()} for it in its]
        for g, its in groups.items()
    }
    all_confs = [c for its in mean_groups.values() for it in its for c in it.values()]
    overall = round(sum(all_confs) / len(all_confs), 4) if all_confs else None
    return mean_groups, overall


def predict(image: Image.Image) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run Donut on one image. Returns ``(parsed, confidence)``.

    ``parsed`` is the raw CORD-shape dict from ``token2json``.
    ``confidence`` is ``{"groups": {...}, "overall": float | None}``.

    On any decoding failure, returns ``({}, {})`` so the caller surfaces an
    empty receipt rather than a 500 — keeping the frontend recoverable.
    """
    if not bundle.is_loaded or bundle.model is None or bundle.processor is None:
        raise RuntimeError("Model bundle not loaded.")

    if image.mode != "RGB":
        image = image.convert("RGB")

    processor = bundle.processor
    model = bundle.model
    tokenizer = processor.tokenizer

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(bundle.device)

    with torch.inference_mode():
        out = model.generate(
            pixel_values,
            decoder_input_ids=bundle.decoder_start_ids,
            max_length=bundle.max_length,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Per-token probability = softmax over each step's logits at the chosen id.
    # (compute_transition_scores can't read vocab_size off VisionEncoderDecoderConfig.)
    gen_ids = out.sequences[0, bundle.decoder_start_ids.shape[1]:].tolist()
    probs = [
        float(torch.softmax(logits[0], dim=-1)[gen_ids[i]])
        for i, logits in enumerate(out.scores)
    ]
    # Defensive: align lengths (sequences may carry an extra leading prefix token).
