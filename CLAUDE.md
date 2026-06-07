# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-notebook ML project that fine-tunes Microsoft's **LayoutLMv3** on the **CORD v2** receipt dataset for token-classification-based invoice/receipt field extraction. Everything lives in `invoice_extraction_v3.ipynb`; `invoice_extraction_v3.html` is a static export of an executed run. The trained model + processor are committed under `model/cord_layoutlmv3/`.

There is no build system, package manifest, test suite, or linter — the "interface" is running the notebook top to bottom.

## Running

Designed for **Lightning.ai Studio** (or any CUDA GPU notebook host): upload the notebook, pick a GPU machine, run all cells. The first cell installs dependencies:

```
!pip install -q "transformers>=4.30" datasets accelerate easyocr gradio Pillow seqeval scikit-learn
```

- Training cell (`cell id="train"`) runs 5 epochs; ~9 min on a T4 in the recorded run.
- CORD v2 downloads from the HuggingFace hub at runtime (`naver-clova-ix/cord-v2`, 800/100/100 split). Set `HF_TOKEN` to avoid rate-limit warnings.
- To use the committed model instead of retraining, load from `./model/cord_layoutlmv3` with `LayoutLMv3ForTokenClassification.from_pretrained` / `LayoutLMv3Processor.from_pretrained` and skip section 4.

## Pipeline architecture

```
image → EasyOCR (text + boxes) → LayoutLMv3Processor → LayoutLMv3 token classification → grouped JSON + annotated image
```

The notebook is organized in numbered sections:
1. Setup / imports — `DEVICE` picks CUDA if available.
2. Load + parse CORD. `parse_cord()` flattens CORD's `valid_line[].words[]` into `(words, normalized_boxes, label_strs)`; the per-line `category` becomes each word's label. Boxes are normalized to LayoutLMv3's 0–1000 coordinate space.
3. Preprocessing. `preprocess_batch()` runs through `dataset.map(batched=True)` and feeds the processor with `apply_ocr=False` (OCR is supplied, not re-derived).
4. Fine-tune with HF `Trainer`; `compute_metrics` reports accuracy + macro/weighted F1 (sklearn, masking `-100`). Best model selected on `macro_f1`.
5. Inference. `predict(image)` does EasyOCR → processor → argmax, aggregates subword tokens to words (**first subword wins**), and groups words by predicted label into a flat dict.
6. Demo / visualization. `visualize()` draws per-class colored boxes with a legend.

The label set is **30 CORD classes** (`menu.*`, `sub_total.*`, `total.*`, `void_menu.*`, plus `O`) — built dynamically from the training split and persisted in `model/cord_layoutlmv3/config.json` as `id2label`/`label2id`. Recorded test metrics: accuracy 0.965, macro F1 0.81, weighted F1 0.962.

> Note: the title markdown mentions "CORD v2 + SROIE" and "~33 classes," but the code only loads CORD v2 and trains 30 classes. Treat the code as ground truth.

## Non-obvious constraints (preserve these when editing)

- **Do not call `processed.set_format("torch")`.** It triggers a buggy `from torchvision.io import VideoReader` import in `datasets`' torch formatter. The notebook deliberately wraps the HF dataset in a manual `TensorWrapper` (`__getitem__` → dict of `torch.as_tensor`) instead.
- **`apply_ocr=False`** on the processor is required — OCR comes from CORD ground truth (training) or EasyOCR (inference); letting the processor re-OCR would discard the real boxes.
- Preprocessing must stay **batched via `.map()`** (cached to disk). Eager full-dataset preprocessing crashed the kernel.
- Training uses `per_device_train_batch_size=1` + `gradient_accumulation_steps=4` (effective batch 4) because the v3 forward pass is heavy on a T4.
- Empty parses get a single throwaway `["empty"]/[[0,0,0,0]]/["O"]` token so batching doesn't break.
- Boxes are clamped to `[0,1000]` and forced monotonic (`x2>=x1`, `y2>=y1`); keep this in any new box-handling code or the processor will error.
- Inference captures `enc.word_ids(0)` **before** moving tensors to the device.
