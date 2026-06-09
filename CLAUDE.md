# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Branch state

`donut-approach` — the project pivoted from a LayoutLMv3 + OCR pipeline to a single **Donut** encoder-decoder model trained on CORD v2.

| Piece | State |
|-------|-------|
| Training notebook `donut_cord_finetune.ipynb` | Ready — 16-section pipeline, Lightning AI compatible |
| Old LayoutLMv3 + PaddleOCR sidecar | **Deleted on this branch** |
| New Donut sidecar | **Not yet implemented** — `backend/inference_sidecar/` only has `.gitignore` + `.venv/` left |
| Frontend (`frontend/`) | Unchanged |
| Express API (`backend/src/`) | Unchanged — response shape will need a small tweak once Donut sidecar lands |

## What this is

A receipt field extraction project with these components:

1. **Training notebook** — `donut_cord_finetune.ipynb` fine-tunes `naver-clova-ix/donut-base` on CORD v2. Output: weights + processor in `./donut-cord-finetuned/final/`.
2. **Express API** (`backend/src/`) — handles uploads, CORS, validation, and proxies to the sidecar on port 8001.
3. **Inference sidecar** (`backend/inference_sidecar/`) — placeholder. Future: FastAPI app that loads fine-tuned Donut and exposes `POST /infer`.
4. **Frontend** (`frontend/`) — Vite + React + TanStack Router; uploads to `/api/extract`.

## Project structure

```
.
├── backend/
│   ├── src/
│   │   ├── index.js              CORS, health, static files, error handler
│   │   ├── config.js             Env-driven settings
│   │   ├── routes/extract.js     POST /api/extract — multer upload → sidecar
│   │   └── services/inference.js HTTP proxy to sidecar; polls /health on startup
│   ├── inference_sidecar/        empty — Donut sidecar TBD
│   ├── storage/annotated/        legacy annotated-image directory (unused under Donut)
│   ├── package.json
│   └── .env.example
├── frontend/                     Vite + React + TanStack
├── test_images/                  sample receipts
└── donut_cord_finetune.ipynb     training notebook
```

## Running

Until the Donut sidecar is rebuilt, only the frontend and Express run locally. Training happens on Lightning AI (the notebook installs its own deps).

**Terminal 1 — Express API:**
```bash
cd backend
cp .env.example .env
npm install
npm run dev   # nodemon, or: npm start
```

**Terminal 2 — frontend:**
```bash
cd frontend
npm install
npm run dev   # defaults to port 8080; falls back to 8081 if taken
```

Express listens on `http://localhost:8000`. It still polls `http://localhost:8001/health` on startup for up to 5 minutes — this currently times out until the new sidecar is built. Add the frontend port to `CORS_ORIGINS` in `backend/.env` — Vite uses 8080 by default but falls back to 8081 if taken.

## Training the Donut model

Open `donut_cord_finetune.ipynb` on **Lightning AI Studio** (single GPU, ≥24 GB VRAM — L4 / A10 / A100). The first cell pins all Python deps:

```
transformers==4.44.2  datasets==2.20.0  accelerate==0.34.2  protobuf>=3.20  sentencepiece  pillow
```

Default training config (`Cfg` class in section 1):
- `EPOCHS = 6`
- `IMAGE_SIZE = [1280, 960]` (downsized from Donut's default 2560×1920 to fit 24 GB)
- `MAX_LENGTH = 768`
- `PER_DEVICE_BATCH = 1`, `GRAD_ACC = 4` → effective batch 4
- `LR = 3e-5`, warmup 10 %, weight decay 0.01
- Light augmentation: brightness/contrast ±15 %, ±3° rotation
- bf16 on Ampere+, fp16 fallback
- Gradient checkpointing (non-reentrant) for memory

Outputs after a run:
- `donut-cord-finetuned/final/` — weights + processor
- `loss_curve.png`, `f1_distribution.png`
- `report_data.json` — full config + Trainer `log_history` + all metrics + every (gt, prediction, latency) triple

Notebook sections:

| § | Content |
|---|---------|
| 1 | Setup, pip install, imports, `Cfg`, `set_seed` |
| 2 | Load CORD v2 + inspect one sample |
| 3 | Load Donut processor + model + resize encoder image / decoder length |
| 4 | `json2token` + `collect_keys` + sanity check |
| 5 | `MultiDonutDataset` (multi-source-capable) + light augmentation |
| 6 | `Seq2SeqTrainingArguments` + `Trainer.train()` |
| 7 | Save fine-tuned model + processor |
| 8 | One-shot inference demo |
| 9 | Full validation evaluation: exact match, micro/macro F1, nTED, latency, per-field |
| 10 | Train + eval loss curve plot |
| 11 | 5 random qualitative examples |
| 12 | Persist everything to `report_data.json` |
| 13 | Test-split evaluation (the report's headline numbers) |
| 14 | Per-epoch table (loss, eval_loss, learning rate) |
| 15 | F1 distribution histogram + 10 worst predictions |
| 16 | Final scorecard comparing validation vs test |

## API (current, pre-rewrite)

The Express API contract is unchanged from the old LayoutLMv3 pipeline. It will change once the Donut sidecar is wired up; below is the *current* contract.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Returns `{ status, version, model_loaded }` |
| POST | `/api/extract` | `multipart/form-data`, field `file`. Query `?annotate=false` to skip annotation. |
| GET | `/files/annotated/:filename` | Serves a saved annotated image (legacy; no annotations under Donut). |

Accepted types: `image/jpeg`, `image/png`, `image/webp`. Max 10 MB.

## Config (`backend/src/config.js`)

| Env var | Default | Purpose |
|---------|---------|---------|
| `PORT` | `8000` | Express listen port |
| `CORS_ORIGINS` | `localhost:3000,5173` | Allowed origins (add 8080/8081 for the Vite frontend) |
| `MAX_UPLOAD_MB` | `10` | Upload size limit |
| `INFERENCE_SIDECAR_URL` | `http://localhost:8001` | Sidecar base URL |
| `OUTPUT_DIR` | `backend/storage/annotated` | Legacy annotated-image directory |

## Donut inference pipeline (what the future sidecar will do)

```
image (PIL)
  → DonutProcessor.image_processor (resize 1280×960, normalize → pixel_values)
  → VisionEncoderDecoderModel.generate(decoder_input_ids=[<s_cord-v2>])
  → emits sequence: <s_cord-v2><s_menu><s_nm>...</s_nm>...</s_menu>...</s>
  → processor.token2json(sequence) → nested dict matching CORD schema
```

No separate OCR step. No box clamping. No row clustering. No `normalize.py`. The model takes responsibility for the full image-to-JSON mapping.

## Non-obvious constraints

### Donut training notebook
- **`use_fast=False`** when loading `DonutProcessor` — XLMRoberta's slow→fast conversion needs `protobuf`, which Lightning AI's `cloudspace` env doesn't ship. Using the slow tokenizer avoids the conversion entirely. Slightly slower tokenization, but GPU is the bottleneck so wall-clock is unaffected.
- **Resize encoder image size** to 1280×960 by setting **both** `processor.image_processor.size` *and* `model.config.encoder.image_size` — default 2560×1920 OOMs on 24 GB.
- **Walk the entire training set once** to collect every `<s_KEY>` / `</s_KEY>` token, then `model.decoder.resize_token_embeddings(len(tokenizer))` *once* after all tokens are added. Don't resize per-token.
- Set **`model.config.pad_token_id`**, **`decoder_start_token_id`**, **`eos_token_id`** explicitly — `generate()` otherwise produces gibberish or never stops.
- **`predict_with_generate=False`** during training — generation in eval makes each epoch ~5× slower. Generation only at the end for the evaluation cells.
- **`gradient_checkpointing_kwargs={"use_reentrant": False}`** — newer PyTorch warns / breaks on the old reentrant default.
- **`report_to=[]`** to suppress W&B / MLflow autodetection on Lightning AI.
- **CORD-v2 ground truth** lives at `json.loads(ground_truth)["gt_parse"]` — the surrounding object also has `valid_line`, `meta`, etc. that we don't use.
- **Augmentation must be light**: receipts need to remain readable. ±15 % brightness/contrast and ±3° rotation are safe; heavier transforms degrade accuracy.
- **`MAX_LENGTH = 768`** is enough for every CORD parse but tight for invoices — bump it if mixing in a larger-schema dataset.

### Branch state
- `model/cord_layoutlmv3(v1)/` (legacy LayoutLMv3 weights) is gitignored and may still exist on disk. Safe to delete locally.
- A leftover `donut_cord_finetune(1).ipynb` (if present) is the Lightning AI download of an earlier 5-epoch run — *not* the current local 6-epoch version with sections 13–16. Treat the unsuffixed `donut_cord_finetune.ipynb` as authoritative.
- `backend/inference_sidecar/.venv/` still has PaddleOCR installed from the previous approach. Recreate it cleanly when building the new Donut sidecar.
- `backend/inference_sidecar/.gitignore` was kept on purpose (still useful for the new sidecar).
