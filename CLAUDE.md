# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Branch state

`donut-approach` — the project pivoted from a LayoutLMv3 + OCR pipeline to a single **Donut** encoder-decoder model trained on CORD v2.

| Piece | State |
| ----- | ----- |
| Training notebook `donut_cord_finetune.ipynb` | Ready — 16-section pipeline, Lightning AI compatible |
| Old LayoutLMv3 + PaddleOCR sidecar | **Deleted on this branch** |
| Donut sidecar | **Implemented** — `main.py`, `model_loader.py`, `inference.py`, `normalize.py` (+ `test_normalize.py`). Loads fine-tuned Donut, exposes `POST /infer`, `GET /health`, `GET /metrics`. Returns per-field + overall confidence scores. |
| Frontend (`frontend/`) | Upload → extract, editable fields, SQLite-backed Saved Documents, Donut evaluation dashboard on `/models`, active-learning stats panel |
| Express API (`backend/src/`) | Proxies to sidecar; persists extractions + corrections + uploaded images in SQLite (`db.js`); `/api/metrics`, `/api/stats`, `/api/corrections/export` |
| Docker | `docker-compose.yml` + per-service `Dockerfile`s — one-command stack |

## What this is

A receipt field extraction project with these components:

1. **Training notebook** — `donut_cord_finetune.ipynb` fine-tunes `naver-clova-ix/donut-base` on CORD v2. Output: weights + processor in `./donut-cord-finetuned/final/`.
2. **Express API** (`backend/src/`) — handles uploads, CORS, validation, proxies to the sidecar on port 8001, persists extractions + user corrections + uploaded images in SQLite (`backend/storage/app.db`).
3. **Inference sidecar** (`backend/inference_sidecar/`) — FastAPI app that loads fine-tuned Donut and exposes `GET /health`, `POST /infer`, `GET /metrics`. No OCR — pure image → `generate()` → `token2json` → CORD normaliser → confidence scoring.
4. **Frontend** (`frontend/`) — Vite + React + TanStack Router; uploads to `/api/extract`, lets the user correct fields, shows evaluation dashboard on `/models` and active-learning stats.

## Project structure

```
.
├── docker-compose.yml            One-command stack: sidecar + express + frontend
├── backend/
│   ├── Dockerfile
│   ├── src/
│   │   ├── index.js              CORS, health, static files, error handler
│   │   ├── config.js             Env-driven settings (PORT, CORS_ORIGINS, UPLOAD_DIR, …)
│   │   ├── routes/extract.js     POST /api/extract — multer upload → sidecar → persist image + extraction
│   │   ├── routes/documents.js   GET/PATCH /api/documents; /api/stats; /api/corrections/export; /api/metrics proxy
│   │   ├── db.js                 SQLite (better-sqlite3): documents + corrections tables; receiptToCordGtParse export helper
│   │   └── services/inference.js HTTP proxy to sidecar; polls /health on startup
│   ├── inference_sidecar/
│   │   ├── Dockerfile
│   │   ├── ARCHITECTURE.md       Data-flow + key-mapping reference
│   │   ├── main.py               FastAPI app — /health, /infer, /metrics
│   │   ├── model_loader.py       ModelBundle singleton — loads DonutProcessor + VisionEncoderDecoderModel
│   │   ├── inference.py          predict() — generate() + token2json + per-field confidence from softmax scores
│   │   ├── normalize.py          CORD-shape dict → camelCase receipt with float values + confidence fields
│   │   ├── test_normalize.py     56 pure-Python assertions for the normaliser
│   │   ├── requirements.txt      fastapi, uvicorn, torch, transformers>=4.44,<4.50, …
│   │   └── .env.example          MODEL_DIR, FORCE_CPU, TASK_TOKEN, MAX_LENGTH, USE_SLOW_TOKENIZER
│   ├── storage/
│   │   ├── annotated/            legacy annotated-image directory (unused under Donut)
│   │   └── uploads/              uploaded receipt images (saved per-extraction for active learning)
│   ├── package.json
│   └── .env.example
├── frontend/
│   ├── Dockerfile
│   └── src/
├── test_images/                  sample receipts
├── model/
│   └── donut-cord-finetuned/
│       ├── final/                fine-tuned weights + processor (from training notebook)
│       └── report_data.json      evaluation report — consumed by /metrics
└── donut_cord_finetune.ipynb     training notebook
```

## Running

### Option A — Docker (recommended for clean environments)

```bash
# Requires fine-tuned weights at ./model/donut-cord-finetuned/
docker compose up --build
# Open http://localhost:8080
```

The compose file mounts `./model` into the sidecar (not baked into the image). Sidecar healthcheck retries 30× (2-minute model load). Express waits for `service_healthy` before starting.

### Option B — Local (three terminals)

Three pieces run locally: the Donut sidecar (port 8001), the Express API (port 8000), and the frontend (port 8080).

**Terminal 1 — Donut sidecar:**

```bash
cd backend/inference_sidecar
cp .env.example .env          # set MODEL_DIR if weights aren't at default path
python -m venv .venv
.venv/Scripts/activate        # Windows; Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --port 8001
```

Requires fine-tuned weights at `model/donut-cord-finetuned/` (or `final/` subdir). Set `MODEL_DIR` in `.env` to override. Sidecar is ready when `/health` returns `{ "model_loaded": true }`.

**Terminal 2 — Express API:**

```bash
cd backend
cp .env.example .env
# Add http://localhost:8080 (and 8081) to CORS_ORIGINS
npm install
npm run dev   # nodemon, or: npm start
```

**Terminal 3 — frontend:**

```bash
cd frontend
npm install
npm run dev   # defaults to port 8080; falls back to 8081 if taken
```

Express listens on `http://localhost:8000` and polls `http://localhost:8001/health` every 2 s for up to 5 minutes on startup. Add the frontend port to `CORS_ORIGINS` in `backend/.env`.

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

| §   | Content                                                                           |
| --- | --------------------------------------------------------------------------------- |
| 1   | Setup, pip install, imports, `Cfg`, `set_seed`                                    |
| 2   | Load CORD v2 + inspect one sample                                                 |
| 3   | Load Donut processor + model + resize encoder image / decoder length              |
| 4   | `json2token` + `collect_keys` + sanity check                                      |
| 5   | `MultiDonutDataset` (multi-source-capable) + light augmentation                   |
| 6   | `Seq2SeqTrainingArguments` + `Trainer.train()`                                    |
| 7   | Save fine-tuned model + processor                                                 |
| 8   | One-shot inference demo                                                           |
| 9   | Full validation evaluation: exact match, micro/macro F1, nTED, latency, per-field |
| 10  | Train + eval loss curve plot                                                      |
| 11  | 5 random qualitative examples                                                     |
| 12  | Persist everything to `report_data.json`                                          |
| 13  | Test-split evaluation (the report's headline numbers)                             |
| 14  | Per-epoch table (loss, eval_loss, learning rate)                                  |
| 15  | F1 distribution histogram + 10 worst predictions                                  |
| 16  | Final scorecard comparing validation vs test                                      |

## API

### Express API (`localhost:8000`)

| Method | Path                          | Description |
| ------ | ----------------------------- | ----------- |
| GET    | `/health`                     | Returns `{ status, version, model_loaded }` |
| POST   | `/api/extract`                | `multipart/form-data`, field `file`. Saves image to `storage/uploads/`, persists extraction. Returns `{ id, receipt, processing_ms, … }` |
| GET    | `/api/documents`              | List saved documents (most recent first) |
| GET    | `/api/documents/:id`          | Fetch one document + its field-level corrections |
| PATCH  | `/api/documents/:id`          | Save a user-corrected receipt (body: `{ receipt: {...} }`). Marks as reviewed, records diffs in `corrections`. |
| GET    | `/api/stats`                  | Aggregate counts for active-learning panel: `{ documents, reviewed, corrections, trainable }` |
| GET    | `/api/corrections/export`     | Download a `.zip` of reviewed images + `metadata.jsonl` in CORD format, ready to feed back into `donut_cord_finetune.ipynb` |
| GET    | `/api/metrics`                | Proxy to sidecar `/metrics` |
| GET    | `/files/annotated/:filename`  | Legacy annotated-image route (unused under Donut) |

Accepted types: `image/jpeg`, `image/png`, `image/webp`. Max 10 MB.

### Sidecar API (`localhost:8001`)

| Method | Path       | Description |
| ------ | ---------- | ----------- |
| GET    | `/health`  | `{ status, model_loaded, device, model_dir }` — Express polls this on startup |
| POST   | `/infer`   | `multipart/form-data`, field `file`. Returns `{ receipt, raw, num_words, annotated_filename, processing_ms }` |
| GET    | `/metrics` | Reshapes `report_data.json` into the frontend `Metrics` type. Returns `{ available: false }` if no report found. |

`/infer` receipt shape includes confidence fields when the model produces them:
- `receipt.overall_confidence` — mean softmax probability across all decoded fields (0–1)
- `receipt.field_confidence` — `{ subtotal, tax, total, … }` per summary field
- `receipt.line_items[i].confidence` — `{ name, price, … }` per line-item field

## Config

### `backend/src/config.js`

| Env var                 | Default                       | Purpose |
| ----------------------- | ----------------------------- | ------- |
| `PORT`                  | `8000`                        | Express listen port |
| `CORS_ORIGINS`          | `localhost:3000,5173`         | Allowed origins (add 8080/8081 for the Vite frontend) |
| `MAX_UPLOAD_MB`         | `10`                          | Upload size limit |
| `INFERENCE_SIDECAR_URL` | `http://localhost:8001`       | Sidecar base URL |
| `OUTPUT_DIR`            | `backend/storage/annotated`   | Legacy annotated-image directory |
| `UPLOAD_DIR`            | `backend/storage/uploads`     | Where uploaded images are saved (for active-learning export) |

### `backend/inference_sidecar/.env`

| Env var              | Default                            | Purpose |
| -------------------- | ---------------------------------- | ------- |
| `MODEL_DIR`          | `../../model/donut-cord-finetuned` | Path to fine-tuned model (auto-detects `final/` subdir) |
| `FORCE_CPU`          | `false`                            | Disable CUDA even when available |
| `TASK_TOKEN`         | `<s_cord-v2>`                      | Must match the training-time task token |
| `MAX_LENGTH`         | `768`                              | Decoder cap — match `Cfg.MAX_LENGTH` in notebook |
| `USE_SLOW_TOKENIZER` | `true`                             | Avoids slow→fast XLMRoberta conversion (needs protobuf) |
| `METRICS_PATH`       | auto-detected from `MODEL_DIR`     | Override path to `report_data.json` (used in Docker) |

## Donut inference pipeline

```
image (PIL)
  → DonutProcessor.image_processor (resize 1280×960, normalize → pixel_values)
  → VisionEncoderDecoderModel.generate(decoder_input_ids=[<s_cord-v2>], output_scores=True)
  → emits sequence + per-step logits
  → processor.token2json(sequence) → nested CORD-shape dict
  → _field_confidences(gen_ids, softmax_probs, tokenizer) → { groups, overall }
  → normalize(cord_dict, conf) → { line_items, subtotal, tax, total, …, overall_confidence, field_confidence }
```

No separate OCR step. `normalize.py` handles CORD key → camelCase mapping, float parsing, both list and single-dict `menu` shapes, and attaches confidence dicts to the output.

## Active learning / corrections export

Every uploaded image is stored in `backend/storage/uploads/` under a UUID filename. When the user corrects a field on the frontend and saves, the corrected receipt is stored in `documents.corrected_json` and each changed field is recorded in the `corrections` table.

`GET /api/corrections/export` produces a `donut_corrections_dataset.zip` containing:
- `images/` — the original uploaded images
- `metadata.jsonl` — one row per reviewed document: `{ file_name, ground_truth }` in CORD `gt_parse` schema

This is a drop-in extra training source for `donut_cord_finetune.ipynb`. Load it via:
```python
from datasets import load_dataset
ds = load_dataset("imagefolder", data_dir="donut_corrections_dataset")
```

## Non-obvious constraints

### Donut training notebook

- **`use_fast=False`** when loading `DonutProcessor` — XLMRoberta's slow→fast conversion needs `protobuf`, which Lightning AI's `cloudspace` env doesn't ship. Using the slow tokenizer avoids the conversion entirely.
- **Resize encoder image size** to 1280×960 by setting **both** `processor.image_processor.size` _and_ `model.config.encoder.image_size` — default 2560×1920 OOMs on 24 GB.
- **Walk the entire training set once** to collect every `<s_KEY>` / `</s_KEY>` token, then `model.decoder.resize_token_embeddings(len(tokenizer))` _once_ after all tokens are added. Don't resize per-token.
- Set **`model.config.pad_token_id`**, **`decoder_start_token_id`**, **`eos_token_id`** explicitly — `generate()` otherwise produces gibberish or never stops.
- **`predict_with_generate=False`** during training — generation in eval makes each epoch ~5× slower.
- **`gradient_checkpointing_kwargs={"use_reentrant": False}`** — newer PyTorch warns / breaks on the old reentrant default.
- **`report_to=[]`** to suppress W&B / MLflow autodetection on Lightning AI.
- **CORD-v2 ground truth** lives at `json.loads(ground_truth)["gt_parse"]`.
- **Augmentation must be light**: ±15 % brightness/contrast and ±3° rotation are safe.
- **`MAX_LENGTH = 768`** is enough for every CORD parse but tight for invoices — bump it if mixing in a larger-schema dataset.

### Branch state

- `model/cord_layoutlmv3(v1)/` (legacy LayoutLMv3 weights) is gitignored and may still exist on disk. Safe to delete.
- `backend/inference_sidecar/.venv/` may still have PaddleOCR installed from the previous approach. Recreate it cleanly if packages seem wrong.

### Donut sidecar

- **`model_loader.py` raises `FileNotFoundError`** at startup if no `config.json` is found — sidecar will not start without fine-tuned weights.
- **`menu` can be a single dict** (one-item receipt) or a list. `normalize()` handles both.
- **`token2json` is fallible** — on malformed sequences it raises; `inference.py` catches and returns `({}, {})`. Frontend shows an empty receipt rather than a 500.
- **`bad_words_ids=[[unk_token_id]]`** in `generate()` prevents `<unk>` tokens that would crash `token2json`.
- **`decoder_start_ids` pre-encoded once** at load time — re-encoding per request adds noticeable latency on CPU.
- **`FORCE_CPU=true`** in `.env` disables CUDA even when a GPU is available.
- **`output_scores=True`** in `generate()` is required for confidence scoring — per-step logits are collected and softmax'd to get per-token probabilities. Removing it disables all confidence output.
- **`predict()` returns a tuple `(parsed, conf)`** — `conf` is `{"groups": {...}, "overall": float}`. Both `normalize()` and the `/infer` handler expect this shape. The `overall` field can be `None` if no fields were decoded.
- **`METRICS_PATH`** env var lets Docker override where `report_data.json` is found (the compose file sets it to `/models/donut-cord-finetuned/report_data.json`).

### SQLite schema

`documents` table carries an `overall_confidence REAL` column and an `image_filename TEXT` column (added via `ALTER TABLE` guard for pre-existing databases). `corrections` records per-field diffs as `(document_id, field_path, original_value, corrected_value)`. A document is "trainable" when `reviewed=1 AND corrected_json IS NOT NULL AND image_filename IS NOT NULL`.
