# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Branch state

`donut-approach` — the project pivoted from a LayoutLMv3 + OCR pipeline to a single **Donut** encoder-decoder model trained on CORD v2.

| Piece                                         | State                                                                                                                                                                                |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Training notebook `donut_cord_finetune.ipynb` | Ready — 16-section pipeline, Lightning AI compatible                                                                                                                                 |
| Old LayoutLMv3 + PaddleOCR sidecar            | **Deleted on this branch**                                                                                                                                                           |
| New Donut sidecar                             | **Implemented** — `main.py`, `model_loader.py`, `inference.py`, `normalize.py` (+ `test_normalize.py`). Loads fine-tuned Donut, exposes `POST /infer`, `GET /health`, `GET /metrics` |
| Frontend (`frontend/`)                        | Upload → extract, editable fields, SQLite-backed Saved Documents, and a Donut evaluation dashboard on `/models`                                                                      |
| Express API (`backend/src/`)                  | Proxies to sidecar; persists extractions + corrections in SQLite (`db.js`, `routes/documents.js`); `/api/metrics` proxy                                                              |

## What this is

A receipt field extraction project with these components:

1. **Training notebook** — `donut_cord_finetune.ipynb` fine-tunes `naver-clova-ix/donut-base` on CORD v2. Output: weights + processor in `./donut-cord-finetuned/final/`.
   <<<<<<< HEAD
2. **Express API** (`backend/src/`) — handles uploads, CORS, validation, proxies to the sidecar on port 8001, and persists extractions + user corrections in SQLite (`backend/storage/app.db`).
3. **Inference sidecar** (`backend/inference_sidecar/`) — FastAPI app that loads the fine-tuned Donut model and exposes `POST /infer`, `GET /health`, `GET /metrics`.
4. # **Frontend** (`frontend/`) — Vite + React + TanStack Router; uploads to `/api/extract`, lets the user correct fields, and shows the evaluation dashboard on `/models`.
5. **Express API** (`backend/src/`) — handles uploads, CORS, validation, and proxies to the sidecar on port 8001.
6. **Inference sidecar** (`backend/inference_sidecar/`) — FastAPI app that loads fine-tuned Donut and exposes `GET /health`, `POST /infer`, and `GET /metrics`. No OCR step — pure image → `generate()` → `token2json` → CORD normaliser.
7. **Frontend** (`frontend/`) — Vite + React + TanStack Router; uploads to `/api/extract`.
   > > > > > > > c8dfa2b895ded5b7c19a80a7a900b797351580cf

## Project structure

```
.
├── backend/
│   ├── src/
│   │   ├── index.js              CORS, health, static files, error handler
│   │   ├── config.js             Env-driven settings
│   │   ├── routes/extract.js     POST /api/extract — multer upload → sidecar → persist
│   │   ├── routes/documents.js   GET/PATCH /api/documents — saved docs + corrections; /api/metrics
│   │   ├── db.js                 SQLite (better-sqlite3): documents + corrections tables
│   │   └── services/inference.js HTTP proxy to sidecar; polls /health on startup
│   ├── inference_sidecar/
│   │   ├── main.py               FastAPI app — /health, /infer, /metrics
│   │   ├── model_loader.py       ModelBundle singleton — loads DonutProcessor + VisionEncoderDecoderModel
│   │   ├── inference.py          predict() — generate() + token2json
│   │   ├── normalize.py          CORD-shape dict → camelCase receipt with float values
│   │   ├── test_normalize.py     56 pure-Python assertions for the normaliser
│   │   ├── requirements.txt      fastapi, uvicorn, torch, transformers>=4.44,<4.50, …
│   │   └── .env.example          MODEL_DIR, FORCE_CPU, TASK_TOKEN, MAX_LENGTH, USE_SLOW_TOKENIZER
│   ├── storage/annotated/        legacy annotated-image directory (unused under Donut)
│   ├── package.json
│   └── .env.example
├── frontend/                     Vite + React + TanStack
├── test_images/                  sample receipts
└── donut_cord_finetune.ipynb     training notebook
```

## Running

<<<<<<< HEAD
Three pieces run locally: the Donut sidecar (port 8001), the Express API (port 8000), and the frontend (port 8080). Training happens on Lightning AI (the notebook installs its own deps).

**Terminal 1 — Donut sidecar** (needs `transformers>=4.44,<4.50`, `sentencepiece`, `protobuf`):

```bash
cd backend/inference_sidecar
.venv/bin/pip install -r requirements.txt   # first time
cp .env.example .env                         # set MODEL_DIR to model/donut-cord-finetuned
.venv/bin/python -m uvicorn main:app --port 8001
```

# On macOS, prefix with `SSL_CERT_FILE=$(.venv/bin/python -c 'import certifi;print(certifi.where())')` if HF downloads hit cert errors, and set `FORCE_CPU=true`.

Three processes run locally. Training happens on Lightning AI (the notebook installs its own deps).

**Terminal 1 — Donut inference sidecar (Python):**

```bash
cd backend/inference_sidecar
cp .env.example .env          # set MODEL_DIR if weights aren't at default path
python -m venv .venv
.venv/Scripts/activate        # Windows; Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --port 8001
```

Requires fine-tuned weights at `model/donut-cord-finetuned/` (or `final/` subdir). Set `MODEL_DIR` in `.env` to override. Sidecar is ready when `/health` returns `{ "model_loaded": true }`.

> > > > > > > c8dfa2b895ded5b7c19a80a7a900b797351580cf

**Terminal 2 — Express API:**

```bash
cd backend
cp .env.example .env
npm install
npm run dev   # nodemon, or: npm start
```

**Terminal 3 — frontend:**

```bash
cd frontend
npm install
npm run dev   # defaults to port 8080; falls back to 8081 if taken
```

<<<<<<< HEAD
Express listens on `http://localhost:8000` and polls `http://localhost:8001/health` on startup for up to 5 minutes (covers the sidecar's model load). Add the frontend port to `CORS_ORIGINS` in `backend/.env` — Vite uses 8080 by default but falls back to 8081 if taken.
=======
Express listens on `http://localhost:8000` and polls `http://localhost:8001/health` every 2 s for up to 5 minutes on startup. Add the frontend port to `CORS_ORIGINS` in `backend/.env` — Vite uses 8080 by default but falls back to 8081 if taken.

> > > > > > > c8dfa2b895ded5b7c19a80a7a900b797351580cf

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

| Method | Path                         | Description                                                                                |
| ------ | ---------------------------- | ------------------------------------------------------------------------------------------ |
| GET    | `/health`                    | Returns `{ status, version, model_loaded }`                                                |
| POST   | `/api/extract`               | `multipart/form-data`, field `file`. Query `?annotate=false` accepted (no-op under Donut). |
| GET    | `/files/annotated/:filename` | Legacy annotated-image route (unused under Donut).                                         |

Accepted types: `image/jpeg`, `image/png`, `image/webp`. Max 10 MB.

### Sidecar API (`localhost:8001`)

| Method | Path       | Description                                                                                                      |
| ------ | ---------- | ---------------------------------------------------------------------------------------------------------------- |
| GET    | `/health`  | `{ status, model_loaded, device, model_dir }` — Express polls this on startup                                    |
| POST   | `/infer`   | `multipart/form-data`, field `file`. Returns `{ receipt, raw, num_words, annotated_filename, processing_ms }`    |
| GET    | `/metrics` | Reshapes `report_data.json` into the frontend `Metrics` type. Returns `{ available: false }` if no report found. |

## Config (`backend/src/config.js`)

| Env var                 | Default                     | Purpose                                               |
| ----------------------- | --------------------------- | ----------------------------------------------------- |
| `PORT`                  | `8000`                      | Express listen port                                   |
| `CORS_ORIGINS`          | `localhost:3000,5173`       | Allowed origins (add 8080/8081 for the Vite frontend) |
| `MAX_UPLOAD_MB`         | `10`                        | Upload size limit                                     |
| `INFERENCE_SIDECAR_URL` | `http://localhost:8001`     | Sidecar base URL                                      |
| `OUTPUT_DIR`            | `backend/storage/annotated` | Legacy annotated-image directory                      |

## Donut inference pipeline

```
image (PIL)
  → DonutProcessor.image_processor (resize 1280×960, normalize → pixel_values)
  → VisionEncoderDecoderModel.generate(decoder_input_ids=[<s_cord-v2>])
  → emits sequence: <s_cord-v2><s_menu><s_nm>...</s_nm>...</s_menu>...</s>
  → processor.token2json(sequence) → nested CORD-shape dict
  → normalize(cord_dict) → { line_items, subtotal, tax, total, … }
```

No separate OCR step. No box clamping. No row clustering. The model takes responsibility for the full image-to-JSON mapping. `normalize.py` handles CORD key → camelCase mapping, float parsing, and both list and single-dict `menu` shapes.

## Non-obvious constraints

### Donut training notebook

- **`use_fast=False`** when loading `DonutProcessor` — XLMRoberta's slow→fast conversion needs `protobuf`, which Lightning AI's `cloudspace` env doesn't ship. Using the slow tokenizer avoids the conversion entirely. Slightly slower tokenization, but GPU is the bottleneck so wall-clock is unaffected.
- **Resize encoder image size** to 1280×960 by setting **both** `processor.image_processor.size` _and_ `model.config.encoder.image_size` — default 2560×1920 OOMs on 24 GB.
- **Walk the entire training set once** to collect every `<s_KEY>` / `</s_KEY>` token, then `model.decoder.resize_token_embeddings(len(tokenizer))` _once_ after all tokens are added. Don't resize per-token.
- Set **`model.config.pad_token_id`**, **`decoder_start_token_id`**, **`eos_token_id`** explicitly — `generate()` otherwise produces gibberish or never stops.
- **`predict_with_generate=False`** during training — generation in eval makes each epoch ~5× slower. Generation only at the end for the evaluation cells.
- **`gradient_checkpointing_kwargs={"use_reentrant": False}`** — newer PyTorch warns / breaks on the old reentrant default.
- **`report_to=[]`** to suppress W&B / MLflow autodetection on Lightning AI.
- **CORD-v2 ground truth** lives at `json.loads(ground_truth)["gt_parse"]` — the surrounding object also has `valid_line`, `meta`, etc. that we don't use.
- **Augmentation must be light**: receipts need to remain readable. ±15 % brightness/contrast and ±3° rotation are safe; heavier transforms degrade accuracy.
- **`MAX_LENGTH = 768`** is enough for every CORD parse but tight for invoices — bump it if mixing in a larger-schema dataset.

### Branch state

- `model/cord_layoutlmv3(v1)/` (legacy LayoutLMv3 weights) is gitignored and may still exist on disk. Safe to delete locally.
- A leftover `donut_cord_finetune(1).ipynb` (if present) is the Lightning AI download of an earlier 5-epoch run — _not_ the current local 6-epoch version with sections 13–16. Treat the unsuffixed `donut_cord_finetune.ipynb` as authoritative.
- `backend/inference_sidecar/.venv/` may still have PaddleOCR installed from the previous approach. Recreate it cleanly if packages seem wrong.

### Donut sidecar

- **`model_loader.py` raises `FileNotFoundError`** at startup if no `config.json` is found — the sidecar will not start without fine-tuned weights. Set `MODEL_DIR` in `.env` to the correct path.
- **`menu` can be a single dict** (one-item receipt) or a list. `normalize()` handles both.
- **`token2json` is fallible** — on malformed sequences it raises; `inference.py` catches and returns `{}`. Frontend shows an empty receipt rather than a 500.
- **`bad_words_ids=[[unk_token_id]]`** in `generate()` prevents `<unk>` tokens that would crash `token2json`.
- **`decoder_start_ids` pre-encoded once** at load time in `model_loader.py` — re-encoding per request adds noticeable latency on CPU.
- **`FORCE_CPU=true`** in `.env` disables CUDA even when a GPU is available — useful for testing on a CPU-only machine.
