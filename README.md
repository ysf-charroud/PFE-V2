# Receipt Field Extraction — PFE V2

Structured field extraction (line items, totals, taxes, etc.) from receipt images using a fine-tuned **Donut** vision-language model trained on **CORD v2**.

## Branch status (`donut-approach`)

This branch replaces the previous LayoutLMv3 + PaddleOCR pipeline with a single end-to-end Donut model — no separate OCR engine, no token-classification post-processing.

| Component | State |
|-----------|-------|
| Training notebook (`donut_cord_finetune.ipynb`) | Ready — Lightning AI compatible, full evaluation suite |
| Frontend (`frontend/`) | Unchanged — Vite + React 19 + TanStack Router + shadcn/ui |
| Express API (`backend/src/`) | Unchanged — needs a response-shape tweak once the Donut sidecar lands |
| Inference sidecar (`backend/inference_sidecar/`) | **Empty** — old sidecar removed, Donut sidecar TBD |

## Overview

Where the old approach was OCR → LayoutLMv3 token classification → normalised JSON, Donut does it all in one model:

```
Browser / client
      │  POST /api/extract  (image upload)
      ▼
 Express :8000  (Node.js)
      │  POST /infer  (forwarded)
      ▼
 Sidecar :8001  (Python / FastAPI)  ← to be rewritten for Donut
      └─ Donut (Swin encoder + BART decoder)
              ↳ emits CORD-schema JSON directly
```

## Project structure

```
.
├── backend/
│   ├── src/
│   │   ├── index.js              CORS, health, static files
│   │   ├── config.js             Env-driven settings
│   │   ├── routes/extract.js     POST /api/extract — multer upload → sidecar
│   │   └── services/inference.js HTTP proxy to sidecar
│   ├── inference_sidecar/        empty — Donut sidecar pending
│   ├── storage/annotated/        legacy annotated-image directory
│   ├── package.json
│   └── .env.example
├── frontend/                     Vite + React + TanStack
├── test_images/                  sample receipts
└── donut_cord_finetune.ipynb     training notebook
```

## Training the Donut model

Open `donut_cord_finetune.ipynb` on **Lightning AI Studio** (single GPU, ≥24 GB VRAM). The notebook is self-contained: the first cell pins all Python deps.

Recommended GPUs (Lightning AI):
- **A100** — ~50 min for 6 epochs (best speed/cost)
- **L4** — ~2 h 30 m for 6 epochs (cheapest viable)
- L40S / H100 — faster, but diminishing returns at this dataset size

Default training config (`Cfg` class):

| Setting | Value |
|---------|-------|
| Base model | `naver-clova-ix/donut-base` |
| Dataset | `naver-clova-ix/cord-v2` (800 train / 100 val / 100 test) |
| Epochs | 6 |
| Image size | 1280 × 960 (downsized from Donut's default 2560 × 1920) |
| Effective batch | 4 (`PER_DEVICE_BATCH=1`, `GRAD_ACC=4`) |
| Learning rate | 3e-5, warmup 10 %, weight decay 0.01 |
| Precision | bf16 on Ampere+, fp16 fallback |
| Augmentation | ±15 % brightness/contrast, ±3° rotation |

Outputs after a run:
- `donut-cord-finetuned/final/` — fine-tuned weights + processor
- `loss_curve.png`, `f1_distribution.png` — report figures
- `report_data.json` — full config + Trainer log history + all val/test metrics + every (gt, prediction, latency) triple

Metrics computed (sections 9–16 of the notebook):
- Exact-match accuracy
- Field-level micro / macro F1 (multiset comparison — list order ignored, the way Donut's CORD evaluation does it)
- **Normalized Tree Edit Distance (nTED)** — Donut's canonical metric
- Latency p50 / p95 / mean
- Per-top-level-field breakdown (`menu`, `sub_total`, `total`, …)
- Test-split evaluation (separate from validation)
- 10 worst predictions for qualitative error analysis
- Final scorecard comparing validation vs test

## Local development

Until the Donut sidecar exists, only the frontend and Express run locally.

### Terminal 1 — Express API
```bash
cd backend
cp .env.example .env
npm install
npm run dev
```

### Terminal 2 — Frontend
```bash
cd frontend
npm install
npm run dev   # defaults to 8080; falls back to 8081 if taken
```

Express listens on `http://localhost:8000` and polls `http://localhost:8001/health` on startup — this currently times out (no sidecar yet). The frontend port (8080 or its fallback) must be in `CORS_ORIGINS` in `backend/.env`.

### Terminal 3 — Inference sidecar (TBD)
The sidecar will be rebuilt as a thin FastAPI app that loads the fine-tuned Donut model. Stub:

```python
# backend/inference_sidecar/main.py  (planned)
processor = DonutProcessor.from_pretrained(MODEL_DIR, use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR)

# POST /infer accepts an image, runs model.generate, returns processor.token2json(sequence)
```

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (Express; reports sidecar state) |
| `POST` | `/api/extract` | `multipart/form-data`, field `file` |

**Accepted types:** `image/jpeg`, `image/png`, `image/webp` — max 10 MB.

```bash
curl -X POST http://localhost:8000/api/extract -F "file=@receipt.jpg"
```

### Response shape (planned, CORD schema)

```jsonc
{
  "filename": "receipt.jpg",
  "receipt": {
    "menu": [
      { "nm": "Caffe Latte", "cnt": "2", "unitprice": "4,500", "price": "9,000" }
    ],
    "sub_total": { "subtotal_price": "9,000", "tax_price": "900" },
    "total":     { "total_price": "10,300", "cashprice": "20,000", "changeprice": "9,700" }
  },
  "processing_ms": 850
}
```

> Donut emits the raw CORD schema (string values, comma thousands separators). The old pipeline used `normalize.py` to flatten this into camelCase keys with floats — that translation will be re-added later, either on the Express side or in a small normaliser on the sidecar.

## Configuration

### Express (`backend/.env`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | `8000` | Express listen port |
| `CORS_ORIGINS` | `localhost:3000,5173` | Allowed origins — add 8080/8081 for Vite |
| `MAX_UPLOAD_MB` | `10` | Upload size limit |
| `INFERENCE_SIDECAR_URL` | `http://localhost:8001` | Sidecar base URL |
| `OUTPUT_DIR` | `backend/storage/annotated` | Legacy annotated-image directory |

## Requirements

- **Node.js** 18+
- **Python** 3.10–3.12 (only needed for training / the future sidecar)
- A CUDA-capable GPU for training (Lightning AI handles this); CPU is sufficient for inference at the cost of latency

## Tech stack

| Layer | Technology |
|-------|-----------|
| Frontend | Vite + React 19 + TanStack Router + shadcn/ui |
| API server | Node.js + Express + Multer |
| Inference server | Python + FastAPI + Uvicorn (to be rebuilt for Donut) |
| Model | Donut (VisionEncoderDecoderModel — Swin encoder + BART decoder) |
| Training dataset | CORD v2 (Naver-Clova) |
| Deep learning | PyTorch, HuggingFace Transformers |
