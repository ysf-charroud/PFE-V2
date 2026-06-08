# Receipt Field Extraction — PFE V2

Automated extraction of structured fields (line items, totals, taxes, etc.) from receipt/invoice images using a fine-tuned LayoutLMv3 model.

## Overview

The system has three components:

1. **ML notebook** — `invoice_extraction_v3.ipynb` — fine-tunes Microsoft's [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base) on the [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) dataset. Trained weights live in `model/cord_layoutlmv3(v1)/`.
2. **Inference sidecar** — `backend/inference_sidecar/` — a FastAPI app that loads LayoutLMv3 + EasyOCR and exposes `POST /infer`.
3. **Express API** — `backend/src/` — handles uploads, CORS, validation, and proxies requests to the sidecar.

```
Browser / client
      │  POST /api/extract  (image upload)
      ▼
 Express :8000  (Node.js)
      │  POST /infer  (forwarded)
      ▼
 Sidecar :8001  (Python / FastAPI)
      ├─ EasyOCR
      ├─ LayoutLMv3Processor
      └─ LayoutLMv3ForTokenClassification
```

## Project Structure

```
.
├── backend/
│   ├── src/
│   │   ├── index.js              Entry point — CORS, health, static files, error handler
│   │   ├── config.js             Env-driven settings
│   │   ├── routes/extract.js     POST /api/extract — multer upload → sidecar → JSON
│   │   └── services/inference.js HTTP proxy to sidecar; polls /health on startup
│   ├── inference_sidecar/
│   │   ├── main.py               FastAPI app — GET /health, POST /infer
│   │   ├── model_loader.py       Loads LayoutLMv3 + EasyOCR; singleton bundle
│   │   ├── inference.py          predict() — OCR → LayoutLMv3 → grouped JSON
│   │   ├── normalize.py          Maps CORD labels → clean receipt dict
│   │   ├── visualize.py          Draws label boxes, saves annotated PNG
│   │   ├── ARCHITECTURE.md       Detailed per-file walkthrough
│   │   └── requirements.txt
│   ├── storage/annotated/        Annotated images served at /files/annotated/*
│   ├── package.json
│   └── .env.example
├── model/
│   └── cord_layoutlmv3(v1)/      Fine-tuned weights + processor config
├── test_images/                  Sample receipt images for testing
└── invoice_extraction_v3.ipynb   Training notebook
```

## Requirements

- **Node.js** 18+
- **Python** 3.10–3.12 (tested on 3.10)
- A CUDA-capable GPU is optional but speeds up inference significantly

## Setup & Running

Two processes must run simultaneously.

### Terminal 1 — Inference sidecar

```bash
cd backend/inference_sidecar

# First time only — create venv and install deps (~5 min)
py -3.10 -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# Start
.venv\Scripts\uvicorn main:app --port 8001
```

The sidecar loads the model and EasyOCR on startup (~15–30 s after a cold start). On first run, EasyOCR downloads its language models — this is covered by Express's 5-minute startup poll window.

### Terminal 2 — Express API

```bash
cd backend
cp .env.example .env   # adjust if needed
npm install
npm run dev            # or: npm start
```

Express listens on `http://localhost:8000` and polls `http://localhost:8001/health` every 2 s until the sidecar is ready (up to 5 minutes).

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Returns `{ status, version, model_loaded }` |
| `POST` | `/api/extract` | `multipart/form-data`, field `file`. Add `?annotate=false` to skip annotation. |
| `GET` | `/files/annotated/:filename` | Serves a saved annotated image. |

**Accepted types:** `image/jpeg`, `image/png`, `image/webp` — max 10 MB.

### Example request

```bash
curl -X POST http://localhost:8000/api/extract \
  -F "file=@receipt.jpg"
```

### Response shape

```jsonc
{
  "filename": "receipt.jpg",
  "receipt": {
    "line_items": [
      {
        "name": "Caffe Latte",
        "item_num": "1",       // optional
        "quantity": 2.0,       // optional
        "unit_price": 4500.0,  // optional
        "price": 9000.0,
        "discount": 500.0      // optional
      }
    ],
    "subtotal": 9000.0,
    "discount": 500.0,
    "service_charge": 900.0,
    "tax": 900.0,
    "total": 10300.0,
    "cash_paid": 20000.0,
    "change": 9700.0,
    "credit_card": null,
    "e_money": null
  },
  "num_words": 14,
  "processing_ms": 5910.2,
  "annotated_image_url": "/files/annotated/receipt_6fa3e7fa93.png"
}
```

All monetary values are `float`. Fields absent from the receipt are omitted (not `null`). The annotated image URL is `null` when `?annotate=false`.

## Configuration

### Express (`backend/.env`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | `8000` | Express listen port |
| `CORS_ORIGINS` | `localhost:3000,5173` | Comma-separated allowed origins |
| `MAX_UPLOAD_MB` | `10` | Upload size limit |
| `INFERENCE_SIDECAR_URL` | `http://localhost:8001` | Sidecar base URL |
| `OUTPUT_DIR` | `backend/storage/annotated` | Directory for annotated images |

### Sidecar (`backend/inference_sidecar/.env`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_DIR` | `../../model/cord_layoutlmv3(v1)` | Path to fine-tuned model |
| `OUTPUT_DIR` | `../storage/annotated` | Must match Express `OUTPUT_DIR` |
| `FORCE_CPU` | `false` | Disable CUDA even if available |
| `OCR_LANG` | `en` | Comma-separated EasyOCR language codes |
| `OCR_MIN_CONFIDENCE` | `0.5` | Drop OCR words below this score |
| `OCR_UPSCALE_TO` | `1600` | Upscale short images before OCR (px, 0 = off) |

## Inference Pipeline

```
image
  → EasyOCR (phrase-level regions)
  → _split_phrase() (per-word boxes, proportional split)
  → normalize boxes to [0, 1000]
  → LayoutLMv3Processor (apply_ocr=False)
  → LayoutLMv3ForTokenClassification
  → majority-vote subword → word label aggregation
  → spatial row clustering
  → normalize() — CORD labels → clean keys, number strings → float
  → (optional) annotate() → save PNG
```

**Label set:** 30 CORD classes (`menu.*`, `sub_total.*`, `total.*`, `void_menu.*`, `O`).

See [`backend/inference_sidecar/ARCHITECTURE.md`](backend/inference_sidecar/ARCHITECTURE.md) for a detailed per-file walkthrough with code snippets.

## Training

Open `invoice_extraction_v3.ipynb` in Jupyter and run all cells. The notebook:

- Loads the CORD v2 dataset via HuggingFace Datasets
- Fine-tunes `microsoft/layoutlmv3-base` for token classification
- Saves the trained model + processor to `model/cord_layoutlmv3(v1)/`

Key training settings: `per_device_train_batch_size=1`, `gradient_accumulation_steps=4` (effective batch 4).

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API server | Node.js, Express, Multer |
| Inference server | Python, FastAPI, Uvicorn |
| OCR | EasyOCR |
| Document understanding | Microsoft LayoutLMv3 (HuggingFace Transformers) |
| Training dataset | CORD v2 (Clova AI) |
| Deep learning | PyTorch |
