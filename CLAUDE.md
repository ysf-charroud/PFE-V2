# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An invoice/receipt field extraction project with three components:

1. **ML notebook** — `invoice_extraction_v3.ipynb` fine-tunes Microsoft's **LayoutLMv3** on the **CORD v2** dataset. The trained model + processor live in `model/cord_layoutlmv3(v1)/`.
2. **Express API** (`backend/src/`) — handles uploads, CORS, validation, and proxies inference requests to the sidecar.
3. **Inference sidecar** (`backend/inference_sidecar/`) — minimal FastAPI app that loads LayoutLMv3 + EasyOCR and exposes `POST /infer`.

## Project structure

```
.
├── backend/
│   ├── src/
│   │   ├── index.js              Entry point — CORS, health, static files, error handler
│   │   ├── config.js             Env-driven settings (port, CORS, upload limits, sidecar URL, output dir)
│   │   ├── routes/
│   │   │   └── extract.js        POST /api/extract — multer upload → sidecar → JSON
│   │   └── services/
│   │       └── inference.js      HTTP proxy to sidecar; polls /health on startup (5 min timeout)
│   ├── inference_sidecar/
│   │   ├── main.py               FastAPI app — GET /health, POST /infer
│   │   ├── model_loader.py       Loads LayoutLMv3 + EasyOCR; exposes bundle singleton
│   │   ├── inference.py          predict() — OCR → LayoutLMv3 → grouped JSON
│   │   ├── normalize.py          normalize() — maps CORD labels → clean DB-ready receipt dict
│   │   ├── visualize.py          annotate() — draws label boxes, saves PNG to disk
│   │   ├── ARCHITECTURE.md       Detailed per-file explanation with code snippets
│   │   ├── requirements.txt
│   │   └── .venv/                Python 3.10 virtualenv (git-ignored)
│   ├── storage/annotated/        Saved annotated images served at /files/annotated/*
│   ├── package.json
│   └── .env.example
├── model/
│   └── cord_layoutlmv3(v1)/      Fine-tuned weights + processor
└── invoice_extraction_v3.ipynb
```

## Running

Two processes must be running at the same time. **Use Python 3.10+** — tested on 3.10; EasyOCR supports 3.10–3.12.

**Terminal 1 — inference sidecar:**
```bash
cd backend/inference_sidecar

# First time only — create venv and install deps (~5 min, downloads PaddleOCR models on first start)
py -3.10 -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# Start
.venv\Scripts\uvicorn main:app --port 8001
```

**Terminal 2 — Express API:**
```bash
cd backend
cp .env.example .env   # adjust if needed
npm install
npm run dev            # nodemon, or: npm start
```

Express listens on `http://localhost:8000`. On startup it polls `http://localhost:8001/health` every 2 s for up to **5 minutes**, logging when the sidecar is ready. The 5-minute window covers the one-time EasyOCR model download on first run; subsequent starts are fast (~15 s).

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Returns `{ status, version, model_loaded }` |
| POST | `/api/extract` | `multipart/form-data`, field `file`. Query `?annotate=false` to skip annotation. |
| GET | `/files/annotated/:filename` | Serves a saved annotated image by filename. |

Accepted types: `image/jpeg`, `image/png`, `image/webp`. Max size: 10 MB (`MAX_UPLOAD_MB`).

**Response shape:**
```jsonc
{
  "filename": "receipt.jpg",
  "receipt": {
    "line_items": [
      {
        "name": "Caffe Latte",       // menu.nm
        "item_num": "1",             // menu.num  (optional)
        "quantity": 2.0,             // menu.cnt  (optional)
        "unit_price": 4500.0,        // menu.unitprice (optional)
        "price": 9000.0,             // menu.price
        "discount": 500.0            // menu.discountprice (optional)
      }
    ],
    "subtotal": 9000.0,              // sub_total.subtotal_price
    "discount": 500.0,               // sub_total.discount_price
    "service_charge": 900.0,         // sub_total.service_price
    "tax": 900.0,                    // sub_total.tax_price
    "total": 10300.0,                // total.total_price
    "cash_paid": 20000.0,            // total.cashprice
    "change": 9700.0,                // total.changeprice
    "credit_card": null,             // total.creditcardprice
    "e_money": null                  // total.emoneyprice
  },
  "num_words": 14,
  "processing_ms": 5910.2,
  "annotated_image_url": "/files/annotated/receipt_6fa3e7fa93.png"  // null if annotate=false
}
```

All monetary values are `float` (Indonesian `.` thousands separator resolved: `23.000` → `23000.0`). Fields absent from the receipt are omitted entirely (not null) — only fields the model detected are included.

The frontend displays the annotated image by fetching `http://localhost:8000/files/annotated/<filename>`.

## Config (`backend/src/config.js`)

| Env var | Default | Purpose |
|---------|---------|---------|
| `PORT` | `8000` | Express listen port |
| `CORS_ORIGINS` | `localhost:3000,5173` | Comma-separated allowed origins |
| `MAX_UPLOAD_MB` | `10` | Upload size limit |
| `INFERENCE_SIDECAR_URL` | `http://localhost:8001` | Sidecar base URL |
| `OUTPUT_DIR` | `backend/storage/annotated` | Where annotated images are saved and served from |

## Sidecar config (`backend/inference_sidecar/.env.example`)

| Env var | Default | Purpose |
|---------|---------|---------|
| `MODEL_DIR` | `../../model/cord_layoutlmv3(v1)` | Path to fine-tuned model |
| `OUTPUT_DIR` | `../storage/annotated` | Must match `OUTPUT_DIR` in Express config |
| `FORCE_CPU` | `false` | Disable CUDA even if available |
| `OCR_LANG` | `en` | Comma-separated EasyOCR language codes (e.g. `en,fr`) |
| `OCR_MIN_CONFIDENCE` | `0.5` | Drop OCR results below this score |
| `OCR_UPSCALE_TO` | `1600` | Upscale short images before OCR (px, 0 = off) |

## Inference pipeline (sidecar)

```
image → EasyOCR (phrase regions) → _split_phrase() (per-word boxes)
      → normalize boxes to [0,1000]
      → LayoutLMv3Processor (apply_ocr=False) → LayoutLMv3ForTokenClassification
      → majority vote subword → word mapping
      → spatial row clustering → { fields, line_items, summary, words }
      → normalize() — CORD labels → clean keys, number strings → float
      → (optional) annotate() → save PNG → return filename
```

**Label set:** 30 CORD classes (`menu.*`, `sub_total.*`, `total.*`, `void_menu.*`, `O`).

See `backend/inference_sidecar/ARCHITECTURE.md` for a full per-file walkthrough with code snippets.

## Non-obvious constraints

### Sidecar
- EasyOCR returns **phrase-level** boxes; `_split_phrase()` in `inference.py` splits them into per-word boxes proportionally by character count — required because the model was trained on CORD's per-word boxes.
- **Majority vote** is used for subword-to-word label aggregation (not first-subword-wins) — more robust for long words split into many BPE tokens.
- **Spatial row clustering** groups labeled words by y-centre proximity into rows; rows with any `menu.*` label become line items, everything else goes into summary.
- Boxes must be clamped to `[0,1000]` and monotonic (`x2>=x1`, `y2>=y1`) — the processor errors otherwise.
- `enc.word_ids(0)` must be captured **before** moving tensors to the device.
- When OCR upscaling is on, box coordinates are scaled back to original image dimensions before normalising.
- `OUTPUT_DIR` must resolve to the same filesystem path in both the sidecar and Express.

### Notebook
- **Do not call `processed.set_format("torch")`** — triggers a buggy `torchvision.io.VideoReader` import. Use the manual `TensorWrapper` instead.
- **`apply_ocr=False`** is required — OCR comes from CORD ground truth (training) or PaddleOCR (inference).
- Preprocessing must stay **batched via `.map()`** (cached to disk) — eager preprocessing crashed the kernel.
- Training: `per_device_train_batch_size=1` + `gradient_accumulation_steps=4` (effective batch 4).
- Empty parses get a throwaway `["empty"]/[[0,0,0,0]]/["O"]` token so batching doesn't break.
