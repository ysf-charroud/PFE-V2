# Donut inference sidecar — architecture

A thin FastAPI app that wraps a fine-tuned **Donut** model
(`naver-clova-ix/donut-base`, fine-tuned on CORD v2 via
`donut_cord_finetune.ipynb`) and exposes it on `:8001` for the Express API
at `:8000` to consume.

Where the previous sidecar was OCR → LayoutLMv3 → label aggregation →
spatial reasoning, this one is just **image → `model.generate` →
`token2json` → CORD normaliser**. No OCR, no boxes, no clustering.

## Files

```
backend/inference_sidecar/
├── main.py             FastAPI app — /health + /infer
├── model_loader.py     Loads DonutProcessor + VisionEncoderDecoderModel
├── inference.py        predict() — runs generate() and parses the output
├── normalize.py        CORD-schema → camelCase/float receipt
├── test_normalize.py   pure-Python tests for the normaliser
├── requirements.txt    pinned Python deps
└── .env.example        env vars (MODEL_DIR, FORCE_CPU, TASK_TOKEN, …)
```

## Data flow

```
image bytes (multipart upload via Express)
  │
  ▼
main.py: parse image → PIL.Image
  │
  ▼
inference.predict(image)
  ├─ processor(image).pixel_values
  ├─ model.generate(pixel_values, decoder_input_ids=<s_cord-v2>, …)
  ├─ batch_decode → strip eos/pad/leading-task-tag
  └─ processor.token2json(seq) → CORD-shape dict
  │
  ▼
normalize(cord_dict)
  ├─ menu → line_items[]   (CORD keys → name/sub_name/item_num/quantity/…)
  ├─ sub_total → flat fields (subtotal, discount, service_charge, tax)
  └─ total → flat fields (total, cash_paid, change, credit_card, e_money)
  │
  ▼
main.py: JSON response →
  { receipt, raw, num_words=0, annotated_filename=null, processing_ms }
```

## CORD → clean key mapping

Mirrors what the frontend's `ExtractResponse` / `Receipt` types in
`frontend/src/lib/api.ts` expect.

### Per line-item

| CORD key        | Clean key      | Type   |
|-----------------|----------------|--------|
| `nm`            | `name`         | string |
| `sub_nm`        | `sub_name`     | string |
| `num`           | `item_num`     | string |
| `cnt`           | `quantity`     | float  |
| `unitprice`     | `unit_price`   | float  |
| `price`         | `price`        | float  |
| `discountprice` | `discount`     | float  |

### Sub-total block

| CORD key          | Clean key        | Type  |
|-------------------|------------------|-------|
| `subtotal_price`  | `subtotal`       | float |
| `discount_price`  | `discount`       | float |
| `service_price`   | `service_charge` | float |
| `tax_price`       | `tax`            | float |

### Total block

| CORD key           | Clean key      | Type  |
|--------------------|----------------|-------|
| `total_price`      | `total`        | float |
| `cashprice`        | `cash_paid`    | float |
| `changeprice`      | `change`       | float |
| `creditcardprice`  | `credit_card`  | float |
| `emoneyprice`      | `e_money`      | float |

Any CORD key not in these tables is silently dropped. Any value that parses
as empty (`""`, `None`, unparseable string) is omitted entirely from the
output — the frontend treats missing as "not on receipt".

## Number parsing

`_parse_number()` in `normalize.py` handles every separator convention seen
in CORD-v2 by inspecting the rightmost separator:

| Input         | Output    | Reasoning                       |
|---------------|-----------|---------------------------------|
| `"23,000"`    | `23000.0` | comma + 3-digit tail = thousand |
| `"23.000"`    | `23000.0` | dot + 3-digit tail = thousand   |
| `"1,250.50"`  | `1250.5`  | dot rightmost = decimal         |
| `"1.250,50"`  | `1250.5`  | comma rightmost = decimal       |
| `"23.5"`      | `23.5`    | non-3-digit tail = decimal      |
| `"Rp 23.000"` | `23000.0` | strips currency symbols         |

Validation: `python test_normalize.py` runs 56 assertions covering each
case (no model required — pure Python).

## /health response

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "model_dir": "/workspace/PFE-V2/model/donut-cord-finetuned/final"
}
```

The Express layer polls `/health` every 2 s for up to 5 minutes on startup.

## /infer response

```json
{
  "receipt": { "line_items": [...], "subtotal": ..., "total": ..., ... },
  "raw":     { "menu": [...], "sub_total": {...}, "total": {...} },
  "num_words": 0,
  "annotated_filename": null,
  "processing_ms": 850.0
}
```

- `receipt` — what the frontend renders (camelCase + floats).
- `raw` — the unparsed Donut output (string values, CORD schema). Useful
  for debugging when something looks wrong in the rendered output.
- `num_words` and `annotated_filename` — legacy fields, kept so the Express
  layer doesn't need branching logic per backend.

## Model directory resolution

`MODEL_DIR` defaults to `../../model/donut-cord-finetuned` (relative to this
file). The loader accepts either layout:

```
model/donut-cord-finetuned/         model/donut-cord-finetuned/
├── config.json                     └── final/
├── pytorch_model.bin                   ├── config.json
└── ...                                 ├── pytorch_model.bin
                                        └── ...
```

The notebook saves to the `final/` form, so dropping the entire
`donut-cord-finetuned/` directory from Lightning AI into `model/` works
without renaming.

## Configuration

| Env var              | Default                            | Purpose                                       |
|----------------------|------------------------------------|-----------------------------------------------|
| `MODEL_DIR`          | `../../model/donut-cord-finetuned` | Path to fine-tuned model (auto-detects `final/`) |
| `FORCE_CPU`          | `false`                            | Disable CUDA even when available              |
| `TASK_TOKEN`         | `<s_cord-v2>`                      | Must match the training-time task token       |
| `MAX_LENGTH`         | `768`                              | Decoder cap — match `Cfg.MAX_LENGTH` in notebook |
| `USE_SLOW_TOKENIZER` | `true`                             | Avoids slow→fast conversion (needs protobuf)  |

## Non-obvious constraints

- **`use_fast=False`** on the processor avoids the slow→fast XLMRoberta
  conversion that requires `protobuf`. We pin `protobuf>=3.20` in
  `requirements.txt` regardless, but keeping the slow tokenizer means the
  sidecar works on minimal Docker images and Lightning AI's `cloudspace`.
- **`bundle.decoder_start_ids`** is pre-encoded once at load time. Re-encoding
  the task token per request adds noticeable latency on CPU.
- **`bad_words_ids=[[unk_token_id]]`** prevents `generate` from emitting
  `<unk>` tokens that would then crash `token2json`.
- **`token2json` is fallible**. We catch and return `{}` rather than 500.
  The frontend then shows an empty receipt — recoverable.
- **`menu` can be a single dict** when the receipt has exactly one item.
  `normalize()` handles both list and single-dict shapes.
- The **task token must match training**. If `<s_cord-v2>` resolves to
  `<unk>` at load time, `model_loader.py` prints a warning — usually means
  the wrong model was placed in `MODEL_DIR`.
