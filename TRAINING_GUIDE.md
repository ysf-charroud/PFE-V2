# Training Guide — Donut Receipt Fine-Tuning

This document explains what `donut_cord_finetune.ipynb` does, why each step
exists, and how to interpret the results. No prior knowledge of transformers
assumed.

---

## What problem are we solving?

We have images of receipts. We want a model that reads an image and outputs a
structured JSON with the relevant fields (items, prices, totals) — without any
OCR step in between.

The traditional pipeline was:

```
image → OCR (detect text + positions) → layout model (classify fields) → JSON
```

The Donut approach collapses this into one step:

```
image → Donut → JSON
```

Donut reads the image directly and generates the JSON token by token, the same
way a language model generates text. There is no intermediate text extraction,
no bounding boxes, no coordinate logic.

---

## What is Donut?

Donut (**Do**cument **u**nderstanding **t**ransformer) is a Vision
Encoder–Decoder model from Naver Clova. It has two parts:

```
┌─────────────────────────────────────────────────┐
│  Encoder: Donut-Swin (Vision Transformer)        │
│  • Splits the image into patches (like tiles)    │
│  • Builds a rich feature map of the whole image  │
└──────────────────────┬──────────────────────────┘
                       │  feature vectors
┌──────────────────────▼──────────────────────────┐
│  Decoder: BART-style Transformer                 │
│  • Reads the encoder features                    │
│  • Generates one token at a time                 │
│  • Each token is conditioned on all prior tokens │
└─────────────────────────────────────────────────┘
```

The decoder generates a sequence of XML-like tags that encode the JSON
structure. For example, a receipt with one item becomes:

```
<s_cord-v2>
  <s_menu>
    <s_nm>Latte</s_nm>
    <s_cnt>1</s_cnt>
    <s_price>9,000</s_price>
  </s_menu>
  <s_total>
    <s_total_price>9,000</s_total_price>
  </s_total>
</s>
```

The processor's `token2json()` method converts this tag sequence back into a
Python dict at inference time.

---

## The dataset — CORD v2

**`naver-clova-ix/cord-v2`** is a collection of scanned receipts from
Indonesian cafes and restaurants, annotated by Naver Clova.

| Split | Samples |
|---|---|
| Train | 800 |
| Validation | 100 |
| Test | 100 |

Ground truth is a nested JSON dict with three top-level blocks:

```json
{
  "menu": [
    { "nm": "Latte", "cnt": "1", "price": "9,000" },
    { "nm": "Croissant", "cnt": "2", "price": "14,000" }
  ],
  "sub_total": {
    "subtotal_price": "23,000",
    "tax_price": "2,300"
  },
  "total": {
    "total_price": "25,300",
    "cashprice": "30,000",
    "changeprice": "4,700"
  }
}
```

All values are raw strings as they appear on the receipt — the normaliser in
`normalize.py` parses them into floats later.

The ground truth is stored as a JSON string inside the `ground_truth` column:
`json.loads(row["ground_truth"])["gt_parse"]`.

---

## The notebook — section by section

### §1 Setup

Installs pinned dependencies and defines two things:

**`DATASET_SPECS`** — a list of dataset descriptors. Currently one entry for
CORD v2. The structure supports adding more datasets later without changing the
training code.

**`Cfg`** — all hyperparameters in one place:

| Parameter | Value | Meaning |
|---|---|---|
| `BASE_MODEL` | `naver-clova-ix/donut-base` | The pre-trained checkpoint we start from |
| `TASK_TOKEN` | `<s_cord-v2>` | The decoder prompt token for this schema |
| `IMAGE_SIZE` | `[1280, 960]` | Height × width the encoder expects (see §3) |
| `MAX_LENGTH` | `768` | Maximum decoder output length in tokens |
| `EPOCHS` | `6` | Full passes over the training set |
| `LR` | `3e-5` | Peak learning rate |
| `PER_DEVICE_BATCH` | `1` | Images per GPU step |
| `GRAD_ACC` | `4` | Effective batch = 1 × 4 = 4 |
| `WARMUP_RATIO` | `0.1` | Fraction of steps used to ramp LR from 0 |
| `USE_AUGMENT` | `True` | Light image augmentation during training |

---

### §2 Load dataset

Downloads CORD v2 from HuggingFace and shows one sample to confirm the
`gt_parser` lambda works. The output confirms image dimensions and the JSON
structure so you catch format issues before training starts.

---

### §3 Model + processor

Loads the pre-trained `naver-clova-ix/donut-base` checkpoint and resizes it.

**Why resize?** The base model was pre-trained on 2560×1920 images, which
requires ~80 GB of GPU memory during training. We shrink to **1280×960**, which
fits in 24 GB. This must be set in two places — both must match or generation
produces garbage:

```python
processor.image_processor.size = {'height': 1280, 'width': 960}
model.config.encoder.image_size = [1280, 960]
```

The decoder cap is also set here: `model.config.decoder.max_length = 768`.
768 tokens is enough for the longest CORD parse (verified in §4's sanity check).

---

### §4 JSON ↔ token conversion

This is the most important setup step.

**The problem:** Donut's base vocabulary knows nothing about CORD field names
like `nm`, `cnt`, `sub_total`, `total_price`, etc. If these are tokenised as
regular text, each tag like `<s_nm>` gets split into multiple subword tokens —
`<`, `s`, `_`, `nm`, `>` — and the structure becomes ambiguous.

**The fix:** We walk every training sample once, collect every unique field key,
and add the open/close tag pairs as dedicated vocabulary entries:

```
<s_nm>  </s_nm>  <s_cnt>  </s_cnt>  <s_price>  </s_price>  ...
```

The task token `<s_cord-v2>` is added as a **special token** (never split).
The list separator `<sep/>` is also added.

After adding all tokens, `resize_token_embeddings` is called **once**. The new
rows start with random weights — training teaches the model what each token means.

The sanity check at the end of §4 serialises one ground truth sample into its
full target sequence and verifies it fits within `MAX_LENGTH=768`. If it doesn't,
you'd need to raise the cap.

---

### §5 PyTorch dataset

`MultiDonutDataset` converts every sample into the two tensors the Trainer needs:

**`pixel_values`** `(3, 1280, 960)`:
The image is resized, normalised to zero-mean/unit-variance, and converted to a
float tensor. This is what the encoder reads.

**`labels`** `(768,)`:
The ground-truth JSON is serialised to a tag sequence, prefixed with
`<s_cord-v2>`, suffixed with `</s>`, then tokenised and padded to 768. Padding
positions are replaced with `-100` so PyTorch's cross-entropy loss ignores them.

**Light augmentation** is applied to training images only (never validation):
- 50% chance: random brightness ±15%
- 50% chance: random contrast ±15%
- 30% chance: random rotation ±3° (white fill)

This makes the model more robust to scanning conditions without distorting the
text enough to change the correct answer.

The targets are pre-computed in `__init__` — the DataLoader workers only do
image loading, keeping the hot path fast.

---

### §6 Training

Standard seq2seq cross-entropy loop run by HuggingFace `Seq2SeqTrainer`:

1. Batch of `pixel_values` → encoder → feature vectors.
2. Feature vectors + shifted target tokens → decoder → logits over vocabulary.
3. At each decoder position, predict the next token.
4. Loss = average cross-entropy over all non-`-100` positions.
5. Backpropagate, update weights.

Key training choices:

| Setting | Value | Why |
|---|---|---|
| bf16 | on (Ampere+) | Half-precision arithmetic — faster, less memory |
| Gradient checkpointing | on | Recomputes activations on backward pass instead of storing them — cuts encoder memory ~40% |
| `predict_with_generate=False` | — | Running full generation during each eval epoch would be ~5× slower; cross-entropy loss is used instead to pick the best checkpoint |
| `load_best_model_at_end=True` | — | The checkpoint with the lowest validation loss is restored at the end |
| `report_to=[]` | — | Suppresses W&B / MLflow autodetection on Lightning AI |

The learning rate follows a linear warmup (first 10% of steps: 0 → 3e-5)
followed by a linear decay back to 0. This avoids large destructive updates
early in training when the new token embeddings are still random.

**What a healthy loss curve looks like:**

```
epoch 1:  train ~0.40   eval ~0.48   ← model quickly learns tag structure
epoch 2:  train ~0.20   eval ~0.23   ← learning field values
epoch 3:  train ~0.13   eval ~0.17
epoch 4:  train ~0.07   eval ~0.15   ← eval loss flattens (best checkpoint here)
epoch 5:  train ~0.09   eval ~0.14
epoch 6:  train ~0.09   eval ~0.15   ← train continues falling, eval plateaus
```

The gap between train and eval loss in later epochs is normal — the model has
seen the training images 6 times but the validation images never.

---

### §7 Save

Final model and processor are saved to `./donut-cord-finetuned/final/`:

```
donut-cord-finetuned/final/
├── config.json              model architecture + generation settings
├── pytorch_model.bin        trained weights (~800 MB)
├── tokenizer.json           vocabulary including all field tag tokens
├── tokenizer_config.json
└── preprocessor_config.json  image size, normalisation params
```

This directory is what you copy to `model/donut-cord-finetuned/` in the repo.
The sidecar auto-detects the `final/` subdirectory.

---

### §8 Inference demo

Runs the trained model on one validation sample without any metrics — just to
confirm it produces reasonable output. Useful as a quick sanity check before
running the full evaluation.

---

### §9–§11 Validation evaluation

Generates predictions for all 100 validation samples and computes:

- **Exact-match accuracy** — full JSON equality (strict, usually 30–45%)
- **Field micro F1** — pooled precision/recall over all `(path, value)` pairs
- **Field macro F1** — average per-receipt F1
- **Normalized TED** — tree edit distance metric (0 = perfect, lower is better)
- **Latency** — mean / p50 / p95 per-receipt inference time
- **Per-field breakdown** — P/R/F1 per top-level block (`menu`, `sub_total`, `total`)

The per-field breakdown is the most useful for debugging — `menu` is typically
the weakest field because it's a variable-length list.

---

### §12 Persist `report_data.json`

Saves everything — config, full training log, all metrics, every
(ground-truth, prediction, latency) triple — to
`donut-cord-finetuned/report_data.json`.

This file is read by the sidecar's `/metrics` endpoint and served to the
frontend dashboard. No manual steps needed.

---

### §13 Test evaluation

Validation was used to select the best checkpoint (`load_best_model_at_end`),
so its numbers are slightly optimistic. The **test split** (100 receipts the
model never saw during training or checkpoint selection) is the headline number
for the report.

---

### §14–§16 Analysis

- **§14** Per-epoch table — train loss, eval loss, learning rate at each epoch.
- **§15** F1 distribution histogram + 10 worst predictions. The histogram shows
  whether failures are concentrated (a few hard receipts) or spread out
  (systematic weakness). The worst-10 list reveals common error patterns —
  e.g. the model flattening nested `sub` items, or decimal separator confusion.
- **§16** Final scorecard — validation vs test metrics side by side.

---

## How inference works after training

The sidecar (`backend/inference_sidecar/`) loads the saved model and runs:

```
image (PIL)
  → resize to 1280×960, normalise → pixel_values
  → encoder → feature vectors
  → decoder starts with <s_cord-v2>
  → generates tokens until </s> or MAX_LENGTH
  → token2json() → raw CORD dict
  → normalize() → { line_items, subtotal, tax, total, … }
```

`output_scores=True` captures per-token softmax probabilities during generation.
These are aggregated into **per-field confidence scores** — a field the model
was uncertain about gets a low score (shown in the frontend UI).

---

## Understanding the metrics

### Exact-match accuracy
Fraction of receipts where prediction == ground truth exactly.
A single wrong character counts as failure. Even a well-trained model scores
30–45% here because minor value differences (e.g. `"9,000"` vs `"9.000"`)
count as mismatches.

### Field micro F1
Compares the multiset of `(field_path, value)` pairs between ground truth and
prediction. Partial credit is given — getting 8 out of 10 fields right scores
higher than getting 0.

```
precision = TP / (TP + FP)    # of what was extracted, how much was correct
recall    = TP / (TP + FN)    # of what was in the GT, how much was found
F1        = harmonic mean
```

List items are compared **order-insensitively** — the model is not penalised
for outputting menu items in a different order.

### Normalized Tree Edit Distance (nTED)
The Donut paper's canonical metric. Converts both GT and prediction to trees and
counts insert/delete/relabel operations to transform one into the other,
normalised by `max(|GT tree|, |pred tree|)`.

- 0 = perfect match. 1 = completely different structure.
- `TED_accuracy = 1 − nTED` (reported alongside, higher is better).

### Results from the current model (6 epochs)

| Metric | Validation | Test |
|---|---|---|
| Exact-match accuracy | 0.39 | 0.29 |
| Field micro F1 | 0.823 | 0.802 |
| Field macro F1 | 0.847 | 0.817 |
| TED accuracy | 0.522 | 0.299 |
| Avg latency | 373 ms | 387 ms |

The gap between validation and test exact-match (0.39 vs 0.29) is expected —
test is a true held-out set. The F1 gap is smaller (0.823 vs 0.802) because F1
gives partial credit.

---

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Sidecar won't start | `MODEL_DIR` points to an empty or wrong directory | Verify `config.json` exists at `MODEL_DIR` (or its `final/` subdir) |
| Model outputs `{}` for every image | Task token `<s_cord-v2>` resolves to `<unk>` | Check TASK_TOKEN in `.env` matches exactly what was used at training |
| Prices appear as `0.0` | Number parsing issue in `normalize.py` | Check raw output via the `raw` field in the `/infer` response |
| Training OOM on GPU | Default 1280×960 + gradient checkpointing is tight | Reduce `IMAGE_SIZE` to `[960, 720]` for a 16 GB GPU |
| Eval loss stops improving before epoch 4 | Learning rate too high or dataset too small | Try `LR=1e-5` or more augmentation |
| `token2json` exception in sidecar | Malformed generated sequence (unclosed tag) | Already caught — returns `{}`. Happens occasionally on unusual receipts. |
