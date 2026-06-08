# Inference Sidecar — Architecture Guide

This document explains every file in the sidecar, what it does, and how they all talk to each other. Written for a new contributor who has never touched this code before.

---

## Why a sidecar?

The LayoutLMv3 model is PyTorch/HuggingFace — it cannot run inside Node.js. Rather than a fragile stdin/stdout pipe, we run a tiny FastAPI process alongside the Express server. Express receives the upload, forwards it here over HTTP, and gets structured JSON back.

```
Browser / client
      │
      │  POST /api/extract  (multipart image)
      ▼
 Express :8000  (Node.js)
      │
      │  POST /infer        (multipart image, forwarded as-is)
      ▼
 Sidecar :8001  (Python / FastAPI)   ← you are here
      │
      ├─ EasyOCR
      ├─ LayoutLMv3Processor
      └─ LayoutLMv3ForTokenClassification
```

---

## File overview

```
inference_sidecar/
├── model_loader.py   Loads and holds the three heavy objects (model, processor, OCR)
├── inference.py      Full prediction pipeline — OCR → LayoutLMv3 → grouped JSON
├── visualize.py      Draws coloured label boxes on the image, saves PNG to disk
├── main.py           FastAPI app — the only HTTP entry point
└── requirements.txt  Python dependencies
```

Dependency direction (nothing goes backwards):

```
main.py
  ├── model_loader.py   (calls bundle.load() once at startup)
  ├── inference.py      (calls predict())
  └── visualize.py      (calls annotate() + save_image())

inference.py
  └── model_loader.py   (reads bundle.model / .processor / .ocr)

visualize.py
  └── (no project imports — pure Pillow)
```

---

## `model_loader.py` — shared state

**Role:** owns the three expensive objects and exposes them as a single singleton (`bundle`) that every other module imports.

Loading the model, processor, and OCR engine takes 10–30 seconds. Doing it once at startup and reusing the same objects on every request is the entire point of this file.

```python
class ModelBundle:
    def __init__(self):
        self.model = None        # LayoutLMv3ForTokenClassification
        self.processor = None    # LayoutLMv3Processor
        self.ocr = None          # easyocr.Reader
        self.device = None       # torch.device("cpu") or ("cuda")
        self.is_loaded = False
        self.ocr_min_confidence = OCR_MIN_CONFIDENCE
        self.ocr_upscale_to = OCR_UPSCALE_TO

bundle = ModelBundle()   # created at import time, empty
```

`bundle.load()` is called exactly once, from `main.py`'s lifespan hook:

```python
def load(self):
    use_gpu = not FORCE_CPU and torch.cuda.is_available()
    self.device = torch.device("cuda" if use_gpu else "cpu")

    self.processor = LayoutLMv3Processor.from_pretrained(MODEL_DIR, apply_ocr=False)

    self.model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR)
    self.model.to(self.device)
    self.model.eval()   # disables dropout — required for inference

    # OCR_LANGS is a list so EasyOCR can handle multi-language receipts
    self.ocr = easyocr.Reader(OCR_LANGS, gpu=use_gpu)
    self.is_loaded = True
```

**All settings come from environment variables** (with sensible defaults):

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | `../../model/cord_layoutlmv3(v1)` | Path to fine-tuned weights |
| `FORCE_CPU` | `false` | Ignore CUDA even if available |
| `OCR_LANG` | `en` | Comma-separated EasyOCR language codes (e.g. `en,fr`) |
| `OCR_MIN_CONFIDENCE` | `0.5` | Drop OCR words below this score |
| `OCR_UPSCALE_TO` | `1600` | Upscale short images before OCR (px) |

---

## `inference.py` — prediction pipeline

**Role:** takes a PIL `Image`, runs the full pipeline, and returns a structured dict. No HTTP, no file I/O — just data in, data out.

Imports `bundle` from `model_loader` to access the loaded objects.

```python
from model_loader import bundle

def predict(image: Image.Image) -> dict:
    ...
```

### Step 1 — upscale if needed

Small or low-resolution images produce poor OCR results. If the image's longest side is below `ocr_upscale_to`, we enlarge it (capped at 2× to avoid blurry artefacts). The original size is saved so boxes can be mapped back later.

```python
def _upscale_if_needed(image):
    target = bundle.ocr_upscale_to          # default 1600 px
    w, h = image.size
    if max(w, h) < target:
        scale = min(target / max(w, h), 2.0)
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image
```

### Step 2 — EasyOCR

EasyOCR returns phrase-level text regions. Each result is a tuple:

```python
(
  [[x1,y1],[x2,y1],[x2,y2],[x1,y2]],   # 4 corner points (pixels)
  "Caffe Latte",                         # recognized text (may be multiple words)
  0.97                                   # confidence score
)
```

Because the model was fine-tuned on CORD's **per-word** boxes, we split each phrase into individual words with proportionally-sized boxes via `_split_phrase()`. This is the key alignment step — without it, multi-word EasyOCR regions produce misaligned layout embeddings.

```python
def _split_phrase(text, box):
    toks = text.split()
    if len(toks) <= 1:
        return [(text, box)]
    x1, y1, x2, y2 = box
    span = max(1, x2 - x1)
    total = sum(len(t) for t in toks) + (len(toks) - 1)  # chars + spaces
    out, pos = [], 0
    for t in toks:
        wx1 = int(x1 + span * pos / total)
        pos += len(t)
        wx2 = int(x1 + span * pos / total)
        pos += 1
        out.append((t, [wx1, y1, max(wx1, wx2), y2]))
    return out
```

OCR result parsing:

```python
ocr_result = ocr.readtext(np.array(ocr_image))

for (quad, text, conf) in ocr_result:
    if conf < bundle.ocr_min_confidence or not text.strip():
        continue
    scaled = [[p[0] * scale_x, p[1] * scale_y] for p in quad]  # back to original coords
    norm_box = _normalize_box(scaled, width, height)
    for wtext, wbox in _split_phrase(text.strip(), norm_box):
        words.append(wtext)
        boxes.append(wbox)
```

### Step 3 — box normalisation

LayoutLMv3 expects boxes in a `[0, 1000]` coordinate space, not pixels. We convert and also clamp + enforce monotonicity (`x2 >= x1`, `y2 >= y1`) — the processor raises an error otherwise.

```python
def _normalize_box(pts, width, height):
    x0 = min(p[0] for p in pts)
    y0 = min(p[1] for p in pts)
    x1 = max(p[0] for p in pts)
    y1 = max(p[1] for p in pts)
    x0 = max(0, min(int(x0 * 1000 / width),  1000))
    y0 = max(0, min(int(y0 * 1000 / height), 1000))
    x1 = max(x0, min(int(x1 * 1000 / width),  1000))  # x1 >= x0
    y1 = max(y0, min(int(y1 * 1000 / height), 1000))  # y1 >= y0
    return [x0, y0, x1, y1]
```

### Step 4 — LayoutLMv3 encoding

The processor takes the image + words + boxes together and turns them into the token tensors the model expects. `apply_ocr=False` tells it to trust our boxes, not re-run its own OCR.

```python
encoding = processor(
    image, words, boxes=boxes,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=512,
)

# word_ids maps each token position back to the original word index.
# MUST be captured before moving to GPU — it disappears after .to(device).
word_ids = encoding.word_ids(0)

encoding = {k: v.to(bundle.device) for k, v in encoding.items()}
```

### Step 5 — model inference

```python
with torch.no_grad():
    logits = model(**encoding).logits    # shape: (1, seq_len, num_labels)

predictions = logits.argmax(-1).squeeze().tolist()
```

### Step 6 — majority vote subword → word

The tokeniser splits long words into multiple subword tokens. We collect all token predictions for each original word and take the **majority vote** — more robust than first-subword-wins, especially for multi-syllable words that get split into many tokens.

```python
votes = defaultdict(list)
for tok_idx, w_id in enumerate(word_ids):
    if w_id is not None:
        votes[w_id].append(predictions[tok_idx])

word_preds = {
    w_id: Counter(v).most_common(1)[0][0]   # most-voted label ID per word
    for w_id, v in votes.items()
}
```

### Step 7 — spatial row clustering

Rather than walking labels sequentially (fragile for multi-column receipts), we sort all labeled words by their **y-centre then x** and group them into rows. Words whose y-centres are within 60% of one line-height of the previous row centre are considered the same row.

```python
labeled_idxs.sort(key=lambda i: (
    (boxes[i][1] + boxes[i][3]) / 2,   # y-centre
    boxes[i][0],                         # x (left-to-right within same row)
))

rows, cur, last_yc = [], [], None
for i in labeled_idxs:
    yc = (boxes[i][1] + boxes[i][3]) / 2
    h = max(1, boxes[i][3] - boxes[i][1])
    if last_yc is None or abs(yc - last_yc) <= max(8, h * 0.6):
        cur.append(i)
    else:
        rows.append(cur); cur = [i]
    last_yc = yc
if cur:
    rows.append(cur)
```

Each row is then classified: rows containing any `menu.*` label become **line items** (one dict per row); all others accumulate into **summary**.

```python
for row in rows:
    row_fields = defaultdict(list)
    for i in sorted(row, key=lambda j: boxes[j][0]):   # left-to-right
        row_fields[word_list[i]["label"]].append(word_list[i]["text"])
    row_fields = {k: " ".join(v) for k, v in row_fields.items()}

    if any(k.startswith("menu") for k in row_fields):
        line_items.append(row_fields)
    else:
        for k, v in row_fields.items():
            summary[k] = (summary[k] + " " + v).strip() if k in summary else v
```

**Return value:**

```python
{
    "fields":     { "menu.nm": "Latte Cappuccino", ... },   # flat all-labels dict
    "line_items": [
        { "menu.nm": "Latte",      "menu.cnt": "1", "menu.price": "4.50" },
        { "menu.nm": "Cappuccino", "menu.cnt": "1", "menu.price": "3.75" },
    ],
    "summary": {
        "sub_total.subtotal_price": "8.25",
        "sub_total.tax_price":      "0.66",
        "total.total_price":        "8.91",
    },
    "words": [
        { "text": "Latte", "box": [42, 233, 200, 251], "label": "menu.nm" },
        ...
    ]
}
```

---

## `visualize.py` — annotation

**Role:** pure image utility. Takes the `words` list from `predict()` and draws coloured boxes on a copy of the original image. Has no knowledge of the model, the bundle, or HTTP.

### Colour assignment

Each label class gets a unique, deterministic colour derived from its name. The same label always renders in the same colour regardless of call order.

```python
def _label_color(label: str) -> tuple:
    h = int(hashlib.md5(label.encode()).hexdigest()[:6], 16)
    return ((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF)   # (R, G, B)
```

### Drawing

Boxes are stored in `[0, 1000]` space. We denormalise them back to pixel coordinates, then draw a semi-transparent fill + solid outline on a separate RGBA overlay to avoid modifying the source image.

```python
def annotate(image: Image.Image, words: list) -> Image.Image:
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = image.size

    for word in words:
        if word["label"] == "O":
            continue
        b = word["box"]
        x0, y0 = b[0] * w / 1000, b[1] * h / 1000
        x1, y1 = b[2] * w / 1000, b[3] * h / 1000
        r, g, b_ = _label_color(word["label"])
        draw.rectangle([x0, y0, x1, y1],
                       outline=(r, g, b_, 255), fill=(r, g, b_, 50), width=2)

    return Image.alpha_composite(img, overlay).convert("RGB")
```

### Saving to disk

The annotated image is written to `OUTPUT_DIR` with a UUID-based filename. The filename (not the full path) is returned to `main.py`, which includes it in the JSON response. Express serves the file at `/files/annotated/<filename>`.

```python
def save_image(image: Image.Image, output_dir: str, stem: str = None) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{stem + '_' if stem else ''}{uuid.uuid4().hex[:10]}.png"
    image.save(os.path.join(output_dir, filename))
    return filename
```

---

## `main.py` — HTTP entry point

**Role:** the only file that knows about HTTP. Wires the three modules above into two endpoints.

### Startup

FastAPI's `lifespan` hook runs `bundle.load()` before the server accepts any requests.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle.load()   # blocks until model + OCR are ready (~15-30 s)
    yield           # server starts accepting requests here
```

### `GET /health`

Used by Express on startup to know when the sidecar is ready.

```python
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bundle.is_loaded, "device": bundle.device_str}
```

### `POST /infer`

```python
@app.post("/infer")
async def infer(file: UploadFile = File(...), do_annotate: bool = Query(True, alias="annotate")):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image.load()

    result = predict(image)   # { fields, line_items, summary, words }

    annotated_filename = None
    if do_annotate and result["words"]:
        stem = Path(file.filename).stem if file.filename else None
        annotated_filename = save_image(annotate(image, result["words"]), OUTPUT_DIR, stem)

    return {**result, "num_words": len(result["words"]), "annotated_filename": annotated_filename}
```

---

## Full request flow (end to end)

```
Express receives POST /api/extract
  │
  │  multer parses the multipart body → req.file.buffer
  │
  └─► fetch("http://localhost:8001/infer?annotate=true", { body: FormData })
            │
            │  main.py receives POST /infer
            │    │
            │    ├─ Image.open(bytes)
            │    │
            │    ├─ inference.predict(image)
            │    │     ├─ _upscale_if_needed(image)
            │    │     ├─ bundle.ocr.readtext(...)       → phrase-level regions
            │    │     ├─ _split_phrase()                → per-word tokens + boxes
            │    │     ├─ _normalize_box()               → [0,1000] coords
            │    │     ├─ bundle.processor(...)          → token tensors
            │    │     ├─ bundle.model(**encoding)       → logits
            │    │     ├─ majority vote + word_ids       → label per word
            │    │     └─ spatial row clustering         → line_items / summary
            │    │
            │    ├─ visualize.annotate(image, words)     → PIL Image with boxes
            │    ├─ visualize.save_image(...)            → PNG saved to OUTPUT_DIR
            │    │
            │    └─ return JSON  { ..., annotated_filename: "receipt_abc123.png" }
            │
  Express builds annotated_image_url = "/files/annotated/<filename>"
  Express returns full JSON to client
  Client fetches image from http://localhost:8000/files/annotated/<filename>
```

---

## Adding a new label class

If you retrain the model with extra CORD classes or a different dataset:

1. The `id2label` / `label2id` mapping is in `model/cord_layoutlmv3(v1)/config.json` — update it after retraining.
2. If the new label starts with a prefix not yet handled (not `menu.*`, `total.*`, `sub_total.*`, `void_menu.*`), add a branch in the row-classification block in `inference.py` (the `for row in rows` loop).
3. `visualize.py` needs no changes — colours are derived automatically from the label name.
