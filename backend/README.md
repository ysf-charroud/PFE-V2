# Invoice Extraction API (backend)

FastAPI service that wraps the fine-tuned **LayoutLMv3 + CORD** model from
`invoice_extraction_v3.ipynb`. It receives a receipt/invoice image from the
React frontend and returns the extracted fields as JSON (plus, later, an
annotated image).

```
image (multipart upload)
    -> PaddleOCR (text + boxes)
    -> LayoutLMv3Processor
    -> LayoutLMv3 token classification
    -> grouped JSON
```

## Project layout

```
backend/
├── app/
│   ├── main.py            # FastAPI app, CORS, router wiring
│   ├── core/
│   │   └── config.py      # settings (model path, CORS, upload limits)
│   ├── api/               # route handlers (e.g. /api/extract)  [next step]
│   ├── services/          # model loading + inference logic       [next step]
│   └── schemas/           # pydantic request/response models      [next step]
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

From the `backend/` directory:

```powershell
# 1. Create + activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

> Note: `torch` installs the CPU build by default. The model is small enough to
> run on CPU for inference. For CUDA, install torch from the PyTorch index URL.

## Run

```powershell
# from backend/ with the venv active
uvicorn app.main:app --reload
```

Then open http://127.0.0.1:8000/docs (Swagger UI) or hit the health check:

```
GET http://127.0.0.1:8000/health  ->  {"status": "ok", "version": "0.1.0"}
```
