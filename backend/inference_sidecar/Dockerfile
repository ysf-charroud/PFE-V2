# Donut inference sidecar (FastAPI + transformers, CPU).
FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first (avoids pulling the multi-GB CUDA build),
# then the rest of the deps (torch>=2.0 in requirements is already satisfied).
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY *.py ./

# The model is mounted at /models (see docker-compose.yml); report_data.json
# (for GET /metrics) lives alongside it.
ENV FORCE_CPU=true \
    MODEL_DIR=/models/donut-cord-finetuned \
    METRICS_PATH=/models/donut-cord-finetuned/report_data.json \
    TASK_TOKEN="<s_cord-v2>" \
    MAX_LENGTH=768 \
    USE_SLOW_TOKENIZER=true

EXPOSE 8001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
