#!/usr/bin/env bash
# One-command local dev launcher: Donut sidecar (8001) + Express (8000) + frontend (8080).
# Usage:  ./start-dev.sh      (Ctrl+C stops all three)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIDECAR="$ROOT/backend/inference_sidecar"
VENV_PY="$SIDECAR/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "Sidecar venv not found at $VENV_PY"
  echo "Create it first:  cd backend/inference_sidecar && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

pids=()
cleanup() { echo; echo "Stopping…"; kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "→ Donut sidecar  on :8001"
( cd "$SIDECAR" && \
  SSL_CERT_FILE="$("$VENV_PY" -c 'import certifi;print(certifi.where())')" \
  MODEL_DIR="$ROOT/model/donut-cord-finetuned" \
  FORCE_CPU=true TASK_TOKEN="<s_cord-v2>" MAX_LENGTH=768 USE_SLOW_TOKENIZER=true \
  "$VENV_PY" -m uvicorn main:app --host 0.0.0.0 --port 8001 ) &
pids+=($!)

echo "→ Express API    on :8000"
( cd "$ROOT/backend" && npm run dev ) &
pids+=($!)

echo "→ Frontend       on :8080"
( cd "$ROOT/frontend" && npm run dev ) &
pids+=($!)

echo
echo "All three starting. Open  http://localhost:8080  (give the sidecar ~30s to load the model)."
wait
