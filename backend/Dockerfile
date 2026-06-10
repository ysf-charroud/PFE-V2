# Express API (uploads, proxy to sidecar, SQLite persistence).
FROM node:20-slim

WORKDIR /app

# better-sqlite3 ships prebuilt binaries; add build tools as a fallback in
# case a prebuild isn't available for this platform.
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 make g++ \
 && rm -rf /var/lib/apt/lists/*

COPY package*.json ./
RUN npm ci --omit=dev

COPY src ./src

ENV PORT=8000 \
    INFERENCE_SIDECAR_URL=http://sidecar:8001 \
    CORS_ORIGINS=http://localhost:8080

EXPOSE 8000
CMD ["node", "src/index.js"]
