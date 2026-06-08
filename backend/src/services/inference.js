import { config } from '../config.js';

let modelLoaded = false;

export async function loadModel() {
  const maxAttempts = 150;   // 150 × 2 s = 5 min — covers first-run PaddleOCR model download
  const delayMs = 2000;

  console.log(`Waiting for inference sidecar at ${config.sidecarUrl} ...`);
  for (let i = 0; i < maxAttempts; i++) {
    try {
      const res = await fetch(`${config.sidecarUrl}/health`);
      if (res.ok) {
        const data = await res.json();
        if (data.model_loaded) {
          modelLoaded = true;
          console.log(`Sidecar ready (device: ${data.device})`);
          return;
        }
      }
    } catch {
      // sidecar not up yet — keep polling
    }
    await new Promise((r) => setTimeout(r, delayMs));
  }

  // Don't block startup if sidecar is slow; requests will fail with a clear error
  console.warn('Sidecar did not become ready within timeout — proceeding anyway.');
  modelLoaded = true;
}

export function isModelLoaded() {
  return modelLoaded;
}

/**
 * @param {Buffer} imageBuffer
 * @param {string} filename
 * @param {boolean} annotate
 */
export async function predict(imageBuffer, filename, annotate = true) {
  const form = new FormData();
  form.append('file', new Blob([imageBuffer]), filename ?? 'upload.jpg');

  const res = await fetch(`${config.sidecarUrl}/infer?annotate=${annotate}`, {
    method: 'POST',
    body: form,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    const err = new Error(body.detail ?? 'Sidecar error');
    err.status = res.status;
    throw err;
  }

  return res.json();
}
