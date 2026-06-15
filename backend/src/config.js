import 'dotenv/config';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BACKEND_DIR = path.resolve(__dirname, '..');

export const config = {
  port: parseInt(process.env.PORT ?? '8000', 10),
  corsOrigins: (process.env.CORS_ORIGINS ?? 'http://localhost:3000,http://localhost:5173')
    .split(',')
    .map((s) => s.trim()),
  maxUploadMb: parseInt(process.env.MAX_UPLOAD_MB ?? '10', 10),
  allowedMimeTypes: ['image/jpeg', 'image/png', 'image/webp'],
  sidecarUrl: process.env.INFERENCE_SIDECAR_URL ?? 'http://localhost:8001',
  outputDir: process.env.OUTPUT_DIR ?? path.join(BACKEND_DIR, 'storage', 'annotated'),
  outputUrlPath: '/files/annotated',
  uploadDir: process.env.UPLOAD_DIR ?? path.join(BACKEND_DIR, 'storage', 'uploads'),
};
