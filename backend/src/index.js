import fs from 'fs';
import express from 'express';
import cors from 'cors';
import { config } from './config.js';
import { loadModel, isModelLoaded } from './services/inference.js';
import extractRouter from './routes/extract.js';
import documentsRouter from './routes/documents.js';

fs.mkdirSync(config.outputDir, { recursive: true });

const app = express();

app.use(cors({ origin: config.corsOrigins, credentials: true }));
app.use(express.json());

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', version: '0.1.0', model_loaded: isModelLoaded() });
});

app.use('/api', extractRouter);
app.use('/api', documentsRouter);

app.use(config.outputUrlPath, express.static(config.outputDir));

app.use((err, _req, res, _next) => {
  const status = err.status ?? 500;
  res.status(status).json({ detail: err.message ?? 'Internal server error' });
});

async function start() {
  await loadModel();
  app.listen(config.port, () => {
    console.log(`Invoice Extraction API listening on http://localhost:${config.port}`);
  });
}

start().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
