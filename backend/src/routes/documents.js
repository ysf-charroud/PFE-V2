import { Router } from 'express';
import { getDocument, listDocuments, getCorrections, saveCorrection } from '../db.js';
import { getMetrics } from '../services/inference.js';

const router = Router();

// Model evaluation metrics (CORD test split), proxied from the sidecar.
router.get('/metrics', async (_req, res, next) => {
  try {
    res.json(await getMetrics());
  } catch (err) {
    next(err);
  }
});

// List saved documents (most recent first).
router.get('/documents', (_req, res) => {
  res.json({ documents: listDocuments() });
});

// Fetch one document plus its recorded field-level corrections.
router.get('/documents/:id', (req, res) => {
  const doc = getDocument(Number(req.params.id));
  if (!doc) return res.status(404).json({ detail: 'Document not found.' });
  res.json({ ...doc, corrections: getCorrections(doc.id) });
});

// Save a user-corrected receipt: stores the corrected JSON, marks the document
// reviewed, and records each changed field as a correction.
router.patch('/documents/:id', (req, res) => {
  const receipt = req.body?.receipt;
  if (!receipt || typeof receipt !== 'object') {
    return res.status(400).json({ detail: 'Body must include a `receipt` object.' });
  }
  const updated = saveCorrection(Number(req.params.id), receipt);
  if (!updated) return res.status(404).json({ detail: 'Document not found.' });
  res.json({ ...updated, corrections: getCorrections(updated.id) });
});

export default router;
