import fs from "fs";
import path from "path";
import { createRequire } from "module";
import { Router } from "express";
import { config } from "../config.js";

// archiver is CommonJS without an ESM default export — load it via require.
const require = createRequire(import.meta.url);
const archiver = require("archiver");
import {
  getDocument,
  listDocuments,
  getCorrections,
  saveCorrection,
  listReviewedDocuments,
  getStats,
  receiptToCordGtParse,
} from "../db.js";
import { getMetrics } from "../services/inference.js";

const router = Router();

// Model evaluation metrics (CORD test split), proxied from the sidecar.
router.get("/metrics", async (_req, res, next) => {
  try {
    res.json(await getMetrics());
  } catch (err) {
    next(err);
  }
});

// List saved documents (most recent first).
router.get("/documents", (_req, res) => {
  res.json({ documents: listDocuments() });
});

// Active-learning stats for the dashboard panel.
router.get("/stats", (_req, res) => {
  res.json(getStats());
});

// Export every reviewed (corrected) document as a Donut/CORD fine-tuning set:
// a zip of the original images + a metadata.jsonl mapping each image to its
// human-corrected `gt_parse` label. Drop-in for HF imagefolder loading.
router.get("/corrections/export", (_req, res, next) => {
  try {
    const docs = listReviewedDocuments();

    res.setHeader("Content-Type", "application/zip");
    res.setHeader(
      "Content-Disposition",
      'attachment; filename="donut_corrections_dataset.zip"',
    );

    const archive = archiver("zip", { zlib: { level: 9 } });
    archive.on("error", next);
    archive.pipe(res);

    const lines = [];
    for (const d of docs) {
      const imgPath = path.join(config.uploadDir, d.image_filename);
      if (!fs.existsSync(imgPath)) continue;
      archive.file(imgPath, { name: `images/${d.image_filename}` });
      const gt_parse = receiptToCordGtParse(d.corrected_receipt ?? d.receipt);
      lines.push(
        JSON.stringify({
          file_name: `images/${d.image_filename}`,
          ground_truth: JSON.stringify({ gt_parse }),
        }),
      );
    }

    archive.append(lines.join("\n") + (lines.length ? "\n" : ""), {
      name: "metadata.jsonl",
    });
    archive.append(EXPORT_README, { name: "README.md" });
    archive.finalize();
  } catch (err) {
    next(err);
  }
});

// Fetch one document plus its recorded field-level corrections.
router.get("/documents/:id", (req, res) => {
  const doc = getDocument(Number(req.params.id));
  if (!doc) return res.status(404).json({ detail: "Document not found." });
  res.json({ ...doc, corrections: getCorrections(doc.id) });
});

// Save a user-corrected receipt: stores the corrected JSON, marks the document
// reviewed, and records each changed field as a correction.
router.patch("/documents/:id", (req, res) => {
  const receipt = req.body?.receipt;
  if (!receipt || typeof receipt !== "object") {
    return res
      .status(400)
      .json({ detail: "Body must include a `receipt` object." });
  }
  const updated = saveCorrection(Number(req.params.id), receipt);
  if (!updated) return res.status(404).json({ detail: "Document not found." });
  res.json({ ...updated, corrections: getCorrections(updated.id) });
});

const EXPORT_README = `# Donut correction dataset

Human-reviewed receipts exported from the IDP app, ready to fine-tune Donut.

- \`images/\` — the original uploaded receipt images.
- \`metadata.jsonl\` — one row per image: \`file_name\` + \`ground_truth\`
  (a JSON string of \`{"gt_parse": {...}}\` in the CORD schema), matching the
  format of \`naver-clova-ix/cord-v2\`.

Load it as an extra training source in donut_cord_finetune.ipynb, e.g.:

    from datasets import load_dataset
    ds = load_dataset("imagefolder", data_dir="donut_corrections_dataset")
`;

export default router;
