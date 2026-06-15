import fs from "fs";
import path from "path";
import crypto from "crypto";
import { Router } from "express";
import multer from "multer";
import { config } from "../config.js";
import { predict } from "../services/inference.js";
import { insertDocument } from "../db.js";

const EXT_BY_MIME = {
  "image/jpeg": "jpg",
  "image/png": "png",
  "image/webp": "webp",
};

const router = Router();

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: config.maxUploadMb * 1024 * 1024 },
  fileFilter(_req, file, cb) {
    if (config.allowedMimeTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      const err = new Error(`Unsupported file type: ${file.mimetype}`);
      err.status = 415;
      cb(err, false);
    }
  },
});

router.post("/extract", upload.single("file"), async (req, res, next) => {
  try {
    if (!req.file) {
      return res.status(400).json({ detail: "No file uploaded." });
    }

    const annotate = req.query.annotate !== "false";
    const start = performance.now();

    const result = await predict(
      req.file.buffer,
      req.file.originalname,
      annotate,
    );

    const elapsed_ms = +(performance.now() - start).toFixed(1);

    const annotated_image_url = result.annotated_filename
      ? `${config.outputUrlPath}/${result.annotated_filename}`
      : null;

    // Save the uploaded image so a corrected document becomes a complete
    // (image, label) training pair we can export later.
    const ext = EXT_BY_MIME[req.file.mimetype] ?? "bin";
    const imageFilename = `${crypto.randomUUID()}.${ext}`;
    fs.mkdirSync(config.uploadDir, { recursive: true });
    fs.writeFileSync(
      path.join(config.uploadDir, imageFilename),
      req.file.buffer,
    );

    // Persist the extraction so it survives reloads and can be corrected later.
    const id = insertDocument({
      filename: req.file.originalname,
      annotated_filename: result.annotated_filename ?? null,
      image_filename: imageFilename,
      num_words: result.num_words,
      processing_ms: elapsed_ms,
      overall_confidence: result.receipt?.overall_confidence ?? null,
      receipt: result.receipt,
    });

    return res.json({
      id,
      filename: req.file.originalname,
      receipt: result.receipt,
      num_words: result.num_words,
      processing_ms: elapsed_ms,
      annotated_image_url,
    });
  } catch (err) {
    next(err);
  }
});

export default router;
