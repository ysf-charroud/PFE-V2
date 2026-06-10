import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import Database from 'better-sqlite3';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BACKEND_DIR = path.resolve(__dirname, '..');
const DB_PATH = process.env.DB_PATH ?? path.join(BACKEND_DIR, 'storage', 'app.db');

fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });

const db = new Database(DB_PATH);
db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');

db.exec(`
  CREATE TABLE IF NOT EXISTS documents (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    filename           TEXT    NOT NULL,
    annotated_filename TEXT,
    num_words          INTEGER,
    processing_ms      REAL,
    overall_confidence REAL,
    receipt_json       TEXT    NOT NULL,   -- original model output
    corrected_json     TEXT,               -- user-corrected receipt (nullable)
    reviewed           INTEGER NOT NULL DEFAULT 0,
    created_at         TEXT    NOT NULL DEFAULT (datetime('now')),
    reviewed_at        TEXT
  );

  -- Field-level corrections: the human-in-the-loop training signal.
  CREATE TABLE IF NOT EXISTS corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    field_path      TEXT    NOT NULL,      -- e.g. "total" or "line_items[0].price"
    original_value  TEXT,
    corrected_value TEXT,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
  );

  CREATE INDEX IF NOT EXISTS idx_corrections_document ON corrections(document_id);
`);

// Added after the initial schema shipped — guard for existing databases.
try {
  db.exec('ALTER TABLE documents ADD COLUMN image_filename TEXT');
} catch {
  /* column already exists */
}

/** Insert a freshly-extracted document. Returns the new row id. */
export function insertDocument({
  filename,
  annotated_filename,
  image_filename,
  num_words,
  processing_ms,
  overall_confidence,
  receipt,
}) {
  const stmt = db.prepare(`
    INSERT INTO documents
      (filename, annotated_filename, image_filename, num_words, processing_ms, overall_confidence, receipt_json)
    VALUES (@filename, @annotated_filename, @image_filename, @num_words, @processing_ms, @overall_confidence, @receipt_json)
  `);
  const info = stmt.run({
    filename,
    annotated_filename: annotated_filename ?? null,
    image_filename: image_filename ?? null,
    num_words: num_words ?? null,
    processing_ms: processing_ms ?? null,
    overall_confidence: overall_confidence ?? null,
    receipt_json: JSON.stringify(receipt ?? {}),
  });
  return info.lastInsertRowid;
}

function rowToDocument(row) {
  if (!row) return null;
  return {
    id: row.id,
    filename: row.filename,
    annotated_filename: row.annotated_filename,
    image_filename: row.image_filename,
    num_words: row.num_words,
    processing_ms: row.processing_ms,
    overall_confidence: row.overall_confidence,
    receipt: JSON.parse(row.receipt_json),
    corrected_receipt: row.corrected_json ? JSON.parse(row.corrected_json) : null,
    reviewed: !!row.reviewed,
    created_at: row.created_at,
    reviewed_at: row.reviewed_at,
  };
}

export function getDocument(id) {
  const row = db.prepare('SELECT * FROM documents WHERE id = ?').get(id);
  return rowToDocument(row);
}

export function listDocuments(limit = 100) {
  const rows = db
    .prepare('SELECT * FROM documents ORDER BY id DESC LIMIT ?')
    .all(limit);
  return rows.map(rowToDocument);
}

export function getCorrections(documentId) {
  return db
    .prepare('SELECT * FROM corrections WHERE document_id = ? ORDER BY id')
    .all(documentId);
}

/**
 * Persist a user-corrected receipt: store the corrected JSON, mark the document
 * reviewed, and record one row per changed field in `corrections`.
 * Runs in a transaction. Returns the updated document.
 */
export const saveCorrection = db.transaction((id, correctedReceipt) => {
  const existing = getDocument(id);
  if (!existing) return null;

  db.prepare(
    `UPDATE documents
       SET corrected_json = ?, reviewed = 1, reviewed_at = datetime('now')
     WHERE id = ?`,
  ).run(JSON.stringify(correctedReceipt), id);

  // Replace prior corrections for this document (idempotent re-saves).
  db.prepare('DELETE FROM corrections WHERE document_id = ?').run(id);

  const diffs = diffReceipts(existing.receipt, correctedReceipt);
  const insert = db.prepare(`
    INSERT INTO corrections (document_id, field_path, original_value, corrected_value)
    VALUES (?, ?, ?, ?)
  `);
  for (const d of diffs) {
    insert.run(id, d.field_path, d.original_value, d.corrected_value);
  }

  return getDocument(id);
});

const toStr = (v) => (v == null ? null : String(v));

/** Compute field-level diffs between the original and corrected receipt. */
export function diffReceipts(original = {}, corrected = {}) {
  const diffs = [];
  const SUMMARY_KEYS = [
    'subtotal',
    'discount',
    'service_charge',
    'other_charges',
    'tax',
    'total',
    'cash_paid',
    'change',
    'credit_card',
    'e_money',
  ];

  for (const key of SUMMARY_KEYS) {
    const a = original[key];
    const b = corrected[key];
    if (toStr(a) !== toStr(b)) {
      diffs.push({ field_path: key, original_value: toStr(a), corrected_value: toStr(b) });
    }
  }

  const oItems = original.line_items ?? [];
  const cItems = corrected.line_items ?? [];
  const n = Math.max(oItems.length, cItems.length);
  const ITEM_KEYS = ['name', 'sub_name', 'item_num', 'quantity', 'unit_price', 'price', 'discount'];
  for (let i = 0; i < n; i++) {
    const a = oItems[i] ?? {};
    const b = cItems[i] ?? {};
    for (const key of ITEM_KEYS) {
      if (toStr(a[key]) !== toStr(b[key])) {
        diffs.push({
          field_path: `line_items[${i}].${key}`,
          original_value: toStr(a[key]),
          corrected_value: toStr(b[key]),
        });
      }
    }
  }

  return diffs;
}

/** Reviewed documents that have both a correction and a stored image —
 * i.e. complete (image, label) pairs ready to export as training data. */
export function listReviewedDocuments() {
  const rows = db
    .prepare(
      `SELECT * FROM documents
        WHERE reviewed = 1 AND corrected_json IS NOT NULL AND image_filename IS NOT NULL
        ORDER BY id`,
    )
    .all();
  return rows.map(rowToDocument);
}

/** Aggregate counts for the active-learning panel. */
export function getStats() {
  const documents = db.prepare('SELECT COUNT(*) AS n FROM documents').get().n;
  const reviewed = db.prepare('SELECT COUNT(*) AS n FROM documents WHERE reviewed = 1').get().n;
  const corrections = db.prepare('SELECT COUNT(*) AS n FROM corrections').get().n;
  const trainable = db
    .prepare(
      `SELECT COUNT(*) AS n FROM documents
        WHERE reviewed = 1 AND corrected_json IS NOT NULL AND image_filename IS NOT NULL`,
    )
    .get().n;
  return { documents, reviewed, corrections, trainable };
}

// Inverse of the sidecar's normalize: clean receipt → raw CORD gt_parse schema.
const _ITEM_OUT = {
  name: 'nm',
  sub_name: 'sub_nm',
  item_num: 'num',
  quantity: 'cnt',
  unit_price: 'unitprice',
  price: 'price',
  discount: 'discountprice',
};
const _SUBTOTAL_OUT = {
  subtotal: 'subtotal_price',
  discount: 'discount_price',
  service_charge: 'service_price',
  tax: 'tax_price',
};
const _TOTAL_OUT = {
  total: 'total_price',
  cash_paid: 'cashprice',
  change: 'changeprice',
  credit_card: 'creditcardprice',
  e_money: 'emoneyprice',
};

const _str = (v) => (v == null ? null : String(v));

function _mapOut(obj, mapping) {
  const out = {};
  for (const [clean, cord] of Object.entries(mapping)) {
    const v = obj?.[clean];
    if (v != null && v !== '') out[cord] = _str(v);
  }
  return out;
}

/** Convert a clean receipt back to Donut's CORD `gt_parse` dict. */
export function receiptToCordGtParse(receipt = {}) {
  const gt = {};
  const menu = (receipt.line_items ?? [])
    .map((it) => _mapOut(it, _ITEM_OUT))
    .filter((m) => Object.keys(m).length > 0);
  if (menu.length) gt.menu = menu;

  const sub = _mapOut(receipt, _SUBTOTAL_OUT);
  if (Object.keys(sub).length) gt.sub_total = sub;

  const total = _mapOut(receipt, _TOTAL_OUT);
  if (Object.keys(total).length) gt.total = total;

  return gt;
}

export default db;
