// Client for the Express extraction backend (see backend/src/routes/extract.js).
// Base URL comes from VITE_API_URL, falling back to the local dev backend.

export const API_BASE_URL =
  (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ??
  "http://localhost:8000";

export type LineItem = {
  name?: string;
  sub_name?: string;
  item_num?: string;
  quantity?: number;
  unit_price?: number;
  price?: number;
  discount?: number;
  /** Per-field model confidence in [0, 1], keyed by field name. */
  confidence?: Record<string, number>;
};

export type Receipt = {
  line_items?: LineItem[];
  subtotal?: number;
  discount?: number;
  service_charge?: number;
  tax?: number;
  total?: number;
  cash_paid?: number;
  change?: number;
  credit_card?: number;
  e_money?: number;
  /** Per-field model confidence in [0, 1] for summary fields. */
  field_confidence?: Record<string, number>;
  /** Mean confidence across every detected field in [0, 1]. */
  overall_confidence?: number;
};

export type ExtractResponse = {
  id: number;
  filename: string;
  receipt: Receipt;
  num_words: number;
  processing_ms: number;
  annotated_image_url: string | null;
};

export type Correction = {
  field_path: string;
  original_value: string | null;
  corrected_value: string | null;
};

export type SavedDocument = {
  id: number;
  filename: string;
  annotated_filename: string | null;
  num_words: number | null;
  processing_ms: number | null;
  overall_confidence: number | null;
  receipt: Receipt;
  corrected_receipt: Receipt | null;
  reviewed: boolean;
  created_at: string;
  reviewed_at: string | null;
  corrections?: Correction[];
};

/** Resolve a backend-relative path (e.g. annotated image URL) to an absolute URL. */
export function resolveApiUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  if (/^https?:\/\//.test(path)) return path;
  return `${API_BASE_URL}${path.startsWith("/") ? "" : "/"}${path}`;
}

/** Upload an image to POST /api/extract and return the structured receipt. */
export async function extractReceipt(
  file: File,
  opts: { annotate?: boolean } = {},
): Promise<ExtractResponse> {
  const annotate = opts.annotate ?? true;
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE_URL}/api/extract?annotate=${annotate}`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    let detail = `Extraction failed (HTTP ${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // non-JSON error body — keep the generic message
    }
    throw new Error(detail);
  }

  return (await res.json()) as ExtractResponse;
}

export type HealthResponse = {
  status: string;
  version: string;
  model_loaded: boolean;
};

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/health`);
  if (!res.ok) throw new Error(`Health check failed (HTTP ${res.status})`);
  return (await res.json()) as HealthResponse;
}

export type LabelMetric = {
  label: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
};

export type AvgMetric = { precision: number; recall: number; f1: number };

export type Metrics = {
  available: boolean;
  detail?: string;
  model_dir?: string;
  dataset?: string;
  split?: string;
  num_receipts?: number;
  num_tokens?: number;
  num_labels?: number;
  generated_at?: string;
  overall?: {
    accuracy: number;
    macro: AvgMetric;
    micro: AvgMetric;
    weighted: AvgMetric;
  };
  per_label?: LabelMetric[];
};

/** Model evaluation metrics on the CORD test split. */
export async function getMetrics(): Promise<Metrics> {
  const res = await fetch(`${API_BASE_URL}/api/metrics`);
  if (!res.ok) throw new Error(`Failed to load metrics (HTTP ${res.status})`);
  return (await res.json()) as Metrics;
}

/** List persisted documents, most recent first. */
export async function listDocuments(): Promise<SavedDocument[]> {
  const res = await fetch(`${API_BASE_URL}/api/documents`);
  if (!res.ok) throw new Error(`Failed to load documents (HTTP ${res.status})`);
  return ((await res.json()).documents ?? []) as SavedDocument[];
}

/** Save a user-corrected receipt for a document; records field-level diffs. */
export async function saveCorrection(id: number, receipt: Receipt): Promise<SavedDocument> {
  const res = await fetch(`${API_BASE_URL}/api/documents/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ receipt }),
  });
  if (!res.ok) {
    let detail = `Save failed (HTTP ${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // keep generic message
    }
    throw new Error(detail);
  }
  return (await res.json()) as SavedDocument;
}
