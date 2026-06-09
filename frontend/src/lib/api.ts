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
};

export type ExtractResponse = {
  filename: string;
  receipt: Receipt;
  num_words: number;
  processing_ms: number;
  annotated_image_url: string | null;
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
