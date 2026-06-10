import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import invoiceScan from "@/assets/invoice-scan.jpg";
import {
  correctionsExportUrl,
  extractReceipt,
  getHealth,
  getStats,
  listDocuments,
  resolveApiUrl,
  saveCorrection,
  type ExtractResponse,
  type LineItem,
  type Receipt,
  type SavedDocument,
} from "@/lib/api";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Documents — Vector IDP" },
      {
        name: "description",
        content:
          "Extract receipt fields with a fine-tuned Donut model. Review and correct the results, then commit to local SQLite.",
      },
    ],
  }),
  component: DocumentsPage,
});

const fmt = (n?: number | null) =>
  n == null ? "—" : n.toLocaleString(undefined, { maximumFractionDigits: 2 });

const PAGE_SIZE = 8; // rows per page in the Saved Documents table

/** Mean of a per-field confidence map, or null if empty. */
function meanConfidence(conf?: Record<string, number>): number | null {
  if (!conf) return null;
  const vals = Object.values(conf);
  if (vals.length === 0) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

// --- Editable form model -------------------------------------------------
// Fields are edited as strings (free typing, incl. partial decimals) and
// coerced back to a clean Receipt only when saving.

type DraftItem = {
  name: string;
  sub_name: string;
  quantity: string;
  unit_price: string;
  price: string;
  discount: string;
};

type Draft = {
  items: DraftItem[];
  subtotal: string;
  discount: string;
  service_charge: string;
  tax: string;
  total: string;
  cash_paid: string;
  change: string;
  credit_card: string;
};

const s = (n?: number | string | null) => (n == null ? "" : String(n));

function num(v: string): number | undefined {
  const t = v.trim();
  if (t === "") return undefined;
  const n = Number(t.replace(/,/g, ""));
  return Number.isFinite(n) ? n : undefined;
}

function toDraft(r: Receipt): Draft {
  return {
    items: (r.line_items ?? []).map((it) => ({
      name: s(it.name),
      sub_name: s(it.sub_name),
      quantity: s(it.quantity),
      unit_price: s(it.unit_price),
      price: s(it.price),
      discount: s(it.discount),
    })),
    subtotal: s(r.subtotal),
    discount: s(r.discount),
    service_charge: s(r.service_charge),
    tax: s(r.tax),
    total: s(r.total),
    cash_paid: s(r.cash_paid),
    change: s(r.change),
    credit_card: s(r.credit_card),
  };
}

function fromDraft(d: Draft): Receipt {
  const r: Receipt = {
    line_items: d.items.map((it) => {
      const item: LineItem = {};
      if (it.name.trim()) item.name = it.name.trim();
      if (it.sub_name.trim()) item.sub_name = it.sub_name.trim();
      if (num(it.quantity) != null) item.quantity = num(it.quantity);
      if (num(it.unit_price) != null) item.unit_price = num(it.unit_price);
      if (num(it.price) != null) item.price = num(it.price);
      if (num(it.discount) != null) item.discount = num(it.discount);
      return item;
    }),
  };
  const summary: [keyof Draft, keyof Receipt][] = [
    ["subtotal", "subtotal"],
    ["discount", "discount"],
    ["service_charge", "service_charge"],
    ["tax", "tax"],
    ["total", "total"],
    ["cash_paid", "cash_paid"],
    ["change", "change"],
    ["credit_card", "credit_card"],
  ];
  for (const [dk, rk] of summary) {
    const v = num(d[dk] as string);
    if (v != null) (r as Record<string, unknown>)[rk] = v;
  }
  return r;
}

// --- Unified view model over a fresh extraction or a persisted document ---

type DocView = {
  id: number;
  filename: string;
  receipt: Receipt; // original model output
  corrected: Receipt | null;
  num_words: number;
  processing_ms: number;
  annotated_image_url: string | null;
  reviewed: boolean;
};

function viewFromExtract(r: ExtractResponse): DocView {
  return {
    id: r.id,
    filename: r.filename,
    receipt: r.receipt,
    corrected: null,
    num_words: r.num_words,
    processing_ms: r.processing_ms,
    annotated_image_url: r.annotated_image_url,
    reviewed: false,
  };
}

function viewFromSaved(d: SavedDocument): DocView {
  return {
    id: d.id,
    filename: d.filename,
    receipt: d.receipt,
    corrected: d.corrected_receipt,
    num_words: d.num_words ?? 0,
    processing_ms: d.processing_ms ?? 0,
    annotated_image_url: d.annotated_filename ? `/files/annotated/${d.annotated_filename}` : null,
    reviewed: d.reviewed,
  };
}

function DocumentsPage() {
  const qc = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [active, setActive] = useState<DocView | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [baseline, setBaseline] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [page, setPage] = useState(1);

  useEffect(() => setMounted(true), []);
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  // Reset the editable draft whenever the active document changes.
  useEffect(() => {
    if (active) {
      const d = toDraft(active.corrected ?? active.receipt);
      setDraft(d);
      setBaseline(JSON.stringify(d));
    } else {
      setDraft(null);
      setBaseline("");
    }
  }, [active]);

  const health = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    enabled: mounted,
    refetchInterval: 15000,
    retry: false,
  });

  const documents = useQuery({
    queryKey: ["documents"],
    queryFn: listDocuments,
    enabled: mounted,
  });

  const stats = useQuery({
    queryKey: ["stats"],
    queryFn: getStats,
    enabled: mounted,
  });

  const extract = useMutation({
    mutationFn: (file: File) => extractReceipt(file),
    onSuccess: (data) => {
      setActive(viewFromExtract(data));
      qc.invalidateQueries({ queryKey: ["documents"] });
      qc.invalidateQueries({ queryKey: ["stats"] });
    },
  });

  const save = useMutation({
    mutationFn: () => saveCorrection(active!.id, fromDraft(draft!)),
    onSuccess: (doc) => {
      setActive(viewFromSaved(doc));
      qc.invalidateQueries({ queryKey: ["documents"] });
      qc.invalidateQueries({ queryKey: ["stats"] });
    },
  });

  function handleFile(file: File | undefined) {
    if (!file) return;
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(file);
    });
    extract.mutate(file);
  }

  const setItem = (i: number, key: keyof DraftItem, value: string) =>
    setDraft((d) => {
      if (!d) return d;
      const items = d.items.map((it, j) => (j === i ? { ...it, [key]: value } : it));
      return { ...d, items };
    });
  const setSummary = (key: keyof Draft, value: string) =>
    setDraft((d) => (d ? { ...d, [key]: value } : d));

  const base = active?.receipt; // original model output — carries confidence
  const items = draft?.items ?? [];
  const displayImage = resolveApiUrl(active?.annotated_image_url) ?? previewUrl ?? invoiceScan;
  const dirty = draft != null && JSON.stringify(draft) !== baseline;

  const status: "idle" | "pending" | "success" | "error" = extract.isPending
    ? "pending"
    : extract.isError
      ? "error"
      : active
        ? "success"
        : "idle";

  const totalQty = items.reduce((sum, it) => sum + (num(it.quantity) ?? 0), 0);
  const savedDocs = documents.data ?? [];

  // Client-side pagination over the saved documents list.
  const pageCount = Math.max(1, Math.ceil(savedDocs.length / PAGE_SIZE));
  const safePage = Math.min(Math.max(1, page), pageCount);
  const pageStart = (safePage - 1) * PAGE_SIZE;
  const pageDocs = savedDocs.slice(pageStart, pageStart + PAGE_SIZE);

  return (
    <main className="max-w-7xl mx-auto px-6 py-8">
      {/* Header / Stats */}
      <header className="mb-8">
        <h1 className="text-2xl font-semibold text-foreground tracking-tight leading-tight text-balance mb-1">
          Document Processing Pipeline
        </h1>
        <p className="text-sm text-muted-foreground mb-6">
          OCR-free receipt understanding. Donut (image → structured fields) → SQLite.
        </p>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: "Saved Documents", value: String(savedDocs.length) },
            {
              label: "Line Items",
              value: active ? String(items.length) : "—",
              mono: true,
            },
            {
              label: "Last Latency",
              value: active ? `${(active.processing_ms / 1000).toFixed(1)}s` : "—",
              mono: true,
            },
          ].map((stat) => (
            <div key={stat.label} className="p-4 bg-panel rounded-xl ring-1 ring-border">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">
                {stat.label}
              </p>
              <p className={`text-2xl font-medium text-foreground ${stat.mono ? "font-mono" : ""}`}>
                {stat.value}
              </p>
            </div>
          ))}
          <div className="p-4 bg-panel rounded-xl ring-1 ring-border">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">
              Model Status
            </p>
            {health.data?.model_loaded ? (
              <div className="flex items-center gap-2 text-success mt-1">
                <div className="size-2 bg-success rounded-full animate-pulse" />
                <span className="text-sm font-medium">Operational</span>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-destructive mt-1">
                <div className="size-2 bg-destructive rounded-full" />
                <span className="text-sm font-medium">
                  {health.isError ? "Offline" : "Loading…"}
                </span>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main Editor Surface */}
      <div className="grid grid-cols-12 gap-6 items-start">
        {/* Document Scan View */}
        <div className="col-span-12 lg:col-span-7 bg-panel rounded-2xl ring-1 ring-border overflow-hidden flex flex-col min-h-[720px]">
          <div className="px-4 py-3 border-b border-border flex items-center justify-between">
            <span className="text-xs font-medium font-mono text-muted-foreground uppercase truncate">
              {active?.filename ?? "No document loaded"}
            </span>
            <div className="flex gap-2">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                className="hidden"
                onChange={(e) => handleFile(e.target.files?.[0])}
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={extract.isPending}
                className="px-3 py-1 rounded bg-brand text-brand-foreground text-[11px] font-semibold uppercase tracking-wide hover:opacity-90 transition-opacity disabled:opacity-50"
              >
                Upload Receipt
              </button>
            </div>
          </div>

          <div
            className={`relative flex-1 p-8 bg-surface/60 flex items-center justify-center transition-colors ${
              dragOver ? "ring-2 ring-inset ring-brand bg-brand/5" : ""
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              handleFile(e.dataTransfer.files?.[0]);
            }}
          >
            {!active && !previewUrl ? (
              <button
                onClick={() => fileInputRef.current?.click()}
                className="flex flex-col items-center gap-3 text-center px-10 py-16 rounded-xl border-2 border-dashed border-border hover:border-brand transition-colors"
              >
                <div className="size-12 rounded-full bg-surface flex items-center justify-center text-2xl text-muted-foreground">
                  ↑
                </div>
                <span className="text-sm font-medium text-foreground">
                  Drop a receipt here, or click to upload
                </span>
                <span className="text-xs text-muted-foreground">JPG, PNG or WebP · max 10 MB</span>
              </button>
            ) : (
              <div className="relative w-full max-w-md aspect-[1/1.7] bg-panel ring-1 ring-border overflow-hidden shadow-2xl rounded-sm">
                <img
                  src={displayImage}
                  alt={active?.filename ?? "Uploaded receipt"}
                  className="absolute inset-0 w-full h-full object-contain bg-white"
                />
                {extract.isPending && (
                  <div className="absolute inset-0 bg-ink/40 backdrop-blur-[1px] flex items-center justify-center">
                    <span className="text-xs font-mono text-panel uppercase tracking-widest animate-pulse">
                      Running Donut…
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Pipeline Steps */}
          <div className="p-4 border-t border-border bg-panel">
            <div className="flex flex-wrap gap-4">
              {[
                { label: "Upload", done: status !== "idle" },
                { label: "Donut Encode", done: status === "success", active: status === "pending" },
                {
                  label: "Decode Fields",
                  done: status === "success",
                  active: status === "pending",
                },
                { label: "Structured Output", done: status === "success" },
              ].map((step) => (
                <div key={step.label} className="flex flex-col gap-1">
                  <span className="text-[10px] font-bold text-subtle uppercase tracking-tight">
                    {step.label}
                  </span>
                  <div
                    className={`h-1 w-28 rounded-full ${
                      step.done ? "bg-brand" : step.active ? "bg-brand animate-pulse" : "bg-surface"
                    }`}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Extraction Panel */}
        <div className="col-span-12 lg:col-span-5 flex flex-col gap-6">
          <section className="bg-panel rounded-2xl ring-1 ring-border overflow-hidden">
            <div className="px-6 py-4 border-b border-border flex items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-semibold text-foreground">Structured Data</h2>
                {active?.reviewed && (
                  <span className="px-2 py-0.5 rounded-full bg-success/15 text-success text-[10px] font-bold uppercase tracking-wide">
                    Reviewed
                  </span>
                )}
              </div>
              {active && (
                <div className="flex items-center gap-3">
                  {base?.overall_confidence != null && (
                    <ConfidenceBar value={base.overall_confidence} label="conf" />
                  )}
                  <span className="text-[10px] font-mono text-subtle uppercase tracking-wider whitespace-nowrap">
                    {items.length} items · {(active.processing_ms / 1000).toFixed(1)}s
                  </span>
                </div>
              )}
            </div>

            {extract.isError ? (
              <div className="p-6 text-sm text-destructive">
                {(extract.error as Error)?.message ?? "Extraction failed."}
              </div>
            ) : !active || !draft ? (
              <div className="p-6 text-sm text-muted-foreground">
                Upload a receipt to see extracted line items and totals. Fields are editable —
                correct any mistakes and save them back to the database.
              </div>
            ) : (
              <div className="p-6 space-y-5">
                {/* Document block */}
                <div>
                  <p className="text-[10px] font-bold text-subtle uppercase tracking-widest mb-2">
                    Document
                  </p>
                  <ReadField label="File Name" value={active.filename} mono />
                </div>

                {/* Line Items */}
                <div className="pt-2 border-t border-border">
                  <p className="text-[10px] font-bold text-subtle uppercase tracking-widest mb-2 mt-3">
                    Line Items
                  </p>
                  {items.length === 0 ? (
                    <p className="text-xs text-muted-foreground py-2">No line items detected.</p>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-sm">
                        <thead>
                          <tr className="text-[10px] font-bold text-subtle uppercase tracking-wider border-b border-border">
                            <th className="py-2 pr-2">Item</th>
                            <th className="py-2 px-1 text-right">Qty</th>
                            <th className="py-2 px-1 text-right">Unit</th>
                            <th className="py-2 px-1 text-right">Total</th>
                            <th className="py-2 text-right">Conf</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border/40">
                          {items.map((it, i) => {
                            const conf = meanConfidence(base?.line_items?.[i]?.confidence);
                            return (
                              <tr key={i}>
                                <td className="py-1 pr-2">
                                  <EditInput
                                    value={it.name}
                                    onChange={(v) => setItem(i, "name", v)}
                                    placeholder="item name"
                                  />
                                </td>
                                <td className="py-1 px-1 w-14">
                                  <EditInput
                                    value={it.quantity}
                                    onChange={(v) => setItem(i, "quantity", v)}
                                    align="right"
                                    mono
                                  />
                                </td>
                                <td className="py-1 px-1 w-20">
                                  <EditInput
                                    value={it.unit_price}
                                    onChange={(v) => setItem(i, "unit_price", v)}
                                    align="right"
                                    mono
                                  />
                                </td>
                                <td className="py-1 px-1 w-20">
                                  <EditInput
                                    value={it.price}
                                    onChange={(v) => setItem(i, "price", v)}
                                    align="right"
                                    mono
                                  />
                                </td>
                                <td className="py-1 text-right">
                                  {conf != null ? (
                                    <ConfidenceBar value={conf} compact />
                                  ) : (
                                    <span className="text-subtle">—</span>
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>

                {/* Totals */}
                <div className="pt-3 border-t border-border space-y-1 text-sm">
                  <ReadTotalRow label="Items" value={String(items.length)} />
                  <ReadTotalRow label="Total Qty" value={fmt(totalQty)} />
                  <EditTotalRow
                    label="Subtotal"
                    value={draft.subtotal}
                    onChange={(v) => setSummary("subtotal", v)}
                    conf={base?.field_confidence?.subtotal}
                  />
                  {draft.discount !== "" && (
                    <EditTotalRow
                      label="Discount"
                      value={draft.discount}
                      onChange={(v) => setSummary("discount", v)}
                      conf={base?.field_confidence?.discount}
                    />
                  )}
                  {draft.service_charge !== "" && (
                    <EditTotalRow
                      label="Service Charge"
                      value={draft.service_charge}
                      onChange={(v) => setSummary("service_charge", v)}
                      conf={base?.field_confidence?.service_charge}
                    />
                  )}
                  <EditTotalRow
                    label="Tax"
                    value={draft.tax}
                    onChange={(v) => setSummary("tax", v)}
                    conf={base?.field_confidence?.tax}
                  />
                  <div className="flex justify-between items-center pt-2 mt-2 border-t border-border gap-3">
                    <span className="text-sm font-semibold text-foreground">Grand Total</span>
                    <div className="flex items-center gap-3">
                      {base?.field_confidence?.total != null && (
                        <ConfidenceBar value={base.field_confidence.total} compact />
                      )}
                      <input
                        value={draft.total}
                        onChange={(e) => setSummary("total", e.target.value)}
                        className="w-28 bg-secondary ring-1 ring-border rounded-md px-2 py-1 text-right text-lg font-medium font-mono tracking-tight text-foreground focus:ring-brand focus:ring-2 outline-none"
                      />
                    </div>
                  </div>
                  {draft.cash_paid !== "" && (
                    <EditTotalRow
                      label="Cash Tendered"
                      value={draft.cash_paid}
                      onChange={(v) => setSummary("cash_paid", v)}
                    />
                  )}
                  {draft.change !== "" && (
                    <EditTotalRow
                      label="Change Due"
                      value={draft.change}
                      onChange={(v) => setSummary("change", v)}
                    />
                  )}
                  {draft.credit_card !== "" && (
                    <EditTotalRow
                      label="Credit Card"
                      value={draft.credit_card}
                      onChange={(v) => setSummary("credit_card", v)}
                    />
                  )}
                </div>

                {save.isError && (
                  <p className="text-xs text-destructive">
                    {(save.error as Error)?.message ?? "Save failed."}
                  </p>
                )}
              </div>
            )}

            <div className="px-6 py-4 bg-secondary flex items-center justify-between gap-3">
              <div className="flex items-center gap-4">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  disabled={extract.isPending}
                  className="text-xs font-semibold text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
                >
                  {extract.isPending ? "Processing…" : "Upload Another"}
                </button>
                <button
                  disabled={!active || !draft}
                  onClick={() => {
                    if (!active || !draft) return;
                    const payload = { filename: active.filename, receipt: fromDraft(draft) };
                    const blob = new Blob([JSON.stringify(payload, null, 2)], {
                      type: "application/json",
                    });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = `${active.filename.replace(/\.[^.]+$/, "")}.json`;
                    a.click();
                    URL.revokeObjectURL(url);
                  }}
                  className="text-xs font-semibold text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
                >
                  Export JSON
                </button>
              </div>
              <button
                disabled={!active || !dirty || save.isPending}
                onClick={() => save.mutate()}
                className="px-4 py-2 bg-brand text-brand-foreground text-xs font-semibold rounded-md shadow-sm hover:opacity-90 transition-opacity ring-1 ring-brand disabled:opacity-50"
              >
                {save.isPending
                  ? "Saving…"
                  : dirty
                    ? "Save Corrections"
                    : active?.reviewed
                      ? "Saved ✓"
                      : "Save to Database"}
              </button>
            </div>
          </section>
        </div>
      </div>

      {/* Active Learning */}
      <section className="mt-12">
        <div className="flex items-end justify-between mb-4 flex-wrap gap-3">
          <div>
            <h2 className="text-lg font-semibold text-foreground tracking-tight">
              Active Learning
            </h2>
            <p className="text-sm text-muted-foreground">
              Every correction becomes a labelled (image, fields) pair. Export them as a CORD
              fine-tuning set to retrain Donut.
            </p>
          </div>
          <a
            href={correctionsExportUrl}
            className={`px-4 py-2 rounded-md text-xs font-semibold ring-1 ring-brand transition-opacity ${
              (stats.data?.trainable ?? 0) > 0
                ? "bg-brand text-brand-foreground hover:opacity-90"
                : "bg-secondary text-muted-foreground pointer-events-none opacity-50"
            }`}
          >
            Export retraining set
            {(stats.data?.trainable ?? 0) > 0 ? ` (${stats.data?.trainable})` : ""}
          </a>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: "Documents", value: stats.data?.documents },
            { label: "Reviewed", value: stats.data?.reviewed },
            { label: "Field Corrections", value: stats.data?.corrections },
            { label: "Trainable Pairs", value: stats.data?.trainable, accent: true },
          ].map((c) => (
            <div key={c.label} className="p-4 bg-panel rounded-xl ring-1 ring-border">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">
                {c.label}
              </p>
              <p
                className={`text-2xl font-medium font-mono ${
                  c.accent ? "text-brand" : "text-foreground"
                }`}
              >
                {c.value ?? "—"}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Saved Documents Table */}
      <section className="mt-12">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-foreground tracking-tight">Saved Documents</h2>
          {documents.isFetching && (
            <span className="text-xs text-muted-foreground">Refreshing…</span>
          )}
        </div>
        <div className="bg-panel rounded-2xl ring-1 ring-border overflow-hidden shadow-sm">
          <table className="w-full text-left">
            <thead>
              <tr className="text-xs font-bold text-subtle uppercase tracking-wider border-b border-border">
                <th className="px-6 py-4">Document</th>
                <th className="px-6 py-4 text-right">Items</th>
                <th className="px-6 py-4 text-right">Total</th>
                <th className="px-6 py-4 text-right">Confidence</th>
                <th className="px-6 py-4 text-center">Status</th>
                <th className="px-6 py-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {savedDocs.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-6 py-8 text-sm text-center text-muted-foreground">
                    No documents saved yet. Upload a receipt to get started.
                  </td>
                </tr>
              ) : (
                pageDocs.map((d) => {
                  const r = d.corrected_receipt ?? d.receipt;
                  return (
                    <tr
                      key={d.id}
                      onClick={() => setActive(viewFromSaved(d))}
                      className={`hover:bg-secondary/60 transition-colors cursor-pointer group ${
                        active?.id === d.id ? "bg-secondary/40" : ""
                      }`}
                    >
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <div className="size-8 bg-surface rounded flex items-center justify-center font-mono text-[10px] text-muted-foreground uppercase">
                            {d.filename.split(".").pop()?.slice(0, 3)}
                          </div>
                          <span className="text-sm font-medium text-foreground group-hover:text-brand truncate max-w-[220px]">
                            {d.filename}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-right text-sm font-mono text-muted-foreground">
                        {r.line_items?.length ?? 0}
                      </td>
                      <td className="px-6 py-4 text-right text-sm font-mono text-foreground">
                        {fmt(r.total)}
                      </td>
                      <td className="px-6 py-4">
                        {d.overall_confidence != null ? (
                          <ConfidenceBar value={d.overall_confidence} />
                        ) : (
                          <span className="text-sm text-subtle block text-right">—</span>
                        )}
                      </td>
                      <td className="px-6 py-4 text-center">
                        {d.reviewed ? (
                          <span className="px-2 py-0.5 rounded-full bg-success/15 text-success text-[10px] font-bold uppercase tracking-wide">
                            Reviewed
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 rounded-full bg-surface text-subtle text-[10px] font-bold uppercase tracking-wide">
                            Extracted
                          </span>
                        )}
                      </td>
                      <td className="px-6 py-4 text-right">
                        <button className="text-sm font-medium text-subtle hover:text-foreground">
                          {d.reviewed ? "Review" : "Correct"}
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>

          {savedDocs.length > PAGE_SIZE && (
            <div className="flex items-center justify-between gap-3 px-6 py-3 border-t border-border">
              <span className="text-xs text-muted-foreground font-mono">
                {pageStart + 1}–{Math.min(pageStart + PAGE_SIZE, savedDocs.length)} of{" "}
                {savedDocs.length}
              </span>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setPage(safePage - 1)}
                  disabled={safePage <= 1}
                  className="px-3 py-1 rounded-md text-xs font-semibold ring-1 ring-border text-foreground hover:bg-secondary transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Prev
                </button>
                <span className="text-xs text-muted-foreground font-mono">
                  Page {safePage} / {pageCount}
                </span>
                <button
                  onClick={() => setPage(safePage + 1)}
                  disabled={safePage >= pageCount}
                  className="px-3 py-1 rounded-md text-xs font-semibold ring-1 ring-border text-foreground hover:bg-secondary transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </div>
      </section>
    </main>
  );
}

function EditInput({
  value,
  onChange,
  align = "left",
  mono,
  placeholder,
}: {
  value: string;
  onChange: (v: string) => void;
  align?: "left" | "right";
  mono?: boolean;
  placeholder?: string;
}) {
  return (
    <input
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
      className={`w-full bg-transparent rounded px-1.5 py-1 text-sm text-foreground hover:bg-surface focus:bg-secondary focus:ring-1 focus:ring-brand outline-none ${
        mono ? "font-mono" : "font-medium"
      } ${align === "right" ? "text-right" : ""}`}
    />
  );
}

function EditTotalRow({
  label,
  value,
  onChange,
  conf,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  conf?: number | null;
}) {
  return (
    <div className="flex justify-between items-center gap-3">
      <span className="text-muted-foreground">{label}</span>
      <div className="flex items-center gap-2">
        {conf != null && <ConfidenceBar value={conf} compact />}
        <input
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-28 bg-secondary ring-1 ring-border rounded-md px-2 py-1 text-right font-mono text-sm text-foreground focus:ring-brand focus:ring-2 outline-none"
        />
      </div>
    </div>
  );
}

/** Small bar visualizing a model-confidence value in [0, 1]. */
function ConfidenceBar({
  value,
  label,
  compact,
}: {
  value: number;
  label?: string;
  compact?: boolean;
}) {
  const pct = Math.round(value * 100);
  const color = pct >= 90 ? "bg-success" : pct >= 70 ? "bg-warning" : "bg-destructive";
  const text = pct >= 90 ? "text-success" : pct >= 70 ? "text-warning" : "text-destructive";
  return (
    <div className="flex items-center justify-end gap-1.5" title={`Model confidence: ${pct}%`}>
      <div className={`${compact ? "w-10" : "w-14"} h-1.5 bg-surface rounded-full overflow-hidden`}>
        <div className={`h-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className={`text-[10px] font-mono ${text}`}>
        {pct}%{label ? ` ${label}` : ""}
      </span>
    </div>
  );
}

function ReadTotalRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-center py-1">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono text-foreground">{value}</span>
    </div>
  );
}

function ReadField({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <label className="block text-[11px] font-bold text-subtle uppercase mb-1.5">{label}</label>
      <input
        type="text"
        readOnly
        value={value}
        className={`w-full bg-secondary border-none ring-1 ring-border rounded-md px-3 py-2 text-sm text-foreground focus:ring-brand focus:ring-2 outline-none ${
          mono ? "font-mono" : "font-medium"
        }`}
      />
    </div>
  );
}
