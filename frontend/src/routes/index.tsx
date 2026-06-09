import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import invoiceScan from "@/assets/invoice-scan.jpg";
import { extractReceipt, getHealth, resolveApiUrl, type ExtractResponse } from "@/lib/api";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Documents — Vector IDP" },
      {
        name: "description",
        content:
          "Process invoices and receipts with OpenCV, EasyOCR and LayoutLM. View extracted fields and commit to local SQLite.",
      },
    ],
  }),
  component: DocumentsPage,
});

const fmt = (n?: number | null) =>
  n == null ? "—" : n.toLocaleString(undefined, { maximumFractionDigits: 2 });

function DocumentsPage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [active, setActive] = useState<ExtractResponse | null>(null);
  const [history, setHistory] = useState<ExtractResponse[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);
  // Revoke the last object URL when it changes or on unmount.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const health = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    enabled: mounted,
    refetchInterval: 15000,
    retry: false,
  });

  const extract = useMutation({
    mutationFn: (file: File) => extractReceipt(file),
    onSuccess: (data) => {
      setActive(data);
      setHistory((h) => [data, ...h].slice(0, 8));
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

  const receipt = active?.receipt;
  const lineItems = receipt?.line_items ?? [];
  const displayImage = resolveApiUrl(active?.annotated_image_url) ?? previewUrl ?? invoiceScan;

  const status: "idle" | "pending" | "success" | "error" = extract.isPending
    ? "pending"
    : extract.isError
      ? "error"
      : active
        ? "success"
        : "idle";

  const totalQty = lineItems.reduce((sum, it) => sum + (it.quantity ?? 0), 0);

  return (
    <main className="max-w-7xl mx-auto px-6 py-8">
      {/* Header / Stats */}
      <header className="mb-8">
        <h1 className="text-2xl font-semibold text-foreground tracking-tight leading-tight text-balance mb-1">
          Document Processing Pipeline
        </h1>
        <p className="text-sm text-muted-foreground mb-6">
          Hybrid Computer Vision + NLP pipeline. OpenCV → EasyOCR → LayoutLM → SQLite.
        </p>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: "Documents This Session", value: String(history.length) },
            {
              label: "Words Detected",
              value: active ? String(active.num_words) : "—",
              mono: true,
            },
            {
              label: "Last Latency",
              value: active ? `${(active.processing_ms / 1000).toFixed(1)}s` : "—",
              mono: true,
            },
          ].map((s) => (
            <div key={s.label} className="p-4 bg-panel rounded-xl ring-1 ring-border">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">
                {s.label}
              </p>
              <p className={`text-2xl font-medium text-foreground ${s.mono ? "font-mono" : ""}`}>
                {s.value}
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
                      Running OCR + LayoutLM…
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
                {
                  label: "EasyOCR Scan",
                  done: status === "success",
                  active: status === "pending",
                },
                {
                  label: "LayoutLM Analysis",
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
            <div className="px-6 py-4 border-b border-border flex items-center justify-between">
              <h2 className="text-sm font-semibold text-foreground">Structured Data</h2>
              {active && (
                <span className="text-[10px] font-mono text-subtle uppercase tracking-wider">
                  {active.num_words} words · {(active.processing_ms / 1000).toFixed(1)}s
                </span>
              )}
            </div>

            {extract.isError ? (
              <div className="p-6 text-sm text-destructive">
                {(extract.error as Error)?.message ?? "Extraction failed."}
              </div>
            ) : !active ? (
              <div className="p-6 text-sm text-muted-foreground">
                Upload a receipt to see extracted line items and totals.
              </div>
            ) : (
              <div className="p-6 space-y-5">
                {/* Document block */}
                <div>
                  <p className="text-[10px] font-bold text-subtle uppercase tracking-widest mb-2">
                    Document
                  </p>
                  <Field label="File Name" value={active.filename} mono />
                </div>

                {/* Line Items */}
                <div className="pt-2 border-t border-border">
                  <p className="text-[10px] font-bold text-subtle uppercase tracking-widest mb-2 mt-3">
                    Line Items
                  </p>
                  {lineItems.length === 0 ? (
                    <p className="text-xs text-muted-foreground py-2">No line items detected.</p>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-sm">
                        <thead>
                          <tr className="text-[10px] font-bold text-subtle uppercase tracking-wider border-b border-border">
                            <th className="py-2 pr-3">Item</th>
                            <th className="py-2 pr-3 text-right">Qty</th>
                            <th className="py-2 pr-3 text-right">Unit</th>
                            <th className="py-2 text-right">Total</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border/40">
                          {lineItems.map((item, i) => (
                            <tr key={i}>
                              <td className="py-2 pr-3 text-foreground font-medium">
                                {[item.name, item.sub_name].filter(Boolean).join(" ") || "—"}
                              </td>
                              <td className="py-2 pr-3 text-right font-mono text-muted-foreground">
                                {fmt(item.quantity)}
                              </td>
                              <td className="py-2 pr-3 text-right font-mono text-muted-foreground">
                                {fmt(item.unit_price)}
                              </td>
                              <td className="py-2 text-right font-mono text-foreground">
                                {fmt(item.price)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>

                {/* Totals */}
                <div className="pt-3 border-t border-border space-y-1.5 text-sm">
                  <TotalRow label="Items" value={String(lineItems.length)} />
                  <TotalRow label="Total Qty" value={fmt(totalQty)} />
                  <TotalRow label="Subtotal" value={fmt(receipt?.subtotal)} />
                  {receipt?.discount != null && (
                    <TotalRow label="Discount" value={fmt(receipt.discount)} />
                  )}
                  {receipt?.service_charge != null && (
                    <TotalRow label="Service Charge" value={fmt(receipt.service_charge)} />
                  )}
                  <TotalRow label="Tax" value={fmt(receipt?.tax)} />
                  <div className="flex justify-between items-center pt-2 mt-2 border-t border-border">
                    <span className="text-sm font-semibold text-foreground">Grand Total</span>
                    <span className="text-lg font-medium text-foreground font-mono tracking-tight">
                      {fmt(receipt?.total)}
                    </span>
                  </div>
                  {receipt?.cash_paid != null && (
                    <TotalRow label="Cash Tendered" value={fmt(receipt.cash_paid)} />
                  )}
                  {receipt?.change != null && (
                    <TotalRow label="Change Due" value={fmt(receipt.change)} />
                  )}
                  {receipt?.credit_card != null && (
                    <TotalRow label="Credit Card" value={fmt(receipt.credit_card)} />
                  )}
                </div>
              </div>
            )}

            <div className="px-6 py-4 bg-secondary flex items-center justify-between">
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={extract.isPending}
                className="text-xs font-semibold text-muted-foreground hover:text-foreground transition-colors disabled:opacity-50"
              >
                {extract.isPending ? "Processing…" : "Upload Another"}
              </button>
              <button
                disabled={!active}
                onClick={() => {
                  if (!active) return;
                  const blob = new Blob([JSON.stringify(active, null, 2)], {
                    type: "application/json",
                  });
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement("a");
                  a.href = url;
                  a.download = `${active.filename.replace(/\.[^.]+$/, "")}.json`;
                  a.click();
                  URL.revokeObjectURL(url);
                }}
                className="px-4 py-2 bg-brand text-brand-foreground text-xs font-semibold rounded-md shadow-sm hover:opacity-90 transition-opacity ring-1 ring-brand disabled:opacity-50"
              >
                Export JSON
              </button>
            </div>
          </section>
        </div>
      </div>

      {/* Session Documents Table */}
      <section className="mt-12">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-foreground tracking-tight">
            Session Documents
          </h2>
        </div>
        <div className="bg-panel rounded-2xl ring-1 ring-border overflow-hidden shadow-sm">
          <table className="w-full text-left">
            <thead>
              <tr className="text-xs font-bold text-subtle uppercase tracking-wider border-b border-border">
                <th className="px-6 py-4">Document</th>
                <th className="px-6 py-4 text-right">Items</th>
                <th className="px-6 py-4 text-right">Total</th>
                <th className="px-6 py-4 text-right">Words</th>
                <th className="px-6 py-4 text-right">Latency</th>
                <th className="px-6 py-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60">
              {history.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-6 py-8 text-sm text-center text-muted-foreground">
                    No documents processed yet this session.
                  </td>
                </tr>
              ) : (
                history.map((d, i) => (
                  <tr
                    key={i}
                    onClick={() => setActive(d)}
                    className="hover:bg-secondary/60 transition-colors cursor-pointer group"
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
                      {d.receipt.line_items?.length ?? 0}
                    </td>
                    <td className="px-6 py-4 text-right text-sm font-mono text-foreground">
                      {fmt(d.receipt.total)}
                    </td>
                    <td className="px-6 py-4 text-right text-sm font-mono text-muted-foreground">
                      {d.num_words}
                    </td>
                    <td className="px-6 py-4 text-right text-sm font-mono text-muted-foreground">
                      {(d.processing_ms / 1000).toFixed(1)}s
                    </td>
                    <td className="px-6 py-4 text-right">
                      <button className="text-sm font-medium text-subtle hover:text-foreground">
                        View
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}

function TotalRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono text-foreground">{value}</span>
    </div>
  );
}

function Field({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
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
