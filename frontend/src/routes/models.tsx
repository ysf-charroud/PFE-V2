import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getMetrics, type LabelMetric } from "@/lib/api";

export const Route = createFileRoute("/models")({
  head: () => ({
    meta: [
      { title: "Models — Vector IDP" },
      {
        name: "description",
        content: "Configure the OCR and document understanding models powering the IDP pipeline.",
      },
    ],
  }),
  component: ModelsPage,
});

const models = [
  {
    name: "OpenCV",
    role: "Pre-processing",
    version: "4.10.0",
    desc: "Deskew, denoise, grayscale and adaptive thresholding to maximize OCR accuracy.",
    status: "Active",
  },
  {
    name: "EasyOCR",
    role: "Optical Character Recognition",
    version: "1.7.2",
    desc: "Detects and recognizes printed English text. Outputs words with bounding boxes and confidence scores.",
    status: "Active",
  },
  {
    name: "LayoutLM",
    role: "Document Understanding",
    version: "v3-base",
    desc: "Joint text + layout transformer for entity extraction (vendor, date, totals, line items).",
    status: "Active",
  },
  {
    name: "SQLite",
    role: "Local Persistence",
    version: "3.45",
    desc: "Stores normalized extraction results. Fully on-device, no external service.",
    status: "Active",
  },
];

const pct = (n?: number) => (n == null ? "—" : `${(n * 100).toFixed(1)}%`);

// F1 → color (matches the confidence-bar thresholds used elsewhere).
const f1Hex = (f1: number) => (f1 >= 0.9 ? "#22c55e" : f1 >= 0.7 ? "#f59e0b" : "#ef4444");
const f1Text = (f1: number) =>
  f1 >= 0.9 ? "text-success" : f1 >= 0.7 ? "text-warning" : "text-destructive";

function ModelsPage() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const metrics = useQuery({
    queryKey: ["metrics"],
    queryFn: getMetrics,
    enabled: mounted,
    retry: false,
  });

  const m = metrics.data;
  const perLabel = m?.per_label ?? [];
  const chartData = [...perLabel]
    .sort((a, b) => b.f1 - a.f1)
    .map((l) => ({ label: l.label.replace(/^menu\./, "m."), f1: l.f1, full: l }));

  return (
    <main className="max-w-7xl mx-auto px-6 py-8">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight mb-1">Model Configuration</h1>
        <p className="text-sm text-muted-foreground">
          Hybrid pipeline combining Computer Vision and Natural Language Processing. Entirely local
          and Open Source.
        </p>
      </header>

      {/* Evaluation dashboard */}
      <section className="mb-12">
        <div className="flex items-end justify-between mb-4 flex-wrap gap-2">
          <div>
            <h2 className="text-lg font-semibold text-foreground tracking-tight">
              LayoutLMv3 Evaluation
            </h2>
            <p className="text-sm text-muted-foreground">
              Token-level metrics on the held-out{" "}
              <span className="font-mono">{m?.dataset ?? "CORD v2"}</span> {m?.split ?? "test"}{" "}
              split.
            </p>
          </div>
          {m?.generated_at && (
            <span className="text-[11px] font-mono text-subtle">
              {m.num_receipts} receipts · {m.num_tokens} tokens · evaluated{" "}
              {new Date(m.generated_at).toLocaleString()}
            </span>
          )}
        </div>

        {metrics.isLoading ? (
          <div className="bg-panel rounded-2xl ring-1 ring-border p-8 text-sm text-muted-foreground">
            Loading evaluation metrics…
          </div>
        ) : metrics.isError ? (
          <div className="bg-panel rounded-2xl ring-1 ring-border p-8 text-sm text-destructive">
            Could not reach the metrics endpoint. Is the backend running?
          </div>
        ) : !m?.available ? (
          <div className="bg-panel rounded-2xl ring-1 ring-border p-8 text-sm text-muted-foreground">
            No metrics yet. Generate them by running{" "}
            <code className="font-mono text-foreground">evaluate.py</code> in the inference sidecar.
          </div>
        ) : (
          <>
            {/* Overall metric cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              {[
                { label: "Accuracy", value: m.overall?.accuracy, hint: "token-level" },
                { label: "Macro F1", value: m.overall?.macro.f1, hint: "all classes equal" },
                { label: "Weighted F1", value: m.overall?.weighted.f1, hint: "by frequency" },
                { label: "Micro F1", value: m.overall?.micro.f1, hint: "global" },
              ].map((s) => (
                <div key={s.label} className="p-4 bg-panel rounded-xl ring-1 ring-border">
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">
                    {s.label}
                  </p>
                  <p className="text-2xl font-medium text-foreground font-mono">{pct(s.value)}</p>
                  <p className="text-[10px] text-subtle uppercase tracking-wide mt-0.5">{s.hint}</p>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-12 gap-6 items-start">
              {/* Per-label F1 chart */}
              <div className="col-span-12 lg:col-span-7 bg-panel rounded-2xl ring-1 ring-border p-5">
                <h3 className="text-sm font-semibold text-foreground mb-4">Per-label F1 score</h3>
                <ResponsiveContainer width="100%" height={Math.max(360, chartData.length * 22)}>
                  <BarChart
                    data={chartData}
                    layout="vertical"
                    margin={{ top: 0, right: 16, bottom: 0, left: 8 }}
                  >
                    <XAxis
                      type="number"
                      domain={[0, 1]}
                      tickFormatter={(v) => `${Math.round(v * 100)}%`}
                      tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
                    />
                    <YAxis
                      type="category"
                      dataKey="label"
                      width={150}
                      tick={{
                        fontSize: 10,
                        fill: "var(--muted-foreground)",
                        fontFamily: "monospace",
                      }}
                    />
                    <Tooltip
                      cursor={{ fill: "var(--secondary)" }}
                      formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, "F1"]}
                      contentStyle={{
                        background: "var(--panel)",
                        border: "1px solid var(--border)",
                        borderRadius: 8,
                        fontSize: 12,
                      }}
                    />
                    <Bar dataKey="f1" radius={[0, 3, 3, 0]} barSize={12}>
                      {chartData.map((d) => (
                        <Cell key={d.label} fill={f1Hex(d.f1)} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <p className="text-[11px] text-subtle mt-3">
                  Green ≥ 90% · amber ≥ 70% · red &lt; 70%. The long tail of rare classes (low
                  support) is why macro-F1 ({pct(m.overall?.macro.f1)}) sits well below weighted-F1
                  ({pct(m.overall?.weighted.f1)}).
                </p>
              </div>

              {/* Per-label table */}
              <div className="col-span-12 lg:col-span-5 bg-panel rounded-2xl ring-1 ring-border overflow-hidden">
                <div className="px-5 py-4 border-b border-border">
                  <h3 className="text-sm font-semibold text-foreground">Per-label breakdown</h3>
                </div>
                <div className="max-h-[520px] overflow-y-auto">
                  <table className="w-full text-left text-sm">
                    <thead className="sticky top-0 bg-panel">
                      <tr className="text-[10px] font-bold text-subtle uppercase tracking-wider border-b border-border">
                        <th className="px-5 py-2">Label</th>
                        <th className="px-2 py-2 text-right">P</th>
                        <th className="px-2 py-2 text-right">R</th>
                        <th className="px-2 py-2 text-right">F1</th>
                        <th className="px-5 py-2 text-right">n</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border/40">
                      {perLabel.map((l: LabelMetric) => (
                        <tr key={l.label}>
                          <td className="px-5 py-2 font-mono text-xs text-foreground">{l.label}</td>
                          <td className="px-2 py-2 text-right font-mono text-muted-foreground">
                            {l.precision.toFixed(2)}
                          </td>
                          <td className="px-2 py-2 text-right font-mono text-muted-foreground">
                            {l.recall.toFixed(2)}
                          </td>
                          <td
                            className={`px-2 py-2 text-right font-mono font-medium ${f1Text(l.f1)}`}
                          >
                            {l.f1.toFixed(2)}
                          </td>
                          <td className="px-5 py-2 text-right font-mono text-subtle">
                            {l.support}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </>
        )}
      </section>

      {/* Pipeline components */}
      <h2 className="text-lg font-semibold text-foreground tracking-tight mb-4">
        Pipeline Components
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {models.map((mdl) => (
          <article
            key={mdl.name}
            className="bg-panel rounded-2xl ring-1 ring-border p-6 flex flex-col gap-3"
          >
            <div className="flex items-start justify-between">
              <div>
                <h2 className="text-base font-semibold text-foreground">{mdl.name}</h2>
                <p className="text-[11px] font-bold text-subtle uppercase tracking-wider mt-0.5">
                  {mdl.role}
                </p>
              </div>
              <div className="flex items-center gap-2 text-success">
                <span className="size-2 bg-success rounded-full" />
                <span className="text-xs font-medium">{mdl.status}</span>
              </div>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">{mdl.desc}</p>
            <div className="pt-3 mt-auto border-t border-border flex items-center justify-between text-xs font-mono text-subtle">
              <span>v{mdl.version}</span>
            </div>
          </article>
        ))}
      </div>
    </main>
  );
}
