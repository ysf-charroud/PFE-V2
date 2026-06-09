import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/logs")({
  head: () => ({
    meta: [
      { title: "Logs — Vector IDP" },
      { name: "description", content: "Pipeline execution logs from the local IDP engine." },
    ],
  }),
  component: LogsPage,
});

const logs = [
  { t: "14:22:01", level: "INFO", msg: "LayoutLM entity extraction completed (0.4s)" },
  { t: "14:22:01", level: "SUCCESS", msg: "Entities normalized to SQLite (doc_id=8829)" },
  { t: "14:21:58", level: "WARN", msg: "Low confidence on Date field (0.76)" },
  { t: "14:21:57", level: "INFO", msg: "EasyOCR detected 42 text regions" },
  { t: "14:21:55", level: "INFO", msg: "OpenCV pre-processing: deskew 1.2° applied" },
  { t: "14:21:54", level: "INFO", msg: "Uploaded INV_8829_SUPPLY.PDF (412 KB)" },
  { t: "14:18:11", level: "SUCCESS", msg: "Committed doc_id=8828 (4 line items)" },
  { t: "14:18:09", level: "INFO", msg: "LayoutLM entity extraction completed (0.5s)" },
  { t: "14:18:04", level: "ERROR", msg: "EasyOCR timeout on receipt_blur_44.jpg, retrying..." },
  { t: "14:17:48", level: "INFO", msg: "Started batch (3 documents)" },
];

const levelColor: Record<string, string> = {
  INFO: "text-panel/70",
  SUCCESS: "text-success",
  WARN: "text-warning",
  ERROR: "text-destructive",
};

function LogsPage() {
  return (
    <main className="max-w-7xl mx-auto px-6 py-8">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight mb-1">Pipeline Logs</h1>
        <p className="text-sm text-muted-foreground">
          Real-time output from the local processing engine.
        </p>
      </header>

      <section className="bg-ink rounded-2xl ring-1 ring-white/10 overflow-hidden">
        <div className="px-6 py-3 border-b border-white/10 flex items-center justify-between">
          <div className="flex items-center gap-2 text-panel">
            <span className="size-2 bg-success rounded-full animate-pulse" />
            <span className="text-xs font-mono uppercase tracking-widest">Live Stream</span>
          </div>
          <button className="text-[11px] font-mono uppercase tracking-widest text-panel/60 hover:text-panel">
            Download .log
          </button>
        </div>
        <div className="p-6 font-mono text-[12px] space-y-2">
          {logs.map((l, i) => (
            <div key={i} className="flex gap-4">
              <span className="text-panel/40 shrink-0">[{l.t}]</span>
              <span className={`shrink-0 w-16 font-semibold ${levelColor[l.level]}`}>
                {l.level}
              </span>
              <span className="text-panel/80">{l.msg}</span>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
