import { createFileRoute } from "@tanstack/react-router";

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

function ModelsPage() {
  return (
    <main className="max-w-7xl mx-auto px-6 py-8">
      <header className="mb-8">
        <h1 className="text-2xl font-semibold tracking-tight mb-1">Model Configuration</h1>
        <p className="text-sm text-muted-foreground">
          Hybrid pipeline combining Computer Vision and Natural Language Processing. Entirely local
          and Open Source.
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {models.map((m) => (
          <article
            key={m.name}
            className="bg-panel rounded-2xl ring-1 ring-border p-6 flex flex-col gap-3"
          >
            <div className="flex items-start justify-between">
              <div>
                <h2 className="text-base font-semibold text-foreground">{m.name}</h2>
                <p className="text-[11px] font-bold text-subtle uppercase tracking-wider mt-0.5">
                  {m.role}
                </p>
              </div>
              <div className="flex items-center gap-2 text-success">
                <span className="size-2 bg-success rounded-full" />
                <span className="text-xs font-medium">{m.status}</span>
              </div>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed">{m.desc}</p>
            <div className="pt-3 mt-auto border-t border-border flex items-center justify-between text-xs font-mono text-subtle">
              <span>v{m.version}</span>
            </div>
          </article>
        ))}
      </div>
    </main>
  );
}
