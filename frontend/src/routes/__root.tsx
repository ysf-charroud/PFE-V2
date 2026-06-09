import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  Outlet,
  Link,
  createRootRouteWithContext,
  useRouter,
  useRouterState,
  HeadContent,
  Scripts,
} from "@tanstack/react-router";
import { useEffect, type ReactNode } from "react";

import appCss from "../styles.css?url";
import { reportLovableError } from "../lib/lovable-error-reporting";

function NotFoundComponent() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="max-w-md text-center">
        <h1 className="text-7xl font-semibold text-foreground tracking-tight">404</h1>
        <h2 className="mt-4 text-xl font-semibold text-foreground">Page not found</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <div className="mt-6">
          <Link
            to="/"
            className="inline-flex items-center justify-center rounded-md bg-brand px-4 py-2 text-sm font-medium text-brand-foreground transition-opacity hover:opacity-90"
          >
            Back to Documents
          </Link>
        </div>
      </div>
    </div>
  );
}

function ErrorComponent({ error, reset }: { error: Error; reset: () => void }) {
  console.error(error);
  const router = useRouter();
  useEffect(() => {
    reportLovableError(error, { boundary: "tanstack_root_error_component" });
  }, [error]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="max-w-md text-center">
        <h1 className="text-xl font-semibold tracking-tight text-foreground">
          This page didn't load
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Something went wrong. Try refreshing or head back home.
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          <button
            onClick={() => {
              router.invalidate();
              reset();
            }}
            className="inline-flex items-center justify-center rounded-md bg-brand px-4 py-2 text-sm font-medium text-brand-foreground"
          >
            Try again
          </button>
          <a
            href="/"
            className="inline-flex items-center justify-center rounded-md border border-border bg-panel px-4 py-2 text-sm font-medium text-foreground"
          >
            Go home
          </a>
        </div>
      </div>
    </div>
  );
}

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  head: () => ({
    meta: [
      { charSet: "utf-8" },
      { name: "viewport", content: "width=device-width, initial-scale=1" },
      { title: "Vector IDP — Intelligent Document Processing" },
      {
        name: "description",
        content:
          "Local, open-source intelligent document processing for invoices and receipts. OpenCV + EasyOCR + LayoutLM, stored in SQLite.",
      },
    ],
    links: [
      { rel: "stylesheet", href: appCss },
      {
        rel: "preconnect",
        href: "https://fonts.googleapis.com",
      },
      {
        rel: "preconnect",
        href: "https://fonts.gstatic.com",
        crossOrigin: "anonymous",
      },
      {
        rel: "stylesheet",
        href: "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap",
      },
    ],
  }),
  shellComponent: RootShell,
  component: RootComponent,
  notFoundComponent: NotFoundComponent,
  errorComponent: ErrorComponent,
});

function RootShell({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <HeadContent />
      </head>
      <body>
        {children}
        <Scripts />
      </body>
    </html>
  );
}

function TopNav() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const link = (to: string, label: string) => (
    <Link
      to={to}
      className={`text-sm font-medium transition-colors ${
        pathname === to ? "text-foreground" : "text-muted-foreground hover:text-foreground"
      }`}
    >
      {label}
    </Link>
  );

  return (
    <nav className="border-b border-border bg-panel/70 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
        <div className="flex items-center gap-8">
          <Link to="/" className="flex items-center gap-2">
            <div className="size-6 bg-brand rounded flex items-center justify-center">
              <div className="size-2 bg-panel rounded-full" />
            </div>
            <span className="font-semibold tracking-tight text-foreground"> Receipts Ocr </span>
          </Link>
          <div className="flex items-center gap-6">
            {link("/", "Documents")}
            {link("/models", "Models")}
          </div>
        </div>
        {/* <div className="flex items-center gap-3">
          <button className="text-sm font-medium px-3 py-1.5 rounded-md bg-secondary text-secondary-foreground ring-1 ring-border hover:bg-muted transition-colors">
            Export JSON
          </button>
          <button className="text-sm font-semibold px-3 py-1.5 rounded-md bg-brand text-brand-foreground ring-1 ring-brand hover:opacity-90 transition-opacity">
            New Upload
          </button>
        </div> */}
      </div>
    </nav>
  );
}

function RootComponent() {
  const { queryClient } = Route.useRouteContext();

  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-screen bg-background text-foreground selection:bg-brand/10">
        <TopNav />
        <Outlet />
        <footer className="max-w-7xl mx-auto px-6 py-10 border-t border-border mt-12">
          <div className="flex justify-between items-center gap-6">
            <div className="flex gap-8 text-xs font-medium text-muted-foreground uppercase tracking-widest">
              <span>SQLite Local Engine</span>
              <span>v4.2.1-stable</span>
            </div>
            <div className="text-xs text-subtle max-w-[44ch] text-right text-pretty">
              Local, open-source document extraction. Processing happens entirely on-device to
              maintain financial data sovereignty.
            </div>
          </div>
        </footer>
      </div>
    </QueryClientProvider>
  );
}
