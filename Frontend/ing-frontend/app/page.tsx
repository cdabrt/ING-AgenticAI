"use client";

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import axios from "axios";
import { AlertCircleIcon, ChevronDown, Eye, RefreshCw, Trash2, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ButtonGroup } from "@/components/ui/button-group";
import RequirementDetailView from "@/components/custom/RequirementDetailView";
import RequirementListSkeleton from "@/components/custom/RequirementListSkeleton";
import RequirementList from "@/components/custom/RequirementList";
import RequirementBundleList from "@/components/custom/RequirementBundleList";
import BundleDetailView from "@/components/custom/BundleDetailView";
import FileUploadDialog from "@/components/custom/FileUploadDialog";
import { RequirementBundle, RequirementItem } from "@/lib/types";

const PDFViewer = dynamic(() => import("@/components/custom/PDFViewer").then(mod => ({ default: mod.PDFViewer })), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center p-8">
      <p className="text-sm text-gray-500">Loading PDF viewer...</p>
    </div>
  ),
});

const requirementTabs = ["Business", "Data"];

interface PdfItem {
  id: number;
  filename: string;
  size: number;
  embedded: boolean;
  embedded_at?: string | null;
  chunk_count?: number | null;
}

interface EmbeddingStatus {
  state: "idle" | "running" | "completed" | "error";
  stage?: string | null;
  message?: string | null;
  progress?: number | null;
}

interface PipelineStatus {
  state: "idle" | "running" | "completed" | "error";
  stage?: string | null;
  message?: string | null;
  progress?: number | null;
  retry_after_seconds?: number | null;
  updated_at?: string | null;
  current_doc?: string | null;
  llm_step?: string | null;
  llm_detail?: string | null;
  llm_calls_done?: number | null;
  llm_calls_total?: number | null;
}

interface VectorStoreStatus {
  exists: boolean;
  backend?: string;
  collection_exists?: boolean;
  collection_count?: number;
  config?: {
    vector_store?: string;
    collection_name?: string;
    chunk_count?: number;
    embedding_model_name?: string;
    generated_at?: string;
  };
}

function formatBytes(value: number): string {
  if (!value && value !== 0) {
    return "--";
  }
  const units = ["B", "KB", "MB", "GB"];
  let size = value;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(size >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatTimestamp(value?: string | null): string {
  if (!value) {
    return "--";
  }
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) {
    return value;
  }
  return date.toLocaleString();
}

function statusTone(state?: string | null): string {
  switch (state) {
    case "running":
      return "bg-amber-500";
    case "completed":
      return "bg-emerald-500";
    case "error":
      return "bg-rose-500";
    default:
      return "bg-slate-400";
  }
}

export default function Home() {
  const [activeView, setActiveView] = useState<"library" | "requirements">("library");
  const [selected, setSelected] = useState<RequirementItem | null>(null);
  const [selectedBundle, setSelectedBundle] = useState<RequirementBundle | null>(null);
  const [selectedList, setSelectedList] = useState<number>(0);
  const [results, setResults] = useState<RequirementBundle[]>([]);

  const [pdfs, setPdfs] = useState<PdfItem[]>([]);
  const [selectedPdfIds, setSelectedPdfIds] = useState<number[]>([]);

  const [loading, setLoading] = useState(false);
  const [embeddingLoading, setEmbeddingLoading] = useState(false);
  const [embeddingStatus, setEmbeddingStatus] = useState<EmbeddingStatus | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus | null>(null);
  const [vectorStoreStatus, setVectorStoreStatus] = useState<VectorStoreStatus | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [lastSyncAt, setLastSyncAt] = useState<string | null>(null);
  const [lastBundleRefreshAt, setLastBundleRefreshAt] = useState<string | null>(null);
  const [vectorRefreshing, setVectorRefreshing] = useState(false);
  const [vectorCheckedAt, setVectorCheckedAt] = useState<string | null>(null);
  const [embeddingSummary, setEmbeddingSummary] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showUploadDialog, setShowUploadDialog] = useState(false);
  const [pdfData, setPdfData] = useState<File | null>(null);
  const [showPdfViewer, setShowPdfViewer] = useState(false);
  const [webSearchEnabled, setWebSearchEnabled] = useState(true);
  const [webSearchLimit, setWebSearchLimit] = useState(3);

  const selectedPdfCount = selectedPdfIds.length;
  const allSelected = pdfs.length > 0 && selectedPdfCount === pdfs.length;
  const embeddedCount = pdfs.filter(pdf => pdf.embedded).length;
  const unembeddedCount = pdfs.length - embeddedCount;

  const selectedPdfNames = useMemo(() => {
    const selectedSet = new Set(selectedPdfIds);
    return pdfs.filter(pdf => selectedSet.has(pdf.id)).map(pdf => pdf.filename);
  }, [pdfs, selectedPdfIds]);

  function getRequirementList(): RequirementItem[] {
    if (selectedList === 1) {
      return selectedBundle ? selectedBundle.business_requirements : [];
    }
    if (selectedList === 2) {
      return selectedBundle ? selectedBundle.data_requirements : [];
    }
    return [];
  }

  function getRequirementBundles(): Promise<void> {
    setError(null);
    return axios.get("/api/bundles").catch((error) => {
      setError(error.message);
    }).then((response) => {
      if (response && response.data) {
        const rawBundles = Array.isArray(response.data) ? response.data : [response.data];
        const bundles = rawBundles.map((bundle: RequirementBundle, index: number) => {
          if (!bundle.id) {
            return { ...bundle, id: index + 1 };
          }
          return bundle;
        });
        setResults(bundles);
        if (bundles.length > 0) {
          setSelectedBundle(bundles[0]);
        }
        setSelectedList(0);
      } else {
        setResults([]);
        setError("No data received!");
      }
    });
  }

  function getPdfs(): Promise<void> {
    return axios.get("/api/embeddings/index").then((response) => {
      if (response && response.data) {
        const items = Array.isArray(response.data?.pdfs) ? response.data.pdfs : [];
        setPdfs(items);
        setSelectedPdfIds((current) => current.filter(id => items.some((pdf: PdfItem) => pdf.id === id)));
      } else {
        setPdfs([]);
      }
    }).catch((error) => {
      setError(error.message);
    });
  }

  function getEmbeddingStatus(): Promise<void> {
    return axios.get("/api/embeddings/status").then((response) => {
      if (response && response.data) {
        setEmbeddingStatus(response.data);
      }
    }).catch(() => {
      // ignore polling errors
    });
  }

  function getPipelineStatus(): Promise<void> {
    return axios.get("/api/pipeline/status").then((response) => {
      if (response && response.data) {
        setPipelineStatus(response.data);
      }
    }).catch(() => {
      // ignore polling errors
    });
  }

  function getVectorStoreStatus(): Promise<void> {
    return axios.get("/api/vector-store").then((response) => {
      if (response && response.data) {
        setVectorStoreStatus(response.data);
      }
    }).catch(() => {
      // ignore polling errors
    });
  }

  function refreshVectorStoreStatus(): Promise<void> {
    setVectorRefreshing(true);
    return getVectorStoreStatus().finally(() => {
      setVectorCheckedAt(new Date().toISOString());
      setVectorRefreshing(false);
    });
  }

  function syncStatus(): Promise<void> {
    setSyncing(true);
    return Promise.all([
      getEmbeddingStatus(),
      getPipelineStatus(),
      refreshVectorStoreStatus(),
    ]).then(() => {
      return undefined;
    }).finally(() => {
      setLastSyncAt(new Date().toISOString());
      setSyncing(false);
    });
  }

  function togglePdfSelection(id: number) {
    setSelectedPdfIds((current) => {
      if (current.includes(id)) {
        return current.filter(item => item !== id);
      }
      return [...current, id];
    });
  }

  function toggleSelectAll() {
    if (allSelected) {
      setSelectedPdfIds([]);
      return;
    }
    setSelectedPdfIds(pdfs.map(pdf => pdf.id));
  }

  function embedSelectedPdfs(): Promise<void> {
    setEmbeddingSummary(null);
    setError(null);
    setEmbeddingLoading(true);
    const payload = selectedPdfCount > 0 ? { pdf_ids: selectedPdfIds } : {};
    return axios.post("/api/embeddings", payload).then((response) => {
      const payload = response?.data;
      if (payload) {
        const summary = `Embedded ${payload.chunks ?? 0} chunks from ${payload.pdf_files ?? 0} PDFs.`;
        setEmbeddingSummary(summary);
      }
      return getEmbeddingStatus();
    }).catch((error) => {
      const detail = error?.response?.data?.detail;
      setError(detail || error.message);
    }).finally(() => {
      setEmbeddingLoading(false);
      void refreshVectorStoreStatus();
    });
  }

  function removeEmbeddedPdf(pdfId: number): Promise<void> {
    setError(null);
    return axios.delete(`/api/embeddings/${pdfId}`).then(() => {
      setEmbeddingSummary("Embedding removed for selected PDF.");
      return getPdfs();
    }).catch((error) => {
      const detail = error?.response?.data?.detail;
      setError(detail || error.message);
    }).finally(() => {
      void refreshVectorStoreStatus();
    });
  }

  function generateRequirementBundle(): Promise<void> {
    setError(null);
    setLoading(true);
    void getPipelineStatus();
    return axios.post("/api/pipeline", {
      skip_ingestion: true,
      web_search_enabled: webSearchEnabled,
      max_web_queries_per_doc: webSearchLimit,
    }).catch((error) => {
      setError(error.message);
    }).then(async () => {
      await getRequirementBundles();
    }).finally(() => {
      setLoading(false);
      void getPipelineStatus();
    });
  }

  async function openPdf(pdfId: number): Promise<void> {
    setError(null);
    try {
      const response = await axios.get(`/api/pdfs/${pdfId}/download`, {
        responseType: "arraybuffer",
      });
      const contentType = response.headers["content-type"] || "application/pdf";
      const disposition = response.headers["content-disposition"] as string | undefined;
      let filename = `document-${pdfId}.pdf`;
      if (disposition) {
        const match = disposition.match(/filename=\"?([^\";]+)\"?/i);
        if (match && match[1]) {
          filename = match[1];
        }
      }
      const blob = new Blob([response.data], { type: contentType });
      const file = new File([blob], filename, { type: contentType });
      setPdfData(file);
      setShowPdfViewer(true);
    } catch (error: any) {
      setError(error.message || "Failed to load PDF");
    }
  }

  useEffect(() => {
    getRequirementBundles();
    getPdfs();
    syncStatus();
  }, []);

  useEffect(() => {
    if (embeddingStatus?.state !== "running") {
      return;
    }
    const interval = setInterval(() => {
      getEmbeddingStatus();
    }, 1500);
    return () => clearInterval(interval);
  }, [embeddingStatus?.state]);

  useEffect(() => {
    if (pipelineStatus?.state !== "running") {
      return;
    }
    const interval = setInterval(() => {
      getPipelineStatus();
    }, 1500);
    return () => clearInterval(interval);
  }, [pipelineStatus?.state]);

  useEffect(() => {
    if (pipelineStatus?.state !== "completed") {
      return;
    }
    if (pipelineStatus.updated_at && pipelineStatus.updated_at !== lastBundleRefreshAt) {
      setLastBundleRefreshAt(pipelineStatus.updated_at);
      void getRequirementBundles();
    }
  }, [pipelineStatus?.state, pipelineStatus?.updated_at, lastBundleRefreshAt]);

  const embeddingProgress = Math.min(100, Math.max(0, (embeddingStatus?.progress ?? 0) * 100));
  const pipelineProgress = Math.min(100, Math.max(0, (pipelineStatus?.progress ?? 0) * 100));
  const storeConfig = vectorStoreStatus?.config;
  const storeBackend = vectorStoreStatus?.backend ?? storeConfig?.vector_store ?? "--";
  const storeCollection = storeConfig?.collection_name ?? "--";
  const storeCount = vectorStoreStatus?.collection_count ?? storeConfig?.chunk_count;
  const canGenerate = Boolean(vectorStoreStatus?.exists && typeof storeCount === "number" && storeCount > 0);
  const pipelineStage = pipelineStatus?.stage ?? "idle";
  const pipelineStageDescriptions: Record<string, string> = {
    init: "Warming up models and tools.",
    ingestion: "Embedding PDFs into the vector store.",
    parsing: "Extracting structured sections from PDFs.",
    mcp_start: "Starting external tools for regulatory context.",
    documents: "Analyzing each document for obligations.",
    requirements: "Drafting requirement bundles.",
    output: "Saving results and PDF output.",
    complete: "Generation complete.",
    error: "Generation failed.",
    idle: "Waiting for a run.",
  };
  const pipelineStageLabel = pipelineStageDescriptions[pipelineStage] ?? "Processing pipeline stage.";

  return (
    <div className="min-h-screen relative overflow-hidden bg-[#f7f3eb] text-zinc-900">
      <div className="absolute inset-0 -z-10">
        <div className="absolute -top-20 -right-24 h-72 w-72 rounded-full bg-emerald-200/60 blur-3xl" />
        <div className="absolute top-32 -left-24 h-80 w-80 rounded-full bg-amber-200/50 blur-3xl" />
        <div className="absolute bottom-4 right-10 h-64 w-64 rounded-full bg-sky-200/40 blur-3xl" />
        <div className="absolute inset-0 opacity-40 bg-[radial-gradient(circle_at_1px_1px,rgba(15,23,42,0.08)_1px,transparent_0)] [background-size:28px_28px]" />
      </div>

      <header className="sticky top-0 z-20 border-b border-zinc-200/70 bg-[#f7f3eb]/90 backdrop-blur">
        <div className="mx-auto flex w-full max-w-[1480px] flex-wrap items-center justify-between gap-6 px-8 py-4">
          <div className="flex items-center gap-6">
            <h1 className="font-display text-2xl font-semibold tracking-tight">ING Agentic AI</h1>
            <nav className="flex items-center gap-6 text-sm font-medium">
              <button
                className={`pb-1 transition ${activeView === "library" ? "border-b-2 border-zinc-900 text-zinc-900" : "text-zinc-500 hover:text-zinc-900"}`}
                onClick={() => setActiveView("library")}
              >
                Library
              </button>
              <button
                className={`pb-1 transition ${activeView === "requirements" ? "border-b-2 border-zinc-900 text-zinc-900" : "text-zinc-500 hover:text-zinc-900"}`}
                onClick={() => setActiveView("requirements")}
              >
                Requirements
              </button>
            </nav>
          </div>
          <div className="flex flex-col items-end gap-1">
            <Button
              variant="outline"
              size="sm"
              onClick={syncStatus}
              disabled={syncing}
            >
              <RefreshCw className={syncing ? "animate-spin" : ""} />
              {syncing ? "Syncing..." : "Sync Status"}
            </Button>
            <span className="text-[10px] text-zinc-500">Refreshes embedding + pipeline status.</span>
            <span className="text-[10px] text-zinc-400">
              Last sync: {lastSyncAt ? formatTimestamp(lastSyncAt) : "--"}
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-[1480px] px-8 py-8">
        <div className="grid gap-8 lg:grid-cols-[280px_minmax(0,1fr)]">
          <aside className="space-y-4">
            <details className="group rounded-2xl border border-zinc-200 bg-white/90 shadow-sm animate-fade-in-up" style={{ animationDelay: "0ms" }}>
              <summary className="flex cursor-pointer list-none items-center justify-between px-4 py-3 text-[11px] uppercase tracking-[0.2em] text-zinc-500">
                Live Status
                <ChevronDown className="h-4 w-4 transition-transform group-open:rotate-180" />
              </summary>
              <div className="border-t border-zinc-200/70 px-4 pb-4 pt-3">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-zinc-400">Auto refresh</span>
                </div>
                <div className="mt-4 space-y-5">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <span className={`h-2 w-2 rounded-full ${statusTone(embeddingStatus?.state)}`} />
                        <span className="font-medium">Embedding</span>
                      </div>
                      <span className="text-xs text-zinc-500">{embeddingStatus?.state ?? "idle"}</span>
                    </div>
                    <div className="h-1.5 w-full rounded-full bg-zinc-100">
                      <div
                        className="h-full rounded-full bg-zinc-900/80 transition-[width]"
                        style={{ width: `${embeddingProgress}%` }}
                      />
                    </div>
                    <p className="text-xs text-zinc-500">{embeddingStatus?.message || "Ready for embedding."}</p>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center gap-2">
                        <span className={`h-2 w-2 rounded-full ${statusTone(pipelineStatus?.state)}`} />
                        <span className="font-medium">Generation</span>
                      </div>
                      <span className="text-xs text-zinc-500">{pipelineStatus?.state ?? "idle"}</span>
                    </div>
                    <div className="h-1.5 w-full rounded-full bg-zinc-100">
                      <div
                        className="h-full rounded-full bg-zinc-900/80 transition-[width]"
                        style={{ width: `${pipelineProgress}%` }}
                      />
                    </div>
                    <p className="text-xs text-zinc-500">
                      {pipelineStatus?.message || "Waiting for a generation run."}
                    </p>
                    <p className="text-[11px] text-zinc-500">Stage: {pipelineStage} - {pipelineStageLabel}</p>
                    {pipelineStatus?.retry_after_seconds && (
                      <p className="text-[11px] text-amber-600">Retry after {pipelineStatus.retry_after_seconds}s</p>
                    )}
                  </div>
                </div>
              </div>
            </details>

            <details className="group rounded-2xl border border-zinc-200 bg-white/90 shadow-sm animate-fade-in-up" style={{ animationDelay: "80ms" }}>
              <summary className="flex cursor-pointer list-none items-center justify-between px-4 py-3 text-[11px] uppercase tracking-[0.2em] text-zinc-500">
                Vector Store
                <ChevronDown className="h-4 w-4 transition-transform group-open:rotate-180" />
              </summary>
              <div className="border-t border-zinc-200/70 px-4 pb-4 pt-3">
                <div className="flex items-center justify-between">
                  <span className={`text-[10px] ${vectorStoreStatus?.exists ? "text-emerald-600" : "text-zinc-400"}`}>
                    {vectorStoreStatus?.exists ? "Ready" : "Missing"}
                  </span>
                </div>
                <div className="mt-4 space-y-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-zinc-500">Backend</span>
                    <span className="font-medium">{storeBackend}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-zinc-500">Collection</span>
                    <span className="font-medium">{storeCollection}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-zinc-500">Chunks</span>
                    <span className="font-medium">{storeCount ?? "--"}</span>
                  </div>
                  {storeConfig?.embedding_model_name && (
                    <div className="flex items-center justify-between">
                      <span className="text-zinc-500">Model</span>
                      <span className="font-medium">{storeConfig.embedding_model_name}</span>
                    </div>
                  )}
                  <div className="flex items-center justify-between">
                    <span className="text-zinc-500">Updated</span>
                    <span className="text-xs text-zinc-500">{formatTimestamp(storeConfig?.generated_at)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-zinc-500">Last checked</span>
                    <span className="text-xs text-zinc-500">{formatTimestamp(vectorCheckedAt)}</span>
                  </div>
                </div>
                <div className="mt-4 flex flex-wrap gap-2">
                  <Button size="sm" variant="outline" onClick={refreshVectorStoreStatus} disabled={vectorRefreshing}>
                    <RefreshCw className={vectorRefreshing ? "animate-spin" : ""} />
                    {vectorRefreshing ? "Refreshing..." : "Refresh"}
                  </Button>
                  <Button size="sm" variant="ghost" asChild>
                    <a href="http://localhost:8001" target="_blank" rel="noreferrer">
                      Milvus UI
                    </a>
                  </Button>
                </div>
              </div>
            </details>
          </aside>

          <section className="space-y-6">
            {activeView === "library" ? (
              <div className="grid gap-8 xl:grid-cols-[minmax(0,1fr)_320px]">
                <div
                  className="rounded-2xl border border-zinc-200 bg-white/90 p-6 shadow-sm animate-fade-in-up"
                  style={{ animationDelay: "40ms" }}
                >
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    <div>
                      <h2 className="font-display text-lg font-semibold text-zinc-900">Document Library</h2>
                      <p className="text-xs text-zinc-500">Curate PDFs before embedding into Milvus.</p>
                    </div>
                    <Button variant="outline" size="sm" onClick={() => setShowUploadDialog(true)}>
                      <Upload />
                      Upload PDF
                    </Button>
                  </div>
                  <div className="mt-4 flex flex-wrap items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={toggleSelectAll}
                      disabled={pdfs.length === 0}
                    >
                      {allSelected ? "Clear Selection" : "Select All"}
                    </Button>
                    <Button variant="ghost" size="icon-sm" onClick={getPdfs} aria-label="Refresh PDF list">
                      <RefreshCw />
                    </Button>
                  </div>
                  <div className="mt-3 text-xs text-zinc-500">
                    {selectedPdfCount > 0 ? `${selectedPdfCount} selected` : "No PDFs selected"}
                    {selectedPdfNames.length > 0 && selectedPdfNames.length <= 3 && (
                      <span className="ml-2 text-zinc-400">{selectedPdfNames.join(", ")}</span>
                    )}
                  </div>
                  <div className="mt-1 text-[11px] text-zinc-500">
                    {pdfs.length > 0 ? `${embeddedCount} embedded / ${unembeddedCount} not embedded` : "Upload PDFs to start embedding."}
                  </div>
                  <div className="mt-4 overflow-hidden rounded-xl border border-zinc-200 bg-white/80">
                    {pdfs.length === 0 ? (
                      <div className="p-6 text-sm text-zinc-500">No PDFs uploaded yet.</div>
                    ) : (
                      <ul className="divide-y divide-zinc-100">
                        {pdfs.map((pdf) => {
                          const isSelected = selectedPdfIds.includes(pdf.id);
                          return (
                            <li
                              key={pdf.id}
                              className={`flex items-center justify-between gap-3 px-4 py-3 transition ${isSelected ? "bg-amber-50/70" : "hover:bg-white"}`}
                            >
                              <label className="flex items-center gap-3 min-w-0">
                                <input
                                  type="checkbox"
                                  className="h-4 w-4 rounded border-zinc-300 text-zinc-900"
                                  checked={isSelected}
                                  onChange={() => togglePdfSelection(pdf.id)}
                                />
                                <div className="min-w-0">
                                  <p className="text-sm font-medium text-zinc-900 truncate">{pdf.filename}</p>
                                  <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-zinc-500">
                                    <span>{formatBytes(pdf.size)}</span>
                                    <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${pdf.embedded ? "bg-emerald-100 text-emerald-700" : "bg-zinc-100 text-zinc-500"}`}>
                                      {pdf.embedded ? "Embedded" : "Not embedded"}
                                    </span>
                                    {pdf.embedded && pdf.chunk_count !== null && (
                                      <span className="text-[10px] text-zinc-400">{pdf.chunk_count} chunks</span>
                                    )}
                                    {pdf.embedded && pdf.embedded_at && (
                                      <span className="text-[10px] text-zinc-400">{formatTimestamp(pdf.embedded_at)}</span>
                                    )}
                                  </div>
                                </div>
                              </label>
                              <div className="flex items-center gap-2">
                                {pdf.embedded && (
                                  <Button
                                    size="icon-sm"
                                    variant="ghost"
                                    onClick={() => removeEmbeddedPdf(pdf.id)}
                                    aria-label="Remove embedding"
                                  >
                                    <Trash2 />
                                  </Button>
                                )}
                                <Button size="icon-sm" variant="ghost" onClick={() => openPdf(pdf.id)} aria-label="Preview PDF">
                                  <Eye />
                                </Button>
                              </div>
                            </li>
                          );
                        })}
                      </ul>
                    )}
                  </div>
                </div>

                <div
                  className="rounded-2xl border border-zinc-200 bg-white/90 p-6 shadow-sm animate-fade-in-up"
                  style={{ animationDelay: "120ms" }}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-display text-base font-semibold text-zinc-900">Embedding Run</h3>
                      <p className="text-xs text-zinc-500">Stage embeddings before generation.</p>
                    </div>
                    <span className="text-xs text-zinc-500">{embeddingStatus?.state ?? "idle"}</span>
                  </div>
                  <div className="mt-4 space-y-3 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-zinc-500">Selected</span>
                      <span className="font-medium">{selectedPdfCount > 0 ? `${selectedPdfCount} PDFs` : "All PDFs"}</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-zinc-500">Vector Store</span>
                      <span className="font-medium">{storeBackend}</span>
                    </div>
                  </div>
                  <Button
                    className="mt-4 w-full"
                    onClick={embedSelectedPdfs}
                    disabled={embeddingLoading || pdfs.length === 0}
                  >
                    {embeddingLoading ? "Embedding..." : selectedPdfCount > 0 ? "Embed Selection" : "Embed All PDFs"}
                  </Button>
                  <div className="mt-4 space-y-2 text-xs text-zinc-500">
                    <div className="flex items-center justify-between">
                      <span>{embeddingStatus?.message || "Ready to embed selections."}</span>
                      <span>{embeddingProgress.toFixed(0)}%</span>
                    </div>
                    <div className="h-1.5 w-full rounded-full bg-zinc-100">
                      <div
                        className="h-full rounded-full bg-zinc-900/80 transition-[width]"
                        style={{ width: `${embeddingProgress}%` }}
                      />
                    </div>
                    {embeddingSummary && <p className="text-xs text-zinc-500">{embeddingSummary}</p>}
                  </div>
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                <div
                  className="rounded-2xl border border-zinc-200 bg-white/90 p-5 shadow-sm animate-fade-in-up"
                  style={{ animationDelay: "40ms" }}
                >
                  <div className="flex flex-wrap items-center justify-between gap-4">
                    <div>
                      <h2 className="font-display text-lg font-semibold text-zinc-900">Generate Requirements</h2>
                      <p className="text-xs text-zinc-500">Uses embedded PDFs only - no re-ingestion.</p>
                    </div>
                    <Button
                      onClick={generateRequirementBundle}
                      disabled={loading || !canGenerate}
                    >
                      {loading ? "Generating..." : "Generate Requirements"}
                    </Button>
                  </div>
                  <div className="mt-4 rounded-xl border border-zinc-200/80 bg-zinc-50/70 p-4 text-xs text-zinc-600">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div>
                        <p className="text-[11px] font-semibold text-zinc-700">Online search</p>
                        <p className="text-[10px] text-zinc-500">Optional enrichment from trusted sources.</p>
                      </div>
                      <label className="flex items-center gap-2 text-[11px]">
                        <span className="text-zinc-500">Disabled</span>
                        <span className="relative inline-flex items-center">
                          <input
                            type="checkbox"
                            className="peer sr-only"
                            checked={webSearchEnabled}
                            onChange={(event) => setWebSearchEnabled(event.target.checked)}
                            aria-label="Enable online search"
                          />
                          <span className="h-5 w-9 rounded-full bg-zinc-200 transition peer-checked:bg-emerald-500" />
                          <span className="absolute left-1 top-1 h-3 w-3 rounded-full bg-white transition peer-checked:translate-x-4" />
                        </span>
                        <span className="text-zinc-700">Enabled</span>
                      </label>
                    </div>
                    <div className="mt-3 flex items-center justify-between gap-3">
                      <div>
                        <p className="text-[11px] font-semibold text-zinc-700">Max searches per document</p>
                        <p className="text-[10px] text-zinc-500">Set to 0 to skip online lookups.</p>
                      </div>
                      <input
                        type="number"
                        min={0}
                        max={10}
                        value={webSearchLimit}
                        onChange={(event) => {
                          const value = Number(event.target.value);
                          if (Number.isNaN(value)) {
                            return;
                          }
                          setWebSearchLimit(Math.max(0, Math.min(10, value)));
                        }}
                        disabled={!webSearchEnabled}
                        className="h-8 w-16 rounded-md border border-zinc-200 bg-white px-2 text-center text-xs text-zinc-700"
                        aria-label="Max searches per document"
                      />
                    </div>
                  </div>
                  <div className="mt-4 flex flex-wrap items-center gap-6 text-sm text-zinc-600">
                    <div className="flex items-center gap-2">
                      <span className={`h-2 w-2 rounded-full ${statusTone(pipelineStatus?.state)}`} />
                      <span>{pipelineStatus?.state ?? "idle"}</span>
                    </div>
                    <div>Chunks: <span className="font-medium text-zinc-900">{storeCount ?? "--"}</span></div>
                    <div>Embeddings: <span className="font-medium text-zinc-900">{canGenerate ? "Ready" : "Missing"}</span></div>
                  </div>
                  <div className="mt-3 space-y-2 text-xs text-zinc-500">
                    <div className="flex items-center justify-between">
                      <span>{pipelineStatus?.message || "Waiting for a run."}</span>
                      <span>{pipelineProgress.toFixed(0)}%</span>
                    </div>
                    <div className="rounded-lg border border-zinc-200/70 bg-zinc-50/80 px-3 py-2 text-[11px] text-zinc-600">
                      <span className="font-semibold text-zinc-800">Stage:</span> {pipelineStage} - {pipelineStageLabel}
                    </div>
                    {(pipelineStatus?.llm_step || pipelineStatus?.current_doc) && (
                      <div className="rounded-lg border border-amber-200/70 bg-amber-50/70 px-3 py-2 text-[11px] text-amber-800">
                        <div>
                          <span className="font-semibold">LLM:</span>{" "}
                          {pipelineStatus?.llm_step || "Working"}
                          {pipelineStatus?.llm_detail ? ` (${pipelineStatus.llm_detail})` : ""}
                        </div>
                        {pipelineStatus?.current_doc && (
                          <div className="text-[10px] text-amber-700">
                            <span className="font-semibold">Doc:</span> {pipelineStatus.current_doc}
                          </div>
                        )}
                        {pipelineStatus?.llm_calls_done !== null && pipelineStatus?.llm_calls_done !== undefined && (
                          <div className="text-[10px] text-amber-700">
                            <span className="font-semibold">Calls:</span>{" "}
                            {pipelineStatus.llm_calls_done}
                            {pipelineStatus?.llm_calls_total ? ` / ${pipelineStatus.llm_calls_total}` : ""}
                          </div>
                        )}
                      </div>
                    )}
                    <div className="h-1.5 w-full rounded-full bg-zinc-100">
                      <div
                        className="h-full rounded-full bg-zinc-900/80 transition-[width]"
                        style={{ width: `${pipelineProgress}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="grid gap-6 lg:grid-cols-[320px_minmax(0,1fr)] items-start">
                  <div
                    className="rounded-2xl border border-zinc-200 bg-white/90 shadow-sm flex flex-col min-h-[520px] max-h-[calc(100vh-320px)] overflow-hidden animate-fade-in-up"
                    style={{ animationDelay: "80ms" }}
                  >
                    <div className="border-b border-zinc-200/80 p-4">
                      <h3 className="font-display text-base font-semibold text-zinc-900">Results</h3>
                      <div className="mt-3 flex flex-wrap gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setSelectedList(0)}
                          className={selectedList === 0 ? "bg-zinc-100" : ""}
                        >
                          Bundles
                        </Button>
                        {requirementTabs.map((btn, index) => (
                          <Button
                            key={btn}
                            variant="outline"
                            size="sm"
                            onClick={() => setSelectedList(index + 1)}
                            className={selectedList === index + 1 ? "bg-zinc-100" : ""}
                          >
                            {btn}
                          </Button>
                        ))}
                      </div>
                    </div>
                    <div className="flex-1 p-4 overflow-y-auto">
                      {!results && (
                        <RequirementListSkeleton loading={loading} setLoading={setLoading} getRequirementBundle={getRequirementBundles} />
                      )}
                      {selectedList > 0 && (
                        <RequirementList
                          data={getRequirementList()}
                          onRowClick={(requirement) => setSelected(requirement)}
                        />
                      )}
                      {selectedList === 0 && (
                        <RequirementBundleList
                          data={results}
                          onRowClick={(bundle) => setSelectedBundle(bundle)}
                        />
                      )}
                    </div>
                  </div>

                  <div
                    className="rounded-2xl border border-zinc-200 bg-white/90 shadow-sm flex flex-col min-h-[520px] max-h-[calc(100vh-320px)] overflow-hidden animate-fade-in-up"
                    style={{ animationDelay: "120ms" }}
                  >
                    <div className="border-b border-zinc-200/80 p-4">
                      <h3 className="font-display text-base font-semibold text-zinc-900">Detail</h3>
                      <p className="text-xs text-zinc-500">
                        {selectedList === 0
                          ? (selectedBundle ? `Bundle: ${selectedBundle.document}` : "Select a bundle to view details.")
                          : (selected ? "Viewing selected requirement." : "Select a requirement to view details.")}
                      </p>
                    </div>
                    <div className="flex-1 p-4 overflow-y-auto overflow-x-hidden min-w-0">
                      {selectedList !== 0 && <RequirementDetailView requirement={selected} />}
                      {selectedList === 0 && <BundleDetailView bundle={selectedBundle} />}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </section>
        </div>
      </main>

      {error && (
        <div className="absolute w-full flex flex-row justify-end items-center p-4 pointer-events-none">
          <div className="w-max-100 y-max-120 y-overflow-scroll pointer-events-auto">
            <Alert variant="destructive">
              <AlertCircleIcon />
              <AlertTitle>Action failed</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          </div>
        </div>
      )}

      {showUploadDialog && (
        <>
          <div className="fixed inset-0 bg-gray-500 z-1000 opacity-25" onClick={() => setShowUploadDialog(false)} />
          <div className="fixed inset-0 flex justify-center items-center pointer-events-none z-1001 p-12">
            <div className="w-full max-w-80 pointer-events-auto">
              <FileUploadDialog
                onUploadComplete={async () => {
                  await getPdfs();
                  setShowUploadDialog(false);
                }}
                onClose={() => setShowUploadDialog(false)}
              />
            </div>
          </div>
        </>
      )}

      {showPdfViewer && pdfData && (
        <>
          <div className="fixed inset-0 bg-gray-500 z-1000 opacity-25" onClick={() => setShowPdfViewer(false)} />
          <div className="fixed inset-0 flex justify-center items-center pointer-events-none z-1001 p-12">
            <div className="w-full h-full pointer-events-auto bg-white overflow-hidden rounded-lg shadow-2xl">
              <div className="h-full flex flex-col">
                <div className="flex justify-between items-center p-4 border-b bg-white">
                  <h2 className="text-lg font-bold">PDF Viewer</h2>
                  <Button onClick={() => setShowPdfViewer(false)} variant="outline">
                    Close
                  </Button>
                </div>
                <div className="flex-1 overflow-auto">
                  <PDFViewer file={pdfData} />
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
