export type Mode = "bm25" | "dense" | "hybrid" | "rerank";
export interface Paper {
  paper_id: string;
  title: string;
  abstract: string;
  source: string;
  authors: string[];
  publication_year: number | null;
  url: string | null;
  fields_of_study: string[];
  score?: number;
  score_type?: string;
  shared_title_terms?: string[];
}
export interface Results {
  results: Paper[];
  mode: Mode;
  dataset: string;
  timings: Record<string, number>;
  returned_count: number;
}
export interface Health {
  status: string;
  index_ready: boolean;
  dataset: string;
  documents: number;
}
export interface Benchmark {
  available: boolean;
  dataset?: string;
  split?: string;
  query_count?: number;
  partial?: boolean;
  timestamp?: string;
  summary?: Record<string, Record<string, number>>;
  comparisons?: Record<
    string,
    {
      mean_difference: number;
      ci95: [number, number];
      queries: number;
      bootstrap_samples: number;
      seed: number;
    }
  >;
}
export async function api<T>(
  path: string,
  init?: RequestInit,
  timeoutMs = 180000,
): Promise<T> {
  const controller = new AbortController();
  const abort = () => controller.abort(init?.signal?.reason);
  if (init?.signal?.aborted) abort();
  init?.signal?.addEventListener("abort", abort, { once: true });
  const timer = setTimeout(
    () =>
      controller.abort(
        new DOMException(
          "The request timed out. The server may still be loading a model; try again shortly.",
          "TimeoutError",
        ),
      ),
    timeoutMs,
  );
  try {
    const response = await fetch(`/api${path}`, {
      ...init,
      signal: controller.signal,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(
        typeof body.detail === "string"
          ? body.detail
          : `Request failed (${response.status}). Please try again.`,
      );
    }
    return (await response.json()) as T;
  } catch (error) {
    if (controller.signal.aborted) throw controller.signal.reason;
    if (error instanceof TypeError)
      throw new Error(
        "Cannot reach the backend. Check that the local server is running, then retry.",
      );
    throw error;
  } finally {
    clearTimeout(timer);
    init?.signal?.removeEventListener("abort", abort);
  }
}
export function safeUrl(url: string | null) {
  if (!url) return undefined;
  try {
    const u = new URL(url);
    return ["https:", "http:"].includes(u.protocol) ? u.href : undefined;
  } catch {
    return undefined;
  }
}
