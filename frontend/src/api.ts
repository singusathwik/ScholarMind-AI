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
}
export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : `Request failed (${response.status}). Please try again.`,
    );
  }
  return response.json() as Promise<T>;
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
