import { useEffect, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Check,
  ChevronRight,
  Clock3,
  ExternalLink,
  FlaskConical,
  Layers3,
  Search,
  Sparkles,
  X,
} from "lucide-react";
import {
  api,
  safeUrl,
  type Benchmark,
  type Health,
  type Mode,
  type Paper,
  type Results,
} from "./api";

const modes: { id: Mode; name: string; caption: string }[] = [
  { id: "bm25", name: "Keyword", caption: "BM25 lexical matching" },
  { id: "dense", name: "Semantic", caption: "MiniLM embeddings" },
  { id: "hybrid", name: "Hybrid", caption: "Reciprocal rank fusion" },
  {
    id: "rerank",
    name: "Hybrid + reranking",
    caption: "Cross-encoder refinement",
  },
];
const examples = [
  "How do immune cells respond to cancer?",
  "Vitamin D and bone health",
  "Genetic factors in cardiovascular disease",
];

export default function App() {
  const [page, setPage] = useState<"search" | "benchmarks" | "about">("search");
  const [health, setHealth] = useState<Health | null>(null);
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<Mode>("hybrid");
  const [results, setResults] = useState<Results | null>(null);
  const [submitted, setSubmitted] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<Paper | null>(null);
  const [related, setRelated] = useState<Paper[]>([]);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState("");
  const [benchmark, setBenchmark] = useState<Benchmark | null>(null);
  const [benchmarkError, setBenchmarkError] = useState("");
  const [resultPage, setResultPage] = useState(0);
  const [topK, setTopK] = useState(20);
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    api<Health>("/health")
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);
  useEffect(() => {
    if (page === "benchmarks") {
      setBenchmarkError("");
      api<Benchmark>("/benchmark-results")
        .then(setBenchmark)
        .catch((e) => setBenchmarkError(e.message));
    }
  }, [page]);
  useEffect(() => {
    if (!selected) return;
    dialogRef.current?.showModal();
    let current = true;
    setRelated([]);
    setDetailError("");
    setDetailLoading(true);
    api<{ results: Paper[] }>(
      `/papers/${encodeURIComponent(selected.paper_id)}/recommendations`,
    )
      .then((response) => {
        if (current) setRelated(response.results);
      })
      .catch((e) => {
        if (current) setDetailError(e.message);
      })
      .finally(() => {
        if (current) setDetailLoading(false);
      });
    return () => {
      current = false;
    };
  }, [selected]);

  async function search(text = query) {
    if (!text.trim() || loading) return;
    setQuery(text);
    setLoading(true);
    setError("");
    setResultPage(0);
    try {
      const data = await api<Results>("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text.trim(), mode, top_k: topK }),
      });
      setResults(data);
      setSubmitted(text);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }
  async function openPaper(paper: Paper) {
    setSelected(paper);
  }

  return (
    <div className="app">
      <header>
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setPage("search");
          }}
        >
          <span className="brand-icon">
            <BookOpen size={23} />
          </span>
          ScholarMind <span className="ai">AI</span>
        </a>
        <nav aria-label="Main navigation">
          <button
            className={page === "search" ? "active" : ""}
            onClick={() => setPage("search")}
          >
            <Search size={16} />
            Discover
          </button>
          <button
            className={page === "benchmarks" ? "active" : ""}
            onClick={() => setPage("benchmarks")}
          >
            <FlaskConical size={16} />
            Evaluation
          </button>
          <button
            className={page === "about" ? "active" : ""}
            onClick={() => setPage("about")}
          >
            How it works
          </button>
        </nav>
        <span className="local-badge">
          <span className={health?.index_ready ? "dot ready" : "dot"} />
          {health?.index_ready ? "Local index ready" : "Index not connected"}
        </span>
      </header>
      <main>
        {page === "search" && (
          <>
            <section className={`hero ${results ? "compact" : ""}`}>
              <div className="eyebrow">
                <span /> EVIDENCE STARTS WITH DISCOVERY
              </div>
              <h1>
                Follow your curiosity.
                <br />
                <em>Find the research.</em>
              </h1>
              <p>
                Explore scientific literature with keyword precision and
                semantic understanding.
                <br className="desktop-break" /> Four retrieval methods. Real
                papers. Transparent results.
              </p>
            </section>
            <section className="search-panel" aria-label="Research search">
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  void search();
                }}
              >
                <label className="sr-only" htmlFor="query">
                  Research question
                </label>
                <Search size={23} />
                <input
                  id="query"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  maxLength={1000}
                  placeholder="What would you like to explore?"
                  required
                />
                <button
                  className="primary"
                  type="submit"
                  disabled={loading || !query.trim()}
                >
                  {loading ? "Searching…" : "Find papers"}
                  <ArrowRight size={17} />
                </button>
              </form>
              <div className="search-options">
                <div
                  className="mode-options"
                  role="group"
                  aria-label="Retrieval method"
                >
                  {modes.map((m) => (
                    <button
                      key={m.id}
                      title={m.caption}
                      aria-pressed={mode === m.id}
                      className={mode === m.id ? "chosen" : ""}
                      onClick={() => setMode(m.id)}
                    >
                      {mode === m.id && <Check size={13} />} {m.name}
                    </button>
                  ))}
                </div>
                <label className="top-k">
                  Results{" "}
                  <select
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                  >
                    <option>10</option>
                    <option>20</option>
                    <option>50</option>
                  </select>
                </label>
              </div>
            </section>
            {!results && (
              <div className="examples">
                <span>TRY A QUESTION</span>
                {examples.map((q) => (
                  <button
                    key={q}
                    onClick={() => void search(q)}
                    disabled={loading}
                  >
                    {q}
                    <ArrowRight size={13} />
                  </button>
                ))}
              </div>
            )}
            <div className="corpus-strip">
              <span>
                <span className="dot ready" /> ACTIVE CORPUS{" "}
                <strong>{health?.dataset ?? "Unavailable"}</strong>
              </span>
              <span>
                {health ? health.documents.toLocaleString() : "—"} indexed
                documents
              </span>
              <span>Local inference · No external LLM</span>
            </div>
            {error && (
              <div className="error" role="alert">
                {error}{" "}
                <button onClick={() => void search()}>Retry search</button>
              </div>
            )}
            {loading ? (
              <div className="loading" role="status">
                <Sparkles size={19} /> Searching the corpus. First use may load
                model weights.
                <div className="skeleton" />
                <div className="skeleton" />
              </div>
            ) : results ? (
              <section className="results">
                <div className="section-title">
                  <div>
                    <span className="eyebrow">SEARCH RESULTS</span>
                    <h2>Research for your question</h2>
                    <p>“{submitted}”</p>
                  </div>
                  <span className="latency">
                    <Clock3 size={14} />
                    {results.timings.total_ms.toFixed(0)} ms ·{" "}
                    {results.returned_count} retrieved ·{" "}
                    {modes.find((m) => m.id === results.mode)?.name}
                  </span>
                </div>
                {results.results.length === 0 && (
                  <div className="empty">
                    <Search />
                    <h3>No papers found</h3>
                    <p>
                      Try a different question or a broader retrieval method.
                    </p>
                  </div>
                )}
                {results.results
                  .slice(resultPage * 10, (resultPage + 1) * 10)
                  .map((paper, i) => (
                    <PaperCard
                      key={paper.paper_id}
                      paper={paper}
                      rank={resultPage * 10 + i + 1}
                      open={() => void openPaper(paper)}
                    />
                  ))}
                {results.results.length > 10 && (
                  <div className="pagination">
                    <button
                      disabled={resultPage === 0}
                      onClick={() => setResultPage((p) => p - 1)}
                    >
                      <ArrowLeft size={15} />
                      Previous
                    </button>
                    <span>
                      Page {resultPage + 1} of{" "}
                      {Math.ceil(results.results.length / 10)}
                    </span>
                    <button
                      disabled={(resultPage + 1) * 10 >= results.results.length}
                      onClick={() => setResultPage((p) => p + 1)}
                    >
                      Next
                      <ArrowRight size={15} />
                    </button>
                  </div>
                )}
              </section>
            ) : (
              <section className="explain-grid">
                <div className="explain-intro">
                  <span className="eyebrow">A BETTER PATH TO THE PAPER</span>
                  <h2>
                    Search beyond
                    <br />
                    the exact words.
                  </h2>
                  <p>
                    Choose how you discover. Compare how each method ranks the
                    same scientific corpus.
                  </p>
                  <button onClick={() => setPage("about")}>
                    Inside the retrieval pipeline
                    <ArrowRight size={16} />
                  </button>
                </div>
                <Feature
                  number="01"
                  icon={<Search size={22} />}
                  title="Start with meaning"
                  text="Ask a natural-language question. Transformer embeddings connect your query to related research."
                />
                <Feature
                  number="02"
                  icon={<Layers3 size={22} />}
                  title="Bring methods together"
                  text="Hybrid search combines lexical and semantic rankings. Optional reranking scores each query–paper pair."
                />
                <Feature
                  number="03"
                  icon={<FlaskConical size={22} />}
                  title="Inspect the evidence"
                  text="Explore benchmark measurements, latency and trade-offs. Improvements are measured, never assumed."
                />
              </section>
            )}
          </>
        )}
        {page === "benchmarks" && (
          <section className="dashboard">
            <span className="eyebrow">MEASURED, NOT ASSUMED</span>
            <h1>Retrieval, under evaluation.</h1>
            <p>
              Saved experiments from real benchmark queries and relevance
              judgments.
            </p>
            {benchmarkError && (
              <p role="alert" className="error">
                {benchmarkError}
              </p>
            )}
            {!benchmark && !benchmarkError && (
              <p role="status">Loading evaluation…</p>
            )}
            {benchmark && !benchmark.available && (
              <div className="empty">
                <FlaskConical />
                <h2>Not yet evaluated</h2>
                <p>Run the benchmark CLI to generate verified results.</p>
              </div>
            )}
            {benchmark?.available && (
              <>
                <div className="corpus-strip">
                  <span>
                    {benchmark.dataset} · {benchmark.split}
                  </span>
                  <span>
                    {benchmark.query_count} queries{" "}
                    {benchmark.partial && "(subset)"}
                  </span>
                  <span>CPU · warmed models · retrieval depth 100</span>
                </div>
                <div className="chart" aria-label="NDCG at 10 comparison">
                  {Object.entries(benchmark.summary ?? {}).map(
                    ([name, row]) => (
                      <div key={name}>
                        <span>{name}</span>
                        <div className="bar-track">
                          <div style={{ width: `${row["NDCG@10"] * 100}%` }} />
                        </div>
                        <strong>{row["NDCG@10"].toFixed(3)}</strong>
                      </div>
                    ),
                  )}
                </div>
                <div className="table-scroll">
                  <table>
                    <caption>Effectiveness and warmed query latency</caption>
                    <thead>
                      <tr>
                        <th>Method</th>
                        {[
                          "NDCG@10",
                          "Recall@10",
                          "Recall@100",
                          "MAP@100",
                          "P50_ms",
                          "P95_ms",
                        ].map((m) => (
                          <th key={m}>{m}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(benchmark.summary ?? {}).map(
                        ([name, row]) => (
                          <tr key={name}>
                            <th>{name}</th>
                            {[
                              "NDCG@10",
                              "Recall@10",
                              "Recall@100",
                              "MAP@100",
                              "P50_ms",
                              "P95_ms",
                            ].map((m) => (
                              <td key={m}>
                                {row[m].toFixed(m.includes("ms") ? 1 : 3)}
                              </td>
                            ))}
                          </tr>
                        ),
                      )}
                    </tbody>
                  </table>
                </div>
                <p className="note">
                  MAP and MRR are truncated at 100 retrieved documents. Unjudged
                  documents count as nonrelevant. Latency depends on hardware;
                  see the technical report for paired confidence intervals and
                  limitations.
                </p>
              </>
            )}
          </section>
        )}
        {page === "about" && (
          <section className="about">
            <span className="eyebrow">TRANSPARENT BY DESIGN</span>
            <h1>
              From a question
              <br />
              to a reading list.
            </h1>
            <p>
              ScholarMind runs retrieval models locally over an indexed corpus.
              It does not generate answers or invent references.
            </p>
            {modes.map((m, i) => (
              <article key={m.id}>
                <span className="step">0{i + 1}</span>
                <div>
                  <h2>{m.name}</h2>
                  <p>
                    {
                      [
                        "BM25 ranks documents by matching query terms, accounting for term frequency, rarity and document length.",
                        "A Sentence Transformer encodes your question and paper text. FAISS searches normalized vectors using exact cosine similarity.",
                        "Reciprocal Rank Fusion combines BM25 and dense rankings with a rank constant of 60 and 100 candidates per retriever.",
                        "A cross-encoder scores the question alongside each of the top candidates. This adds computation and may improve or worsen relevance.",
                      ][i]
                    }
                  </p>
                </div>
                <ArrowDown size={20} />
              </article>
            ))}
            <div className="note">
              SciFact metadata is limited. Missing authors, dates and source
              URLs remain missing. MiniLM truncates text to its supported token
              limit; recommendations reflect content similarity, not a
              probability or personalized advice.
            </div>
          </section>
        )}
      </main>
      <footer>
        <span>
          <BookOpen size={16} />
          ScholarMind AI
        </span>
        <span>Scientific literature. Thoughtfully explored.</span>
        <button onClick={() => setPage("benchmarks")}>
          View evaluation
          <ChevronRight size={14} />
        </button>
      </footer>
      {selected && (
        <dialog
          ref={dialogRef}
          onCancel={() => setSelected(null)}
          aria-labelledby="paper-title"
          className="detail"
        >
          <div className="detail-inner">
            <button
              className="close"
              autoFocus
              aria-label="Close paper details"
              onClick={() => setSelected(null)}
            >
              <X />
            </button>
            <span className="eyebrow">
              {selected.source} · PAPER {selected.paper_id}
            </span>
            <h2 id="paper-title">{selected.title}</h2>
            <p className="metadata">
              {selected.authors.length
                ? selected.authors.join(", ")
                : "Author metadata unavailable"}{" "}
              · {selected.publication_year ?? "Year unavailable"}
            </p>
            <h3>Abstract</h3>
            <p className="full-abstract">
              {selected.abstract || "Abstract unavailable."}
            </p>
            {safeUrl(selected.url) && (
              <a href={safeUrl(selected.url)} target="_blank" rel="noreferrer">
                Open original source
                <ExternalLink size={15} />
              </a>
            )}
            <hr />
            <h3>Continue your reading</h3>
            <p className="note">
              Related by embedding similarity. Scores are cosine similarities,
              not relevance probabilities.
            </p>
            {detailLoading && <p role="status">Finding related papers…</p>}
            {detailError && <p role="alert">{detailError}</p>}
            {related.map((p) => (
              <article className="related" key={p.paper_id}>
                <button onClick={() => void openPaper(p)}>
                  {p.title}
                  <ArrowRight size={16} />
                </button>
                <span>Cosine similarity {p.score?.toFixed(3)}</span>
                {!!p.shared_title_terms?.length && (
                  <p>Shared title terms: {p.shared_title_terms.join(", ")}</p>
                )}
              </article>
            ))}
          </div>
        </dialog>
      )}
    </div>
  );
}
function Feature({
  number,
  icon,
  title,
  text,
}: {
  number: string;
  icon: React.ReactNode;
  title: string;
  text: string;
}) {
  return (
    <article className="feature">
      <div className="feature-top">
        {icon}
        <span>{number}</span>
      </div>
      <h3>{title}</h3>
      <p>{text}</p>
    </article>
  );
}
function PaperCard({
  paper,
  rank,
  open,
}: {
  paper: Paper;
  rank: number;
  open: () => void;
}) {
  return (
    <article className="paper-card">
      <span className="rank">{String(rank).padStart(2, "0")}</span>
      <div className="paper-content">
        <div className="paper-meta">
          {paper.source}
          <span>•</span>
          {paper.publication_year ?? "Year unavailable"}
          <span className="score">
            {paper.score_type}: {paper.score?.toFixed(4)}
          </span>
        </div>
        <h3>
          <button onClick={open}>{paper.title}</button>
        </h3>
        {paper.authors.length > 0 && (
          <p className="authors">{paper.authors.join(", ")}</p>
        )}
        <p className="abstract">{paper.abstract || "Abstract unavailable"}</p>
        <div className="paper-actions">
          <button onClick={open}>
            View details
            <ArrowRight size={14} />
          </button>
          <button onClick={open}>
            <Layers3 size={14} />
            Find related papers
          </button>
          {safeUrl(paper.url) && (
            <a href={safeUrl(paper.url)} target="_blank" rel="noreferrer">
              Original paper
              <ExternalLink size={14} />
            </a>
          )}
        </div>
      </div>
    </article>
  );
}
