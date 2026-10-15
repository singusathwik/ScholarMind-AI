import { useState } from "react";
import { ArrowRight, Download, Trash2 } from "lucide-react";
import type { Paper } from "./api";
import {
  downloadText,
  paperKey,
  toBibTeX,
  type ReadingStatus,
  type useLibrary,
} from "./library";

export function ReadingList({
  library,
  open,
}: {
  library: ReturnType<typeof useLibrary>;
  open: (paper: Paper, dataset: string) => void;
}) {
  const [filter, setFilter] = useState("all");
  const [query, setQuery] = useState("");
  const [exportError, setExportError] = useState("");
  const items = library.items.filter(
    (e) =>
      (filter === "all" || e.status === filter) &&
      `${e.paper.title} ${e.notes} ${e.paper.authors.join(" ")}`
        .toLocaleLowerCase()
        .includes(query.toLocaleLowerCase()),
  );
  function exportList(format: "bib" | "json") {
    try {
      downloadText(
        `scholarmind-reading-list.${format}`,
        format === "bib"
          ? toBibTeX(library.items)
          : JSON.stringify(library.items, null, 2),
        format === "bib"
          ? "application/x-bibtex;charset=utf-8"
          : "application/json",
      );
      setExportError("");
    } catch {
      setExportError(
        "Download failed. Please try again in a browser that supports file downloads.",
      );
    }
  }
  return (
    <section className="reading-list">
      <span className="eyebrow">YOUR RESEARCH, WITH CONTEXT</span>
      <h1>A place for your next read.</h1>
      <p>
        Save papers, keep notes, and build a reading list. Stored in this
        browser only; export a backup before clearing browser data. Up to 200
        papers.
      </p>
      <div className="library-toolbar">
        <label>
          Search saved papers
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Title, author or note"
          />
        </label>
        <label>
          Reading status
          <select value={filter} onChange={(e) => setFilter(e.target.value)}>
            <option value="all">All papers</option>
            <option value="to-read">To read</option>
            <option value="reading">Reading</option>
            <option value="read">Read</option>
          </select>
        </label>
        <button
          disabled={!library.items.length}
          onClick={() => exportList("bib")}
        >
          <Download size={15} />
          Export BibTeX
        </button>
        <button
          disabled={!library.items.length}
          onClick={() => exportList("json")}
        >
          <Download size={15} />
          Export JSON
        </button>
      </div>
      <p className="note">
        BibTeX exports the entire list using available metadata only. Missing
        authors, year, DOI and journal are not invented. Personal notes are
        included in JSON, but excluded from BibTeX.
      </p>
      {exportError && (
        <p role="alert" className="error">
          {exportError}
        </p>
      )}
      <p className="library-count">
        {items.length} of {library.items.length} saved papers
      </p>
      {!items.length && (
        <div className="empty">
          <h2>
            {library.items.length
              ? "No saved papers match"
              : "Your reading list starts with a paper"}
          </h2>
          <p>
            {library.items.length
              ? "Change the search or status filter."
              : "Use Save paper on a search result to keep it here."}
          </p>
        </div>
      )}
      {items.map((entry) => {
        const key = paperKey(entry.dataset, entry.paper.paper_id);
        return (
          <article className="saved-paper" key={key}>
            <span className="eyebrow">
              {entry.dataset} · {entry.status.replace("-", " ")}
            </span>
            <h2>
              <button onClick={() => open(entry.paper, entry.dataset)}>
                {entry.paper.title}
                <ArrowRight size={16} />
              </button>
            </h2>
            <p className="abstract">
              {entry.paper.abstract || "Abstract unavailable."}
            </p>
            <div className="saved-controls">
              <label>
                Status
                <select
                  aria-label={`Status for ${entry.paper.title}`}
                  value={entry.status}
                  onChange={(e) =>
                    library.edit(key, {
                      status: e.target.value as ReadingStatus,
                    })
                  }
                >
                  <option value="to-read">To read</option>
                  <option value="reading">Reading</option>
                  <option value="read">Read</option>
                </select>
              </label>
              <button
                className="remove-paper"
                onClick={() => library.toggle(entry.paper, entry.dataset)}
              >
                <Trash2 size={14} />
                Remove paper
              </button>
            </div>
            <label className="notes-label">
              Notes
              <textarea
                aria-label={`Notes for ${entry.paper.title}`}
                value={entry.notes}
                onChange={(e) => library.edit(key, { notes: e.target.value })}
                maxLength={5000}
                rows={3}
                placeholder="What makes this paper useful?"
              />
            </label>
          </article>
        );
      })}
    </section>
  );
}
