import { useEffect, useState } from "react";
import { safeUrl, type Paper } from "./api";

export const LIBRARY_KEY = "scholarmind.reading-list.v1";
export type ReadingStatus = "to-read" | "reading" | "read";
export interface SavedPaper {
  paper: Paper;
  dataset: string;
  savedAt: string;
  status: ReadingStatus;
  notes: string;
}
export const paperKey = (dataset: string, id: string) =>
  JSON.stringify([dataset, id]);

export function parseLibrary(raw: string | null): SavedPaper[] {
  if (!raw) return [];
  const data: unknown = JSON.parse(raw);
  if (!Array.isArray(data) || data.length > 200)
    throw new Error("Invalid reading list");
  const keys = new Set<string>();
  return data.map((entry) => {
    const p = entry?.paper;
    if (
      !p ||
      typeof p.paper_id !== "string" ||
      typeof p.title !== "string" ||
      typeof p.abstract !== "string" ||
      typeof p.source !== "string" ||
      !Array.isArray(p.authors) ||
      !p.authors.every((s: unknown) => typeof s === "string") ||
      !Array.isArray(p.fields_of_study) ||
      !p.fields_of_study.every((s: unknown) => typeof s === "string") ||
      !(p.publication_year === null || Number.isInteger(p.publication_year)) ||
      !(p.url === null || typeof p.url === "string") ||
      typeof entry.dataset !== "string" ||
      !entry.dataset ||
      typeof entry.savedAt !== "string" ||
      !Number.isFinite(Date.parse(entry.savedAt)) ||
      typeof entry.notes !== "string" ||
      entry.notes.length > 5000 ||
      !["to-read", "reading", "read"].includes(entry.status)
    )
      throw new Error("Invalid reading list");
    const key = paperKey(entry.dataset, p.paper_id);
    if (keys.has(key)) throw new Error("Duplicate reading-list entry");
    keys.add(key);
    return entry as SavedPaper;
  });
}

function initialLibrary() {
  try {
    return {
      items: parseLibrary(localStorage.getItem(LIBRARY_KEY)),
      error: "",
    };
  } catch {
    return {
      items: [] as SavedPaper[],
      error:
        "Your saved reading list could not be read. Existing storage has not been changed. Export any recoverable browser data before saving new papers.",
    };
  }
}

export function useLibrary() {
  const [state, setState] = useState(initialLibrary);
  useEffect(() => {
    function sync(event: StorageEvent) {
      if (event.key !== LIBRARY_KEY && event.key !== null) return;
      try {
        setState({ items: parseLibrary(event.newValue), error: "" });
      } catch {
        setState((previous) => ({
          ...previous,
          error:
            "A reading-list change in another tab could not be read. Your current list has been kept.",
        }));
      }
    }
    window.addEventListener("storage", sync);
    return () => window.removeEventListener("storage", sync);
  }, []);
  function update(items: SavedPaper[]) {
    let error = "";
    try {
      localStorage.setItem(LIBRARY_KEY, JSON.stringify(items));
    } catch {
      error =
        "Browser storage is unavailable or full. Changes are kept for this session only; export JSON to keep a copy.";
    }
    setState({ items, error });
  }
  return {
    ...state,
    has: (dataset: string, id: string) =>
      state.items.some(
        (e) => paperKey(e.dataset, e.paper.paper_id) === paperKey(dataset, id),
      ),
    toggle(paper: Paper, dataset: string) {
      const key = paperKey(dataset, paper.paper_id);
      if (
        state.items.some((e) => paperKey(e.dataset, e.paper.paper_id) === key)
      ) {
        update(
          state.items.filter(
            (e) => paperKey(e.dataset, e.paper.paper_id) !== key,
          ),
        );
      } else if (state.items.length >= 200) {
        setState({
          ...state,
          error:
            "Your reading list holds up to 200 papers. Export or remove an entry before adding another.",
        });
      } else {
        // Ranking scores describe a particular query, not the saved paper itself.
        const {
          score: _score,
          score_type: _type,
          shared_title_terms: _terms,
          ...metadata
        } = paper;
        update([
          {
            paper: metadata,
            dataset,
            savedAt: new Date().toISOString(),
            status: "to-read",
            notes: "",
          },
          ...state.items,
        ]);
      }
    },
    edit(key: string, changes: Partial<Pick<SavedPaper, "status" | "notes">>) {
      update(
        state.items.map((e) =>
          paperKey(e.dataset, e.paper.paper_id) === key
            ? { ...e, ...changes }
            : e,
        ),
      );
    },
  };
}

function tex(value: string): string {
  const escapes: Record<string, string> = {
    "\\": "\\textbackslash{}",
    "{": "\\{",
    "}": "\\}",
    "%": "\\%",
    "&": "\\&",
    _: "\\_",
    $: "\\$",
    "#": "\\#",
    "~": "\\textasciitilde{}",
    "^": "\\textasciicircum{}",
  };
  return value
    .replace(/[\\{}%&_\$#~^]/g, (c) => escapes[c])
    .replace(/\s+/g, " ")
    .trim();
}

export function toBibTeX(entries: SavedPaper[]): string {
  return (
    entries
      .map((entry, i) => {
        const p = entry.paper;
        const fields = [`  title = {{${tex(p.title)}}}`];
        if (p.authors.length)
          fields.push(`  author = {${p.authors.map(tex).join(" and ")}}`);
        if (p.publication_year !== null)
          fields.push(`  year = {${p.publication_year}}`);
        const url = safeUrl(p.url);
        if (url) fields.push(`  url = {${tex(url)}}`);
        fields.push(
          `  note = {Corpus: ${tex(entry.dataset)}; source: ${tex(p.source)}; paper ID: ${tex(p.paper_id)}. Bibliographic metadata may be incomplete.}`,
        );
        return `@misc{scholarmind_${i + 1}_${p.paper_id.replace(/[^a-zA-Z0-9]/g, "") || "paper"},\n${fields.join(",\n")}\n}`;
      })
      .join("\n\n") + "\n"
  );
}

export function downloadText(filename: string, content: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
