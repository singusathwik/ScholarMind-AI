import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import {
  LIBRARY_KEY,
  parseLibrary,
  toBibTeX,
  useLibrary,
  type SavedPaper,
} from "./library";

const entry: SavedPaper = {
  dataset: "scifact",
  savedAt: "2026-10-15T00:00:00Z",
  status: "to-read",
  notes: "private note",
  paper: {
    paper_id: "42",
    title: "Cells & {DNA} 50%_#",
    abstract: "abstract",
    authors: [],
    publication_year: null,
    source: "scifact",
    url: null,
    fields_of_study: [],
  },
};
beforeEach(() => localStorage.clear());
afterEach(() => vi.restoreAllMocks());
it("exports escaped titles without invented metadata or personal notes", () => {
  const bib = toBibTeX([entry]);
  expect(bib).toContain("Cells \\& \\{DNA\\} 50\\%\\_\\#");
  for (const field of [
    "author =",
    "year =",
    "doi =",
    "journal =",
    "private note",
  ])
    expect(bib).not.toContain(field);
});
it("includes available authors/year and rejects unsafe URLs", () => {
  const bib = toBibTeX([
    {
      ...entry,
      paper: {
        ...entry.paper,
        authors: ["A One", "B Two"],
        publication_year: 2024,
        url: "javascript:alert(1)",
      },
    },
  ]);
  expect(bib).toContain("author = {A One and B Two}");
  expect(bib).toContain("year = {2024}");
  expect(bib).not.toContain("url =");
});
it("keeps the same paper ID in separate corpora and rejects duplicates within one corpus", () => {
  expect(
    parseLibrary(JSON.stringify([entry, { ...entry, dataset: "other" }])),
  ).toHaveLength(2);
  expect(() => parseLibrary(JSON.stringify([entry, entry]))).toThrow();
  expect(() => parseLibrary('{"bad":true}')).toThrow();
  expect(() =>
    parseLibrary(
      JSON.stringify([{ ...entry, paper: { ...entry.paper, authors: null } }]),
    ),
  ).toThrow();
});
it("preserves corrupt storage on load and reports the failure", () => {
  localStorage.setItem(LIBRARY_KEY, "broken");
  const { result } = renderHook(useLibrary);
  expect(result.current.error).toContain("could not be read");
  expect(localStorage.getItem(LIBRARY_KEY)).toBe("broken");
});
it("syncs another tab and retains the current list on invalid updates", () => {
  const { result } = renderHook(useLibrary);
  act(() =>
    window.dispatchEvent(
      new StorageEvent("storage", {
        key: LIBRARY_KEY,
        newValue: JSON.stringify([entry]),
      }),
    ),
  );
  expect(result.current.items).toHaveLength(1);
  act(() =>
    window.dispatchEvent(
      new StorageEvent("storage", { key: LIBRARY_KEY, newValue: "broken" }),
    ),
  );
  expect(result.current.items).toHaveLength(1);
  expect(result.current.error).toContain("another tab");
});
it("retains an in-memory list and warns when browser storage is full", () => {
  const { result } = renderHook(useLibrary);
  vi.spyOn(Object.getPrototypeOf(localStorage), "setItem").mockImplementation(
    () => {
      throw new DOMException("Full", "QuotaExceededError");
    },
  );
  act(() => result.current.toggle(entry.paper, entry.dataset));
  expect(result.current.items).toHaveLength(1);
  expect(result.current.error).toContain("session only");
});
