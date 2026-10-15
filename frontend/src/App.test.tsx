import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import App from "./App";

const paper = {
  paper_id: "a",
  title: "Immune cells and cancer",
  abstract: "A test abstract.",
  source: "fixture",
  authors: [],
  publication_year: null,
  url: null,
  fields_of_study: [],
  score: 0.4,
  score_type: "RRF",
};
beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => ({
      ok: true,
      json: async () =>
        url.endsWith("/health")
          ? { index_ready: true, dataset: "fixture", documents: 4 }
          : url.endsWith("/benchmark-results")
            ? { available: false }
            : url.includes("/recommendations")
              ? { results: [] }
              : {
                  results: [paper],
                  timings: { total_ms: 12 },
                  returned_count: 1,
                  mode: "hybrid",
                  dataset: "fixture",
                },
    })),
  );
});

it("persists saved papers, notes and reading status across remounts", async () => {
  const user = userEvent.setup();
  const view = render(<App />);
  await user.type(screen.getByLabelText("Research question"), "cancer");
  await user.click(screen.getByRole("button", { name: /Find papers/ }));
  await user.click(await screen.findByRole("button", { name: "Save paper" }));
  await user.click(screen.getByRole("button", { name: "Reading list (1)" }));
  await user.type(
    screen.getByLabelText(`Notes for ${paper.title}`),
    "Discuss the methods",
  );
  await user.selectOptions(
    screen.getByLabelText(`Status for ${paper.title}`),
    "reading",
  );
  view.unmount();
  render(<App />);
  await user.click(screen.getByRole("button", { name: "Reading list (1)" }));
  expect(screen.getByLabelText(`Notes for ${paper.title}`)).toHaveValue(
    "Discuss the methods",
  );
  expect(screen.getByLabelText(`Status for ${paper.title}`)).toHaveValue(
    "reading",
  );
  await user.selectOptions(screen.getByLabelText("Reading status"), "read");
  expect(screen.getByText("No saved papers match")).toBeInTheDocument();
  await user.selectOptions(screen.getByLabelText("Reading status"), "all");
  await user.click(screen.getByRole("button", { name: "Remove paper" }));
  expect(
    screen.getByText("Your reading list starts with a paper"),
  ).toBeInTheDocument();
});

it("ignores a late cancelled response after a new search", async () => {
  const user = userEvent.setup();
  render(<App />);
  await user.type(screen.getByLabelText("Research question"), "first");
  let finish: (value: Response) => void = () => {};
  vi.mocked(fetch).mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  await user.click(screen.getByRole("button", { name: /Find papers/ }));
  await user.click(screen.getByRole("button", { name: "Cancel search" }));
  await user.clear(screen.getByLabelText("Research question"));
  await user.type(screen.getByLabelText("Research question"), "second");
  await user.click(screen.getByRole("button", { name: /Find papers/ }));
  expect(await screen.findByText(paper.title)).toBeInTheDocument();
  await act(async () =>
    finish({
      ok: true,
      json: async () => ({
        results: [{ ...paper, title: "Stale result" }],
        timings: { total_ms: 1 },
        returned_count: 1,
        mode: "hybrid",
        dataset: "fixture",
      }),
    } as Response),
  );
  expect(screen.queryByText("Stale result")).not.toBeInTheDocument();
  expect(screen.getByText(paper.title)).toBeInTheDocument();
});

it("shows the measured confidence intervals and inconclusive comparisons", async () => {
  const user = userEvent.setup();
  render(<App />);
  await screen.findByText("Local index ready");
  vi.mocked(fetch).mockResolvedValueOnce({
    ok: true,
    json: async () => ({
      available: true,
      comparisons: {
        "rerank minus hybrid": {
          mean_difference: 0.0055,
          ci95: [-0.0195, 0.0308],
          queries: 300,
          bootstrap_samples: 10000,
          seed: 42,
        },
      },
    }),
  } as Response);
  await user.click(screen.getByRole("button", { name: "Evaluation" }));
  expect(await screen.findByText("Inconclusive")).toBeInTheDocument();
  expect(
    screen.getByText("95% interval [-0.0195, +0.0308]"),
  ).toBeInTheDocument();
});
it("searches the selected mode and displays real response fields", async () => {
  const user = userEvent.setup();
  render(<App />);
  await user.type(screen.getByLabelText("Research question"), "cancer");
  await user.click(screen.getByRole("button", { name: "Semantic" }));
  await user.click(screen.getByRole("button", { name: /Find papers/ }));
  expect(await screen.findByText(paper.title)).toBeInTheDocument();
  expect(fetch).toHaveBeenCalledWith(
    "/api/search",
    expect.objectContaining({
      body: expect.stringContaining('"mode":"dense"'),
    }),
  );
  await user.click(screen.getByText("View details"));
  expect(await screen.findByRole("dialog")).toBeInTheDocument();
  await user.click(screen.getByLabelText("Close paper details"));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});
it("navigates and shows unevaluated state", async () => {
  const user = userEvent.setup();
  render(<App />);
  await user.click(screen.getByRole("button", { name: "Evaluation" }));
  expect(await screen.findByText("Not yet evaluated")).toBeInTheDocument();
  await user.click(screen.getByText("How it works"));
  expect(screen.getByText(/to a reading list/)).toBeInTheDocument();
});
it("shows request errors", async () => {
  const user = userEvent.setup();
  render(<App />);
  await user.type(screen.getByLabelText("Research question"), "test");
  vi.mocked(fetch).mockResolvedValueOnce({
    ok: false,
    status: 503,
    json: async () => ({ detail: "Model unavailable" }),
  } as Response);
  await user.click(screen.getByRole("button", { name: /Find papers/ }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Model unavailable",
  );
});
it("shows a loading state and disables duplicate submissions", async () => {
  const user = userEvent.setup();
  render(<App />);
  await user.type(screen.getByLabelText("Research question"), "test");
  vi.mocked(fetch).mockImplementationOnce(() => new Promise(() => {}));
  await user.click(screen.getByRole("button", { name: /Find papers/ }));
  expect(screen.getByRole("status")).toHaveTextContent("Searching the corpus");
  expect(screen.getByRole("button", { name: "Searching…" })).toBeDisabled();
});
