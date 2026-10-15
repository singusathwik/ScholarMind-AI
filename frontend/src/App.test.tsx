import { render, screen } from "@testing-library/react";
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
