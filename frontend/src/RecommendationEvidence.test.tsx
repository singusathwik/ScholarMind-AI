import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { RecommendationEvidence } from "./RecommendationEvidence";

it("keeps proxy and click evidence separate and reports inconclusive intervals", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: true,
      json: async () => ({
        available: true,
        local: {
          anchors: 58,
          summary: {
            dense: {
              "NDCG@5": 0.57,
              "Recall@5": 0.67,
              "Hit@5": 0.79,
              "MRR@100": 0.62,
            },
          },
          comparison: {
            mean_difference: 0.02,
            ci95: [-0.003, 0.05],
            groups: 22,
          },
        },
        external: {
          events: 1000,
          seed_papers: 862,
          summary: {
            dense: { "Adj-NDCG@10": 0.53, "Adj-P@1": 0.19, "Adj-MRR": 0.39 },
          },
          comparison: {
            mean_difference: 0.001,
            ci95: [-0.004, 0.007],
            groups: 862,
          },
        },
      }),
    })),
  );
  render(<RecommendationEvidence />);
  expect(
    await screen.findByText("Local co-evidence proxy"),
  ).toBeInTheDocument();
  expect(screen.getByText(/58 test papers/)).toBeInTheDocument();
  expect(
    screen.getByText(/1000 SciDocs test click events/),
  ).toBeInTheDocument();
  expect(screen.getAllByText(/Difference remains inconclusive/)).toHaveLength(
    2,
  );
  expect(screen.queryByText(/Positive interval/)).not.toBeInTheDocument();
});

it("shows missing benchmark state without fabricated metrics", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({ ok: true, json: async () => ({ available: false }) })),
  );
  render(<RecommendationEvidence />);
  expect(
    await screen.findByText(/No matching recommendation evaluation/),
  ).toBeInTheDocument();
  expect(screen.queryByRole("table")).not.toBeInTheDocument();
});
