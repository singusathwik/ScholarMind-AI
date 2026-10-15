"""Predeclared A/C/E: methods on test; depth and RRF studies on training judgments only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.evaluation.benchmark import benchmark

if __name__ == "__main__":
    for depth in [10, 20, 50]:
        benchmark(
            split="train",
            limit=50,
            modes=["rerank"],
            rerank_depth=depth,
            output=f"experiments/results/ablation-depth-{depth}",
        )
    for constant in [10, 60, 100]:
        benchmark(
            split="train",
            limit=50,
            modes=["hybrid"],
            constant=constant,
            output=f"experiments/results/ablation-rrf-{constant}",
        )
