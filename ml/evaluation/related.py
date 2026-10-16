"""Proxy construction and metrics. Never imported by the serving recommender."""

import math
import numpy as np


def co_evidence(qrels):
    related = {}
    for judgments in qrels.values():
        positive = {pid for pid, grade in judgments.items() if grade > 0}
        if len(positive) > 1:
            for pid in positive:
                related.setdefault(pid, set()).update(positive - {pid})
    return related


def disjoint_development(train, test):
    forbidden = set(test) | set().union(set(), *test.values())
    result = {p: neighbors - forbidden for p, neighbors in train.items() if p not in forbidden}
    result = {p: neighbors for p, neighbors in result.items() if neighbors}
    assert not forbidden.intersection(set(result) | set().union(set(), *result.values()))
    return result


def components(qrels):
    graph = {p: set(neighbors) for p, neighbors in qrels.items()}
    for p, neighbors in qrels.items():
        for n in neighbors:
            graph.setdefault(n, set()).add(p)
    labels = {}
    for p in sorted(graph):
        if p in labels:
            continue
        pending = [p]
        while pending:
            node = pending.pop()
            if node in labels:
                continue
            labels[node] = p
            pending.extend(graph[node] - labels.keys())
    return labels


def related_metrics(ranking, relevant):
    ids = [p for p, _ in ranking]
    if len(ids) != len(set(ids)) or not relevant:
        raise ValueError("Unique ranked IDs and nonempty relevance labels required")
    top = ids[:5]
    dcg = sum(1 / math.log2(i + 2) for i, p in enumerate(top) if p in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(5, len(relevant))))
    hits = 0
    ap = rr = 0.0
    for i, p in enumerate(ids[:100], 1):
        if p in relevant:
            hits += 1
            ap += hits / i
            rr = rr or 1 / i
    return {
        "NDCG@5": dcg / ideal,
        "Recall@5": len(set(top) & relevant) / len(relevant),
        "Hit@5": float(bool(set(top) & relevant)),
        "Precision@5": len(set(top) & relevant) / 5,
        "Recall@100": hits / len(relevant),
        "MAP@100": ap / len(relevant),
        "MRR@100": rr,
    }


def grouped_bootstrap(left, right, groups, weights=None, seed=42, repetitions=10000):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    weights = np.ones(len(left)) if weights is None else np.asarray(weights, dtype=float)
    if not len(left) or not (len(left) == len(right) == len(groups) == len(weights)):
        raise ValueError("Aligned nonempty observations required")
    if not np.all(np.isfinite(left + right + weights)) or np.any(weights <= 0):
        raise ValueError("Finite observations and positive weights required")
    names = sorted(set(groups))
    if len(names) < 2:
        raise ValueError("At least two independent groups required for a cluster interval")
    totals = np.array([sum(w for w, g in zip(weights, groups) if g == name) for name in names])
    changes = (right - left) * weights
    sums = np.array([sum(d for d, g in zip(changes, groups) if g == name) for name in names])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), (repetitions, len(names)))
    samples = sums[draws].sum(axis=1) / totals[draws].sum(axis=1)
    return {
        "mean_difference": float(changes.sum() / weights.sum()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "groups": len(names),
        "observations": len(left),
        "seed": seed,
        "bootstrap_samples": repetitions,
        "method": "paired cluster percentile bootstrap",
    }
