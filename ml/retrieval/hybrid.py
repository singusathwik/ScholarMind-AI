import math


def rrf(rankings, constant=60, top_k=100, weights=None):
    if constant < 0:
        raise ValueError("RRF constant must be nonnegative")
    weights = [1.0] * len(rankings) if weights is None else weights
    if len(weights) != len(rankings) or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("RRF weights must be finite, nonnegative and match rankings")
    scores = {}
    for ranking, weight in zip(rankings, weights):
        if weight == 0:
            continue
        seen = set()
        for rank, (pid, _) in enumerate(ranking, 1):
            if pid not in seen:
                scores[pid] = scores.get(pid, 0) + weight / (constant + rank)
                seen.add(pid)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
