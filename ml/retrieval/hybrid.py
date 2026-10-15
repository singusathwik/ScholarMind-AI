def rrf(rankings, constant=60, top_k=100):
    if constant < 0:
        raise ValueError("RRF constant must be nonnegative")
    scores = {}
    for ranking in rankings:
        seen = set()
        for rank, (pid, _) in enumerate(ranking, 1):
            if pid not in seen:
                scores[pid] = scores.get(pid, 0) + 1 / (constant + rank)
                seen.add(pid)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
