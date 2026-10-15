import math


def metrics(ranking, qrels):
    """TREC-compatible linear-gain nDCG; binary AP/recall; unjudged treated nonrelevant.

    MAP and MRR are truncated at the supplied retrieval depth (100 in experiments).
    AP denominator is all known relevant documents, including unretrieved ones.
    """
    ids = [pid for pid, _ in ranking]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate run IDs")
    relevant = {pid for pid, grade in qrels.items() if grade > 0}
    ideal = sorted(qrels.values(), reverse=True)[:10]
    idcg = sum(grade / math.log2(i + 2) for i, grade in enumerate(ideal))
    dcg = sum(qrels.get(pid, 0) / math.log2(i + 2) for i, pid in enumerate(ids[:10]))
    hits, ap, rr = 0, 0.0, 0.0
    for rank, pid in enumerate(ids, 1):
        if pid in relevant:
            hits += 1
            ap += hits / rank
            if not rr:
                rr = 1 / rank
    return {
        "NDCG@10": dcg / idcg if idcg else 0.0,
        "Recall@10": len(set(ids[:10]) & relevant) / len(relevant) if relevant else 0.0,
        "Recall@100": len(set(ids[:100]) & relevant) / len(relevant) if relevant else 0.0,
        "MAP@100": ap / len(relevant) if relevant else 0.0,
        "MRR@100": rr,
        "Precision@10": len(set(ids[:10]) & relevant) / 10,
    }
