"""Retrieval evaluation metrics: P@K, R@K, MRR, NDCG@K.

Ground truth format (experiments/ground_truth.json):
{
  "q01": {"relevant_features": ["integration_R500", "Human Scale", ...]},
  ...
}
Relevance is binary (a feature either belongs to the query's ideal feature
set or not), graded relevance can be added later via {"feature": grade}.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence


def precision_at_k(ranked: Sequence[str], relevant: set, k: int) -> float:
    if k <= 0:
        return 0.0
    top = list(ranked)[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in relevant) / k


def recall_at_k(ranked: Sequence[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top = list(ranked)[:k]
    return sum(1 for r in top if r in relevant) / len(relevant)


def mrr(ranked: Sequence[str], relevant: set) -> float:
    for i, r in enumerate(ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set, k: int) -> float:
    dcg = 0.0
    for i, r in enumerate(list(ranked)[:k], start=1):
        if r in relevant:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking(
    ranked: Sequence[str],
    relevant: Sequence[str],
    *,
    ks: Sequence[int] = (5, 10),
) -> Dict[str, float]:
    rel = set(relevant)
    out: Dict[str, float] = {"mrr": mrr(ranked, rel)}
    for k in ks:
        out[f"p@{k}"] = precision_at_k(ranked, rel, k)
        out[f"r@{k}"] = recall_at_k(ranked, rel, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, rel, k)
    return out


def aggregate(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(q[k] for q in per_query) / len(per_query) for k in keys}
