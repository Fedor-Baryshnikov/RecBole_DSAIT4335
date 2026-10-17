"""Top-k ranking metrics implemented independently from RecBole."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Iterable, Mapping, Sequence


MetricRow = dict[str, float]


def _unique_top_k(items: Sequence[str], k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = [str(item) for item in items[:k]]
    if len(top_k) != k:
        raise ValueError(f"a recommendation list contains fewer than {k} items")
    if len(top_k) != len(set(top_k)):
        raise ValueError("a recommendation list contains duplicate item IDs")
    return top_k


def user_accuracy(
    relevant: Iterable[str], recommended: Sequence[str], k: int
) -> MetricRow:
    """Calculate binary-relevance ranking metrics for one user."""
    truth = {str(item) for item in relevant}
    if not truth:
        raise ValueError("accuracy is undefined for a user with no relevant items")

    ranked = _unique_top_k(recommended, k)
    hits = [item in truth for item in ranked]
    hit_count = sum(hits)

    precision = hit_count / k
    recall = hit_count / len(truth)
    hit = float(hit_count > 0)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    reciprocal_rank = next(
        (1.0 / rank for rank, is_hit in enumerate(hits, start=1) if is_hit),
        0.0,
    )

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, is_hit in enumerate(hits, start=1)
        if is_hit
    )
    ideal_length = min(len(truth), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_length + 1))
    ndcg = dcg / idcg if idcg else 0.0

    running_hits = 0
    precision_sum = 0.0
    for rank, is_hit in enumerate(hits, start=1):
        if is_hit:
            running_hits += 1
            precision_sum += running_hits / rank
    average_precision = precision_sum / ideal_length if ideal_length else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit": hit,
        "mrr": reciprocal_rank,
        "ndcg": ndcg,
        "map": average_precision,
    }


def per_user_accuracy(
    ground_truth: Mapping[str, Iterable[str]],
    recommendations: Mapping[str, Sequence[str]],
    k: int = 10,
) -> dict[str, MetricRow]:
    """Evaluate each test user, assigning equal weight to every user."""
    ground_truth_users = {str(user_id) for user_id in ground_truth}
    recommendation_users = {str(user_id) for user_id in recommendations}
    if recommendation_users != ground_truth_users:
        missing = sorted(ground_truth_users - recommendation_users)
        extra = sorted(recommendation_users - ground_truth_users)
        raise ValueError(
            "recommendation users do not match test users "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )

    rows: dict[str, MetricRow] = {}
    for user_id, relevant in ground_truth.items():
        truth = {str(item) for item in relevant}
        if not truth:
            continue
        rows[str(user_id)] = user_accuracy(
            truth, recommendations[str(user_id)], k
        )
    return rows


def aggregate_accuracy(rows: Mapping[str, MetricRow]) -> MetricRow:
    """Macro-average the per-user accuracy metrics."""
    if not rows:
        raise ValueError("no evaluable users")
    metric_names = next(iter(rows.values())).keys()
    return {
        name: fmean(row[name] for row in rows.values()) for name in metric_names
    }
