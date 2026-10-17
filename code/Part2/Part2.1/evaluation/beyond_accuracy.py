"""Beyond-accuracy metrics implemented independently from RecBole."""

from __future__ import annotations

import math
from collections import Counter
from statistics import fmean
from typing import Iterable, Mapping, Sequence


def jaccard_distance(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return 1.0 - len(left_set & right_set) / len(union)


def intra_list_diversity(
    items: Sequence[str], genres: Mapping[str, Iterable[str]]
) -> float:
    """Mean pairwise Jaccard genre distance in a recommendation list."""
    known_items = [item for item in items if item in genres and set(genres[item])]
    if len(known_items) < 2:
        return 0.0
    distances = [
        jaccard_distance(genres[left], genres[right])
        for index, left in enumerate(known_items)
        for right in known_items[index + 1 :]
    ]
    return fmean(distances)


def item_popularity(histories: Mapping[str, Sequence[str]]) -> Counter[str]:
    """Count item interactions using training histories only."""
    return Counter(item for items in histories.values() for item in items)


def mean_self_information(
    items: Sequence[str], popularity: Mapping[str, int], catalog_size: int
) -> float:
    """Mean popularity-based novelty with Laplace smoothing."""
    if not items:
        return 0.0
    total = sum(popularity.values())
    denominator = total + max(catalog_size, 1)
    values = [
        -math.log2((popularity.get(item, 0) + 1) / denominator)
        for item in items
    ]
    return fmean(values)


def genre_distribution(
    items: Sequence[str],
    genres: Mapping[str, Iterable[str]],
    rank_weighted: bool = False,
) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for rank, item in enumerate(items, start=1):
        item_genres = set(genres.get(item, ()))
        if not item_genres:
            continue
        weight = 1.0 / math.log2(rank + 1) if rank_weighted else 1.0
        contribution = weight / len(item_genres)
        for genre in item_genres:
            counts[genre] += contribution
    total = sum(counts.values())
    return {genre: value / total for genre, value in counts.items()} if total else {}


def calibration_kl(
    history: Sequence[str],
    recommendations: Sequence[str],
    genres: Mapping[str, Iterable[str]],
    alpha: float = 0.01,
    rank_weighted: bool = True,
) -> float:
    """KL(profile || recommendations), in bits, with Steck-style smoothing."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    profile = genre_distribution(history, genres)
    recommended = genre_distribution(recommendations, genres, rank_weighted)
    if not profile:
        return 0.0
    result = 0.0
    for genre, probability in profile.items():
        smoothed = (1.0 - alpha) * recommended.get(genre, 0.0) + alpha * probability
        result += probability * math.log2(probability / smoothed)
    return result


def catalog_coverage(
    recommendations: Mapping[str, Sequence[str]], catalog: Iterable[str]
) -> float:
    catalog_set = set(catalog)
    if not catalog_set:
        return 0.0
    exposed = {
        item
        for items in recommendations.values()
        for item in items
        if item in catalog_set
    }
    return len(exposed) / len(catalog_set)


def exposure_counts(
    recommendations: Mapping[str, Sequence[str]],
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for items in recommendations.values():
        counts.update(items)
    return counts


def gini_index(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or sum(ordered) == 0:
        return 0.0
    n = len(ordered)
    weighted_sum = sum((index + 1) * value for index, value in enumerate(ordered))
    return (2.0 * weighted_sum) / (n * sum(ordered)) - (n + 1.0) / n


def normalized_entropy(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    if len(values) <= 1 or sum(values) == 0:
        return 0.0
    total = sum(values)
    entropy = -sum(
        (value / total) * math.log2(value / total) for value in values if value > 0
    )
    return entropy / math.log2(len(values))


def popularity_groups(
    popularity: Mapping[str, int], catalog: Iterable[str]
) -> dict[str, str]:
    """Split catalog items into equal-sized popularity-rank tertiles."""
    ordered = sorted(set(catalog), key=lambda item: (-popularity.get(item, 0), item))
    groups = {}
    size = len(ordered)
    for index, item in enumerate(ordered):
        third = min((3 * index) // max(size, 1), 2)
        groups[item] = ("head", "mid", "tail")[third]
    return groups


def _group_distribution(
    items: Sequence[str], groups: Mapping[str, str]
) -> dict[str, float]:
    counts = Counter(groups[item] for item in items if item in groups)
    total = sum(counts.values())
    return {
        label: counts[label] / total if total else 0.0
        for label in ("head", "mid", "tail")
    }


def jensen_shannon(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    labels = set(left) | set(right)
    midpoint = {
        label: (left.get(label, 0.0) + right.get(label, 0.0)) / 2
        for label in labels
    }

    def kl(first: Mapping[str, float]) -> float:
        return sum(
            value * math.log2(value / midpoint[label])
            for label, value in first.items()
            if value > 0 and midpoint[label] > 0
        )

    return 0.5 * kl(left) + 0.5 * kl(right)


def user_popularity_deviation(
    history: Sequence[str],
    recommendations: Sequence[str],
    groups: Mapping[str, str],
) -> float:
    return jensen_shannon(
        _group_distribution(history, groups),
        _group_distribution(recommendations, groups),
    )


def aggregate_beyond_accuracy(
    recommendations: Mapping[str, Sequence[str]],
    histories: Mapping[str, Sequence[str]],
    genres: Mapping[str, Iterable[str]],
    catalog_items: Iterable[str],
    k: int = 10,
    calibration_alpha: float = 0.01,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    catalog = set(catalog_items)
    popularity = item_popularity(histories)
    pop_groups = popularity_groups(popularity, catalog)
    per_user: dict[str, dict[str, float]] = {}

    for user_id, recommended in recommendations.items():
        top_k = list(recommended[:k])
        history = list(histories.get(user_id, ()))
        per_user[user_id] = {
            "ild": intra_list_diversity(top_k, genres),
            "novelty": mean_self_information(top_k, popularity, len(catalog)),
            "miscalibration": calibration_kl(
                history, top_k, genres, calibration_alpha, rank_weighted=True
            ),
            "popularity_deviation": user_popularity_deviation(
                history, top_k, pop_groups
            ),
        }

    aggregate = {
        name: fmean(row[name] for row in per_user.values())
        for name in next(iter(per_user.values()), {})
    }
    top_k_recommendations = {
        user_id: list(items[:k]) for user_id, items in recommendations.items()
    }
    aggregate["catalog_coverage"] = catalog_coverage(
        top_k_recommendations, catalog
    )
    exposure = exposure_counts(top_k_recommendations)
    aggregate["exposure_gini"] = gini_index(
        exposure.get(item, 0.0) for item in catalog
    )
    aggregate["exposure_entropy"] = normalized_entropy(
        exposure.get(item, 0.0) for item in catalog
    )
    return aggregate, per_user


def group_metric_summary(
    per_user_metrics: Mapping[str, Mapping[str, float]],
    user_attributes: Mapping[str, Mapping[str, str]],
    attribute: str,
    metric: str = "ndcg",
) -> dict[str, object]:
    """Report group means and pairwise Group Recommendation Unfairness."""
    grouped: dict[str, list[float]] = {}
    for user_id, metrics in per_user_metrics.items():
        group = user_attributes.get(user_id, {}).get(attribute)
        if group not in (None, "") and metric in metrics:
            grouped.setdefault(group, []).append(metrics[metric])
    means = {group: fmean(values) for group, values in sorted(grouped.items())}
    sizes = {group: len(values) for group, values in sorted(grouped.items())}
    labels = list(means)
    pairwise_gru = {
        f"{labels[left]}_vs_{labels[right]}": abs(
            means[labels[left]] - means[labels[right]]
        )
        for left in range(len(labels))
        for right in range(left + 1, len(labels))
    }
    return {
        "metric": metric,
        "group_sizes": sizes,
        "means": means,
        "pairwise_gru": pairwise_gru,
    }
