"""Command-line entry point for the standalone Task 2.1 evaluator."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .accuracy import aggregate_accuracy, per_user_accuracy
from .beyond_accuracy import aggregate_beyond_accuracy, group_metric_summary
from .data import (
    ground_truth_by_user,
    history_by_user,
    load_interactions,
    load_item_genres,
    load_recommendations,
    load_user_groups,
)


def validate_protocol(metadata: dict[str, object], k: int) -> None:
    """Reject recommendation files produced under a different protocol."""
    dataset = metadata.get("dataset")
    if dataset not in (None, "ml-100k"):
        raise ValueError(f"expected ml-100k recommendations, found {dataset}")
    seed = metadata.get("seed")
    if seed not in (None, 2020):
        raise ValueError(f"expected seed 2020, found {seed}")
    stored_k = metadata.get("k")
    if stored_k is not None and int(stored_k) < k:
        raise ValueError(f"recommendation file contains top-{stored_k}, not top-{k}")

    eval_args = metadata.get("eval_args")
    if eval_args is None:
        return
    if not isinstance(eval_args, dict):
        raise ValueError("eval_args must be a JSON object")
    if eval_args.get("split", {}).get("RS") != [0.8, 0.1, 0.1]:
        raise ValueError("expected the common 80/10/10 random split")
    if eval_args.get("group_by") != "user" or eval_args.get("order") != "RO":
        raise ValueError("expected user-grouped random-order evaluation")
    mode = eval_args.get("mode")
    full_sort = mode == "full" or (
        isinstance(mode, dict)
        and mode.get("valid") == "full"
        and mode.get("test") == "full"
    )
    if not full_sort:
        raise ValueError("expected full-sort validation and test evaluation")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate repository top-k recommendation JSON independently."
    )
    parser.add_argument("--train", required=True, help="exported training TSV")
    parser.add_argument("--valid", required=True, help="exported validation TSV")
    parser.add_argument("--test", required=True, help="exported test TSV")
    parser.add_argument("--recommendations", required=True, help="test recommendation JSON")
    parser.add_argument(
        "--items", default="dataset/ml-100k/ml-100k.item", help="item metadata"
    )
    parser.add_argument(
        "--users", default="dataset/ml-100k/ml-100k.user", help="user metadata"
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--calibration-alpha", type=float, default=0.01)
    parser.add_argument("--output", required=True, help="aggregate JSON output")
    parser.add_argument("--per-user-output", help="optional per-user TSV output")
    return parser.parse_args()


def write_per_user(path: str | Path, rows: dict[str, dict[str, float]]):
    metric_names = sorted({name for metrics in rows.values() for name in metrics})
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["user_id", *metric_names], delimiter="\t"
        )
        writer.writeheader()
        for user_id in sorted(rows):
            writer.writerow({"user_id": user_id, **rows[user_id]})


def main():
    args = parse_args()
    train_rows = load_interactions(args.train)
    valid_rows = load_interactions(args.valid)
    test_rows = load_interactions(args.test)
    metadata, recommendations = load_recommendations(args.recommendations, args.k)
    validate_protocol(metadata, args.k)
    genres = load_item_genres(args.items)
    user_attributes = load_user_groups(args.users)

    ground_truth = ground_truth_by_user(test_rows)
    histories = history_by_user(train_rows)
    interaction_catalog = {
        row["item_id"] for row in [*train_rows, *valid_rows, *test_rows]
    }
    recommended_items = {
        item for items in recommendations.values() for item in items[: args.k]
    }
    unknown_items = sorted(recommended_items - interaction_catalog)
    if unknown_items:
        raise ValueError(
            f"recommendations contain items outside the interaction catalog: {unknown_items[:5]}"
        )

    accuracy_rows = per_user_accuracy(ground_truth, recommendations, args.k)
    accuracy = aggregate_accuracy(accuracy_rows)
    beyond, beyond_rows = aggregate_beyond_accuracy(
        recommendations,
        histories,
        genres,
        interaction_catalog,
        k=args.k,
        calibration_alpha=args.calibration_alpha,
    )
    per_user = {
        user_id: {**accuracy_rows[user_id], **beyond_rows[user_id]}
        for user_id in accuracy_rows
    }

    result: dict[str, object] = {
        "protocol": {
            "dataset": metadata.get("dataset", "ml-100k"),
            "model": metadata.get("model"),
            "split": "test",
            "k": args.k,
            "seed": metadata.get("seed"),
            "eval_args": metadata.get("eval_args"),
            "relevance": "all held-out test interactions",
            "averaging": "macro over test users",
            "calibration_alpha": args.calibration_alpha,
        },
        "evaluated_users": len(accuracy_rows),
        "accuracy": accuracy,
        "beyond_accuracy": beyond,
        "group_fairness": {
            attribute: group_metric_summary(
                per_user, user_attributes, attribute, "ndcg"
            )
            for attribute in ("gender", "age")
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if args.per_user_output:
        write_per_user(args.per_user_output, per_user)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
