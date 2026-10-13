"""Fit one hybrid with its selected parameters, evaluate it, and save its .pth."""

import argparse
import json
from pathlib import Path

from run_hyper import HYBRIDS, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=HYBRIDS, required=True)
    parser.add_argument("--params_file", type=Path, required=True)
    parser.add_argument("--checkpoint_dir", type=Path, default=ROOT / "run_outputs/Part1/Part1.5/best_models")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    import pandas as pd
    import torch
    from hybrids import HybridExperiment, checkpoint_manifest, metrics, top_items

    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    record = json.loads(args.params_file.read_text(encoding="utf-8"))
    if record["model"] != args.model:
        raise ValueError(f"Selection file does not describe {args.model}")
    if checkpoint_manifest(record["model_dir"]) != record["base_checkpoints"]:
        raise ValueError("Base checkpoints changed since tuning; rerun tune.py")
    experiment = HybridExperiment(record["model_dir"], record["data_dir"],
                                  record["fit_fraction"], record["seed"])
    if (experiment.fit_split.users.cpu().tolist() != record["fit_users"]
            or experiment.tune_split.users.cpu().tolist() != record["tune_users"]):
        raise ValueError("Validation partition changed since tuning; rerun tune.py")

    state = experiment.fit(args.model, record["params"], experiment.data.valid)
    rec = experiment.recommend(args.model, record["params"], state, experiment.data.test)
    result = metrics(rec, experiment.data.test)
    artifact = {"format_version": 1, **record, "fitted_state": state, "test_metrics": result}
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.checkpoint_dir / f"{args.model}.pth")

    rows = [{"model": args.model, "type": "hybrid", "tune_mrr@10": record["validation_score"],
             **{f"test_{metric}": value for metric, value in result.items()}}]
    for name, scores in experiment.scores.items():
        base_result = metrics(top_items(scores, experiment.data.test), experiment.data.test)
        rows.append({"model": name, "type": "base",
                     **{f"test_{metric}": value for metric, value in base_result.items()}})
    table = pd.DataFrame(rows)
    summary_file = args.checkpoint_dir.parent / "summary.csv"
    if summary_file.exists():
        previous = pd.read_csv(summary_file)
        previous = previous[~previous["model"].isin(table["model"])]
        table = pd.concat([previous, table], ignore_index=True)
    table.to_csv(summary_file, index=False)
    print(args.model, "parameters:", record["params"])
    print(args.model, "test:", result)
    print("saved:", args.checkpoint_dir / f"{args.model}.pth")


if __name__ == "__main__":
    main()
