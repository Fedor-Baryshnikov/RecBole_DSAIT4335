"""Part 1.5 equivalent of root run_hyper.py, using a hybrid objective."""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
HYBRIDS = ["WeightedHybrid", "Switching", "Cascade", "FeatureCombination",
           "FeatureAugmentation", "MetaLevel"]


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=HYBRIDS, required=True)
    parser.add_argument("--params_file", type=Path)
    parser.add_argument("--output_file", type=Path)
    parser.add_argument("--model_dir", type=Path, default=ROOT / "run_outputs/Part1/Part1.2/best_models")
    parser.add_argument("--data_dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--fit_fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--algo", choices=["exhaustive", "random", "bayes"], default="exhaustive")
    parser.add_argument("--max_evals", type=int, default=100)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main():
    args = get_parser().parse_args()
    import torch
    from recbole.trainer import HyperTuning
    from hybrids import HybridExperiment

    if args.threads < 1 or args.max_evals < 1 or args.early_stop < 1:
        raise ValueError("threads, max_evals and early_stop must be positive")
    torch.set_num_threads(args.threads)
    # Hyperopt 0.2.5 reads this when creating its search RNG.
    os.environ["HYPEROPT_FMIN_SEED"] = str(args.seed)
    params_file = args.params_file or HERE / f"{args.model}.hyper"
    output_file = args.output_file or ROOT / f"run_outputs/Part1/Part1.5/{args.model}.result"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    experiment = HybridExperiment(args.model_dir, args.data_dir, args.fit_fraction, args.seed)
    trials = []

    def objective_function(config_dict, config_file_list=None):
        result = experiment.objective(args.model, config_dict)
        trials.append({"params": dict(config_dict), "result": result})
        return result

    hp = HyperTuning(objective_function, algo=args.algo, max_evals=args.max_evals,
                     early_stop=args.early_stop, params_file=str(params_file))
    if args.algo == "exhaustive":
        # Unlike the Part 1.2 runner, finish the grid even after ten non-improvements.
        from hyperopt.early_stop import no_progress_loss
        hp.early_stop_fn = no_progress_loss(hp.max_evals + 1)
    hp.run()
    hp.export_result(str(output_file))
    # Select explicitly: HyperTuning treats a best score of zero as uninitialized.
    best = max(trials, key=lambda trial: trial["result"]["best_valid_score"])
    record = {
        "model": args.model, "params": best["params"], "metric": "mrr@10",
        "validation_score": best["result"]["best_valid_score"],
        "validation_metrics": best["result"]["best_valid_result"],
        "seed": args.seed, "fit_fraction": args.fit_fraction,
        "model_dir": str(args.model_dir.resolve()), "data_dir": str(args.data_dir.resolve()),
        "base_checkpoints": experiment.manifest,
        "dataset": experiment.data.config["dataset"],
        "fit_users": experiment.fit_split.users.cpu().tolist(),
        "tune_users": experiment.tune_split.users.cpu().tolist(),
    }
    best_file = output_file.with_suffix(".best.json")
    best_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("best params:", best["params"])
    print("best validation result:", best["result"]["best_valid_result"])
    print("saved:", best_file)


if __name__ == "__main__":
    main()
