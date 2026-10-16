"""Run the existing Part 1.3/1.4 scripts with parameters supplied by Part 1.5."""

import argparse
import ast
import copy
import importlib
import io
import json
import os
import sys
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = ROOT / "run_outputs/Part1/Part1.5"
MODEL_DIR = ROOT / "run_outputs/Part1/Part1.2/best_models"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "Part1.4"))

from utils import get_scores, latest_checkpoints, load, user_metrics

SCRIPTS = {
    "WeightedHybrid": HERE.parent / "Part1.3/weighted_hybrid.py",
    "Switching": HERE.parent / "Part1.4/switching.py",
    "Cascade": HERE.parent / "Part1.4/cascade.py",
    "FeatureCombination": HERE.parent / "Part1.4/feature_combination.py",
    "FeatureAugmentation": HERE.parent / "Part1.4/feature_augmentation.py",
    "MetaLevel": HERE.parent / "Part1.4/metalevel.py",
}


class Experiment:
    def __init__(self):
        self.data, self.models = load(str(MODEL_DIR), str(ROOT / "dataset"))
        self.scores = {id(model): get_scores(model, self.data) for model in self.models.values()}
        self.paths = {Path(p).parent.name: p for p in latest_checkpoints(str(MODEL_DIR))}

        # The hybrids fit weights/model choices on one half of validation users.
        # Hyperparameters are compared on the other half. Test users are not used.
        users = self.data.valid.users
        if len(users) < 2:
            raise ValueError("Need at least two validation users")
        order = np.random.RandomState(42).permutation(len(users))
        self.fit = copy.copy(self.data.valid)
        self.validation = copy.copy(self.data.valid)
        self.fit.users = users[order[:len(users) // 2]]
        self.validation.users = users[order[len(users) // 2:]]

    def run(self, name, params, final=False):
        names = params.get("model_names", "all")
        names = list(self.models) if names == "all" else names.split(",")
        models = {name: self.models[name] for name in names}
        data = copy.copy(self.data)
        data.valid = self.data.valid if final else self.fit
        data.test = self.data.test if final else self.validation
        # Let the tuner print validation results. The original weighted script
        # labels its evaluation group "test", even when we supply validation.
        with redirect_stdout(io.StringIO()):
            if name == "WeightedHybrid":
                state, rec = self.weighted(params, models, data)
            else:
                state, rec = self.part14(name, params, models, data)
        metrics = {f"{key}@10": value.mean().item()
                   for key, value in user_metrics(rec, data.test).items()}
        return state, metrics

    def part14(self, name, params, models, data):
        module = importlib.import_module(SCRIPTS[name].stem)
        captured = {}

        def report(label, valid_rec, test_rec, scores, data, out_dir, details, control=None):
            # The scripts keep fitted choices inside main(). Capture them at
            # their existing report call, without adding a new hybrid algorithm.
            local = sys._getframe(1).f_locals
            keys = {
                "Cascade": ["best_pair"],
                "Switching": ["edges", "best"],
                "FeatureAugmentation": ["best"],
                "FeatureCombination": ["best_weight"],
                "MetaLevel": ["best_name"],
            }[name]
            captured["state"] = {key: local[key] for key in keys}
            captured["rec"] = test_rec

        argv = [str(SCRIPTS[name])]
        for key, value in params.items():
            if key not in ("model_names", "weight"):
                argv.extend(["--" + key, str(value)])
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", argv))
            stack.enter_context(patch.multiple(
                module, load=lambda *args: (data, models),
                get_scores=lambda model, data: self.scores[id(model)], report=report))
            if "weight" in params:
                stack.enter_context(patch.object(module, "WEIGHTS", [params["weight"]]))
            module.main()
        return captured["state"], captured["rec"]

    def weighted(self, params, models, data, state=None):
        # Part 1.3 has hard-coded constants and no main() function. Replace only
        # its input assignments in memory; execute its original fitting/ranking.
        path = SCRIPTS["WeightedHybrid"]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        values = {"paths": list(models), "TOP_N": params["TOP_N"],
                  "REGULARIZATION": params["REGULARIZATION"]}
        if not 0 <= values["REGULARIZATION"] < 0.1:
            raise ValueError("Use weighted-hybrid regularisation in [0, 0.1)")
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                key = node.targets[0].id
                if key in values:
                    node.value = ast.parse(repr(values[key]), mode="eval").body
        # Supply loaded base models instead of reloading them on every trial.
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module == "recbole.quick_start")]

        def bundle(name):
            model = models[name]

            def predict(interaction):
                try:
                    return model.full_sort_predict(interaction)
                except NotImplementedError:
                    # Part 1.4's get_scores already handles predict-only models.
                    return self.scores[id(model)][interaction[data.dataset.uid_field]]

            config = {"model": name, "device": data.config["device"]}
            wrapper = SimpleNamespace(eval=model.eval, full_sort_predict=predict)
            return config, wrapper, data.dataset, None, None, None

        def frame(matrix, users):
            rows, items = matrix[users].nonzero(as_tuple=True)
            return pd.DataFrame({
                "user_id": data.dataset.id2token(data.dataset.uid_field, users[rows].cpu().numpy()),
                "item_id": data.dataset.id2token(data.dataset.iid_field, items.cpu().numpy()),
            })

        def read_split(path, **kwargs):
            # Only the three split files are read by weighted_hybrid.py.
            return frames[Path(path).name.split(".")[-2]].copy()

        namespace = {"__name__": "__main__", "__file__": str(path), "load_data_and_model": bundle}
        if state is None:
            frames = {
                "train": frame(data.train_matrix, torch.arange(data.n_users, device=data.train_matrix.device)),
                "valid": frame(data.valid.truth, data.valid.users),
                "test": frame(data.test.truth, data.test.users),
            }
            with patch.object(pd, "read_csv", read_split):
                exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
        else:
            # Replay the original prediction loop with saved weights. Skip all
            # fitting statements and the evaluation code that reads test labels.
            def assignment(node, key):
                return (isinstance(node, ast.Assign)
                        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == key)

            start = next(i for i, node in enumerate(tree.body) if assignment(node, "TOP_N"))
            setup = [node for node in tree.body[start:]
                     if assignment(node, "TOP_N") or assignment(node, "K")
                     or isinstance(node, ast.FunctionDef) and node.name == "features_for_user"]
            loop = next(node for node in tree.body if isinstance(node, ast.For)
                        and ast.unparse(node.iter) == "test.items()")
            loop.body = loop.body[:next(i for i, node in enumerate(loop.body) if assignment(node, "relevant"))]
            tree.body = tree.body[:start] + setup + [loop]
            namespace.update(
                weights=np.array([state["weights"][name] for name in models]),
                train={uid: set(data.test.seen[uid].nonzero().flatten().tolist())
                       for uid in data.test.users.tolist()}, valid={},
                test=dict.fromkeys(data.test.users.tolist()), recommendations={})
            exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
        users = data.dataset.id2token(data.dataset.uid_field, data.test.users.cpu().numpy())
        tokens = np.array([namespace["recommendations"][str(user)] for user in users])
        rec = torch.as_tensor(data.dataset.token2id(data.dataset.iid_field, tokens),
                              device=data.test.users.device)
        return {"weights": dict(zip(models, namespace["weights"].tolist()))}, rec


def tune(name, params_file, output_dir):
    from hyperopt.early_stop import no_progress_loss
    from recbole.trainer import HyperTuning

    experiment = Experiment()
    trials = []

    def objective(params, config_file_list=None):
        _, metrics = experiment.run(name, params)
        trials.append({"params": dict(params), "validation_metrics": metrics})
        return {"model": name, "best_valid_score": metrics["mrr@10"],
                "valid_score_bigger": True, "best_valid_result": metrics, "test_result": {}}

    def full_grid(*args, **kwargs):
        tuner = HyperTuning(*args, **kwargs)
        tuner.early_stop_fn = no_progress_loss(tuner.max_evals + 1)
        return tuner

    # Use the actual root runner's Hyperopt function. Loading just this function
    # avoids its optional Ray imports and lets us supply the hybrid objective.
    path = ROOT / "run_hyper.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "hyperopt_tune"]
    namespace = {"HyperTuning": full_grid, "objective_function": objective}
    exec(compile(tree, str(path), "exec"), namespace)
    namespace["hyperopt_tune"](SimpleNamespace(
        config_files=None, params_file=str(params_file), display_file=None,
        output_file=str(output_dir / f"{name}.result")))

    best = max(trials, key=lambda trial: trial["validation_metrics"]["mrr@10"])
    best.update(format_version=2, model=name, base_checkpoints=experiment.paths)
    (output_dir / f"{name}.best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")


def train(name, output_dir):
    best = json.loads((output_dir / f"{name}.best.json").read_text(encoding="utf-8"))
    if best.get("format_version") != 2 or best["model"] != name:
        raise ValueError("Rerun tune.py to replace the previous Part 1.5 results")
    experiment = Experiment()
    if best["base_checkpoints"] != experiment.paths:
        raise ValueError("Base-model checkpoints changed; rerun tune.py")
    state, metrics = experiment.run(name, best["params"], final=True)
    folder = output_dir / "best_models"
    folder.mkdir(exist_ok=True)
    # Hybrid weights/choices plus references to the already-trained base models.
    torch.save({**best, "fitted_state": state, "test_metrics": metrics}, folder / f"{name}.pth")
    row = {"model": name, "valid_mrr@10": best["validation_metrics"]["mrr@10"], **metrics}
    summary = output_dir / "summary.csv"
    table = pd.read_csv(summary) if summary.exists() else pd.DataFrame(columns=["model"])
    # Drop rows/columns from the previous Part 1.5 summary format.
    table = table.reindex(columns=list(row)).dropna(subset=["valid_mrr@10"])
    table = table[table["model"] != name]
    pd.concat([table, pd.DataFrame([row])], ignore_index=True).to_csv(summary, index=False)
    print(name, metrics)
    print("Saved:", folder / f"{name}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=SCRIPTS)
    parser.add_argument("--mode", required=True, choices=["tune", "train"])
    parser.add_argument("--params_file", type=Path)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    os.environ["HYPEROPT_FMIN_SEED"] = "42"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "tune":
        console = args.output_dir / f"{args.model}_console.txt"
        print(f"Tuning {args.model}; log: {console}", flush=True)
        with console.open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
            tune(args.model, args.params_file or HERE / f"{args.model}.hyper", args.output_dir)
    else:
        train(args.model, args.output_dir)
