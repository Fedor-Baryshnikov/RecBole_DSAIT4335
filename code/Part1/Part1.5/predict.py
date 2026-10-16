"""Load a saved hybrid and recommend movies for existing users, without refitting.

Example (from the repository root):
python code/Part1/Part1.5/predict.py --model WeightedHybrid --users 42 100

Python: hybrid = SavedHybrid(checkpoint_path); hybrid.recommend(["42"])
The original base checkpoints and dataset must still be available.
"""

import argparse
import ast
import copy
import importlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch

from run_hybrid import Experiment, MODEL_DIR, OUTPUT, SCRIPTS, latest_checkpoints


class SavedHybrid:
    def __init__(self, checkpoint_path):
        self.checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if self.checkpoint.get("format_version") != 2:
            raise ValueError("Use a checkpoint saved by the rebuilt Part 1.5")
        paths = {Path(p).parent.name: p for p in latest_checkpoints(str(MODEL_DIR))}
        if self.checkpoint["base_checkpoints"] != paths:
            raise ValueError("Restore the base checkpoints referenced by the saved hybrid")
        # Loads base-model weights and computes their scores; never calls fit.
        with redirect_stdout(io.StringIO()):
            self.experiment = Experiment()

    @torch.no_grad()
    def recommend(self, user_ids=None, split="test"):
        """Return {original user ID: [10 original item IDs]}.

        test excludes training + validation interactions, matching final evaluation.
        valid excludes training interactions. Neither option uses target labels.
        Omit user_ids to recommend for all users in the chosen split.
        """
        if split not in ("valid", "test"):
            raise ValueError("split must be 'valid' or 'test'")
        experiment = self.experiment
        dataset = experiment.data.dataset
        source = getattr(experiment.data, split)
        users = source.users
        if user_ids is not None:
            tokens = [str(user) for user in user_ids]
            if not tokens:
                return {}
            try:
                ids = dataset.token2id(dataset.uid_field, tokens)
            except (ValueError, KeyError) as error:
                raise ValueError("Specify existing dataset user IDs") from error
            users = torch.as_tensor(ids, dtype=torch.long, device=users.device)
            if (users == 0).any() or not torch.isin(users, source.users).all():
                raise ValueError("Specify users present in the selected split")
        # Deliberately supply no truth labels to the prediction code.
        selection = SimpleNamespace(users=users, seen=source.seen)
        name = self.checkpoint["model"]
        params = self.checkpoint["params"]
        state = self.checkpoint["fitted_state"]
        names = params.get("model_names", "all")
        names = list(experiment.models) if names == "all" else names.split(",")
        models = {name: experiment.models[name] for name in names}
        data = copy.copy(experiment.data)
        data.test = selection
        if name == "WeightedHybrid":
            _, rec = experiment.weighted(params, models, data, state=state)
        else:
            rec = self.part14(name, params, state, models, data)
        user_tokens = dataset.id2token(dataset.uid_field, users.cpu().numpy())
        items = dataset.id2token(dataset.iid_field, rec.cpu().numpy())
        return {str(user): items[i].tolist() for i, user in enumerate(user_tokens)}

    def part14(self, name, params, state, models, data):
        # Execute only prediction statements from the original main() function.
        # The saved state supplies the choices its fitting loop normally learns.
        path = SCRIPTS[name]
        module = importlib.import_module(path.stem)
        nodes = next(node.body for node in ast.parse(path.read_text(encoding="utf-8")).body
                     if isinstance(node, ast.FunctionDef) and node.name == "main")
        namespace = dict(vars(module), **state)
        namespace.update(data=data, models=models, args=SimpleNamespace(**params),
                         scores={name: self.experiment.scores[id(model)] for name, model in models.items()})
        if name == "Switching":
            namespace["edges"] = state["edges"].to(data.test.users.device)
        statements = {
            "Cascade": ["first", "test_rec"],
            "Switching": ["group", "recommend"],
            "FeatureAugmentation": ["recommend", "profiles"],
            "FeatureCombination": ["collab", "genres", "recommend"],
            "MetaLevel": ["meta_scores", "test_rec"],
        }[name]

        def defines(node, key):
            if isinstance(node, ast.FunctionDef):
                return node.name == key
            return isinstance(node, ast.Assign) and any(
                isinstance(child, ast.Name) and child.id == key
                for target in node.targets for child in ast.walk(target))

        selected = [next(node for node in nodes if defines(node, key)) for key in statements]
        tree = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
        exec(compile(tree, str(path), "exec"), namespace)
        if name == "Switching":
            return namespace["recommend"](data.test)
        if name == "FeatureAugmentation":
            return namespace["recommend"](namespace["profiles"], data.test)
        if name == "FeatureCombination":
            return namespace["recommend"](state["best_weight"], data.test)
        return namespace["test_rec"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=SCRIPTS)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--users", nargs="+")
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    args = parser.parse_args()
    torch.set_num_threads(4)
    hybrid = SavedHybrid(args.checkpoint or OUTPUT / "best_models" / f"{args.model}.pth")
    if hybrid.checkpoint["model"] != args.model:
        parser.error("--model does not match the checkpoint")
    print(json.dumps(hybrid.recommend(args.users, args.split), indent=2))
