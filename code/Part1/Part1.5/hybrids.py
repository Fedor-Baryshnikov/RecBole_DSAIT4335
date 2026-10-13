"""Fit and evaluate the Part 1.3/1.4 hybrids without retraining base models."""

import copy
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "Part1.4"))

from cascade import rerank
from feature_augmentation import augment
from feature_combination import genre_matrix
from metalevel import neighbor_scores, user_embedding
from utils import (
    DEVICE, K, get_scores, item_knn_scores, latest_checkpoints, load,
    top_items, user_metrics,
)

DEFAULT_MODELS = ROOT / "run_outputs/Part1/Part1.2/best_models"
DEFAULT_DATA = ROOT / "dataset"


def checkpoint_manifest(model_dir):
    paths = latest_checkpoints(str(model_dir))
    if not paths:
        raise FileNotFoundError(f"No base-model checkpoints found in {model_dir}")
    return [
        {"path": str(Path(p).resolve()), "size": Path(p).stat().st_size,
         "mtime_ns": Path(p).stat().st_mtime_ns}
        for p in paths
    ]


def load_hybrid(path):
    """Return (experiment, checkpoint) without refitting the saved hybrid.

    Requires the original Part 1.2 checkpoints and dataset. For example:
        experiment, checkpoint = load_hybrid("best_models/WeightedHybrid.pth")
        rec = experiment.recommend(
            checkpoint["model"], checkpoint["params"],
            checkpoint["fitted_state"], experiment.data.test,
        )
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported Part 1.5 hybrid checkpoint format")
    if checkpoint_manifest(checkpoint["model_dir"]) != checkpoint["base_checkpoints"]:
        raise ValueError("Base checkpoints differ from those used by the saved hybrid")
    experiment = HybridExperiment(
        checkpoint["model_dir"], checkpoint["data_dir"],
        checkpoint["fit_fraction"], checkpoint["seed"],
    )
    if (experiment.fit_split.users.cpu().tolist() != checkpoint["fit_users"]
            or experiment.tune_split.users.cpu().tolist() != checkpoint["tune_users"]):
        raise ValueError("Validation partition differs from the saved hybrid")
    return experiment, checkpoint


def subset(split, users):
    result = copy.copy(split)
    result.users = users
    return result


def validation_partition(split, fit_fraction, seed):
    if not 0 < fit_fraction < 1:
        raise ValueError("fit_fraction must be between 0 and 1")
    if len(split.users) < 2:
        raise ValueError("At least two validation users are needed")
    order = np.random.RandomState(seed).permutation(len(split.users))
    cut = max(1, min(len(order) - 1, int(len(order) * fit_fraction)))
    users = split.users[torch.as_tensor(order, device=split.users.device)]
    return subset(split, users[:cut]), subset(split, users[cut:])


def metrics(rec, split):
    return {f"{name}@{K}": values.mean().item()
            for name, values in user_metrics(rec, split).items()}


def fit_weights(gram, target, regularization):
    """Solve Part 1.3's balanced squared loss using its sufficient statistics."""
    if regularization < 0:
        raise ValueError("REGULARIZATION must be nonnegative")
    equal = np.ones(len(target)) / len(target)
    fit = minimize(
        fun=lambda w: w @ gram @ w - 2 * target @ w
        + regularization * np.sum((w - equal) ** 2),
        jac=lambda w: 2 * (gram @ w - target)
        + 2 * regularization * (w - equal),
        x0=equal.copy(), method="SLSQP", bounds=[(0, 1)] * len(equal),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                     "jac": lambda w: np.ones_like(w)},
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not fit.success:
        raise RuntimeError(f"Weight fitting failed: {fit.message}")
    return fit.x


class HybridExperiment:
    def __init__(self, model_dir=DEFAULT_MODELS, data_dir=DEFAULT_DATA,
                 fit_fraction=0.5, seed=42):
        self.manifest = checkpoint_manifest(model_dir)
        self.data, models = load(str(model_dir), str(data_dir))
        self.scores = {name: get_scores(model, self.data)
                       for name, model in models.items()}
        self.embeddings = {name: embedding for name, model in models.items()
                           if (embedding := user_embedding(model)) is not None}
        # Base scores and embeddings stay fixed throughout the search.
        self.fit_split, self.tune_split = validation_partition(
            self.data.valid, fit_fraction, seed)
        self._raw_scores = None
        self._weight_statistics = {}
        self._genres = None

    def select(self, candidates, recommend, fit_split):
        if not candidates:
            raise ValueError("No compatible base models for this hybrid")
        return max(candidates, key=lambda candidate:
                   metrics(recommend(candidate, fit_split), fit_split)["mrr@10"])

    def weighted_features(self, split, top_n):
        if top_n < K:
            raise ValueError(f"TOP_N must be at least {K}")
        if self._raw_scores is None:
            self._raw_scores = [s.detach().cpu().numpy().astype(float)
                                for s in self.scores.values()]
        for uid in split.users.tolist():
            eligible = ~split.seen[uid].cpu().numpy().copy()
            eligible[0] = False
            eligible = np.flatnonzero(eligible)
            if len(eligible) < K:
                raise ValueError(f"User {uid} has fewer than {K} eligible items")
            ranked, columns = [], []
            for scores in self._raw_scores:
                values = scores[uid, eligible]
                if not np.isfinite(values).all():
                    raise ValueError("Base model returned nonfinite scores")
                ranked.append(eligible[np.argsort(-values)[:top_n]])
                span = values.max() - values.min()
                normalized = np.zeros(self.data.n_items)
                if span > 0:
                    normalized[eligible] = (values - values.min()) / span
                columns.append(normalized)
            items = np.unique(np.concatenate(ranked))
            yield uid, items, np.column_stack([c[items] for c in columns])

    def weighted_statistics(self, split, top_n):
        key = (top_n, tuple(split.users.tolist()))
        if key not in self._weight_statistics:
            n = len(self.scores)
            grams = [np.zeros((n, n)), np.zeros((n, n))]
            target = np.zeros(n)
            counts = np.zeros(2, dtype=int)
            for uid, items, features in self.weighted_features(split, top_n):
                labels = split.truth[uid, torch.as_tensor(items, device=DEVICE)]
                labels = labels.cpu().numpy().astype(bool)
                for label in (0, 1):
                    x = features[labels == label]
                    grams[label] += x.T @ x
                    counts[label] += len(x)
                    if label == 1:
                        target += x.sum(axis=0)
            if (counts == 0).any():
                raise ValueError("Weight fitting needs positive and negative examples")
            self._weight_statistics[key] = (
                0.5 * (grams[0] / counts[0] + grams[1] / counts[1]),
                0.5 * target / counts[1],
            )
        return self._weight_statistics[key]

    def combination_scores(self, params):
        if self._genres is None:
            self._genres = torch.nn.functional.normalize(genre_matrix(self.data), dim=1)
        collab = torch.nn.functional.normalize(self.data.train_matrix.T, dim=1)
        vectors = torch.cat([collab, params["weight"] * self._genres], dim=1)
        return item_knn_scores(vectors, self.data.train_matrix, params["n_neighbors"])

    def augmentation_scores(self, first, params):
        profiles = augment(self.scores[first], self.data.train_matrix,
                           params["n_pseudo"], params["weight"])
        return item_knn_scores(profiles.T, profiles, params["n_neighbors"])

    def fit(self, name, params, fit_split):
        """Learn weights/model choices using only fit_split's validation labels."""
        if name == "WeightedHybrid":
            gram, target = self.weighted_statistics(fit_split, params["TOP_N"])
            weights = fit_weights(gram, target, params["REGULARIZATION"])
            return {"weights": dict(zip(self.scores, weights.tolist()))}
        if name == "Cascade":
            pair = self.select(list(permutations(self.scores, 2)),
                               lambda pair, split: rerank(
                                   self.scores[pair[0]], self.scores[pair[1]], split,
                                   params["n_candidates"]), fit_split)
            return {"first": pair[0], "second": pair[1]}
        if name == "Switching":
            n = params["n_groups"]
            if n < 1:
                raise ValueError("n_groups must be positive")
            counts = self.data.train_count
            edges = torch.quantile(counts[counts > 0],
                                   torch.linspace(0, 1, n + 1, device=DEVICE)[1:-1])
            groups = torch.bucketize(counts, edges, right=True)
            values = {model: user_metrics(top_items(scores, fit_split), fit_split)["mrr"]
                      for model, scores in self.scores.items()}
            fallback = max(values, key=lambda model: values[model].mean().item())
            best = {}
            for group in range(n):
                mask = groups[fit_split.users] == group
                best[str(group)] = (max(values, key=lambda model:
                                       values[model][mask].mean().item())
                                    if mask.any() else fallback)
            return {"edges": edges.cpu().tolist(), "models": best}
        if name == "FeatureCombination":
            return {}
        if name == "FeatureAugmentation":
            first = self.select(list(self.scores), lambda model, split:
                                top_items(self.augmentation_scores(model, params), split),
                                fit_split)
            return {"first": first}
        if name == "MetaLevel":
            embedding = self.select(list(self.embeddings), lambda model, split:
                                    top_items(neighbor_scores(
                                        self.embeddings[model], self.data.train_matrix,
                                        params["n_neighbors"]), split), fit_split)
            return {"embedding": embedding}
        raise ValueError(f"Unknown hybrid: {name}")

    def recommend(self, name, params, state, split):
        if name == "WeightedHybrid":
            weights = np.array([state["weights"][model] for model in self.scores])
            rows = [items[np.argsort(-(features @ weights))[:K]]
                    for _, items, features in self.weighted_features(split, params["TOP_N"])]
            return torch.as_tensor(np.stack(rows), dtype=torch.long, device=DEVICE)
        if name == "Cascade":
            return rerank(self.scores[state["first"]], self.scores[state["second"]],
                          split, params["n_candidates"])
        if name == "Switching":
            edges = torch.tensor(state["edges"], device=DEVICE)
            groups = torch.bucketize(self.data.train_count[split.users], edges, right=True)
            rec = torch.empty(len(split.users), K, dtype=torch.long, device=DEVICE)
            for group, model in state["models"].items():
                mask = groups == int(group)
                rec[mask] = top_items(self.scores[model], split)[mask]
            return rec
        if name == "FeatureCombination":
            scores = self.combination_scores(params)
        elif name == "FeatureAugmentation":
            scores = self.augmentation_scores(state["first"], params)
        elif name == "MetaLevel":
            scores = neighbor_scores(self.embeddings[state["embedding"]],
                                     self.data.train_matrix, params["n_neighbors"])
        else:
            raise ValueError(f"Unknown hybrid: {name}")
        return top_items(scores, split)

    def objective(self, name, params):
        state = self.fit(name, params, self.fit_split)
        result = metrics(self.recommend(name, params, state, self.tune_split), self.tune_split)
        return {"model": name, "best_valid_score": result["mrr@10"],
                "valid_score_bigger": True, "best_valid_result": result,
                "test_result": {}}
