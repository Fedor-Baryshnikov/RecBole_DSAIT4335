import json
import os

import numpy as np
import pandas as pd
import torch

from cascade import rerank
from feature_combination import genre_matrix
from utils import DEVICE, K, get_parser, get_scores, load, top_items, user_metrics

MONOLITHIC = ["FeatureCombination", "MetaLevel", "FeatureAugmentation"]
ENSEMBLES = ["Switching", "Cascade", "FeatureAugmentation"]


def bootstrap_ci(diff, n_boot=2000):
    rng = np.random.RandomState(0)
    index = rng.randint(0, len(diff), size=(n_boot, len(diff)))
    return np.percentile(diff[index].mean(axis=1), [2.5, 97.5])


def points(values):
    result = {}
    for name in values:
        total = 0
        for other in values:
            if other == name:
                continue
            low, high = bootstrap_ci(values[name] - values[other])
            total += 1 if low > 0 else 0 if high < 0 else 0.5
        result[name] = total
    return result


def main():
    parser = get_parser()
    parser.add_argument("--n_candidates", type=int, default=100)
    args = parser.parse_args()

    data, models = load(args.model_dir, args.data_dir)
    scores = {name: get_scores(model, data) for name, model in models.items()}
    dataset = data.config["dataset"]
    out_dir = os.path.join(args.out_dir, dataset)
    splits = {"valid": data.valid, "test": data.test}

    def saved(name, split_name):
        path = os.path.join(out_dir, "recommendations", f"{dataset}_{name}_{split_name}_top{K}.json")
        with open(path) as f:
            recommendations = json.load(f)["recommendations"]
        users = data.dataset.token2id(data.dataset.uid_field, np.array(list(recommendations)))
        items = data.dataset.token2id(data.dataset.iid_field, np.array([row["items"] for row in recommendations.values()]))
        rec = torch.zeros(data.n_users, K, dtype=torch.long, device=DEVICE)
        rec[torch.tensor(users, device=DEVICE)] = torch.tensor(items, device=DEVICE)
        return rec[splits[split_name].users]

    base = {name: {s: top_items(model_scores, split) for s, split in splits.items()} for name, model_scores in scores.items()}
    best_single = max(base, key=lambda name: user_metrics(base[name]["valid"], data.valid)["mrr"].mean().item())

    cascade_details = pd.read_csv(os.path.join(out_dir, "Cascade_details.csv"))
    chosen = cascade_details[cascade_details["selected"]].iloc[0]
    aligned = cascade_details[cascade_details["second"] == "ItemKNN"].sort_values("valid_mrr@10").iloc[-1]
    print("Switching is compared with", best_single)
    print("Cascade is compared with", chosen["second"])
    print("Cascade_ItemKNN uses", aligned["first"], "->", "ItemKNN")

    pairs = {}
    for name in ["FeatureCombination", "FeatureAugmentation", "MetaLevel"]:
        pairs[name] = ({s: saved(name, s) for s in splits}, {s: saved(f"{name}_control", s) for s in splits})
    pairs["Switching"] = ({s: saved("Switching", s) for s in splits}, base[best_single])
    pairs["Cascade"] = ({s: saved("Cascade", s) for s in splits}, base[chosen["second"]])
    cascade_itemknn = {s: rerank(scores[aligned["first"]], scores["ItemKNN"], split, args.n_candidates) for s, split in splits.items()}
    pairs["Cascade_ItemKNN"] = (cascade_itemknn, base["ItemKNN"])

    counts = data.train_count
    edges = torch.quantile(counts[counts > 0], torch.tensor([1 / 3, 2 / 3], device=DEVICE))
    group = torch.bucketize(counts, edges, right=True)
    sparse = {s: (group[split.users] == 0).cpu().numpy() for s, split in splits.items()}

    gain_rows = []
    gain = {}
    for name, (hybrid, control) in pairs.items():
        for s, split in splits.items():
            hybrid_metrics = user_metrics(hybrid[s], split)
            control_metrics = user_metrics(control[s], split)
            for metric in ["ndcg", "mrr"]:
                hybrid_values = hybrid_metrics[metric].cpu().numpy()
                control_values = control_metrics[metric].cpu().numpy()
                diff = hybrid_values - control_values
                if s == "valid" and metric == "ndcg":
                    gain[name] = diff
                for users, mask in [("all", np.ones_like(sparse[s])), ("sparse", sparse[s])]:
                    low, high = bootstrap_ci(diff[mask])
                    gain_rows.append({
                        "candidate": name,
                        "split": s,
                        "users": users,
                        "metric": f"{metric}@{K}",
                        "hybrid": round(float(hybrid_values[mask].mean()), 4),
                        "control": round(float(control_values[mask].mean()), 4),
                        "gain": round(float(diff[mask].mean()), 4),
                        "gain_ci_low": round(float(low), 4),
                        "gain_ci_high": round(float(high), 4),
                    })

    genres = torch.nn.functional.normalize(genre_matrix(data), dim=1)
    popularity = data.train_matrix.sum(dim=0).clamp(min=1) / (counts > 0).sum()
    item_novelty = -torch.log2(popularity)

    def diversity(rec):
        vectors = genres[rec]
        similarity = vectors @ vectors.transpose(1, 2)
        ild = 1 - (similarity.sum(dim=(1, 2)) - K) / (K * (K - 1))
        novelty = item_novelty[rec].mean(dim=1)
        coverage = rec.unique().numel() / (data.n_items - 1)
        return ild, novelty, coverage

    lists = {name: hybrid for name, (hybrid, control) in pairs.items()}
    lists.update({f"{name}_control": control for name, (hybrid, control) in pairs.items()})
    diversity_rows = []
    diversity_valid = {}
    for name, rec in lists.items():
        for s in splits:
            ild, novelty, coverage = diversity(rec[s])
            diversity_rows.append({
                "model": name,
                "split": s,
                f"ild@{K}": round(ild.mean().item(), 4),
                f"novelty@{K}": round(novelty.mean().item(), 4),
                f"coverage@{K}": round(coverage, 4),
            })
            if s == "valid":
                diversity_valid[name] = (ild.cpu().numpy(), novelty.cpu().numpy())

    def scorecard(names):
        accuracy = points({name: gain[name] for name in names})
        sparse_users = points({name: gain[name][sparse["valid"]] for name in names})
        ild = points({name: diversity_valid[name][0] for name in names})
        novelty = points({name: diversity_valid[name][1] for name in names})
        rows = []
        for name in names:
            row = {
                "candidate": name,
                "accuracy": accuracy[name],
                "sparse_users": sparse_users[name],
                "diversity": (ild[name] + novelty[name]) / 2,
            }
            row["total"] = row["accuracy"] + row["sparse_users"] + row["diversity"]
            rows.append(row)
        return pd.DataFrame(rows).sort_values("total", ascending=False)

    tables = {
        "gains": pd.DataFrame(gain_rows),
        "diversity": pd.DataFrame(diversity_rows),
        "scorecard_monolithic": scorecard(MONOLITHIC),
        "scorecard_ensembles": scorecard(ENSEMBLES),
    }
    os.makedirs(os.path.join(out_dir, "compare"), exist_ok=True)
    for name, table in tables.items():
        table.to_csv(os.path.join(out_dir, "compare", f"{name}.csv"), index=False)
        print(name)
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
