import os

import pandas as pd
import torch

from utils import DEVICE, get_parser, get_scores, item_knn_scores, load, report, top_items, user_metrics

WEIGHTS = [0.1, 0.25, 0.5, 1.0, 2.0]


def genre_matrix(data):
    path = os.path.join(data.config["data_path"], f"{data.config['dataset']}.item")
    items = pd.read_csv(path, sep="\t", usecols=["item_id:token", "class:token_seq"], dtype=str)
    ids = data.dataset.token2id(data.dataset.iid_field, items["item_id:token"].values)
    genres = items["class:token_seq"].str.split()
    names = sorted({genre for row in genres for genre in row})
    matrix = torch.zeros(data.n_items, len(names), device=DEVICE)
    for item, row in zip(ids, genres):
        for genre in row:
            matrix[item, names.index(genre)] = 1
    return matrix


def main():
    parser = get_parser()
    parser.add_argument("--n_neighbors", type=int, default=400)
    args = parser.parse_args()

    data, models = load(args.model_dir, args.data_dir)
    scores = {name: get_scores(model, data) for name, model in models.items()}

    collab = torch.nn.functional.normalize(data.train_matrix.T, dim=1)
    genres = torch.nn.functional.normalize(genre_matrix(data), dim=1)

    def recommend(weight, split):
        item_vectors = torch.cat([collab, weight * genres], dim=1)
        return top_items(item_knn_scores(item_vectors, data.train_matrix, args.n_neighbors), split)

    def valid_mrr(weight):
        return user_metrics(recommend(weight, data.valid), data.valid)["mrr"].mean().item()

    details = [{"weight": 0.0, "valid_mrr@10": round(valid_mrr(0.0), 4), "selected": False}]
    best_weight = None
    best_mrr = -1
    for weight in WEIGHTS:
        mrr = valid_mrr(weight)
        details.append({"weight": weight, "valid_mrr@10": round(mrr, 4)})
        if mrr > best_mrr:
            best_weight, best_mrr = weight, mrr
    print("best weight:", best_weight)
    for row in details[1:]:
        row["selected"] = row["weight"] == best_weight

    report("FeatureCombination", recommend(best_weight, data.valid), recommend(best_weight, data.test), scores, data, args.out_dir, details)


if __name__ == "__main__":
    main()
