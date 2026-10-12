import torch

from utils import get_parser, get_scores, item_knn_scores, load, report, top_items, user_metrics

WEIGHTS = [0.25, 0.5, 1.0]


def augment(first_scores, train_matrix, n_pseudo, weight):
    candidates = first_scores.clone()
    candidates[train_matrix > 0] = -torch.inf
    candidates[:, 0] = -torch.inf
    pseudo = torch.zeros_like(train_matrix).scatter_(1, candidates.topk(n_pseudo, dim=1).indices, 1.0)
    pseudo[0] = 0
    return train_matrix + weight * pseudo


def main():
    parser = get_parser()
    parser.add_argument("--n_pseudo", type=int, default=20)
    parser.add_argument("--n_neighbors", type=int, default=400)
    args = parser.parse_args()

    data, models = load(args.model_dir, args.data_dir)
    scores = {name: get_scores(model, data) for name, model in models.items()}

    def recommend(profiles, split):
        return top_items(item_knn_scores(profiles.T, profiles, args.n_neighbors), split)

    def valid_mrr(profiles):
        return user_metrics(recommend(profiles, data.valid), data.valid)["mrr"].mean().item()

    details = [{"first": "none", "weight": 0.0, "valid_mrr@10": round(valid_mrr(data.train_matrix), 4), "selected": False}]
    best = None
    best_mrr = -1
    for first in scores:
        for weight in WEIGHTS:
            profiles = augment(scores[first], data.train_matrix, args.n_pseudo, weight)
            mrr = valid_mrr(profiles)
            details.append({"first": first, "weight": weight, "valid_mrr@10": round(mrr, 4)})
            if mrr > best_mrr:
                best, best_mrr = (first, weight), mrr
    print("best first model and weight:", best)
    for row in details[1:]:
        row["selected"] = (row["first"], row["weight"]) == best

    profiles = augment(scores[best[0]], data.train_matrix, args.n_pseudo, best[1])
    control = (recommend(data.train_matrix, data.valid), recommend(data.train_matrix, data.test))
    report("FeatureAugmentation", recommend(profiles, data.valid), recommend(profiles, data.test), scores, data, args.out_dir, details, control)


if __name__ == "__main__":
    main()
