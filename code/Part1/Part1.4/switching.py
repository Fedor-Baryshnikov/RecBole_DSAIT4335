import torch

from utils import DEVICE, K, get_parser, get_scores, load, report, top_items, user_metrics


def main():
    parser = get_parser()
    parser.add_argument("--n_groups", type=int, default=3)
    args = parser.parse_args()

    data, models = load(args.model_dir, args.data_dir)
    scores = {name: get_scores(model, data) for name, model in models.items()}

    quantiles = torch.linspace(0, 1, args.n_groups + 1, device=DEVICE)[1:-1]
    edges = torch.quantile(data.train_count[data.train_count > 0], quantiles)
    group = torch.bucketize(data.train_count, edges, right=True)    #Split users into groups based on their training count

    valid_mrr = {}
    for name, model_scores in scores.items():
        valid_mrr[name] = user_metrics(top_items(model_scores, data.valid), data.valid)["mrr"]
    valid_group = group[data.valid.users]

    best = {}
    details = []
    for g in range(args.n_groups):
        in_group = valid_group == g
        group_mrr = {name: valid_mrr[name][in_group].mean().item() for name in scores}
        best[g] = max(group_mrr, key=group_mrr.get)
        print("group", g, "->", best[g])
        counts = data.train_count[(group == g) & (data.train_count > 0)]
        for name, mrr in group_mrr.items():
            details.append({
                "group": g,
                "train_count_min": int(counts.min().item()),
                "train_count_max": int(counts.max().item()),
                "valid_users": int(in_group.sum().item()),
                "model": name,
                "valid_mrr@10": round(mrr, 4),
                "selected": name == best[g],
            })

    def recommend(split):
        user_group = group[split.users]
        rec = torch.empty(len(split.users), K, dtype=torch.long, device=DEVICE)
        for g, name in best.items():
            rec[user_group == g] = top_items(scores[name], split)[user_group == g]
        return rec

    report("Switching", recommend(data.valid), recommend(data.test), scores, data, args.out_dir, details)


if __name__ == "__main__":
    main()
