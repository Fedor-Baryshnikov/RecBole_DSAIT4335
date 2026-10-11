from itertools import permutations

from utils import K, get_parser, get_scores, load, report, top_items, user_metrics


def rerank(first, second, split, n_candidates):
    candidates = top_items(first, split, n_candidates)
    order = second[split.users].gather(1, candidates).topk(K, dim=1).indices
    return candidates.gather(1, order)


def main():
    parser = get_parser()
    parser.add_argument("--n_candidates", type=int, default=100)
    args = parser.parse_args()

    data, models = load(args.model_dir, args.data_dir)
    scores = {name: get_scores(model, data) for name, model in models.items()}

    best_pair = None
    best_mrr = -1
    details = []
    for first, second in permutations(scores, 2):
        rec = rerank(scores[first], scores[second], data.valid, args.n_candidates)
        mrr = user_metrics(rec, data.valid)["mrr"].mean().item()
        details.append({"first": first, "second": second, "valid_mrr@10": round(mrr, 4)})
        if mrr > best_mrr:
            best_pair, best_mrr = (first, second), mrr
    print("best pair:", best_pair)
    for row in details:
        row["selected"] = (row["first"], row["second"]) == best_pair

    first, second = (scores[name] for name in best_pair)
    valid_rec = rerank(first, second, data.valid, args.n_candidates)
    test_rec = rerank(first, second, data.test, args.n_candidates)
    report("Cascade", valid_rec, test_rec, scores, data, args.out_dir, details)


if __name__ == "__main__":
    main()
