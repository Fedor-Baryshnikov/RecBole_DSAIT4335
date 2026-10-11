import torch

from utils import get_parser, get_scores, load, report, top_items, user_metrics


@torch.no_grad()
def user_embedding(model):
    if hasattr(model, "restore_user_e"):
        return model.forward()[0]
    if hasattr(model, "user_mf_embedding"):
        return model.user_mf_embedding.weight
    if hasattr(model, "user_embedding"):
        return model.user_embedding.weight
    return None


def neighbor_scores(embedding, train_matrix, n_neighbors):
    embedding = torch.nn.functional.normalize(embedding.float(), dim=1)
    similarity = embedding @ embedding.T
    similarity.fill_diagonal_(-torch.inf)
    top = similarity.topk(n_neighbors, dim=1)
    weights = torch.zeros_like(similarity).scatter_(1, top.indices, top.values.clamp(min=0))
    return weights @ train_matrix


def main():
    parser = get_parser()
    parser.add_argument("--n_neighbors", type=int, default=100)
    args = parser.parse_args()

    data, models = load(args.model_dir, args.data_dir)
    scores = {name: get_scores(model, data) for name, model in models.items()}

    best_name = None
    best_mrr = -1
    details = []
    for name, model in models.items():
        embedding = user_embedding(model)
        if embedding is None:
            continue
        rec = top_items(neighbor_scores(embedding, data.train_matrix, args.n_neighbors), data.valid)
        mrr = user_metrics(rec, data.valid)["mrr"].mean().item()
        details.append({"embedding": name, "valid_mrr@10": round(mrr, 4)})
        if mrr > best_mrr:
            best_name, best_mrr = name, mrr
    print("best embedding:", best_name)
    for row in details:
        row["selected"] = row["embedding"] == best_name

    meta_scores = neighbor_scores(user_embedding(models[best_name]), data.train_matrix, args.n_neighbors)
    valid_rec = top_items(meta_scores, data.valid)
    test_rec = top_items(meta_scores, data.test)
    report("MetaLevel", valid_rec, test_rec, scores, data, args.out_dir, details)


if __name__ == "__main__":
    main()
