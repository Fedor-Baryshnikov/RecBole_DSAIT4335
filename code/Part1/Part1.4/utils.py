import argparse
import glob
import json
import os
from datetime import datetime

import pandas as pd
import torch
from recbole.data import create_dataset, data_preparation
from recbole.data.interaction import Interaction
from recbole.utils import get_model, init_seed

K = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class Split:
    def __init__(self, loader, n_users, n_items):
        self.users = loader.uid_list.to(DEVICE)
        self.seen = torch.zeros(n_users, n_items, dtype=torch.bool, device=DEVICE)
        self.truth = torch.zeros_like(self.seen)
        for user in loader.uid_list.tolist():
            self.seen[user, loader.uid2history_item[user]] = True
            self.truth[user, loader.uid2positive_item[user]] = True


class Data:
    def __init__(self, config, dataset, train, valid, test):
        self.config = config
        self.dataset = dataset
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        train_matrix = train._dataset.inter_matrix(form="csr").toarray()
        self.train_matrix = torch.tensor(train_matrix, dtype=torch.float32, device=DEVICE)
        self.train_count = self.train_matrix.sum(dim=1)
        self.valid = Split(valid, self.n_users, self.n_items)
        self.test = Split(test, self.n_users, self.n_items)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=os.path.join(ROOT, "run_outputs", "Part1", "Part1.2", "best_models"))
    parser.add_argument("--data_dir", default=os.path.join(ROOT, "dataset"))
    parser.add_argument("--out_dir", default=os.path.join(ROOT, "run_outputs", "Part1", "Part1.4"))
    return parser


def checkpoint_time(path):
    return datetime.strptime(os.path.basename(path)[-24:-4], "%b-%d-%Y_%H-%M-%S")


def latest_checkpoints(model_dir):
    paths = []
    for folder in sorted(glob.glob(os.path.join(model_dir, "*"))):
        files = glob.glob(os.path.join(folder, "*.pth"))
        if files:
            paths.append(max(files, key=checkpoint_time))
    return paths


def load(model_dir, data_dir):
    paths = latest_checkpoints(model_dir)
    checkpoints = [torch.load(path, weights_only=False, map_location=DEVICE) for path in paths]
    for checkpoint in checkpoints:
        config = checkpoint["config"]
        config["data_path"] = os.path.join(data_dir, config["dataset"])
        config["device"] = DEVICE
        config["use_gpu"] = DEVICE.type == "cuda"
        config["show_progress"] = False

    config = checkpoints[0]["config"]
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train, valid, test = data_preparation(config, dataset)
    data = Data(config, dataset, train, valid, test)

    models = {}
    for checkpoint in checkpoints:
        config = checkpoint["config"]
        name = config["model"]
        if name == "ItemKNN" and config["knn_method"] == "user":
            name = "UserKNN"
        print("loading", name)
        init_seed(config["seed"], config["reproducibility"])
        model = get_model(config["model"])(config, train._dataset).to(DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        model.load_other_parameter(checkpoint.get("other_parameter"))
        model.eval()
        models[name] = model
    return data, models


@torch.no_grad()
def get_scores(model, data):
    uid = data.dataset.uid_field
    iid = data.dataset.iid_field
    items = torch.arange(data.n_items, device=DEVICE)
    rows = []
    for start in range(0, data.n_users, 256):
        users = torch.arange(start, min(start + 256, data.n_users), device=DEVICE)
        try:
            scores = model.full_sort_predict(Interaction({uid: users}).to(DEVICE))
        except NotImplementedError:
            pairs = Interaction({uid: users.repeat_interleave(data.n_items), iid: items.repeat(len(users))})
            scores = model.predict(pairs.to(DEVICE))
        rows.append(scores.view(len(users), data.n_items).float().to(DEVICE))
    return torch.cat(rows)


def item_knn_scores(item_vectors, profiles, n_neighbors):
    vectors = torch.nn.functional.normalize(item_vectors.float(), dim=1)
    similarity = vectors @ vectors.T
    similarity.fill_diagonal_(-torch.inf)
    top = similarity.topk(n_neighbors, dim=1)
    weights = torch.zeros_like(similarity).scatter_(1, top.indices, top.values.clamp(min=0))
    return profiles @ weights.T


def top_items(scores, split, n=K):
    scores = scores[split.users]
    scores[split.seen[split.users]] = -torch.inf
    scores[:, 0] = -torch.inf
    return scores.topk(n, dim=1).indices


def user_metrics(rec, split):
    truth = split.truth[split.users]
    hits = truth.gather(1, rec).float()
    n_truth = truth.sum(dim=1)
    ranks = torch.arange(1, K + 1, device=DEVICE)
    discount = 1 / torch.log2(ranks + 1)
    ideal = discount.cumsum(0)[n_truth.clamp(max=K) - 1]
    return {
        "recall": hits.sum(dim=1) / n_truth,
        "mrr": (hits / ranks).max(dim=1).values,
        "ndcg": (hits * discount).sum(dim=1) / ideal,
        "hit": hits.max(dim=1).values,
        "precision": hits.sum(dim=1) / K,
    }


def make_row(label, kind, valid_rec, test_rec, data):
    row = {"model": label, "type": kind}
    for split_name, rec in [("valid", valid_rec), ("test", test_rec)]:
        for metric, values in user_metrics(rec, getattr(data, split_name)).items():
            row[f"{split_name}_{metric}@{K}"] = round(values.mean().item(), 4)
    return row


def save_recommendations(name, split_name, rec, data, out_dir):
    dataset = data.dataset
    users = dataset.id2token(dataset.uid_field, getattr(data, split_name).users.cpu().numpy())
    items = dataset.id2token(dataset.iid_field, rec.cpu().numpy())
    output = {
        "dataset": data.config["dataset"],
        "model": name,
        "split": split_name,
        "k": K,
        "seed": data.config["seed"],
        "eval_args": data.config["eval_args"],
        "recommendations": {str(user): {"items": items[i].tolist()} for i, user in enumerate(users)},
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{data.config['dataset']}_{name}_{split_name}_top{K}.json"), "w") as f:
        json.dump(output, f, indent=2)


def report(name, valid_rec, test_rec, scores, data, out_dir, details, control=None):
    rows = []
    for model_name, model_scores in scores.items():
        valid_base = top_items(model_scores, data.valid)
        test_base = top_items(model_scores, data.test)
        rows.append(make_row(model_name, "base", valid_base, test_base, data))
    rows.append(make_row(name, "hybrid", valid_rec, test_rec, data))

    table = pd.DataFrame(rows)
    out_dir = os.path.join(out_dir, data.config["dataset"])
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, f"{name}_summary.csv"), index=False)
    pd.DataFrame(details).to_csv(os.path.join(out_dir, f"{name}_details.csv"), index=False)
    rec_dir = os.path.join(out_dir, "recommendations")
    save_recommendations(name, "valid", valid_rec, data, rec_dir)
    save_recommendations(name, "test", test_rec, data, rec_dir)
    if control is not None:
        save_recommendations(f"{name}_control", "valid", control[0], data, rec_dir)
        save_recommendations(f"{name}_control", "test", control[1], data, rec_dir)
    print(table.to_string(index=False))
