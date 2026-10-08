"""
save_split.py

Re-creates the exact train/valid/test split that a trained RecBole
checkpoint was trained and evaluated on, and dumps each partition to a
plain TSV file using the ORIGINAL (external) user/item tokens -- not
RecBole's internal integer ids -- so the files can be read by
pandas/Excel or by any other tool outside RecBole.

The split is rebuilt from the config stored inside the checkpoint
(including `seed` and `eval_args`), via load_data_and_model, which seeds
the random number generators before splitting exactly like run_recbole.py
does. This guarantees the exported split is the one the model actually saw.

Usage:
    python save_split.py \
        --model_file saved/<dataset>-<model_name>-<timestamp>.pth \
        --output_dir saved/splits
"""
import argparse
import os

import pandas as pd

from recbole.quick_start import load_data_and_model


def dump_interactions(data_loader, dataset, out_path):
    """Write one partition (train/valid/test) of a dataloader to disk.

    The dataloader's underlying Interaction tensors store *internal* ids.
    We map them back to the original raw ids with dataset.id2token(...)
    so the exported file is human-readable and reproducible outside RecBole.
    """
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field

    inter_feat = data_loader.dataset.inter_feat
    user_ids = inter_feat[uid_field].numpy()
    item_ids = inter_feat[iid_field].numpy()

    user_tokens = dataset.id2token(uid_field, user_ids)
    item_tokens = dataset.id2token(iid_field, item_ids)

    df = pd.DataFrame({uid_field: user_tokens, iid_field: item_tokens})
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved {len(df)} interactions -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True,
                         help="path to a .pth checkpoint produced by run_recbole.py")
    parser.add_argument("--output_dir", type=str, default="saved/splits")
    args = parser.parse_args()

    config, _, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.model_file
    )
    print(f'Split rebuilt from {args.model_file} '
          f'(seed={config["seed"]}, eval_args={config["eval_args"]})')

    os.makedirs(args.output_dir, exist_ok=True)
    name = config["dataset"]
    dump_interactions(train_data, dataset, os.path.join(args.output_dir, f"{name}.train.tsv"))
    dump_interactions(valid_data, dataset, os.path.join(args.output_dir, f"{name}.valid.tsv"))
    dump_interactions(test_data, dataset, os.path.join(args.output_dir, f"{name}.test.tsv"))


if __name__ == "__main__":
    main()
