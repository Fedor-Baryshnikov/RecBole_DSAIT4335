# Task 2.1

Run all commands from the repository root.

## 1. Export the data split

```bash
python save_split.py \
  --model BPR \
  --dataset ml-100k \
  --config_files recbole/config/BPR/ml-100k.yaml \
  --output_dir saved/splits
```

## 2. Export recommendations

```bash
python save_recommendations.py \
  --model_file run_outputs/Part1/Part1.2/best_models/BPR/<checkpoint>.pth \
  --k 10 \
  --output_dir saved/recommendations
```

Replace `<checkpoint>.pth` with the checkpoint filename to evaluate.

## 3. Run the evaluation

```bash
python code/Part2/Part2.1/evaluate.py \
  --train saved/splits/ml-100k.train.tsv \
  --valid saved/splits/ml-100k.valid.tsv \
  --test saved/splits/ml-100k.test.tsv \
  --recommendations saved/recommendations/ml-100k_BPR_top10.json \
  --k 10 \
  --output run_outputs/Part2/Part2.1/aggregate/BPR.json \
  --per-user-output run_outputs/Part2/Part2.1/per_user/BPR.tsv
```

## Tests

```bash
python -m unittest discover -s code/Part2/Part2.1/tests -v
```
