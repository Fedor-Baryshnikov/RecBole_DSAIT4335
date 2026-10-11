"""
Usage (from the repository root): python code/Part1/Part1.2/train_best.py
"""

import subprocess

subprocess.run("python run_recbole.py --model ItemKNN --dataset ml-100k --config_files recbole/config/ItemKNN/ml-100k.yaml --checkpoint_dir=run_outputs/Part1/Part1.2/best_models/ItemKNN --k=400 --shrink=1")
subprocess.run("python run_recbole.py --model ItemKNN --dataset ml-100k --config_files recbole/config/UserKNN/ml-100k.yaml --checkpoint_dir=run_outputs/Part1/Part1.2/best_models/UserKNN --k=50 --shrink=0.0")

subprocess.run("python run_recbole.py --model BPR --dataset ml-100k --config_files recbole/config/BPR/ml-100k.yaml --checkpoint_dir=run_outputs/Part1/Part1.2/best_models/BPR --learning_rate=0.005")

subprocess.run("python run_recbole.py --model EASE --dataset ml-100k --config_files recbole/config/EASE/ml-100k.yaml --checkpoint_dir=run_outputs/Part1/Part1.2/best_models/EASE --reg_weight=500.0")

subprocess.run("python run_recbole.py --model FISM --dataset ml-100k --config_files recbole/config/FISM/ml-100k.yaml --checkpoint_dir=run_outputs/Part1/Part1.2/best_models/FISM --alpha=0.0 --embedding_size=64 --learning_rate=0.001 --reg_weights=[0,0]")

subprocess.run("python run_recbole.py --model SLIMElastic --dataset ml-100k --config_files recbole/config/SLIMElastic/ml-100k.yaml --checkpoint_dir=run_outputs/Part1/Part1.2/best_models/SLIMElastic --alpha=0.2 --hide_item=True --l1_ratio=0.01 --positive_only=True")

subprocess.run("python run_recbole.py --model NeuMF --dataset ml-100k --config_files recbole/config/NeuMF/ml-100k.yaml --checkpoint_dir=run_outputs/Part1/Part1.2/best_models/NeuMF --dropout_prob=0.0 --learning_rate=0.001 --mlp_hidden_size=[64,32,16]")

subprocess.run("python run_recbole.py --model NGCF --dataset ml-100k --config_files recbole/config/NGCF/ml-100k.yaml --checkpoint_dir=run_outputs/Part1/Part1.2/best_models/NGCF --hidden_size_list=[128,128,128] --learning_rate=0.001 --message_dropout=0.0 --node_dropout=0.1 --reg_weight=1e-05")

subprocess.run("python run_recbole.py --model LightGCN --dataset ml-100k --config_files recbole/config/LightGCN/ml-100k.yaml --checkpoint_dir=run_outputs/Part1/Part1.2/best_models/LightGCN --learning_rate=0.002 --n_layers=1 --reg_weight=0.01")
