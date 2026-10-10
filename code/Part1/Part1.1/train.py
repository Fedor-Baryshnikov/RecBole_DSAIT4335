import subprocess

subprocess.run("python run_recbole.py --model Random --dataset ml-100k --config_files recbole/config/Random/ml-100k.yaml --checkpoint_dir=run_outputs/Part1/Part1.1/Random")

subprocess.run("python run_recbole.py --model Pop --dataset ml-100k --config_files recbole/config/Pop/ml-100k.yaml --checkpoint_dir=run_outputs/Part1/Part1.1/Pop")

subprocess.run("python run_recbole.py --model ItemKNN --dataset ml-100k --config_files recbole/config/ItemKNN/ml-100k.yaml --checkpoint_dir=run_outputs/Part1/Part1.1/ItemKNN")
subprocess.run("python run_recbole.py --model ItemKNN --dataset ml-100k --config_files recbole/config/UserKNN/ml-100k.yaml --checkpoint_dir=run_outputs/Part1/Part1.1/UserKNN")

subprocess.run("python run_recbole.py --model BPR --dataset ml-100k --config_files recbole/config/BPR/ml-100k.yaml --checkpoint_dir=run_outputs/Part1/Part1.1/BPR")

subprocess.run("python run_recbole.py --model EASE --dataset ml-100k --config_files recbole/config/EASE/ml-100k.yaml --checkpoint_dir=run_outputs/Part1/Part1.1/EASE")

subprocess.run("python run_recbole.py --model FISM --dataset ml-100k --config_files recbole/config/FISM/ml-100k.yaml --checkpoint_dir=run_outputs/Part1/Part1.1/FISM")

subprocess.run("python run_recbole.py --model SLIMElastic --dataset ml-100k --config_files recbole/config/SLIMElastic/ml-100k.yaml --checkpoint_dir=run_outputs/Part1/Part1.1/SLIMElastic")

subprocess.run("python run_recbole.py --model NeuMF --dataset ml-100k --config_files recbole/config/NeuMF/ml-100k.yaml --checkpoint_dir=run_outputs/Part1/Part1.1/NeuMF")

subprocess.run("python run_recbole.py --model NGCF --dataset ml-100k --config_files recbole/config/NGCF/ml-100k.yaml --checkpoint_dir=run_outputs/Part1/Part1.1/NGCF")

subprocess.run("python run_recbole.py --model LightGCN --dataset ml-100k --config_files recbole/config/LightGCN/ml-100k.yaml --checkpoint_dir=run_outputs/Part1/Part1.1/LightGCN")