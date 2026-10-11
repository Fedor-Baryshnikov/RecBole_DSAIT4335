"""
Usage (from the repository root): python code/Part1/Part1.2/tune.py
"""

import os
import subprocess

os.makedirs("run_outputs/Part1/Part1.2", exist_ok=True)

subprocess.run("python run_hyper.py --config_files recbole/config/ItemKNN/ml-100k.yaml --params_file code/Part1/Part1.2/ItemKNN.hyper --output_file run_outputs/Part1/Part1.2/ItemKNN.result", stdout=open("run_outputs/Part1/Part1.2/ItemKNN_console.txt", "w"), stderr=subprocess.STDOUT)
subprocess.run("python run_hyper.py --config_files recbole/config/UserKNN/ml-100k.yaml --params_file code/Part1/Part1.2/UserKNN.hyper --output_file run_outputs/Part1/Part1.2/UserKNN.result", stdout=open("run_outputs/Part1/Part1.2/UserKNN_console.txt", "w"), stderr=subprocess.STDOUT)

subprocess.run("python run_hyper.py --config_files recbole/config/BPR/ml-100k.yaml --params_file code/Part1/Part1.2/BPR.hyper --output_file run_outputs/Part1/Part1.2/BPR.result", stdout=open("run_outputs/Part1/Part1.2/BPR_console.txt", "w"), stderr=subprocess.STDOUT)

subprocess.run("python run_hyper.py --config_files recbole/config/EASE/ml-100k.yaml --params_file code/Part1/Part1.2/EASE.hyper --output_file run_outputs/Part1/Part1.2/EASE.result", stdout=open("run_outputs/Part1/Part1.2/EASE_console.txt", "w"), stderr=subprocess.STDOUT)

subprocess.run("python run_hyper.py --config_files recbole/config/FISM/ml-100k.yaml --params_file code/Part1/Part1.2/FISM.hyper --output_file run_outputs/Part1/Part1.2/FISM.result", stdout=open("run_outputs/Part1/Part1.2/FISM_console.txt", "w"), stderr=subprocess.STDOUT)

subprocess.run("python run_hyper.py --config_files recbole/config/SLIMElastic/ml-100k.yaml --params_file code/Part1/Part1.2/SLIMElastic.hyper --output_file run_outputs/Part1/Part1.2/SLIMElastic.result", stdout=open("run_outputs/Part1/Part1.2/SLIMElastic_console.txt", "w"), stderr=subprocess.STDOUT)

subprocess.run("python run_hyper.py --config_files recbole/config/NeuMF/ml-100k.yaml --params_file code/Part1/Part1.2/NeuMF.hyper --output_file run_outputs/Part1/Part1.2/NeuMF.result", stdout=open("run_outputs/Part1/Part1.2/NeuMF_console.txt", "w"), stderr=subprocess.STDOUT)

subprocess.run("python run_hyper.py --config_files recbole/config/NGCF/ml-100k.yaml --params_file code/Part1/Part1.2/NGCF.hyper --output_file run_outputs/Part1/Part1.2/NGCF.result", stdout=open("run_outputs/Part1/Part1.2/NGCF_console.txt", "w"), stderr=subprocess.STDOUT)

subprocess.run("python run_hyper.py --config_files recbole/config/LightGCN/ml-100k.yaml --params_file code/Part1/Part1.2/LightGCN.hyper --output_file run_outputs/Part1/Part1.2/LightGCN.result", stdout=open("run_outputs/Part1/Part1.2/LightGCN_console.txt", "w"), stderr=subprocess.STDOUT)
