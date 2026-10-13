"""
Usage (from the repository root): python code/Part1/Part1.5/train_best.py
"""

import subprocess
import sys

subprocess.run([sys.executable, "code/Part1/Part1.5/run_hybrid.py", "--model", "WeightedHybrid", "--params_file", "run_outputs/Part1/Part1.5/WeightedHybrid.best.json", "--checkpoint_dir", "run_outputs/Part1/Part1.5/best_models"], check=True)

subprocess.run([sys.executable, "code/Part1/Part1.5/run_hybrid.py", "--model", "Switching", "--params_file", "run_outputs/Part1/Part1.5/Switching.best.json", "--checkpoint_dir", "run_outputs/Part1/Part1.5/best_models"], check=True)

subprocess.run([sys.executable, "code/Part1/Part1.5/run_hybrid.py", "--model", "Cascade", "--params_file", "run_outputs/Part1/Part1.5/Cascade.best.json", "--checkpoint_dir", "run_outputs/Part1/Part1.5/best_models"], check=True)

subprocess.run([sys.executable, "code/Part1/Part1.5/run_hybrid.py", "--model", "FeatureCombination", "--params_file", "run_outputs/Part1/Part1.5/FeatureCombination.best.json", "--checkpoint_dir", "run_outputs/Part1/Part1.5/best_models"], check=True)

subprocess.run([sys.executable, "code/Part1/Part1.5/run_hybrid.py", "--model", "FeatureAugmentation", "--params_file", "run_outputs/Part1/Part1.5/FeatureAugmentation.best.json", "--checkpoint_dir", "run_outputs/Part1/Part1.5/best_models"], check=True)

subprocess.run([sys.executable, "code/Part1/Part1.5/run_hybrid.py", "--model", "MetaLevel", "--params_file", "run_outputs/Part1/Part1.5/MetaLevel.best.json", "--checkpoint_dir", "run_outputs/Part1/Part1.5/best_models"], check=True)
