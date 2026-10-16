"""Run from the repository root: python code/Part1/Part1.5/train_best.py"""

import subprocess
import sys

# Fit the original hybrids using the best parameters found by tune.py.
subprocess.run([sys.executable, "code/Part1/Part1.5/run_hybrid.py", "--mode", "train", "--model", "WeightedHybrid"], check=True)

subprocess.run([sys.executable, "code/Part1/Part1.5/run_hybrid.py", "--mode", "train", "--model", "Switching"], check=True)

subprocess.run([sys.executable, "code/Part1/Part1.5/run_hybrid.py", "--mode", "train", "--model", "Cascade"], check=True)

subprocess.run([sys.executable, "code/Part1/Part1.5/run_hybrid.py", "--mode", "train", "--model", "FeatureCombination"], check=True)

subprocess.run([sys.executable, "code/Part1/Part1.5/run_hybrid.py", "--mode", "train", "--model", "FeatureAugmentation"], check=True)

subprocess.run([sys.executable, "code/Part1/Part1.5/run_hybrid.py", "--mode", "train", "--model", "MetaLevel"], check=True)
