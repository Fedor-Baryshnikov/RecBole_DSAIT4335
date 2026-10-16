"""Run from the repository root: python code/Part1/Part1.5/tune.py"""

import subprocess
import sys

# Edit the .hyper files to choose model combinations and parameter values.
# model_names is a comma-separated list; 'all' includes every base model.
subprocess.run([sys.executable, "code/Part1/Part1.5/run_hybrid.py", "--mode", "tune", "--model", "WeightedHybrid"], check=True)

subprocess.run([sys.executable, "code/Part1/Part1.5/run_hybrid.py", "--mode", "tune", "--model", "Switching"], check=True)

subprocess.run([sys.executable, "code/Part1/Part1.5/run_hybrid.py", "--mode", "tune", "--model", "Cascade"], check=True)

subprocess.run([sys.executable, "code/Part1/Part1.5/run_hybrid.py", "--mode", "tune", "--model", "FeatureCombination"], check=True)

subprocess.run([sys.executable, "code/Part1/Part1.5/run_hybrid.py", "--mode", "tune", "--model", "FeatureAugmentation"], check=True)

subprocess.run([sys.executable, "code/Part1/Part1.5/run_hybrid.py", "--mode", "tune", "--model", "MetaLevel"], check=True)
