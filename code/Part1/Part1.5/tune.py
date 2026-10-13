"""
Usage (from the repository root): python code/Part1/Part1.5/tune.py
"""

import os
import subprocess
import sys

os.makedirs("run_outputs/Part1/Part1.5", exist_ok=True)

with open("run_outputs/Part1/Part1.5/WeightedHybrid_console.txt", "w") as console:
    subprocess.run([sys.executable, "code/Part1/Part1.5/run_hyper.py", "--model", "WeightedHybrid", "--params_file", "code/Part1/Part1.5/WeightedHybrid.hyper", "--output_file", "run_outputs/Part1/Part1.5/WeightedHybrid.result"], stdout=console, stderr=subprocess.STDOUT, check=True)

with open("run_outputs/Part1/Part1.5/Switching_console.txt", "w") as console:
    subprocess.run([sys.executable, "code/Part1/Part1.5/run_hyper.py", "--model", "Switching", "--params_file", "code/Part1/Part1.5/Switching.hyper", "--output_file", "run_outputs/Part1/Part1.5/Switching.result"], stdout=console, stderr=subprocess.STDOUT, check=True)

with open("run_outputs/Part1/Part1.5/Cascade_console.txt", "w") as console:
    subprocess.run([sys.executable, "code/Part1/Part1.5/run_hyper.py", "--model", "Cascade", "--params_file", "code/Part1/Part1.5/Cascade.hyper", "--output_file", "run_outputs/Part1/Part1.5/Cascade.result"], stdout=console, stderr=subprocess.STDOUT, check=True)

with open("run_outputs/Part1/Part1.5/FeatureCombination_console.txt", "w") as console:
    subprocess.run([sys.executable, "code/Part1/Part1.5/run_hyper.py", "--model", "FeatureCombination", "--params_file", "code/Part1/Part1.5/FeatureCombination.hyper", "--output_file", "run_outputs/Part1/Part1.5/FeatureCombination.result"], stdout=console, stderr=subprocess.STDOUT, check=True)

with open("run_outputs/Part1/Part1.5/FeatureAugmentation_console.txt", "w") as console:
    subprocess.run([sys.executable, "code/Part1/Part1.5/run_hyper.py", "--model", "FeatureAugmentation", "--params_file", "code/Part1/Part1.5/FeatureAugmentation.hyper", "--output_file", "run_outputs/Part1/Part1.5/FeatureAugmentation.result"], stdout=console, stderr=subprocess.STDOUT, check=True)

with open("run_outputs/Part1/Part1.5/MetaLevel_console.txt", "w") as console:
    subprocess.run([sys.executable, "code/Part1/Part1.5/run_hyper.py", "--model", "MetaLevel", "--params_file", "code/Part1/Part1.5/MetaLevel.hyper", "--output_file", "run_outputs/Part1/Part1.5/MetaLevel.result"], stdout=console, stderr=subprocess.STDOUT, check=True)
