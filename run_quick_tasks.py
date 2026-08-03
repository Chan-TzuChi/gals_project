"""The fast, parallelizable-with-everything-else tasks: environment capture,
run config dump, alpha_sweep (Table 2), ideal_ratio_sweep (L_ideal sensitivity).
Runs on a couple of small datasets with few seeds -- these are sanity/
sensitivity checks, not the final per-dataset sweeps.
"""
import os

from gals.experiments import (alpha_sweep, capture_environment,
                              dump_run_config, ideal_ratio_sweep)
from gals.gals import GAConfig
from gals.loader import load_mulan

DATA_ROOT = os.environ.get("GALS_DATA_ROOT", os.path.join("..", "DATA"))

capture_environment("results/environment.json")
print("wrote results/environment.json")

cfg = GAConfig()
dump_run_config(cfg, "results/run_config.json", seeds=range(10))
print("wrote results/run_config.json")

datasets = {
    "emotions": ("emotions.arff", "emotions.xml"),
    "flags": ("flags.arff", "flags.xml"),
}

for name, (arff, xml) in datasets.items():
    X, Y, _, _ = load_mulan(os.path.join(DATA_ROOT, arff), os.path.join(DATA_ROOT, xml))

    df = alpha_sweep(name, X, Y, seeds=range(5))
    df.to_csv(f"results/{name}_alpha_sweep.csv", index=False)
    print(f"[{name}] alpha_sweep -> results/{name}_alpha_sweep.csv")

    df = ideal_ratio_sweep(name, X, Y, seeds=range(5))
    df.to_csv(f"results/{name}_ideal_ratio_sweep.csv", index=False)
    print(f"[{name}] ideal_ratio_sweep -> results/{name}_ideal_ratio_sweep.csv")

print("done.")
