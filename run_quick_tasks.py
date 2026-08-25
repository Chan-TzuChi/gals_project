"""The fast, parallelizable-with-everything-else tasks: environment capture,
run config dump, alpha_sweep (Table 2), ideal_ratio_sweep (L_ideal sensitivity).
Runs on a couple of small datasets with few seeds -- these are sanity/
sensitivity checks, not the final per-dataset sweeps.

Every output here is skipped if the file already exists. results/run_config.json
and results/environment.json as committed describe the runs that produced the
reported results and carry hand-recorded detail (seed plan, wall-clock caps,
revision history) that a fresh dump does not reproduce; regenerating them on a
different machine or from changed defaults would silently replace a record of
what was done with a description of the current checkout. Pass --force to
overwrite anyway.
"""
import argparse
import os

from gals.experiments import (alpha_sweep, capture_environment,
                              dump_run_config, ideal_ratio_sweep)
from gals.gals import GAConfig
from gals.loader import load_mulan

ap = argparse.ArgumentParser()
ap.add_argument("--force", action="store_true",
                help="overwrite outputs that already exist")
args = ap.parse_args()

DATA_ROOT = os.environ.get("GALS_DATA_ROOT", os.path.join("..", "DATA"))


def wanted(path):
    """True if `path` should be written; False (with a notice) if it exists."""
    if os.path.exists(path) and not args.force:
        print(f"skip {path} (already exists; --force to overwrite)")
        return False
    return True


if wanted("results/environment.json"):
    capture_environment("results/environment.json")
    print("wrote results/environment.json")

cfg = GAConfig()
if wanted("results/run_config.json"):
    dump_run_config(cfg, "results/run_config.json", seeds=range(10))
    print("wrote results/run_config.json")

datasets = {
    "emotions": ("emotions.arff", "emotions.xml"),
    "flags": ("flags.arff", "flags.xml"),
}

for name, (arff, xml) in datasets.items():
    alpha_out = f"results/{name}_alpha_sweep.csv"
    ratio_out = f"results/{name}_ideal_ratio_sweep.csv"
    do_alpha, do_ratio = wanted(alpha_out), wanted(ratio_out)
    if not (do_alpha or do_ratio):
        continue

    X, Y, _, _ = load_mulan(os.path.join(DATA_ROOT, arff), os.path.join(DATA_ROOT, xml))

    if do_alpha:
        df = alpha_sweep(name, X, Y, seeds=range(5))
        df.to_csv(alpha_out, index=False)
        print(f"[{name}] alpha_sweep -> {alpha_out}")

    if do_ratio:
        df = ideal_ratio_sweep(name, X, Y, seeds=range(5))
        df.to_csv(ratio_out, index=False)
        print(f"[{name}] ideal_ratio_sweep -> {ratio_out}")

print("done.")
