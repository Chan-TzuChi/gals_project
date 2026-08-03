"""alpha in {0.8, 0.9, 1.0}, 5 seeds, on the 7 batch1 datasets (+ cal500).
Table-2-style parameter-selection support, not full-precision main-table
numbers. Also reports subset-size / M behaviour at alpha=1.0: alpha=1.0
removes the label-subset-ratio term entirely, so this checks whether GALS
then picks degenerate -- too-small or too-large -- subsets.
"""
import os

import pandas as pd

from gals.experiments import alpha_sweep
from gals.loader import load_mulan

DATA_ROOT = os.environ.get("GALS_DATA_ROOT", os.path.join("..", "DATA"))
DATASETS = {
    "emotions": ("emotions.arff", "emotions.xml"),
    "flags": ("flags.arff", "flags.xml"),
    "Gnegative": ("GnegativePseAAC.arff", "GnegativePseAAC.xml"),
    "Gpositive": ("GpositivePseAAC.arff", "GpositivePseAAC.xml"),
    "Plant": ("PlantPseAAC.arff", "PlantPseAAC.xml"),
    "scene": ("scene.arff", "scene.xml"),
    "yeast": ("yeast.arff", "yeast.xml"),
    "cal500": ("CAL500.arff", "CAL500.xml"),
}

frames = []
for name, (arff, xml) in DATASETS.items():
    X, Y, _, _ = load_mulan(os.path.join(DATA_ROOT, arff), os.path.join(DATA_ROOT, xml))
    print(f"=== {name}: X={X.shape} Y={Y.shape} ===", flush=True)
    df = alpha_sweep(name, X, Y, seeds=range(5), alphas=(0.8, 0.9, 1.0))
    frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv("results/alpha_sweep.csv", index=False)
    print(f"  -> checkpointed results/alpha_sweep.csv ({sum(len(f) for f in frames)} rows)",
         flush=True)

df = pd.concat(frames, ignore_index=True)
df.to_csv("results/alpha_sweep.csv", index=False)
print(f"\nwrote results/alpha_sweep.csv ({len(df)} rows)")

# ---- table: 4 metrics x 3 alphas x dataset, mean +/- std ----
METRICS = ["hamming_loss", "subset_accuracy", "weighted_f1", "micro_f1"]
lines = ["# alpha_sweep summary (5 seeds/cell)\n"]
for metric in METRICS:
    lines.append(f"\n## {metric}\n")
    g = df.groupby(["dataset", "alpha"])[metric].agg(["mean", "std"])
    cell = (g["mean"].round(4).astype(str) + "\u00b1" + g["std"].round(4).astype(str))
    piv = cell.unstack("alpha")
    lines.append(piv.to_markdown())

# ---- alpha=1.0 subset-size / M behaviour ----
lines.append("\n## Subset size / M at alpha=1.0 vs 0.9 vs 0.8\n")
sub = df.groupby(["dataset", "alpha"])[["n_models", "mean_subset_size"]].agg(["mean", "min", "max"])
lines.append(sub.round(2).to_markdown())

out = "\n".join(str(x) for x in lines)
with open("results/alpha_sweep_summary.md", "w", encoding="utf-8") as fh:
    fh.write(out)
print("wrote results/alpha_sweep_summary.md")
print("\n" + out)
