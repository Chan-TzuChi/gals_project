"""ideal_ratio in {0.25, 1/3, 0.5, 2/3} (L_ideal = L/4, L/3, L/2, 2L/3),
5 seeds, on 4 datasets spanning a wide label-count range: emotions (L=6),
Gnegative (L=8), Plant (L=12), yeast (L=14). Run in parallel with
run_alpha_sweep.py -- each GALS run is single-threaded (rf n_jobs=1) and
the machine has 22 logical cores, so the two sweeps don't meaningfully
contend.
"""
import os

import pandas as pd

from gals.experiments import ideal_ratio_sweep
from gals.loader import load_mulan

DATA_ROOT = os.environ.get("GALS_DATA_ROOT", os.path.join("..", "DATA"))
DATASETS = {
    "emotions": ("emotions.arff", "emotions.xml"),
    "Gnegative": ("GnegativePseAAC.arff", "GnegativePseAAC.xml"),
    "Plant": ("PlantPseAAC.arff", "PlantPseAAC.xml"),
    "yeast": ("yeast.arff", "yeast.xml"),
}
RATIOS = (0.25, 1/3, 0.5, 2/3)

frames = []
for name, (arff, xml) in DATASETS.items():
    X, Y, _, _ = load_mulan(os.path.join(DATA_ROOT, arff), os.path.join(DATA_ROOT, xml))
    print(f"=== {name}: X={X.shape} Y={Y.shape} L={Y.shape[1]} ===", flush=True)
    df = ideal_ratio_sweep(name, X, Y, seeds=range(5), ratios=RATIOS)
    frames.append(df)
    pd.concat(frames, ignore_index=True).to_csv("results/ideal_ratio_sweep.csv", index=False)
    print(f"  -> checkpointed results/ideal_ratio_sweep.csv "
         f"({sum(len(f) for f in frames)} rows)", flush=True)

df = pd.concat(frames, ignore_index=True)
df.to_csv("results/ideal_ratio_sweep.csv", index=False)
print(f"\nwrote results/ideal_ratio_sweep.csv ({len(df)} rows)")

METRICS = ["hamming_loss", "subset_accuracy", "weighted_f1", "micro_f1"]
lines = ["# ideal_ratio_sweep summary (5 seeds/cell)\n"]

for metric in METRICS:
    lines.append(f"\n## {metric}\n")
    g = df.groupby(["dataset", "ideal_ratio"])[metric].agg(["mean", "std"])
    cell = (g["mean"].round(4).astype(str) + "\u00b1" + g["std"].round(4).astype(str))
    piv = cell.unstack("ideal_ratio")
    lines.append(piv.to_markdown())

# ---- point 1: sensitivity = range of the mean across ideal_ratio, per dataset/metric ----
lines.append("\n## Sensitivity: range (max-min of the mean across ideal_ratio), per dataset\n")
means = df.groupby(["dataset", "ideal_ratio"])[METRICS].mean()
rng = means.groupby("dataset").agg(lambda s: s.max() - s.min())
lines.append(rng.round(4).to_markdown())

# ---- point 2: is L/2 (0.5) best or near-best, per dataset/metric ----
lines.append("\n## Is ideal_ratio=0.5 the best (or near-best) setting?\n")
LOWER_BETTER = {"hamming_loss"}
rows = []
for ds, sub in means.groupby("dataset"):
    sub = sub.droplevel("dataset")
    for metric in METRICS:
        asc = metric in LOWER_BETTER
        ranked = sub[metric].rank(ascending=asc, method="min")
        rows.append(dict(dataset=ds, metric=metric,
                         value_at_0_5=round(sub.loc[0.5, metric], 4),
                         rank_of_0_5=int(ranked.loc[0.5]),
                         best_ratio=sub[metric].idxmin() if asc else sub[metric].idxmax(),
                         best_value=round(sub[metric].min() if asc else sub[metric].max(), 4)))
rank_df = pd.DataFrame(rows)
lines.append(rank_df.to_markdown(index=False))

# ---- point 3: mean_subset_size vs ideal_ratio ----
lines.append("\n## mean_subset_size vs ideal_ratio\n")
sub = df.groupby(["dataset", "ideal_ratio"])[["n_models", "mean_subset_size"]].mean()
lines.append(sub.round(2).to_markdown())

out = "\n".join(str(x) for x in lines)
with open("results/ideal_ratio_sweep_summary.md", "w", encoding="utf-8") as fh:
    fh.write(out)
print("wrote results/ideal_ratio_sweep_summary.md")
print("\n" + out)
