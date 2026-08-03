"""Resume ideal_ratio_sweep for yeast only (emotions/Gnegative/Plant already
complete in results/ideal_ratio_sweep.csv from the prior run that was
killed partway into yeast)."""
import os

import pandas as pd

from gals.experiments import ideal_ratio_sweep
from gals.loader import load_mulan

DATA_ROOT = os.environ.get("GALS_DATA_ROOT", os.path.join("..", "DATA"))
X, Y, _, _ = load_mulan(
    os.path.join(DATA_ROOT, "yeast.arff"),
    os.path.join(DATA_ROOT, "yeast.xml"))
print(f"=== yeast: X={X.shape} Y={Y.shape} ===", flush=True)

df_yeast = ideal_ratio_sweep("yeast", X, Y, seeds=range(5),
                             ratios=(0.25, 1/3, 0.5, 2/3))

prev = pd.read_csv("results/ideal_ratio_sweep.csv")
full = pd.concat([prev, df_yeast], ignore_index=True)
full.to_csv("results/ideal_ratio_sweep.csv", index=False)
print(f"wrote results/ideal_ratio_sweep.csv ({len(full)} rows)")

METRICS = ["hamming_loss", "subset_accuracy", "weighted_f1", "micro_f1"]
lines = ["# ideal_ratio_sweep summary (5 seeds/cell)\n"]
for metric in METRICS:
    lines.append(f"\n## {metric}\n")
    g = full.groupby(["dataset", "ideal_ratio"])[metric].agg(["mean", "std"])
    cell = (g["mean"].round(4).astype(str) + "\u00b1" + g["std"].round(4).astype(str))
    piv = cell.unstack("ideal_ratio")
    lines.append(piv.to_markdown())

lines.append("\n## Sensitivity: range (max-min of the mean across ideal_ratio), per dataset\n")
means = full.groupby(["dataset", "ideal_ratio"])[METRICS].mean()
rng = means.groupby("dataset").agg(lambda s: s.max() - s.min())
lines.append(rng.round(4).to_markdown())

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

lines.append("\n## mean_subset_size vs ideal_ratio\n")
sub = full.groupby(["dataset", "ideal_ratio"])[["n_models", "mean_subset_size"]].mean()
lines.append(sub.round(2).to_markdown())

out = "\n".join(str(x) for x in lines)
with open("results/ideal_ratio_sweep_summary.md", "w", encoding="utf-8") as fh:
    fh.write(out)
print("wrote results/ideal_ratio_sweep_summary.md")
print("\n" + out)
