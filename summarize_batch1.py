"""Combine the 7 batch1 per-dataset CSVs into results/batch1_raw.csv and
produce the summary requested for the batch1 pipeline check."""
import numpy as np
import pandas as pd

from gals.analyze import summary_table

DATASETS = ["emotions", "flags", "Gnegative", "Gpositive", "Plant", "scene", "yeast"]
METRICS = ["hamming_loss", "subset_accuracy", "weighted_f1", "micro_f1"]
G, P = 20, 10   # GAConfig.n_generations, GAConfig.population_size

df = pd.concat([pd.read_csv(f"results/{d}_raw.csv") for d in DATASETS],
               ignore_index=True)
df.to_csv("results/batch1_raw.csv", index=False)
print(f"wrote results/batch1_raw.csv ({len(df)} rows)")

lines = ["# batch1 summary\n"]

# ---- 1. mean +/- std per (dataset, method), each of the 4 headline metrics ----
for metric in METRICS:
    lines.append(f"\n## {metric}\n")
    _, piv = summary_table(df, metric)
    lines.append(piv.to_markdown())
    lines.append("")

# ---- 2. GALS M (n_models) distribution ----
gals = df[df.method == "GALS"]
lines.append("\n## GALS M (n_models) distribution\n")
lines.append(f"overall: min={gals.n_models.min():.0f}  "
             f"median={gals.n_models.median():.0f}  max={gals.n_models.max():.0f}\n")
per_ds = gals.groupby("dataset")["n_models"].agg(["min", "median", "max"])
lines.append(per_ds.to_markdown())

# ---- 3. n_rf_trainings actual vs nominal cap G*P=200 ----
lines.append(f"\n## GALS n_rf_trainings vs nominal cap G*P={G*P}\n")
rf = gals.groupby("dataset")["n_rf_trainings"].agg(["min", "median", "max", "mean"])
rf["pct_of_cap_mean"] = (rf["mean"] / (G * P) * 100).round(1)
lines.append(rf.to_markdown())
lines.append(f"\noverall mean={gals.n_rf_trainings.mean():.1f} "
             f"({gals.n_rf_trainings.mean()/(G*P)*100:.1f}% of cap), "
             f"max={gals.n_rf_trainings.max():.0f} "
             f"(cap reached: {(gals.n_rf_trainings >= G*P).sum()}/{len(gals)} runs)")

# ---- 4. average runtime per method ----
lines.append("\n## mean runtime_sec per method (all 7 datasets pooled)\n")
rt = df.groupby("method")["runtime_sec"].agg(["mean", "std", "count"]).round(2)
lines.append(rt.to_markdown())
lines.append("\n## mean runtime_sec per (dataset, method)\n")
rt2 = df.groupby(["dataset", "method"])["runtime_sec"].mean().unstack().round(2)
lines.append(rt2.to_markdown())

# ---- 5. yeast Hamming Loss / Micro F1 / Weighted F1 ----
lines.append("\n## yeast headline numbers\n")
yeast = df[df.dataset == "yeast"]
means = yeast.groupby("method")[["hamming_loss", "weighted_f1", "micro_f1"]].mean()
gals_hl = means.loc["GALS", "hamming_loss"]
best_hl_method = means["hamming_loss"].idxmin()
best_hl_value = means["hamming_loss"].min()
gals_is_best_hl = best_hl_method == "GALS"

gals_wf1 = means.loc["GALS", "weighted_f1"]
best_wf1_method = means["weighted_f1"].idxmax()
gals_is_best_wf1 = best_wf1_method == "GALS"

gals_mf1 = means.loc["GALS", "micro_f1"]
best_mf1_method = means["micro_f1"].idxmax()
gals_is_best_mf1 = best_mf1_method == "GALS"

lines.append(f"- yeast Hamming Loss, GALS = {gals_hl:.4f}")
lines.append(f"- yeast Hamming Loss, lowest = {best_hl_method} = {best_hl_value:.4f} "
             f"({'GALS is lowest' if gals_is_best_hl else f'GALS is NOT lowest ({best_hl_method} is)'})")
lines.append(f"- yeast Weighted F1, GALS = {gals_wf1:.4f}, "
             f"highest = {best_wf1_method} = {means['weighted_f1'].max():.4f} "
             f"({'GALS is highest' if gals_is_best_wf1 else f'GALS is NOT highest ({best_wf1_method} is)'})")
lines.append(f"- yeast Micro F1, GALS = {gals_mf1:.4f}, "
             f"highest = {best_mf1_method} = {means['micro_f1'].max():.4f} "
             f"({'GALS is highest' if gals_is_best_mf1 else f'GALS is NOT highest ({best_mf1_method} is)'})")
lines.append("\nfull per-method means on yeast:\n")
lines.append(means.round(4).to_markdown())

out = "\n".join(str(x) for x in lines)
with open("results/batch1_summary.md", "w", encoding="utf-8") as fh:
    fh.write(out)
print("wrote results/batch1_summary.md")
print("\n" + out)
