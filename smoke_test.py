"""End-to-end smoke test on synthetic data with planted label structure."""
import numpy as np, pandas as pd
from gals.core import make_synthetic, dataset_stats
from gals.gals import GAConfig
from gals.runner import run_dataset
from gals.analyze import full_report, to_markdown

pd.set_option("display.width", 200)
cfg = GAConfig(population_size=12, n_generations=6,
               rf_params=dict(n_estimators=30, max_depth=None,
                              min_samples_split=2, n_jobs=1))

frames = []
for name, (n, d, L) in {"syn_a": (300, 15, 8),
                        "syn_b": (300, 15, 10),
                        "syn_c": (250, 12, 6)}.items():
    X, Y = make_synthetic(n=n, d=d, L=L, seed=hash(name) % 100)
    print(name, dataset_stats(Y))
    frames.append(run_dataset(name, X, Y, seeds=[0, 1, 2], cfg=cfg,
                              methods=["BR", "CC", "LP", "RAkELD", "RAkELO"]))

df = pd.concat(frames, ignore_index=True)
df.to_csv("results/smoke_raw.csv", index=False)
print("\n=== raw rows:", len(df), "===")
print(df.groupby("method")[["hamming_loss","micro_f1","runtime_sec"]].mean())

rep = full_report(df)
print("\n=== micro_f1 table ===")
print(rep["micro_f1"]["table"])
print("\n=== hamming_loss avg rank ===")
print(rep["hamming_loss"]["avg_rank"])
to_markdown(rep, "results/smoke_report.md")
print("\nGALS diagnostics:")
print(df[df.method=="GALS"][["dataset","seed","n_models","mean_subset_size",
                             "threshold","generations","n_rf_trainings",
                             "n_compensated"]])
