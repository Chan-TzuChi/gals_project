"""End-to-end smoke test on synthetic data with planted label structure."""
import io

import numpy as np, pandas as pd
from gals.core import make_synthetic, dataset_stats
from gals.gals import GAConfig
from gals.runner import run_dataset
from gals.analyze import full_report, to_markdown

SMOKE_HEADER = """# Smoke test report (synthetic data)

Output of `python smoke_test.py`, which runs the pipeline end to end on three
small synthetic datasets (`syn_a`, `syn_b`, `syn_c`) with planted label
structure. Its purpose is to confirm that an installation works.

These are not results from the paper. The datasets are artificial, only three
seeds are used, and every Wilcoxon test reports `too few datasets`, so the
rankings and values below carry no substantive meaning and should not be
compared with the reported tables.
"""


def prepend_smoke_header(path):
    with io.open(path, encoding="utf-8") as fh:
        body = fh.read()
    with io.open(path, "w", encoding="utf-8") as fh:
        fh.write(SMOKE_HEADER + body)


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
prepend_smoke_header("results/smoke_report.md")
print("\nGALS diagnostics:")
print(df[df.method=="GALS"][["dataset","seed","n_models","mean_subset_size",
                             "threshold","generations","n_rf_trainings",
                             "n_compensated"]])
