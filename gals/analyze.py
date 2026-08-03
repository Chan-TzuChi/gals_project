"""Aggregate raw per-seed results into publication tables:
mean +/- std, average ranks, and Wilcoxon signed-rank tests vs GALS."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

LOWER_IS_BETTER = {"hamming_loss", "runtime_sec"}
METHOD_ORDER = ["BR", "CC", "LP", "RAkELD", "RAkELO", "ACkELD", "ACkELO",
                "GALS"]


def _order(methods):
    known = [m for m in METHOD_ORDER if m in methods]
    return known + sorted(set(methods) - set(known))


def summary_table(df, metric, decimals=3):
    """Mean +/- std per (dataset, method), with competition ranking."""
    ok = df[df[metric].notna()]
    g = ok.groupby(["dataset", "method"])[metric].agg(["mean", "std", "count"])
    g = g.reset_index()

    rows = []
    for ds, sub in g.groupby("dataset"):
        asc = metric in LOWER_IS_BETTER
        sub = sub.copy()
        sub["rank"] = sub["mean"].rank(ascending=asc, method="min").astype(int)
        for _, r in sub.iterrows():
            std = 0.0 if np.isnan(r["std"]) else r["std"]
            rows.append(dict(dataset=ds, method=r["method"],
                             mean=r["mean"], std=std,
                             n_runs=int(r["count"]), rank=r["rank"],
                             cell=f"{r['mean']:.{decimals}f}±{std:.{decimals}f} ({r['rank']})"))
    out = pd.DataFrame(rows)
    piv = out.pivot(index="dataset", columns="method", values="cell")
    return out, piv.reindex(columns=_order(piv.columns))


def average_ranks(detail):
    r = detail.pivot(index="dataset", columns="method", values="rank")
    return r.mean().sort_values()


def wilcoxon_vs_gals(df, metric, reference="GALS"):
    """Paired Wilcoxon signed-rank test across datasets, on per-dataset means.

    Requires >=6 datasets for a meaningful two-sided test.
    """
    ok = df[df[metric].notna()]
    means = ok.groupby(["dataset", "method"])[metric].mean().unstack()
    if reference not in means.columns:
        return pd.DataFrame()

    rows = []
    for m in means.columns:
        if m == reference:
            continue
        pair = means[[reference, m]].dropna()
        if pair.shape[0] < 6:
            rows.append(dict(method=m, n_datasets=pair.shape[0],
                             statistic=np.nan, p_value=np.nan,
                             note="too few datasets"))
            continue
        diff = pair[reference] - pair[m]
        if np.allclose(diff, 0):
            rows.append(dict(method=m, n_datasets=pair.shape[0],
                             statistic=np.nan, p_value=1.0, note="identical"))
            continue
        stat, p = wilcoxon(pair[reference], pair[m])
        better = ((diff < 0).sum() if metric in LOWER_IS_BETTER
                  else (diff > 0).sum())
        rows.append(dict(method=m, n_datasets=pair.shape[0],
                         gals_better_on=int(better),
                         statistic=float(stat), p_value=float(p),
                         significant_005=bool(p < 0.05)))
    return pd.DataFrame(rows)


def full_report(df, metrics=None, decimals=3):
    metrics = metrics or ["hamming_loss", "subset_accuracy", "weighted_f1",
                          "micro_f1", "runtime_sec"]
    out = {}
    for m in metrics:
        if m not in df.columns:
            continue
        detail, piv = summary_table(df, m, decimals)
        out[m] = dict(detail=detail, table=piv,
                      avg_rank=average_ranks(detail),
                      wilcoxon=wilcoxon_vs_gals(df, m))
    return out


def record_run_counts(csv_paths, out_path=None):
    """Per (dataset, method) run count and outcome breakdown (ok / DNF /
    not_implemented / out_of_memory / error / skipped_seed_limit), for
    table captions that need to explain why seed counts differ across
    cells. This is a post-hoc report over actual outcomes, not
    a static setting -- kept separate from experiments.dump_run_config(),
    which only records what was *configured* before a sweep ran.

    csv_paths: one or more raw results CSVs (e.g. results/corel5k_raw.csv).
    """
    if isinstance(csv_paths, (str, bytes)):
        csv_paths = [csv_paths]
    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    if "status" not in df.columns:
        df["status"] = np.nan
    df["outcome"] = df["status"].where(df["status"].notna(), "ok")
    counts = (df.groupby(["dataset", "method", "outcome"])
                .size().rename("n").reset_index())
    pivot = counts.pivot_table(index=["dataset", "method"], columns="outcome",
                               values="n", fill_value=0).astype(int)
    pivot["n_total"] = pivot.sum(axis=1)
    pivot = pivot.reset_index()
    if out_path:
        pivot.to_json(out_path, orient="records", indent=2)
    return pivot


def to_markdown(report, path):
    with open(path, "w") as f:
        for metric, r in report.items():
            f.write(f"\n## {metric}\n\n")
            f.write(r["table"].to_markdown())
            f.write("\n\n**Average rank**\n\n")
            f.write(r["avg_rank"].to_frame("avg_rank").to_markdown())
            if not r["wilcoxon"].empty:
                f.write("\n\n**Wilcoxon signed-rank vs GALS**\n\n")
                f.write(r["wilcoxon"].to_markdown(index=False))
            f.write("\n")
    return path
