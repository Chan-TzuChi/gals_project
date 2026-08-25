"""Environment capture and parameter sweeps (Table 2 reproduction, L_ideal sensitivity)."""

from __future__ import annotations

import json
import platform
import sys

import numpy as np
import pandas as pd

from .core import split_data
from .gals import GAConfig, fit_predict_ensemble, run_gals
from .core import evaluate_all


# --------------------------------------------------------------------------
def _windows_cpu_name():
    """platform.processor() on Windows returns the raw
    'Intel64 Family 6 Model ... Stepping ...' identifier, not a model name
    fit for a Methods section. Query WMI via PowerShell for the real one."""
    try:
        import subprocess
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=15)
        name = out.stdout.strip()
        return name or None
    except Exception:
        return None


def capture_environment(path=None, rf_params=None, cfg=None):
    """Record hardware and software details for the Methods section.

    rf_params / cfg: optional -- pass GAConfig.rf_params (or the GAConfig
    itself) to also record whether/how parallelism (n_jobs) is used, since
    that's part of "what ran the experiment", not just what's installed.
    """
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "os_version": platform.version(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count_logical": None,
        "ram_gb": None,
    }
    try:
        import os
        info["cpu_count_logical"] = os.cpu_count()
    except Exception:
        pass
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass
    try:                                   # more informative on Linux
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    info["processor"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    if platform.system() == "Windows":
        friendly = _windows_cpu_name()
        if friendly:
            info["processor"] = friendly

    for mod in ("numpy", "scipy", "sklearn", "pandas"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "not installed"

    rf_params = rf_params or (cfg.rf_params if cfg is not None else None)
    info["parallelism"] = {
        "rf_n_jobs": (rf_params or {}).get("n_jobs") if rf_params else None,
        "note": ("n_jobs=1 means each RandomForestClassifier fit is "
                 "single-threaded; no joblib/multiprocessing parallelism "
                 "is used inside model training. gals.runner uses a "
                 "separate subprocess per method only for the "
                 "method_timeouts wall-clock cap (ACkELD and ACkELO on "
                 "corel5k/mediamill), not for speed."),
    }

    if path:
        with open(path, "w") as fh:
            json.dump(info, fh, indent=2)
    return info


# --------------------------------------------------------------------------
def dump_run_config(cfg: GAConfig, path=None, seeds=None, seed_plan=None):
    """Serialize every setting needed to fully reproduce a run: GA
    hyperparameters, theta base value, RF params, split protocol, and
    whether the GA's validation set is the same as the final test set.
    Complements capture_environment() -- that records hardware/software,
    this records the experiment's own configuration.

    seeds: flat seed list, kept for the simple single-plan case.
    seed_plan: dict describing seed counts per batch/dataset/method when
    they differ (they do from batch2 onward: cal500 gets 10 seeds, corel5k
    gets 3 with ACkELO capped at 1, etc.) -- a flat `seeds` list can't
    represent that, this can. Actual n runs per (dataset, method),
    including outcomes (ok/DNF/error/...), is a post-hoc result of running
    the sweep, not a plan decided beforehand -- see
    gals.analyze.record_run_counts() for that.
    """
    info = {
        "ga": {
            "population_size": cfg.population_size,
            "n_generations": cfg.n_generations,
            "crossover_rate": cfg.crossover_rate,
            "mutation_rate": cfg.mutation_rate,
            "elitism_ratio": cfg.elitism_ratio,
            "alpha": cfg.alpha,
            "fitness_threshold_theta_base": cfg.fitness_threshold,
            "fitness_threshold_is_adaptive": cfg.fitness_threshold is None,
            "compensation_delta": cfg.compensation_delta,
            "theta_after_compensation": (
                None if cfg.fitness_threshold is None
                else cfg.fitness_threshold - cfg.compensation_delta),
            "ideal_ratio": cfg.ideal_ratio,
            "stall_tolerance": cfg.stall_tolerance,
            "stall_generations": cfg.stall_generations,
            "max_best_set": cfg.max_best_set,
        },
        "random_forest": dict(cfg.rf_params, random_state="= per-seed value"),
        "split": {
            "test_size": 0.30,
            "val_size_of_remaining_train": 0.30,
            "effective_ratio": "train_sub 49% / val 21% / test 30%",
            "method": "sklearn.train_test_split, shuffle=True, no stratify",
            "cross_validation": False,
            "ga_validation_set_equals_final_test_set": False,
            "note": ("GA fitness (run_gals) uses split['val'] only; the final "
                     "M-model ensemble is retrained on the full split['train'] "
                     "and split['test'] is touched exactly once, after GA "
                     "selection is frozen. val and test are disjoint index "
                     "sets produced by the same split_data() call."),
        },
        "seeds": list(seeds) if seeds is not None else None,
        "seed_plan": seed_plan,
        "run_counts_note": ("actual per-(dataset, method) execution counts "
                            "and outcomes (ok/DNF/not_implemented/error/"
                            "skipped_seed_limit) are in results/"
                            "run_counts.json, generated post-hoc by "
                            "gals.analyze.record_run_counts() over the raw "
                            "CSVs -- not duplicated here since this file is "
                            "written before the sweep runs."),
    }
    if path:
        with open(path, "w") as fh:
            json.dump(info, fh, indent=2)
    return info


# --------------------------------------------------------------------------
def _one_run(X, Y, seed, cfg):
    split = split_data(X, Y, seed)
    res = run_gals(X, Y, split, cfg, seed)
    P = fit_predict_ensemble(X, Y, split, res["subsets"], seed, cfg.rf_params)
    out = evaluate_all(Y[split["test"]], P)
    out.update(n_models=len(res["subsets"]),
               mean_subset_size=res["mean_subset_size"])
    return out


def alpha_sweep(name, X, Y, seeds, alphas=(0.8, 0.9, 1.0), base=None):
    """Reproduce Table 2: fitness weight configurations."""
    base = base or GAConfig()
    rows = []
    for a in alphas:
        cfg = GAConfig(**{**base.__dict__, "alpha": a})
        for s in seeds:
            r = _one_run(X, Y, s, cfg)
            r.update(dataset=name, alpha=a, seed=s)
            rows.append(r)
    return pd.DataFrame(rows)


def ideal_ratio_sweep(name, X, Y, seeds, ratios=(0.25, 1/3, 0.5, 2/3),
                      base=None):
    """Sensitivity analysis for L_ideal.

    Requires GAConfig.ideal_ratio and label_subset_ratio to honour it.
    """
    base = base or GAConfig()
    rows = []
    for r_ in ratios:
        cfg = GAConfig(**{**base.__dict__, "ideal_ratio": r_})
        for s in seeds:
            r = _one_run(X, Y, s, cfg)
            r.update(dataset=name, ideal_ratio=round(r_, 4), seed=s)
            rows.append(r)
    return pd.DataFrame(rows)


def summarize_sweep(df, key, metrics=("hamming_loss", "subset_accuracy",
                                      "weighted_f1", "micro_f1",
                                      "n_models", "mean_subset_size")):
    g = df.groupby(["dataset", key])[list(metrics)].agg(["mean", "std"])
    return g.round(4)
