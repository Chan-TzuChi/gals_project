"""Re-run ACkELD on corel5k with a uniform 24h wall-clock cap.

WHY THIS EXISTS
---------------
The original sweep gave ACkELO a 12h cap but ACkELD only 6h ("defensive"),
see results/run_config.json. ACkELD hit that 6h cap on corel5k and was
recorded as DNF, while GALS on mediamill ran 11.1-22.4h with no cap at all.
An inconsistent cap across methods makes the runtime table indefensible:
some >6h runs are reported as failures, others as results.

This script removes that inconsistency by re-running ACkELD/corel5k under a
cap that no run in the study has ever exceeded (24h), so the reported
outcome no longer depends on where the cap sits.

Cost estimate: the O(N_train^2 * L^2) cost structure, calibrated on cal500
(the completed dataset with the largest label space, L=174), predicts
~5.8h per seed for corel5k (N_train=2450, L=374). The original run was
killed at 6.0h, i.e. plausibly within minutes of finishing.

Seeds run CONCURRENTLY. Memory is not the constraint: _kernel_sum() works
in row blocks, so peak memory grows linearly with N (order of hundreds of
MB per process here), not as N^2. CPU is: rf_params n_jobs=1 bounds only
scikit-learn's own parallelism, not the BLAS backend underneath numpy, and
OpenBLAS defaults to one thread per logical core. Each seed therefore
spreads across many cores and the three processes contend with each other,
so the wall clock is neither ~max(t) nor sum(t), and the runtime_sec
recorded here is inflated relative to a run that had the machine to itself.

Concurrency is the right trade for establishing whether a seed completes
inside the cap, which is what this re-run is for. It is the wrong basis for
a runtime comparison. Set PARALLEL_SEEDS = False, and run nothing else, to
measure a timing that is comparable with one taken the same way.

Usage:
    python rerun_corel5k_ackeld.py
"""
import os
import shutil
import time
from datetime import datetime

import numpy as np
import pandas as pd

from gals.core import evaluate_all, split_data
from gals.baselines import BASELINES
from gals.gals import GAConfig
from gals.runner import _run_many_with_timeout

# ---------------------------------------------------------------- settings
CSV = "results/corel5k_raw.csv"
DATASET = "corel5k"
METHOD = "ACkELD"
SEEDS = (0, 1, 2)
TIMEOUT = 24 * 3600          # uniform cap; nothing in the study exceeds this
PARALLEL_SEEDS = True        # False -> sequential, cleaner timing, ~3x longer
K = 3

if __name__ == "__main__":
    DATA_ROOT = os.environ.get("GALS_DATA_ROOT", os.path.join("..", "DATA"))
    cfg = GAConfig()

    from gals.loader import load_mulan
    X, Y, _, _ = load_mulan(os.path.join(DATA_ROOT, f"{DATASET}.arff"),
                            os.path.join(DATA_ROOT, f"{DATASET}.xml"))
    print(f"{DATASET}: X={X.shape} Y={Y.shape}", flush=True)
    print(f"cap = {TIMEOUT}s ({TIMEOUT/3600:.0f}h), "
          f"seeds {SEEDS} {'CONCURRENT' if PARALLEL_SEEDS else 'SEQUENTIAL'}",
          flush=True)

    # -------- back up, then drop the stale ACkELD rows so we don't duplicate
    backup = f"{CSV}.bak-{datetime.now():%Y%m%d-%H%M%S}"
    shutil.copy(CSV, backup)
    print(f"backup written to {backup}", flush=True)

    df = pd.read_csv(CSV)
    stale = df[df.method == METHOD]
    print(f"dropping {len(stale)} stale {METHOD} row(s):")
    print(stale[["method", "seed", "runtime_sec", "status"]].to_string(index=False))
    df[df.method != METHOD].to_csv(CSV, index=False)

    def append(row):
        cur = pd.read_csv(CSV)
        pd.concat([cur, pd.DataFrame([row])], ignore_index=True).to_csv(CSV, index=False)

    # -------- build one job per seed
    jobs, Yte_by_seed = [], {}
    fn = BASELINES[METHOD]
    for seed in SEEDS:
        split = split_data(X, Y, seed)
        Xtr, Ytr = X[split["train"]], Y[split["train"]]
        Xte, Yte = X[split["test"]], Y[split["test"]]
        Yte_by_seed[seed] = Yte
        args = (Xtr, Ytr, Xte, seed, cfg.rf_params)
        kwargs = dict(k=K, dataset=DATASET)
        jobs.append((seed, fn, args, kwargs, TIMEOUT))

    def record(seed, status, payload, elapsed):
        if status == "DNF":
            row = dict(method=METHOD, seed=seed, runtime_sec=elapsed,
                       status=f"DNF (timeout={TIMEOUT}s)")
        elif status == "not_implemented":
            # Must be handled explicitly: falling through to the "ok" branch
            # would hand evaluate_all() the exception *message* as if it were
            # a prediction array, which silently produces plausible-looking
            # metrics for a method that never ran.
            row = dict(method=METHOD, seed=seed, runtime_sec=np.nan,
                       status="not_implemented")
        elif status == "out_of_memory":
            row = dict(method=METHOD, seed=seed, runtime_sec=np.nan,
                       status="out_of_memory")
        elif status == "error":
            row = dict(method=METHOD, seed=seed, runtime_sec=np.nan,
                       status=f"error: {payload}")
        else:
            row = dict(method=METHOD, seed=seed, runtime_sec=elapsed,
                       **evaluate_all(Yte_by_seed[seed], payload))
        row["dataset"] = DATASET
        append(row)                      # checkpoint immediately
        print(f"[{datetime.now():%H:%M:%S}] seed {seed}: {status} "
              f"({elapsed/3600:.2f}h)", flush=True)
        return row

    # -------- run
    t0 = time.perf_counter()
    if PARALLEL_SEEDS:
        for seed, status, payload in _run_many_with_timeout(jobs):
            record(seed, status, payload, time.perf_counter() - t0)
    else:
        for job in jobs:
            s0 = time.perf_counter()
            for seed, status, payload in _run_many_with_timeout([job]):
                record(seed, status, payload, time.perf_counter() - s0)

    # -------- report
    out = pd.read_csv(CSV)
    got = out[out.method == METHOD]
    print(f"\n=== {METHOD} on {DATASET}, cap {TIMEOUT/3600:.0f}h ===")
    cols = [c for c in ["seed", "runtime_sec", "hamming_loss", "subset_accuracy",
                        "weighted_f1", "micro_f1", "status"] if c in got.columns]
    print(got[cols].sort_values("seed").to_string(index=False))
    done = got[got.runtime_sec.notna() & got.get("hamming_loss", pd.Series()).notna()] \
        if "hamming_loss" in got.columns else got.iloc[0:0]
    if len(done):
        print(f"\nruntime  mean {done.runtime_sec.mean():.0f}s "
              f"({done.runtime_sec.mean()/3600:.2f}h) "
              f"± {done.runtime_sec.std(ddof=1):.0f}s")
        gals = out[out.method == "GALS"]
        if len(gals):
            print(f"GALS     mean {gals.runtime_sec.mean():.0f}s "
                  f"({gals.runtime_sec.mean()/3600:.2f}h)")
            print(f"ratio    ACkELD / GALS = "
                  f"{done.runtime_sec.mean()/gals.runtime_sec.mean():.1f}x")
