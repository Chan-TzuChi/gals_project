"""Re-run ACkELD on mediamill under the uniform 24h wall-clock cap.

WHY THIS EXISTS
---------------
The mediamill ACkELD rows in results/mediamill_raw.csv were produced under
the 6h cap that step3b_mediamill_continue.py applied to LP, ACkELD and
ACkELO together: seed 0 recorded out_of_memory, seeds 1-2 DNF at that cap.
corel5k's ACkELD carried the same 6h cap and was likewise recorded as DNF,
yet completed in 12.25h once the cap was raised to the uniform 24h
(rerun_corel5k_ackeld.py). This script asks the same question of mediamill
under the same protocol, so that no dataset's outcome depends on which cap
happened to be in force for it.

OUTCOME CRITERION, FIXED IN ADVANCE
-----------------------------------
Decided before the run, so the reading of the result does not depend on
what the result turns out to be:

  - Completes within 24h -> report the measured time.
  - Does not complete within 24h -> the DNF stands. It is not retried at a
    longer cap, and the cap is not raised for this dataset.

A DNF at 24h is defensible precisely because the cap is uniform: no result
adopted anywhere in this study took longer than 24h to produce (the longest
is GALS on mediamill at 22.4h), so "ACkELD did not finish within 24h while
GALS finished in 11.1-22.4h" is a statement made under one protocol rather
than a comparison between two different ones. Raising the cap for this one
dataset would reintroduce the per-dataset caps that made the earlier
runtime table indefensible, in the opposite direction.

COST
----
ACkELD's cost is the number of quality() evaluations times the O(N^2 * n)
cost of each. Against corel5k, mediamill makes about 13.7x fewer calls
(33 subsets over 101 labels, against 124 over 374) but each call is about
18.5x more expensive (N_train 30734 against 3500), for a net factor of
about 1.35. Scaling corel5k's measured 12.25h per seed gives roughly 16.5h
per seed. That estimate rests on a coarse model and leaves only about 1.45x
of headroom under the cap, so a DNF is a real possibility; see the
criterion above.

MEMORY
------
Seed 0's earlier out_of_memory is attributable to what it ran alongside,
not to ACkELD: step3b launched LP, ACkELD and ACkELO together, and LP built
a 6555-class problem from mediamill's 6555 distinct label sets and ran out
of memory on all three seeds. ACkELD seeds 1-2 ran the full 6h under the
same conditions without exhausting memory. Only ACkELD runs here.

_kernel_sum() works in row blocks, so the transient is block x N_train, not
N^2. block=256 rather than the default 1024 puts that at roughly 60MB per
block (about 120MB including the exp() copy) for N_train=30734, so three
concurrent seeds stay well inside available memory. Blocking changes only
how the sum is accumulated, not its value.

CONCURRENCY
-----------
Seeds run concurrently. Measured on the corel5k re-run: a single ACkELD
process uses about 3.85 cores even with the machine to itself, and three
concurrent processes used about 12.4 of 22 logical cores in total, so they
do not starve each other and the wall clock is close to one seed's rather
than three. rf_params n_jobs=1 bounds scikit-learn only, not the BLAS
backend underneath numpy, which is left unrestricted as in every other run.

Concurrency is appropriate for establishing whether a seed completes inside
the cap, which is what this run is for. It is not a basis for a runtime
comparison against a figure measured some other way.

Usage:
    python rerun_mediamill_ackeld.py
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
CSV = "results/mediamill_raw.csv"
DATASET = "mediamill"
METHOD = "ACkELD"
SEEDS = (0, 1, 2)
TIMEOUT = 24 * 3600          # uniform cap; not raised for this dataset
PARALLEL_SEEDS = True        # False -> sequential, cleaner timing, ~3x longer
K = 3
BLOCK = 256                  # _kernel_sum row block; memory only, not values

if __name__ == "__main__":
    DATA_ROOT = os.environ.get("GALS_DATA_ROOT", os.path.join("..", "DATA"))
    cfg = GAConfig()

    from gals.loader import load_mulan
    X, Y, _, _ = load_mulan(os.path.join(DATA_ROOT, f"{DATASET}.arff"),
                            os.path.join(DATA_ROOT, f"{DATASET}.xml"))
    print(f"{DATASET}: X={X.shape} Y={Y.shape}", flush=True)
    print(f"cap = {TIMEOUT}s ({TIMEOUT/3600:.0f}h), "
          f"seeds {SEEDS} {'CONCURRENT' if PARALLEL_SEEDS else 'SEQUENTIAL'}, "
          f"kernel block = {BLOCK}", flush=True)

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
        kwargs = dict(k=K, dataset=DATASET, block=BLOCK)
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
            print("         (the GALS figure this divides by spans 2.03x "
                  "across seeds for identical work -- see "
                  "results/run_config.json, "
                  "runtime_measurement_conditions -- so treat the ratio as "
                  "an order of magnitude, not a measurement)")
    else:
        print("\nNo seed completed within the cap. Under the criterion fixed "
              "in this script's docstring the DNF stands: it is not retried "
              "at a longer cap.")
