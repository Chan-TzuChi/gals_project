"""Measure GALS and ACkELD on corel5k seed 0 with the machine to itself.

WHY THIS EXISTS
---------------
The runtimes recorded for corel5k and mediamill were not taken under
comparable conditions. GALS on corel5k did identical work on all three seeds
(M=34, 34 RF fits, 5 generations) yet took 353.3s, 765.1s and 599.6s, so that
2.17x spread measures the machine's load at the time, not the method. Only
seed 0 has an execution log, and it agrees across two runs (354.2s, 353.3s);
seeds 1-2 were written while the mediamill run was active. See
results/run_config.json, runtime_measurement_conditions.

This script produces one pair of numbers that can be compared with each
other: GALS seed 0 and ACkELD seed 0, run one after the other, in-process,
on an otherwise idle machine, with BLAS left unrestricted exactly as in every
other run in the study. GALS goes first because it takes minutes rather than
hours and confirms the machine is behaving as expected before the long run
starts.

Deliberately NOT set: OMP_NUM_THREADS / MKL_NUM_THREADS /
OPENBLAS_NUM_THREADS. Restricting ACkELD to one core while the GALS figure it
is compared against had the whole machine would bias the comparison the other
way.

Both methods are timed the same way run_one_seed times them: for GALS, the GA
plus the final ensemble fit and predict; for ACkELD, the single call in
gals.baselines. Effective core usage is recorded alongside each timing, so
the measurement condition travels with the number.

Usage:
    python measure_corel5k_solo.py

Refuses to start if another Python process is running, since that would
reproduce the contention this measurement exists to remove.
"""
import argparse
import os
import time

import pandas as pd

from gals.baselines import BASELINES
from gals.core import evaluate_all, split_data
from gals.gals import GAConfig, fit_predict_ensemble, run_gals

DATASET = "corel5k"
SEED = 0
K = 3
OUT = "results/corel5k_solo_timing.csv"


def other_python_processes():
    """PIDs of Python processes other than this one, or None if unknown."""
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name like '%python%'\" "
             "| Select-Object -ExpandProperty ProcessId"],
            capture_output=True, text=True, timeout=60).stdout.split()
        return [int(x) for x in out if x.strip().isdigit() and int(x) != os.getpid()]
    except Exception:
        return None


def timed(label, call):
    """Run `call`, returning (result, wall_seconds, cpu_seconds, cores)."""
    w0, c0 = time.perf_counter(), time.process_time()
    result = call()
    wall = time.perf_counter() - w0
    cpu = time.process_time() - c0
    cores = cpu / wall if wall else float("nan")
    print(f"[{label}] wall {wall:.1f}s ({wall/3600:.2f}h)  "
          f"cpu {cpu:.1f}s  effective cores {cores:.2f}", flush=True)
    return result, wall, cpu, cores


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-busy", action="store_true",
                    help="run even if other Python processes are present")
    args = ap.parse_args()

    others = other_python_processes()
    if others is None:
        print("could not enumerate processes; check by hand that the machine "
              "is idle before trusting these numbers", flush=True)
    elif others and not args.allow_busy:
        raise SystemExit(
            f"other Python processes are running (PIDs {others}). These "
            f"timings are only meaningful on an idle machine. Wait for them "
            f"to finish, or pass --allow-busy to override.")

    DATA_ROOT = os.environ.get("GALS_DATA_ROOT", os.path.join("..", "DATA"))
    cfg = GAConfig()

    from gals.loader import load_mulan
    X, Y, _, _ = load_mulan(os.path.join(DATA_ROOT, f"{DATASET}.arff"),
                            os.path.join(DATA_ROOT, f"{DATASET}.xml"))
    print(f"{DATASET}: X={X.shape} Y={Y.shape}  seed {SEED}, solo, "
          f"BLAS unrestricted", flush=True)

    split = split_data(X, Y, SEED)
    Xtr, Ytr = X[split["train"]], Y[split["train"]]
    Xte, Yte = X[split["test"]], Y[split["test"]]

    rows = []

    # ---- GALS first: minutes, and confirms the machine looks normal -------
    def run_gals_call():
        res = run_gals(X, Y, split, cfg, SEED, verbose=True)
        P = fit_predict_ensemble(X, Y, split, res["subsets"], SEED, cfg.rf_params)
        return res, P

    (res, P), wall, cpu, cores = timed("GALS", run_gals_call)
    m = len(res["subsets"])
    print(f"[GALS] M={m} rf_fits={res['n_rf_trainings']} "
          f"generations={res['generations']}", flush=True)
    if m != 34:
        print(f"[GALS] NOTE: M={m}, but the recorded corel5k runs all had "
              f"M=34. The workload is not the one being compared against.",
              flush=True)
    rows.append(dict(method="GALS", seed=SEED, runtime_sec=wall,
                     cpu_sec=cpu, effective_cores=cores,
                     n_models=m, mean_subset_size=res["mean_subset_size"],
                     threshold=res["threshold"],
                     generations=res["generations"],
                     n_rf_trainings=res["n_rf_trainings"],
                     n_compensated=res["n_missing_compensated"],
                     dataset=DATASET, measurement="solo_sequential",
                     **evaluate_all(Yte, P)))
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"  -> checkpointed {OUT}", flush=True)

    # ---- ACkELD second ----------------------------------------------------
    fn = BASELINES["ACkELD"]

    def run_ackeld_call():
        return fn(Xtr, Ytr, Xte, SEED, cfg.rf_params, k=K, dataset=DATASET)

    Pd, wall, cpu, cores = timed("ACkELD", run_ackeld_call)
    rows.append(dict(method="ACkELD", seed=SEED, runtime_sec=wall,
                     cpu_sec=cpu, effective_cores=cores,
                     dataset=DATASET, measurement="solo_sequential",
                     **evaluate_all(Yte, Pd)))
    pd.DataFrame(rows).to_csv(OUT, index=False)

    # ---- report -----------------------------------------------------------
    df = pd.DataFrame(rows)
    print(f"\n=== corel5k seed {SEED}, solo sequential ===")
    print(df[["method", "runtime_sec", "cpu_sec", "effective_cores"]]
          .to_string(index=False))
    g = df[df.method == "GALS"].runtime_sec.iloc[0]
    a = df[df.method == "ACkELD"].runtime_sec.iloc[0]
    print(f"\nACkELD / GALS = {a/g:.1f}x "
          f"(both measured solo, sequentially, same seed)")
