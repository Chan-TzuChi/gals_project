"""Re-verify corel5k's ACkELD and ACkELO with the fixed, now-parallel
_run_many_with_timeout (2026-08-02): the pipe-buffer deadlock is fixed,
and ACkELD/ACkELO -- independent of each other -- now run CONCURRENTLY,
so seed 0's worst case is 6h total, not 6h+6h=12h sequential.

Uses m_override to skip re-running GALS (already have every seed's M from
the existing CSV) -- GALS is deterministic given the same seed/cfg, so
recomputing it here would just waste time (corel5k: ~6-13 min/seed) for a
row that gets thrown away anyway.

1. Seed 0: run ACkELD and ACkELO together, 6h cap each.
2. For whichever of the two actually finished within 6h: run seeds 1, 2
   for that method too (also run together if both finished).
3. For whichever DNF'd (a real DNF this time, not a masked deadlock):
   mark seeds 1, 2 as not attempted, honestly labeled.
"""
import os

import numpy as np
import pandas as pd

from gals.gals import GAConfig
from gals.loader import load_mulan
from gals.runner import run_one_seed

if __name__ == "__main__":
    CSV = "results/corel5k_raw.csv"
    TIMEOUT = 6 * 3600
    DATA_ROOT = os.environ.get("GALS_DATA_ROOT", os.path.join("..", "DATA"))

    X, Y, _, _ = load_mulan(os.path.join(DATA_ROOT, "corel5k.arff"),
                            os.path.join(DATA_ROOT, "corel5k.xml"))
    print(f"corel5k: X={X.shape} Y={Y.shape}", flush=True)
    cfg = GAConfig()

    gals_rows = pd.read_csv(CSV)
    gals_rows = gals_rows[gals_rows.method == "GALS"].set_index("seed")
    m_by_seed = gals_rows["n_models"].astype(int).to_dict()
    print(f"M per seed (from existing GALS rows, not recomputed): {m_by_seed}")

    def append(rows):
        df = pd.read_csv(CSV)
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        df.to_csv(CSV, index=False)

    def run_methods_seed(methods, seed):
        """Checkpoint each result the moment it arrives (not after the
        whole batch resolves) -- ACkELD and ACkELO run in parallel and can
        finish at very different times; buffering both before writing
        would lose an already-finished one if this process died while
        still waiting on the other."""
        rows = []
        for r in run_one_seed(
                X, Y, seed=seed, cfg=cfg, methods=methods, verbose=True,
                dataset="corel5k", m_override=m_by_seed[seed],
                method_timeouts={m: TIMEOUT for m in methods}):
            if r["method"] not in methods:
                continue
            r["dataset"] = "corel5k"
            append([r])
            rows.append(r)
        return rows

    print("\n=== ACkELD + ACkELO re-verification, seed 0, PARALLEL, 6h cap each ===",
         flush=True)
    rows0 = run_methods_seed(["ACkELD", "ACkELO"], 0)   # already checkpointed row-by-row
    print(f"seed 0 results: {rows0}")
    status_by_method = {r["method"]: r for r in rows0}

    finished = [m for m in ("ACkELD", "ACkELO")
               if "DNF" not in str(status_by_method[m].get("status", ""))]
    dnfd = [m for m in ("ACkELD", "ACkELO") if m not in finished]

    if dnfd:
        print(f"\n{dnfd} DNF'd on seed 0 "
             "-> seeds 1, 2 not attempted for these.")
        # Superseded by rerun_corel5k_ackeld.py, which re-runs all three seeds
        # under the unified 24h cap recorded in results/run_config.json.
        append([dict(method=m, seed=s, dataset="corel5k", runtime_sec=np.nan,
                    status=f"{m} did not complete within the 6h cap "
                           "applied in this batch")
               for m in dnfd for s in (1, 2)])

    if finished:
        print(f"\n{finished} finished within 6h on seed 0 -> running seeds 1, 2 "
             "for real (parallel if more than one method).")
        for seed in (1, 2):
            rows = run_methods_seed(finished, seed)   # already checkpointed row-by-row
            print(f"seed {seed} results: {rows}")

    df = pd.read_csv(CSV)
    print(f"\nfinal {CSV}: {len(df)} rows")
    print(df[["method", "seed", "status"]].sort_values(["seed", "method"]).to_string(index=False))
