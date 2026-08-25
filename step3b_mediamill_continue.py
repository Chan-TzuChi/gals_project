"""Continues the mediamill batch after the initial run. The earlier driver
for this dataset is not retained in the repository; the configuration it
used is recorded in results/run_config.json under batch3_mediamill.

This script:

1. Reads whatever is already in results/mediamill_raw.csv for seed 0
   (likely GALS/BR/CC at minimum).
2. Runs the METHODS STILL MISSING for seed 0 using the parallel runner with
   per-method wall-clock caps (LP/ACkELD/ACkELO launch together, so the worst
   case is max(caps) = 6h rather than their sum),
   using m_override to reuse seed 0's already-known M instead of
   recomputing GALS (which took 16.7h -- redoing it here to throw the row
   away would defeat the entire point of "continuing").
3. Runs seeds 1, 2 in full (GALS included -- these
   haven't run at all yet, so there's nothing to reuse).

Safe to re-run: it only ever adds rows for (seed, method) pairs not
already present, never touches what's already saved.
"""
import os

import numpy as np
import pandas as pd

from gals.gals import GAConfig
from gals.loader import load_mulan
from gals.runner import run_dataset, run_one_seed

ALL_METHODS = ["BR", "CC", "LP", "RAkELD", "RAkELO", "ACkELD", "ACkELO"]
TIMEOUT = 6 * 3600

if __name__ == "__main__":
    CSV = "results/mediamill_raw.csv"
    DATA_ROOT = os.environ.get("GALS_DATA_ROOT", os.path.join("..", "DATA"))

    X, Y, _, _ = load_mulan(os.path.join(DATA_ROOT, "mediamill.arff"),
                            os.path.join(DATA_ROOT, "mediamill.xml"))
    print(f"mediamill: X={X.shape} Y={Y.shape}", flush=True)
    cfg = GAConfig()

    prev = pd.read_csv(CSV) if os.path.exists(CSV) else pd.DataFrame(
        columns=["method", "seed"])

    # --- seed 0: fill in whatever's missing, reusing its known M ---
    done_seed0 = set(prev[prev.seed == 0]["method"]) if len(prev) else set()
    missing_seed0 = [m for m in ALL_METHODS if m not in done_seed0]
    print(f"seed 0 already has: {sorted(done_seed0)}")
    print(f"seed 0 still needs: {missing_seed0}")

    if missing_seed0:
        gals_seed0 = prev[(prev.seed == 0) & (prev.method == "GALS")]
        if len(gals_seed0) == 0:
            raise RuntimeError(
                "seed 0 has no GALS row yet -- nothing to reuse. Don't run "
                "this script until seed 0's GALS row exists in the CSV; it is "
                "produced by a full mediamill seed-0 run.")
        m0 = int(gals_seed0.iloc[0]["n_models"])
        print(f"reusing seed 0's known M={m0} (not recomputing GALS)")

        # Checkpoint each result the instant it arrives -- BR/CC/RAkELD/
        # RAkELO can finish in seconds while LP/ACkELD/ACkELO run in
        # parallel for up to 6h; buffering everything into one list before
        # writing would lose the fast ones too if this process died while
        # still waiting on the slow ones.
        rows = []
        for r in run_one_seed(
                X, Y, seed=0, cfg=cfg, methods=missing_seed0, verbose=True,
                dataset="mediamill", m_override=m0,
                method_timeouts={m: TIMEOUT for m in missing_seed0 if m in
                                ("LP", "ACkELD", "ACkELO")}):
            if r["method"] not in missing_seed0:
                continue
            r["dataset"] = "mediamill"
            df = pd.read_csv(CSV) if os.path.exists(CSV) else pd.DataFrame()
            df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)
            df.to_csv(CSV, index=False)
            rows.append(r)
        print(f"seed 0 completed: {[r['method'] for r in rows]}")

    # --- seeds 1, 2: run in full (nothing to reuse yet) ---
    run_dataset("mediamill", X, Y, seeds=[1, 2], cfg=cfg, methods=ALL_METHODS,
               out_csv=CSV, verbose=True, resume=True,
               method_timeouts={"LP": TIMEOUT, "ACkELD": TIMEOUT, "ACkELO": TIMEOUT})

    df = pd.read_csv(CSV)
    print(f"\nfinal {CSV}: {len(df)} rows")
    print(df[["method", "seed", "status"] if "status" in df else ["method", "seed"]]
         .sort_values(["seed", "method"]).to_string(index=False))
