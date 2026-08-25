"""Batch sweep driver, matching the 3-tier execution plan:

  batch1  7 small datasets (emotions/flags/Gnegative/Gpositive/Plant/scene/
          yeast), 10 seeds, all methods incl. ACkEL -- pipeline sanity check,
          not final numbers. Look at GALS's M range and the 4 headline
          metrics' rough direction before committing to the slow batches.
  batch2  cal500 (10 seeds, all methods) + corel5k (3 seeds; every method
          runs the full seed list under the unified 24h wall-clock cap --
          ACkELD and ACkELO must actually be attempted on corel5k, not
          skipped, to get real numbers for the comparison).
  batch3  mediamill (3 seeds; same ACkELD/ACkELO treatment as corel5k, for
          the same reason).

The corel5k and mediamill results in results/ were not produced by this
driver: see step2b_corel5k_reverify.py, step3b_mediamill_continue.py and
rerun_corel5k_ackeld.py, which apply no per-method seed limit. The seed
plan and caps actually used are recorded in results/run_config.json.

DNF policy: if ACkELO exceeds its timeout it is recorded as status="DNF"
(Madjarov et al. 2012 convention) and the sweep continues -- no max_pool
truncation of ACkELO's candidate pool is used to force it to finish faster.
That would silently shrink the method's search space and undercut an
honest "this method doesn't work here" conclusion. If you want to try
max_pool anyway, do it as a separate, clearly-labeled run -- not this one.

Usage:
  python run_batch.py batch1
  python run_batch.py batch1 --datasets emotions,flags   # subset
  python run_batch.py batch2
  python run_batch.py batch3
  python run_batch.py batch2 --ackelo-timeout-hours 6     # override the cap
"""
import argparse
import os
import sys

from gals.gals import GAConfig
from gals.loader import load_mulan
from gals.runner import run_dataset

DATA_ROOT = os.environ.get("GALS_DATA_ROOT", os.path.join("..", "DATA"))
ALL_METHODS = ["BR", "CC", "LP", "RAkELD", "RAkELO", "ACkELD", "ACkELO"]

BATCH1 = ["emotions", "flags", "Gnegative", "Gpositive", "Plant", "scene", "yeast"]
FILES = {
    "cal500": ("CAL500.arff", "CAL500.xml"),
    "corel5k": ("corel5k.arff", "corel5k.xml"),
    "emotions": ("emotions.arff", "emotions.xml"),
    "flags": ("flags.arff", "flags.xml"),
    "Gnegative": ("GnegativePseAAC.arff", "GnegativePseAAC.xml"),
    "Gpositive": ("GpositivePseAAC.arff", "GpositivePseAAC.xml"),
    "Plant": ("PlantPseAAC.arff", "PlantPseAAC.xml"),
    "scene": ("scene.arff", "scene.xml"),
    "yeast": ("yeast.arff", "yeast.xml"),
    "mediamill": ("mediamill.arff", "mediamill.xml"),
}


def load(name):
    arff, xml = FILES[name]
    X, Y, _, _ = load_mulan(os.path.join(DATA_ROOT, arff), os.path.join(DATA_ROOT, xml))
    return X, Y


def run(name, seeds, methods, cfg, method_timeouts=None, method_seed_limits=None):
    X, Y = load(name)
    print(f"=== {name}: X={X.shape} Y={Y.shape} seeds={list(seeds)} "
          f"methods={methods} timeouts={method_timeouts} "
          f"seed_limits={method_seed_limits} ===", flush=True)
    run_dataset(name, X, Y, seeds=seeds, cfg=cfg, methods=methods,
               out_csv=f"results/{name}_raw.csv", verbose=True, resume=True,
               method_timeouts=method_timeouts,
               method_seed_limits=method_seed_limits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", choices=["batch1", "batch2", "batch3"])
    ap.add_argument("--datasets", default=None,
                    help="comma-separated subset (batch1 only)")
    ap.add_argument("--ackelo-timeout-hours", type=float, default=24.0,
                    help="unified wall-clock cap, applied identically to "
                    "ACkELD and ACkELO so that reporting a run as a result "
                    "or as DNF does not depend on which method it is")
    ap.add_argument("--ackeld-timeout-hours", type=float, default=24.0,
                    help="unified wall-clock cap, applied identically to "
                    "ACkELD and ACkELO so that reporting a run as a result "
                    "or as DNF does not depend on which method it is")
    args = ap.parse_args()

    cfg = GAConfig()   # confirmed defaults, do not change

    if args.batch == "batch1":
        names = args.datasets.split(",") if args.datasets else BATCH1
        for name in names:
            run(name, range(10), ALL_METHODS, cfg)

    elif args.batch == "batch2":
        run("cal500", range(10), ALL_METHODS, cfg)
        timeouts = {"ACkELD": args.ackeld_timeout_hours * 3600,
                   "ACkELO": args.ackelo_timeout_hours * 3600}
        run("corel5k", range(3), ALL_METHODS, cfg,
           method_timeouts=timeouts)

    elif args.batch == "batch3":
        timeouts = {"ACkELD": args.ackeld_timeout_hours * 3600,
                   "ACkELO": args.ackelo_timeout_hours * 3600}
        run("mediamill", range(3), ALL_METHODS, cfg,
           method_timeouts=timeouts)


if __name__ == "__main__":
    sys.exit(main())
