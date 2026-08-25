"""Grid-search ACkEL (gamma, beta) for the 3 PseAAC datasets not covered by
the original paper's Table 1 (different feature representation: PseAAC vs.
the paper's "Go" variant). Writes results/ackel_tuning.json; copy the
winning (gamma, beta_d, beta_o) into ACKEL_PARAMS in gals/baselines.py.

The search results go under the file's "tuned" key. The provenance metadata
alongside them covers all ten datasets, including the six that reuse the
paper's Table 1 values and corel5k's untuned default, so it is written here
too rather than being lost whenever this script is re-run.
"""
import json
import os
import time

from gals.ackel import tune_ackel_params
from gals.loader import load_mulan

DATASETS = {
    "Gnegative": ("GnegativePseAAC.arff", "GnegativePseAAC.xml"),
    "Gpositive": ("GpositivePseAAC.arff", "GpositivePseAAC.xml"),
    "Plant":     ("PlantPseAAC.arff", "PlantPseAAC.xml"),
}
DATA_ROOT = os.environ.get("GALS_DATA_ROOT", os.path.join("..", "DATA"))

# Provenance metadata, written alongside the search results. Kept here so
# that re-running this script cannot silently drop it.
PROVENANCE = {
    "note": "Provenance of the ACkEL (gamma, beta_d, beta_o) values. The values used at run time are hard-coded in gals/baselines.py (ACKEL_PARAMS); this file records where each one came from, because they do not all have the same origin. 'paper_table1' = reused from the original ACkEL paper's Table 1, valid because the dataset uses the same feature representation. 'tuned_locally' = grid-searched here, because the PseAAC feature representation differs from the 'Go' variant the paper used, so its published values do not apply. 'default_not_tuned' = a stated default that was never tuned; see corel5k_exception.",
    "parameter_source": {
        "cal500": "paper_table1",
        "emotions": "paper_table1",
        "flags": "paper_table1",
        "scene": "paper_table1",
        "yeast": "paper_table1",
        "mediamill": "paper_table1",
        "Gnegative": "tuned_locally",
        "Gpositive": "tuned_locally",
        "Plant": "tuned_locally",
        "corel5k": "default_not_tuned"
    },
    "tuning_protocol": "gals.ackel.tune_ackel_params(), 15x5 grid (gamma in 2**-3 .. 2**11, beta in 0.1 .. 0.9), k=3, seed=0, selected by hamming_loss on a held-out split. Run 2026-07-31.",
    "corel5k_exception": "corel5k uses gamma=2**0, beta_d=0.5, beta_o=0.5. These are defaults, not search results: corel5k is absent from the original paper's 30 datasets, and ACkELd is O(M^2 N^2 n), so with L=374 and N=5000 a single (gamma, beta) evaluation runs for hours and the 15x5 grid is not feasible on this hardware. Reported corel5k ACkEL numbers are therefore obtained at an untuned operating point, which should be stated alongside them rather than left implicit.",
}

results = {}
for name, (arff, xml) in DATASETS.items():
    X, Y, _, _ = load_mulan(os.path.join(DATA_ROOT, arff), os.path.join(DATA_ROOT, xml))
    print(f"=== {name}: X={X.shape} Y={Y.shape} ===", flush=True)
    entry = {}
    for mode in ("disjoint", "overlap"):
        t0 = time.perf_counter()
        best = tune_ackel_params(X, Y, mode=mode, k=3, seed=0, metric="hamming_loss")
        elapsed = time.perf_counter() - t0
        print(f"  {mode}: gamma={best['gamma']} beta={best['beta']} "
              f"hamming_loss={best['hamming_loss']:.4f}  ({elapsed:.1f}s)", flush=True)
        entry[mode] = best
    results[name] = dict(gamma=entry["disjoint"]["gamma"],   # both modes searched
                         gamma_overlap=entry["overlap"]["gamma"],
                         beta_d=entry["disjoint"]["beta"],
                         beta_o=entry["overlap"]["beta"])
    with open("results/ackel_tuning.json", "w", encoding="utf-8") as fh:
        json.dump({**PROVENANCE, "tuned": results}, fh, indent=2,
                  ensure_ascii=False)

print("\nDone. results/ackel_tuning.json written.")
print(json.dumps(results, indent=2))
