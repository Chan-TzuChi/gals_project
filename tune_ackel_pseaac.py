"""Grid-search ACkEL (gamma, beta) for the 3 PseAAC datasets not covered by
the original paper's Table 1 (different feature representation: PseAAC vs.
the paper's "Go" variant). Writes results/ackel_tuning.json; copy the
winning (gamma, beta_d, beta_o) into ACKEL_PARAMS in gals/baselines.py.
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
    with open("results/ackel_tuning.json", "w") as fh:
        json.dump(results, fh, indent=2)

print("\nDone. results/ackel_tuning.json written.")
print(json.dumps(results, indent=2))
