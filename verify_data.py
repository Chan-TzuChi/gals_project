"""Load every dataset in ../DATA and check it against Table 1.

Run this once, with all lines showing OK, before starting any experiment.
"""
from __future__ import annotations

import os

from gals.loader import TABLE1, load_mulan, verify_against_table1

DATA_ROOT = os.environ.get("GALS_DATA_ROOT",
                           os.path.join("..", "DATA"))

# dataset key (must match TABLE1) -> (arff, xml) relative to DATA_ROOT
# DATA is a flat folder (no A-F subfolders); HumanPseAAC / foodtruck removed
# since they aren't part of this paper's 10 datasets.
PATHS = {
    "cal500":    ("CAL500.arff", "CAL500.xml"),
    "corel5k":   ("corel5k.arff", "corel5k.xml"),
    "Gnegative": ("GnegativePseAAC.arff", "GnegativePseAAC.xml"),
    "Gpositive": ("GpositivePseAAC.arff", "GpositivePseAAC.xml"),
    "emotions":  ("emotions.arff", "emotions.xml"),
    "flags":     ("flags.arff", "flags.xml"),
    "Plant":     ("PlantPseAAC.arff", "PlantPseAAC.xml"),
    "scene":     ("scene.arff", "scene.xml"),
    "yeast":     ("yeast.arff", "yeast.xml"),
    "mediamill": ("mediamill.arff", "mediamill.xml"),
}


def main():
    missing_keys = set(TABLE1) - set(PATHS)
    if missing_keys:
        raise SystemExit(f"PATHS is missing keys present in TABLE1: {missing_keys}")

    all_ok = True
    for name, (arff_rel, xml_rel) in PATHS.items():
        arff_path = os.path.join(DATA_ROOT, arff_rel)
        xml_path = os.path.join(DATA_ROOT, xml_rel)
        print(f"[{name}] {arff_path}")
        X, Y, feat_names, label_names = load_mulan(arff_path, xml_path)
        ok, lines = verify_against_table1(Y, TABLE1[name])
        for line in lines:
            print(line)
        print(f"  X.shape={X.shape}  -> {'OK' if ok else 'MISMATCH'}\n")
        all_ok &= ok

    print("ALL DATASETS OK" if all_ok else "SOME DATASETS MISMATCHED — do not start experiments")


if __name__ == "__main__":
    main()
