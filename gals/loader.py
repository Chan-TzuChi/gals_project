"""MULAN dataset loader.

The original implementation used liac-arff plus the dataset's .xml label file.
This module reproduces that behaviour with a self-contained parser, so the
project has no dependency that is unavailable or unmaintained. It handles both
the dense and the sparse ARFF formats used by MULAN, and identifies label
columns by name from the .xml file rather than by position, which is what the
original code did.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import numpy as np


# --------------------------------------------------------------------------
def read_label_names(xml_path):
    """Extract label names from a MULAN .xml label file."""
    root = ET.parse(xml_path).getroot()
    ns = {"ns": "http://mulan.sourceforge.net/labels"}
    names = [lab.get("name") for lab in root.findall("ns:label", ns)]
    if not names:                                  # file without namespace
        names = [lab.get("name") for lab in root.findall("label")]
    if not names:
        raise ValueError(f"no <label> entries found in {xml_path}")
    return names


def _parse_attribute(line):
    """Return the attribute name from an @attribute declaration."""
    body = line[len("@attribute"):].strip()
    if body.startswith("'"):
        return body[1:body.index("'", 1)]
    if body.startswith('"'):
        return body[1:body.index('"', 1)]
    return body.split()[0]


def read_arff(path):
    """Minimal ARFF reader returning (attribute_names, data_matrix).

    Supports the dense format and the sparse {index value, ...} format.
    Non-numeric tokens are label-encoded per column. Missing values ('?')
    become NaN.
    """
    names, rows, sparse_rows, in_data = [], [], [], False
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            low = s.lower()
            if not in_data:
                if low.startswith("@attribute"):
                    names.append(_parse_attribute(s))
                elif low.startswith("@data"):
                    in_data = True
                continue
            if s.startswith("{"):                              # sparse row
                entry = {}
                inner = s.strip("{}").strip()
                if inner:
                    for pair in re.split(r",\s*", inner):
                        k, _, v = pair.strip().partition(" ")
                        entry[int(k)] = v.strip().strip("'\"")
                sparse_rows.append(entry)
            else:                                              # dense row
                rows.append([t.strip().strip("'\"")
                             for t in re.split(r",(?=(?:[^']*'[^']*')*[^']*$)", s)])

    n_attr = len(names)
    if sparse_rows:
        rows = [["0"] * n_attr for _ in sparse_rows]
        for r, entry in zip(rows, sparse_rows):
            for k, v in entry.items():
                r[k] = v
    if not rows:
        raise ValueError(f"no data rows parsed from {path}")

    raw = np.array(rows, dtype=object)
    if raw.shape[1] != n_attr:
        raise ValueError(f"{path}: {raw.shape[1]} columns but {n_attr} "
                         f"@attribute declarations")

    mat = np.empty(raw.shape, dtype=float)
    for j in range(n_attr):
        col = raw[:, j]
        try:
            mat[:, j] = np.where(col == "?", np.nan, col).astype(float)
        except ValueError:                        # nominal -> label encode
            uniq = {v: i for i, v in enumerate(sorted(set(col)))}
            mat[:, j] = [uniq[v] for v in col]
    return names, mat


def load_mulan(arff_path, xml_path):
    """Load a MULAN dataset. Returns (X, Y, feature_names, label_names)."""
    names, mat = read_arff(arff_path)
    labels = read_label_names(xml_path)

    missing = [l for l in labels if l not in names]
    if missing:
        raise ValueError(f"labels declared in {xml_path} but absent from the "
                         f"ARFF header: {missing[:5]}")

    lab_idx = [names.index(l) for l in labels]
    feat_idx = [i for i in range(len(names)) if i not in set(lab_idx)]

    X = mat[:, feat_idx]
    Y = (mat[:, lab_idx] > 0.5).astype(np.int8)

    if np.isnan(X).any():                          # mean imputation
        col_mean = np.nanmean(X, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_mean, idx[1])

    return X, Y, [names[i] for i in feat_idx], labels


def verify_against_table1(Y, expected):
    """Compare loaded statistics with the values published in Table 1.

    expected: dict with any of instances, labels, label_sets, cardinality,
    density. Returns (ok, report_lines).
    """
    from .core import dataset_stats
    got = dataset_stats(Y)
    lines, ok = [], True
    for k, exp in expected.items():
        val = got.get(k)
        if val is None:
            continue
        match = (abs(val - exp) < 1e-3 if isinstance(exp, float)
                 else val == exp)
        ok &= match
        lines.append(f"  {k:12} loaded={val!r:<12} paper={exp!r:<12} "
                     f"{'OK' if match else '<-- MISMATCH'}")
    return ok, lines


# Published Table 1 values, for load verification.
TABLE1 = {
    "cal500":    dict(instances=502, labels=174, label_sets=502,
                      cardinality=26.0438, density=0.1497),
    "emotions":  dict(instances=593, labels=6, label_sets=27,
                      cardinality=1.8685, density=0.3114),
    "flags":     dict(instances=194, labels=7, label_sets=54,
                      cardinality=3.3918, density=0.4845),
    "Gnegative": dict(instances=1392, labels=8, label_sets=19,
                      cardinality=1.046, density=0.1307),
    "Gpositive": dict(instances=519, labels=4, label_sets=7,
                      cardinality=1.0077, density=0.2519),
    "Plant":     dict(instances=978, labels=12, label_sets=32,
                      cardinality=1.0787, density=0.0899),
    "scene":     dict(instances=2407, labels=6, label_sets=15,
                      cardinality=1.074, density=0.179),
    "yeast":     dict(instances=2417, labels=14, label_sets=198,
                      cardinality=4.2371, density=0.3026),
    "corel5k":   dict(instances=5000, labels=374, label_sets=3175,
                      cardinality=3.522, density=0.0094),
    "mediamill": dict(instances=43907, labels=101, label_sets=6555,
                      cardinality=4.3756, density=0.0433),
}
