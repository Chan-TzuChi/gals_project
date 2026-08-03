"""Core utilities: data loading, splitting, LP transform, evaluation metrics."""

from __future__ import annotations

import numpy as np
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_arff(path, n_labels, label_location="end"):
    """Load a MULAN-style multi-label .arff file.

    label_location: 'end' if the label attributes are the last n_labels
    columns, 'start' if they are the first n_labels columns.
    """
    data, _ = arff.loadarff(path)
    cols = data.dtype.names
    mat = np.empty((data.shape[0], len(cols)), dtype=float)
    for j, c in enumerate(cols):
        col = data[c]
        if col.dtype.kind in ("S", "O", "U"):          # nominal -> numeric
            col = np.array([float(v) for v in col.astype(str)])
        mat[:, j] = col.astype(float)

    if label_location == "end":
        X, Y = mat[:, :-n_labels], mat[:, -n_labels:]
    elif label_location == "start":
        Y, X = mat[:, :n_labels], mat[:, n_labels:]
    else:
        raise ValueError("label_location must be 'end' or 'start'")
    return X, (Y > 0.5).astype(np.int8)


def make_synthetic(n=400, d=20, L=8, seed=0):
    """Synthetic multi-label data with planted label co-occurrence groups.

    Labels are generated in correlated blocks so that a subset-selection
    method has real structure to find. Used for pipeline testing only.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    Y = np.zeros((n, L), dtype=np.int8)
    for start in range(0, L, 3):                       # blocks of <=3 labels
        block = list(range(start, min(start + 3, L)))
        w = rng.normal(size=d)
        base = X @ w
        for j in block:                                # shared driver
            noise = rng.normal(scale=0.6, size=n)
            Y[:, j] = ((base + noise) > np.median(base)).astype(np.int8)
    return X, Y


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def split_data(X, Y, seed, test_size=0.30, val_size=0.30):
    """Three-way split used throughout the experiments.

    outer : train (70%) / test (30%)
    inner : train -> train_sub (70%) / val (30%)   [= 49% / 21% of all]

    The GA sees only (train_sub, val). The final ensemble is retrained on the
    full train set and evaluated once on test. This prevents any leakage
    between subset selection and final evaluation.
    """
    idx = np.arange(X.shape[0])
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed,
                              shuffle=True)
    tr_sub, va = train_test_split(tr, test_size=val_size, random_state=seed,
                                  shuffle=True)
    return dict(train=tr, test=te, train_sub=tr_sub, val=va)


# --------------------------------------------------------------------------
# Label Powerset + Random Forest
# --------------------------------------------------------------------------
class LPRandomForest:
    """Label Powerset wrapper around a Random Forest classifier.

    Each distinct combination of labels observed in the training data becomes
    a single class of a multi-class problem.
    """

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 random_state=0, n_jobs=1):
        self.params = dict(n_estimators=n_estimators, max_depth=max_depth,
                           min_samples_split=min_samples_split,
                           random_state=random_state, n_jobs=n_jobs)
        self.rf = None
        self.classes_ = None          # array (n_classes, n_labels_in_subset)
        self.n_out = None

    def fit(self, X, Y):
        self.n_out = Y.shape[1]
        combos, inverse = np.unique(Y, axis=0, return_inverse=True)
        self.classes_ = combos
        if combos.shape[0] == 1:                       # degenerate: one class
            self.rf = None
            self._const = combos[0]
            return self
        self.rf = RandomForestClassifier(**self.params)
        self.rf.fit(X, inverse)
        return self

    def predict(self, X):
        if self.rf is None:
            return np.tile(self._const, (X.shape[0], 1)).astype(np.int8)
        cls = self.rf.predict(X)
        return self.classes_[cls].astype(np.int8)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def hamming_loss(Y, P):
    return float(np.mean(Y != P))


def subset_accuracy(Y, P):
    return float(np.mean(np.all(Y == P, axis=1)))


def _prf_per_label(Y, P):
    tp = np.sum((Y == 1) & (P == 1), axis=0).astype(float)
    fp = np.sum((Y == 0) & (P == 1), axis=0).astype(float)
    fn = np.sum((Y == 1) & (P == 0), axis=0).astype(float)
    return tp, fp, fn


def macro_f1(Y, P):
    tp, fp, fn = _prf_per_label(Y, P)
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(np.mean(f1))


def micro_f1(Y, P):
    tp, fp, fn = _prf_per_label(Y, P)
    denom = 2 * tp.sum() + fp.sum() + fn.sum()
    return 0.0 if denom == 0 else float(2 * tp.sum() / denom)


def weighted_f1(Y, P):
    """F1 per label, averaged with weights proportional to label support."""
    tp, fp, fn = _prf_per_label(Y, P)
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    support = np.sum(Y == 1, axis=0).astype(float)
    if support.sum() == 0:
        return 0.0
    return float(np.sum(f1 * support) / support.sum())


def evaluate_all(Y, P):
    return dict(hamming_loss=hamming_loss(Y, P),
                subset_accuracy=subset_accuracy(Y, P),
                weighted_f1=weighted_f1(Y, P),
                macro_f1=macro_f1(Y, P),
                micro_f1=micro_f1(Y, P))


# --------------------------------------------------------------------------
# Ensemble voting (shared by RAkEL*, ACkEL*, and GALS)
# --------------------------------------------------------------------------
def accumulate_votes(n_samples, n_labels, parts, preds):
    """Sum per-model binary predictions into per-label vote/coverage counts."""
    votes = np.zeros((n_samples, n_labels), dtype=float)
    counts = np.zeros(n_labels, dtype=float)
    for idx, p in zip(parts, preds):
        votes[:, idx] += p
        counts[idx] += 1
    return votes, counts


def vote_ensemble(n_samples, n_labels, parts, preds, threshold=0.5):
    votes, counts = accumulate_votes(n_samples, n_labels, parts, preds)
    counts_safe = np.where(counts == 0, 1.0, counts)
    return (votes / counts_safe >= threshold).astype(np.int8)


def dataset_stats(Y):
    n, L = Y.shape
    card = float(Y.sum() / n)
    return dict(instances=n, labels=L,
                label_sets=int(np.unique(Y, axis=0).shape[0]),
                cardinality=card, density=card / L)
