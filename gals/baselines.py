"""Baseline multi-label methods, implemented directly on scikit-learn.

scikit-multilearn is deliberately avoided: it is unmaintained and breaks on
current numpy/scikit-learn versions. Each method here is short enough to be
verified by inspection.

All methods share the same base learner (Random Forest) and the same voting
rule, so that observed differences are attributable to the label-partitioning
strategy alone.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .ackel import ackel_disjoint as _ackel_disjoint_impl
from .ackel import ackel_overlap as _ackel_overlap_impl
from .core import LPRandomForest, vote_ensemble


def _rf(seed, rf_params):
    return RandomForestClassifier(random_state=seed, **rf_params)


# --------------------------------------------------------------------------
def binary_relevance(Xtr, Ytr, Xte, seed, rf_params):
    P = np.zeros((Xte.shape[0], Ytr.shape[1]), dtype=np.int8)
    for j in range(Ytr.shape[1]):
        y = Ytr[:, j]
        if len(np.unique(y)) == 1:
            P[:, j] = y[0]
            continue
        m = _rf(seed, rf_params).fit(Xtr, y)
        P[:, j] = m.predict(Xte)
    return P


def classifier_chains(Xtr, Ytr, Xte, seed, rf_params, order=None):
    L = Ytr.shape[1]
    order = np.arange(L) if order is None else np.asarray(order)
    P = np.zeros((Xte.shape[0], L), dtype=np.int8)
    Atr = Xtr.copy()
    Ate = Xte.copy()
    for j in order:
        y = Ytr[:, j]
        if len(np.unique(y)) == 1:
            pred = np.full(Xte.shape[0], y[0], dtype=np.int8)
        else:
            m = _rf(seed, rf_params).fit(Atr, y)
            pred = m.predict(Ate).astype(np.int8)
        P[:, j] = pred
        # training uses true labels, prediction uses predicted labels
        Atr = np.hstack([Atr, y.reshape(-1, 1)])
        Ate = np.hstack([Ate, pred.reshape(-1, 1)])
    return P


def label_powerset(Xtr, Ytr, Xte, seed, rf_params):
    return LPRandomForest(random_state=seed, **rf_params).fit(Xtr, Ytr).predict(Xte)


# --------------------------------------------------------------------------
def rakel_disjoint(Xtr, Ytr, Xte, seed, rf_params, k=3):
    L = Ytr.shape[1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(L)
    parts = [perm[i:i + k] for i in range(0, L, k)]
    preds = [LPRandomForest(random_state=seed, **rf_params)
             .fit(Xtr, Ytr[:, idx]).predict(Xte) for idx in parts]
    return vote_ensemble(Xte.shape[0], L, parts, preds)


def rakel_overlap(Xtr, Ytr, Xte, seed, rf_params, k=3, m=10):
    L = Ytr.shape[1]
    rng = np.random.default_rng(seed)
    k = min(k, L)
    parts, seen = [], set()
    attempts = 0
    while len(parts) < m and attempts < 20 * m:
        attempts += 1
        idx = np.sort(rng.choice(L, size=k, replace=False))
        key = idx.tobytes()
        if key in seen:
            continue
        seen.add(key)
        parts.append(idx)
    preds = [LPRandomForest(random_state=seed, **rf_params)
             .fit(Xtr, Ytr[:, idx]).predict(Xte) for idx in parts]
    return vote_ensemble(Xte.shape[0], L, parts, preds)


# --------------------------------------------------------------------------
# ACkEL (Wang et al., Pattern Recognition 109 (2021) 107583).
# Algorithms live in gals/ackel.py, ported from the official MATLAB/LIBSVM
# code (https://github.com/xuwangfmc/AkEL) -- see that module's docstring
# for the three places where the official code diverges from the paper
# text. The base classifier (RF vs. SVM) is a deliberate, still-open
# choice -- see the base_classifier parameter below.
#
# gamma is the paper's Table 1 "sigma" column, which is in fact libsvm's -g
# (gamma), not a Gaussian sigma -- confirmed by reading rbf_kernel.m /
# run.m in the official repo. beta is given as a (disjoint, overlap) pair;
# ACKEL_PARAMS below stores them separately as beta_d / beta_o.
#
# Only the 6 datasets that use the SAME feature representation as the
# original paper can reuse its tuned (gamma, beta) directly. The PseAAC
# variants of Gnegative/Gpositive/Plant (paper used the different "Go"
# feature set) were tuned locally with gals.ackel.tune_ackel_params();
# corel5k (absent from the paper's 30 datasets) uses a stated default, for
# the reason given at its entry below. results/ackel_tuning.json records
# the provenance of every dataset's values.
# --------------------------------------------------------------------------
ACKEL_PARAMS = {
    "cal500":    dict(gamma=2.0**0, beta_d=0.9, beta_o=0.3),
    "emotions":  dict(gamma=2.0**0, beta_d=0.3, beta_o=0.3),
    "flags":     dict(gamma=2.0**0, beta_d=0.3, beta_o=0.5),
    "scene":     dict(gamma=2.0**1, beta_d=0.3, beta_o=0.3),
    "yeast":     dict(gamma=2.0**0, beta_d=0.5, beta_o=0.1),
    "mediamill": dict(gamma=2.0**0, beta_d=0.5, beta_o=0.5),
    # Tuned locally via gals.ackel.tune_ackel_params() (2026-07-31), full
    # 15x5 grid, hamming_loss on a held-out split -- see
    # results/ackel_tuning.json. PseAAC != paper's "Go" features, so the
    # paper's own Table 1 values don't apply here.
    "Gnegative": dict(gamma=0.125, beta_d=0.1, beta_o=0.1),
    "Gpositive": dict(gamma=0.125, beta_d=0.1, beta_o=0.1),
    "Plant":     dict(gamma=0.125, beta_d=0.9, beta_o=0.1),
    # corel5k: 374 labels / 5000 samples, ACkELd is O(M^2 N^2 n) -- a single
    # (gamma, beta) run can take hours, so a 15x5 grid search is not
    # feasible on this hardware (the original paper used a 384GB machine
    # and didn't include corel5k anyway). Deliberate default, not a tuned
    # value: gamma=2**0, beta=0.5 for both modes. State this limitation in
    # the write-up rather than leaving the cell blank or forcing a full sweep.
    "corel5k":   dict(gamma=2.0**0, beta_d=0.5, beta_o=0.5),
}


def ackel_disjoint(Xtr, Ytr, Xte, seed, rf_params, k=3, dataset=None, **kw):
    params = ACKEL_PARAMS.get(dataset) if dataset else None
    if params is None:
        raise NotImplementedError(
            f"ACkELD: no tuned (gamma, beta) for dataset={dataset!r} -- run "
            "gals.ackel.tune_ackel_params() and add it to ACKEL_PARAMS")
    return _ackel_disjoint_impl(Xtr, Ytr, Xte, seed, rf_params, k=k,
                                gamma=params["gamma"], beta=params["beta_d"], **kw)


def ackel_overlap(Xtr, Ytr, Xte, seed, rf_params, k=3, m=10, dataset=None, **kw):
    params = ACKEL_PARAMS.get(dataset) if dataset else None
    if params is None:
        raise NotImplementedError(
            f"ACkELO: no tuned (gamma, beta) for dataset={dataset!r} -- run "
            "gals.ackel.tune_ackel_params() and add it to ACKEL_PARAMS")
    return _ackel_overlap_impl(Xtr, Ytr, Xte, seed, rf_params, k=k, m=m,
                               gamma=params["gamma"], beta=params["beta_o"], **kw)


BASELINES = {
    "BR": binary_relevance,
    "CC": classifier_chains,
    "LP": label_powerset,
    "RAkELD": rakel_disjoint,
    "RAkELO": rakel_overlap,
    "ACkELD": ackel_disjoint,
    "ACkELO": ackel_overlap,
}
