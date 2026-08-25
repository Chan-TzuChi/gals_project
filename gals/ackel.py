"""ACkEL (Active k-Labelsets Ensemble), ported from the official MATLAB/LIBSVM
code released with Wang et al., "Active k-labelsets ensemble for multi-label
classification", Pattern Recognition 109 (2021) 107583.
https://github.com/xuwangfmc/AkEL

Ported directly from: fisherRatio.m, jointEntropy.m, rbf_kernel.m,
ml_KLabelset_active_disjoint_Train.m, ml_KLabelset_active_overlap_Train.m,
ml_KLabelset_Test.m, base_svm_OAA_train.m, base_svm_OAA_test.m, run.m.

The official code has three behaviours that do NOT match the paper's own
description (Eq. (1), Algorithm 2, Algorithm 3). They are reproduced as-is
by default so that ACkEL numbers are faithful to what the original authors
actually ran; each is a toggleable parameter so the alternative reading can
be tried too. Each needs an explicit decision (and a line in the methods
section), the same way the S~(b) coefficient (2, not 1) did.

1. Beta is dropped after the first pick in overlap mode. The official
   `ml_KLabelset_active_overlap_Train.m` uses `beta*fRatio + (1-beta)*jEntropy`
   only for the very first (random) labelset; every subsequent candidate is
   scored as the *unweighted* `fRatio + jEntropy`. Controlled by
   `overlap_apply_beta` (default False = reproduce the official code).

2. Overlap-mode acceptance requires full label-pattern coverage. A checked
   candidate is only accepted if its quality beats the running best AND all
   2**k label patterns are observed in the training data for that candidate
   (`jointProbsNonZeroNum == 2**k` in the .m file). On sparse real datasets
   this condition is rarely true, so in practice most iterations fall through
   to "pick the best of the <=5 checked candidates" rather than "accept the
   first improving one". This is not a bug to fix -- it's what the official
   code does -- but it changes the expected behaviour vs. a literal reading of
   Algorithm 2, so it is worth a footnote.

3. Disjoint mode always tests at a fixed threshold of 0.5 (see run.m); only
   overlap mode runs the Algorithm 3 threshold search over {0.3, 0.5, 0.7}
   on the training set. Both are exposed as parameters below.

4. Disjoint mode builds floor(M/k) labelsets, not ceil(M/k); the leftover
   labels are folded entirely into the last labelset (which can then hold
   more than k labels). Reproduced as-is.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from .core import LPRandomForest, accumulate_votes, vote_ensemble

DEFAULT_GAMMA_GRID = tuple(2.0 ** e for e in range(-3, 12))   # paper 4.2
DEFAULT_BETA_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)                  # paper 4.2


# --------------------------------------------------------------------------
# Kernel / quality primitives (fisherRatio.m, jointEntropy.m, rbf_kernel.m)
# --------------------------------------------------------------------------
def _kernel_sum(A, B, gamma, block=1024):
    """sum(exp(-gamma * ||a-b||^2)) over all pairs (a in A, b in B), computed
    in row blocks so an N-sample dataset never needs an N x N matrix in
    memory (needed to keep mediamill / corel5k tractable)."""
    a_sq = np.sum(A * A, axis=1)
    b_sq = np.sum(B * B, axis=1)
    total = 0.0
    for i in range(0, A.shape[0], block):
        a_blk, a_blk_sq = A[i:i + block], a_sq[i:i + block]
        d2 = a_blk_sq[:, None] + b_sq[None, :] - 2.0 * (a_blk @ B.T)
        np.clip(d2, 0.0, None, out=d2)
        total += float(np.sum(np.exp(-gamma * d2)))
    return total


def _label_groups(Y_subset):
    """Partition sample indices by their exact pattern over Y_subset's
    columns (dec2bin grouping in the .m files)."""
    k = Y_subset.shape[1]
    weights = 1 << np.arange(k - 1, -1, -1)
    codes = Y_subset.astype(np.int64) @ weights
    return {int(c): np.flatnonzero(codes == c) for c in np.unique(codes)}


def joint_entropy(Y_subset):
    """jointEntropy.m: entropy of the label-pattern distribution, normalized
    by log2(#observed patterns). Returns (entropy, n_nonempty_patterns)."""
    groups = _label_groups(Y_subset)
    n = Y_subset.shape[0]
    probs = np.array([idx.size / n for idx in groups.values()])
    n_nonzero = probs.size
    if n_nonzero <= 1:
        return 0.0, n_nonzero
    ent = -float(np.sum(probs * np.log2(probs))) / np.log2(n_nonzero)
    return ent, n_nonzero


def fisher_ratio(X, Y_subset, gamma, block=1024):
    """fisherRatio.m: between-class / within-class RBF-kernel scatter ratio
    over the classes induced by Y_subset's label patterns.

    S~(w)_I = 1 - (1/N_I^2) sum_sum K_II
    S~(b)_IJ = (1/N_I^2)ssK_II + (1/N_J^2)ssK_JJ - (2/(N_I N_J))ssK_IJ
    S~(C_L) = sum_{I<J} S~(b)_IJ / sum_I S~(w)_I
    """
    groups = _label_groups(Y_subset)
    items = list(groups.items())
    diag_sum, within = {}, 0.0
    for code, idx in items:
        n_i = idx.size
        s = _kernel_sum(X[idx], X[idx], gamma, block)
        diag_sum[code] = s
        within += 1.0 - s / (n_i * n_i)
    between = 0.0
    for a in range(len(items)):
        code_i, idx_i = items[a]
        n_i = idx_i.size
        for b in range(a + 1, len(items)):
            code_j, idx_j = items[b]
            n_j = idx_j.size
            s_ij = _kernel_sum(X[idx_i], X[idx_j], gamma, block)
            between += (diag_sum[code_i] / (n_i * n_i)
                        + diag_sum[code_j] / (n_j * n_j)
                        - 2.0 * s_ij / (n_i * n_j))
    return 0.0 if within <= 0 else between / within


def quality(X, Y_subset, gamma, beta, block=1024):
    """Eq. (1): Qua(L) = beta * S~(C_L) + (1-beta) * H_norm(L)."""
    fr = fisher_ratio(X, Y_subset, gamma, block)
    je, _ = joint_entropy(Y_subset)
    return beta * fr + (1.0 - beta) * je


# --------------------------------------------------------------------------
# Base classifier: Label Powerset via one-against-all RBF-SVM
# (base_svm_OAA_train.m / base_svm_OAA_test.m)
# --------------------------------------------------------------------------
class LPSVMOAA:
    def __init__(self, gamma=1.0, C=1.0, random_state=0):
        self.gamma, self.C, self.random_state = gamma, C, random_state
        self.clf, self.classes_, self._const = None, None, None

    def fit(self, X, Y):
        combos, inverse = np.unique(Y, axis=0, return_inverse=True)
        self.classes_ = combos
        if combos.shape[0] == 1:
            self._const = combos[0]
            return self
        base = SVC(kernel="rbf", C=self.C, gamma=self.gamma,
                   random_state=self.random_state)
        self.clf = OneVsRestClassifier(base)
        self.clf.fit(X, inverse)
        return self

    def predict(self, X):
        if self.clf is None:
            return np.tile(self._const, (X.shape[0], 1)).astype(np.int8)
        return self.classes_[self.clf.predict(X)].astype(np.int8)


def _make_lp_model(base_classifier, seed, rf_params, gamma, C):
    if base_classifier == "svm":
        return LPSVMOAA(gamma=gamma, C=C, random_state=seed)
    if base_classifier == "rf":
        return LPRandomForest(random_state=seed, **rf_params)
    raise ValueError(f"base_classifier must be 'rf' or 'svm', got {base_classifier!r}")


# --------------------------------------------------------------------------
# Algorithm 3: pick the voting threshold that maximizes example-based
# (per-instance) F-score on the training set.
# --------------------------------------------------------------------------
def _best_threshold_train(Y_true, coverage, votes, candidates=(0.3, 0.5, 0.7)):
    eps = 1e-12
    best_t, best_f = candidates[0], -1.0
    for t in candidates:
        pred = (votes / coverage >= t).astype(np.int8)
        tp = np.sum((pred == 1) & (Y_true == 1), axis=1).astype(float)
        pred_pos = np.sum(pred == 1, axis=1).astype(float)
        true_pos = np.sum(Y_true == 1, axis=1).astype(float)
        prec = np.mean(tp / (pred_pos + eps))
        rec = np.mean(tp / (true_pos + eps))
        f = 2 * prec * rec / (prec + rec + eps)
        if f > best_f:
            best_f, best_t = f, t
    return best_t


# --------------------------------------------------------------------------
# Algorithm 1: ACkELd (disjoint / pool-based)
# --------------------------------------------------------------------------
def ackel_disjoint(Xtr, Ytr, Xte, seed, rf_params, k=3, gamma=1.0, beta=0.5,
                   base_classifier="rf", C=1.0, threshold=0.5, block=1024,
                   **kw):
    M = Ytr.shape[1]
    n_classifiers = max(1, M // k)              # floor(M/k), not ceil
    rng = np.random.default_rng(seed)
    remaining = list(range(M))
    parts = []

    for i in range(n_classifiers):
        if i != n_classifiers - 1:
            first = remaining.pop(int(rng.integers(0, len(remaining))))
            subset = [first]
            for _ in range(k - 1):
                best_q, best_lab = -np.inf, remaining[0]
                for lab in remaining:
                    q = quality(Xtr, Ytr[:, subset + [lab]], gamma, beta, block)
                    if q > best_q:
                        best_q, best_lab = q, lab
                subset.append(best_lab)
                remaining.remove(best_lab)
        else:
            subset = remaining                   # last classifier absorbs the rest
        parts.append(np.array(subset, dtype=int))

    preds = []
    for idx in parts:
        model = _make_lp_model(base_classifier, seed, rf_params, gamma, C)
        model.fit(Xtr, Ytr[:, idx])
        preds.append(model.predict(Xte))
    return vote_ensemble(Xte.shape[0], M, parts, preds, threshold=threshold)


# --------------------------------------------------------------------------
# Algorithm 2: ACkELo (overlapping / stream-based, BALCO selection)
# --------------------------------------------------------------------------
def ackel_overlap(Xtr, Ytr, Xte, seed, rf_params, k=3, m=10, gamma=1.0,
                  beta=0.5, base_classifier="rf", C=1.0, ncheck=5,
                  overlap_apply_beta=False,
                  threshold_candidates=(0.3, 0.5, 0.7), max_pool=None,
                  block=1024, **kw):
    M = Ytr.shape[1]
    k = min(k, M)
    rng = np.random.default_rng(seed)

    all_combos = np.array(list(combinations(range(M), k)), dtype=np.int64)
    if max_pool is not None and all_combos.shape[0] > max_pool:
        sel = rng.choice(all_combos.shape[0], size=max_pool, replace=False)
        all_combos = all_combos[sel]
    n_pool = all_combos.shape[0]
    m = min(m, n_pool)

    label_freq = np.zeros(M, dtype=np.int64)
    mark = np.ones(n_pool, dtype=bool)           # True = still selectable
    cache = {}                                    # i -> (fisher_ratio, entropy, n_nonzero)
    parts = []

    def qual_of(i):
        if i not in cache:
            combo = all_combos[i]
            fr = fisher_ratio(Xtr, Ytr[:, combo], gamma, block)
            je, n_nz = joint_entropy(Ytr[:, combo])
            cache[i] = (fr, je, n_nz)
        return cache[i]

    def score(fr, je):
        return (beta * fr + (1 - beta) * je) if overlap_apply_beta else (fr + je)

    first = int(rng.integers(0, n_pool))
    fr, je, _ = qual_of(first)
    base_quality = beta * fr + (1 - beta) * je   # first pick always uses beta
    label_freq[all_combos[first]] += 1
    mark[first] = False
    parts.append(all_combos[first])

    for _ in range(1, m):
        avail = np.flatnonzero(mark)
        freq_new = label_freq[all_combos[avail]] + 1        # (n_avail, k)
        freq_diff = freq_new.max(axis=1) - freq_new.min(axis=1)
        best_local = avail[freq_diff == freq_diff.min()]
        order = rng.permutation(best_local.size)
        checked = best_local[order[:min(ncheck, best_local.size)]]

        chosen = None
        for i in checked:
            fr, je, n_nz = qual_of(i)
            q = score(fr, je)
            if q > base_quality and n_nz == 2 ** k:
                chosen, base_quality = i, max(base_quality, q)
                break
        if chosen is None:
            scored = [score(*qual_of(i)[:2]) for i in checked]
            best_j = int(np.argmax(scored))
            chosen = checked[best_j]
            base_quality = max(base_quality, scored[best_j])

        label_freq[all_combos[chosen]] += 1
        mark[chosen] = False
        parts.append(all_combos[chosen])

    models = []
    for idx in parts:
        model = _make_lp_model(base_classifier, seed, rf_params, gamma, C)
        model.fit(Xtr, Ytr[:, idx])
        models.append(model)

    tr_votes, tr_cov = accumulate_votes(
        Xtr.shape[0], M, parts, [mdl.predict(Xtr) for mdl in models])
    tr_cov_safe = np.where(tr_cov == 0, 1.0, tr_cov)
    thr = _best_threshold_train(Ytr, tr_cov_safe, tr_votes, threshold_candidates)

    te_preds = [mdl.predict(Xte) for mdl in models]
    return vote_ensemble(Xte.shape[0], M, parts, te_preds, threshold=thr)


# --------------------------------------------------------------------------
# Hyperparameter tuning for datasets not covered by the paper's Table 1
# (our PseAAC variants of Gnegative/Gpositive/Plant, and corel5k).
# --------------------------------------------------------------------------
def tune_ackel_params(X, Y, mode, k=3, gammas=DEFAULT_GAMMA_GRID,
                      betas=DEFAULT_BETA_GRID, seed=0, val_size=0.3,
                      metric="hamming_loss", rf_params=None,
                      base_classifier="rf", m=10, **fixed_kw):
    """Grid search over (gamma, beta) on a single held-out split, scoring
    with `metric` from core.evaluate_all. Expensive (len(gammas)*len(betas)
    full ACkEL runs) -- run once per dataset and hard-code the winner into
    ACKEL_PARAMS in gals/baselines.py, the same way the paper's Table 1
    values are hard-coded for the 6 shared datasets.
    """
    from .core import evaluate_all, split_data

    rf_params = rf_params or dict(n_estimators=100, max_depth=None,
                                  min_samples_split=2, n_jobs=1)
    split = split_data(X, Y, seed, test_size=val_size, val_size=val_size)
    Xtr, Ytr = X[split["train_sub"]], Y[split["train_sub"]]
    Xva, Yva = X[split["val"]], Y[split["val"]]
    lower_is_better = metric in ("hamming_loss",)

    fn = ackel_disjoint if mode == "disjoint" else ackel_overlap
    best = None
    for g in gammas:
        for b in betas:
            kwargs = dict(gamma=g, beta=b, base_classifier=base_classifier, **fixed_kw)
            if mode == "overlap":
                kwargs["m"] = m
            try:
                P = fn(Xtr, Ytr, Xva, seed, rf_params, k=k, **kwargs)
            except Exception:
                continue
            score = evaluate_all(Yva, P)[metric]
            better = best is None or (score < best[0] if lower_is_better else score > best[0])
            if better:
                best = (score, g, b)
    if best is None:
        raise RuntimeError("tune_ackel_params: every (gamma, beta) combination failed")
    return dict(gamma=best[1], beta=best[2], **{metric: best[0]})
