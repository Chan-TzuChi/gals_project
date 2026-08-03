"""GALS - Genetic Algorithm for Label Selection.

Implements Algorithm 1 (GA driver + label compensation) and Algorithm 2
(fitness evaluation) as described in the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import LPRandomForest, hamming_loss


# --------------------------------------------------------------------------
@dataclass
class GAConfig:
    # Defaults recovered from the original 2025 notebooks.
    population_size: int = 10
    n_generations: int = 20
    crossover_rate: float = 0.75
    mutation_rate: float = 0.025
    elitism_ratio: float = 0.40
    alpha: float = 0.90                  # fitness weight on (1 - HammingLoss)
    fitness_threshold: float | None = 0.75   # absolute theta; None -> adaptive
    compensation_delta: float = 0.075    # threshold reduction for compensation
    ideal_ratio: float = 0.5             # L_ideal = ideal_ratio * L  (Eq. 10)
    unique_offspring: bool = True        # reject duplicates when breeding
    stall_tolerance: float = 0.01        # 1% change rate
    stall_generations: int = 5
    max_best_set: int | None = None      # cap on |B|; None = unlimited
    rf_params: dict = field(default_factory=lambda: dict(
        n_estimators=100, max_depth=None, min_samples_split=2, n_jobs=1))


# --------------------------------------------------------------------------
def label_subset_ratio(n_selected: int, n_total: int,
                       ideal_ratio: float = 0.5) -> float:
    """Equation (9): symmetric triangular penalty peaking at L_ideal."""
    ideal = max(1e-9, n_total * ideal_ratio)
    return max(0.0, 1.0 - abs(n_selected - ideal) / ideal)


class FitnessEvaluator:
    """Algorithm 2. Caches results, since elitism re-presents chromosomes."""

    def __init__(self, X_tr, Y_tr, X_val, Y_val, cfg: GAConfig, seed: int):
        self.X_tr, self.Y_tr = X_tr, Y_tr
        self.X_val, self.Y_val = X_val, Y_val
        self.cfg, self.seed = cfg, seed
        self.cache: dict[bytes, float] = {}
        self.n_train_calls = 0

    def __call__(self, chrom: np.ndarray) -> float:
        key = chrom.tobytes()
        if key in self.cache:
            return self.cache[key]

        idx = np.flatnonzero(chrom)
        if idx.size == 0:                       # degenerate chromosome
            self.cache[key] = 0.0
            return 0.0

        model = LPRandomForest(random_state=self.seed, **self.cfg.rf_params)
        model.fit(self.X_tr, self.Y_tr[:, idx])
        pred = model.predict(self.X_val)
        self.n_train_calls += 1

        hl = hamming_loss(self.Y_val[:, idx], pred)
        lsr = label_subset_ratio(idx.size, chrom.size,
                                 self.cfg.ideal_ratio)
        fit = self.cfg.alpha * (1.0 - hl) + (1.0 - self.cfg.alpha) * lsr
        self.cache[key] = fit
        return fit


# --------------------------------------------------------------------------
def _init_population(rng, size, L, forced=None):
    pop = rng.integers(0, 2, size=(size, L)).astype(np.int8)
    for c in pop:                                # avoid empty chromosomes
        if c.sum() == 0:
            c[rng.integers(0, L)] = 1
    if forced is not None:
        pop[:, forced] = 1
    return pop


def _uniform_crossover(rng, p1, p2, rate):
    mask = rng.random(p1.size) < rate
    child = p1.copy()
    child[mask] = p2[mask]
    return child


def _bitflip_mutation(rng, c, rate):
    mask = rng.random(c.size) < rate
    c = c.copy()
    c[mask] = 1 - c[mask]
    if c.sum() == 0:
        c[rng.integers(0, c.size)] = 1
    return c


def _run_ga(evaluator, cfg, L, seed, threshold=None, forced=None, log=None):
    """One GA run. Returns (best_set, threshold_used, generations_run).

    threshold=None -> theta is set to the mean fitness of the initial
    population (adaptive, dataset-independent).
    forced -> index of a label whose gene is pinned to 1 (compensation).
    """
    rng = np.random.default_rng(seed)
    pop = _init_population(rng, cfg.population_size, L, forced)

    fits = np.array([evaluator(c) for c in pop])
    if threshold is None:
        threshold = (float(fits.mean()) if cfg.fitness_threshold is None
                     else cfg.fitness_threshold)

    best_set: dict[bytes, np.ndarray] = {}
    prev_max = fits.max()
    stall = 0
    gen = 0

    while True:
        gen += 1
        qualified = pop[fits >= threshold]
        for c in qualified:
            if cfg.max_best_set is None or len(best_set) < cfg.max_best_set:
                best_set[c.tobytes()] = c.copy()

        # ---- termination check (skipped when no chromosome qualified) ----
        if qualified.shape[0] > 0:
            cur_max = fits.max()
            rate = abs(cur_max - prev_max) / prev_max if prev_max > 0 else 0.0
            stall = stall + 1 if rate < cfg.stall_tolerance else 0
            prev_max = cur_max
            if gen >= cfg.n_generations or stall >= cfg.stall_generations:
                break
        else:
            prev_max = max(prev_max, fits.max())
            if gen >= cfg.n_generations:      # hard cap still applies
                break

        # ---- selection (elitism) ----
        n_elite = max(2, int(np.ceil(cfg.elitism_ratio * cfg.population_size)))
        elite_idx = np.argsort(-fits)[:n_elite]
        elite = pop[elite_idx]

        # ---- crossover + mutation ----
        offspring = []
        existing = {c.tobytes() for c in elite}
        while n_elite + len(offspring) < cfg.population_size:
            i, j = rng.integers(0, n_elite, size=2)
            child = _uniform_crossover(rng, elite[i], elite[j],
                                       cfg.crossover_rate)
            child = _bitflip_mutation(rng, child, cfg.mutation_rate)
            if forced is not None:
                child[forced] = 1
            if cfg.unique_offspring:
                for _ in range(50):            # reject duplicates, bounded
                    if child.tobytes() not in existing:
                        break
                    child = _bitflip_mutation(rng, child, cfg.mutation_rate)
                    if forced is not None:
                        child[forced] = 1
                existing.add(child.tobytes())
            offspring.append(child)

        pop = np.vstack([elite] + [o[None, :] for o in offspring])
        fits = np.array([evaluator(c) for c in pop])

        if log is not None:
            log.append(dict(generation=gen, max_fitness=float(fits.max()),
                            mean_fitness=float(fits.mean()),
                            n_best=len(best_set)))

    return list(best_set.values()), threshold, gen


# --------------------------------------------------------------------------
def run_gals(X, Y, split, cfg: GAConfig, seed: int, verbose=False):
    # NOTE: the final ensemble is retrained on the FULL training set
    # (train_sub + val), matching the original implementation.
    """Full GALS pipeline: subset selection + compensation.

    Returns a dict with the selected subsets and diagnostics.
    """
    L = Y.shape[1]
    Xtr, Ytr = X[split["train_sub"]], Y[split["train_sub"]]
    Xva, Yva = X[split["val"]], Y[split["val"]]

    ev = FitnessEvaluator(Xtr, Ytr, Xva, Yva, cfg, seed)
    log = []
    best, theta, gens = _run_ga(ev, cfg, L, seed, log=log)

    if not best:                       # safety net: never return empty
        best = [_init_population(np.random.default_rng(seed), 1, L)[0]]

    # ---- label compensation, one missing label at a time ----
    covered = np.zeros(L, dtype=bool)
    for c in best:
        covered |= c.astype(bool)
    missing = np.flatnonzero(~covered)

    for k, lab in enumerate(missing):
        extra, _, _ = _run_ga(ev, cfg, L, seed + 1000 + k,
                              threshold=theta - cfg.compensation_delta,
                              forced=int(lab))
        added = [c for c in extra if c[lab] == 1]
        if not added:                  # fall back to a minimal valid subset
            c = np.zeros(L, dtype=np.int8)
            c[lab] = 1
            added = [c]
        best.extend(added)
        covered |= np.any(np.array(added).astype(bool), axis=0)

    # dedup while preserving order
    seen, subsets = set(), []
    for c in best:
        k = c.tobytes()
        if k not in seen:
            seen.add(k)
            subsets.append(c)

    still_missing = int(np.sum(~covered))
    if verbose:
        print(f"  GALS: M={len(subsets)} theta={theta:.4f} gens={gens} "
              f"missing_before_comp={missing.size} uncovered_after={still_missing} "
              f"rf_fits={ev.n_train_calls}")

    return dict(subsets=subsets, threshold=theta, generations=gens,
                n_missing_compensated=int(missing.size),
                n_uncovered_after=still_missing,
                n_rf_trainings=ev.n_train_calls,
                mean_subset_size=float(np.mean([c.sum() for c in subsets])),
                log=log)


# --------------------------------------------------------------------------
def fit_predict_ensemble(X, Y, split, subsets, seed, rf_params):
    """Train one LP+RF per subset on the FULL training set, predict on test,
    then combine with the voting rule of Equation (11)."""
    Xtr, Ytr = X[split["train"]], Y[split["train"]]
    Xte = X[split["test"]]
    L = Y.shape[1]

    votes = np.zeros((Xte.shape[0], L), dtype=float)
    counts = np.zeros(L, dtype=float)

    for c in subsets:
        idx = np.flatnonzero(c)
        if idx.size == 0:
            continue
        model = LPRandomForest(random_state=seed, **rf_params)
        model.fit(Xtr, Ytr[:, idx])
        p = model.predict(Xte)
        votes[:, idx] += p
        counts[idx] += 1

    counts[counts == 0] = 1.0
    return (votes / counts >= 0.5).astype(np.int8)
