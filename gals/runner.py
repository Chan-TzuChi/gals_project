"""Experiment runner: repeats every method over multiple random seeds and
records one tidy CSV row per (dataset, method, seed)."""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback

import numpy as np
import pandas as pd

from .baselines import BASELINES
from .core import evaluate_all, split_data
from .gals import GAConfig, fit_predict_ensemble, run_gals

METRICS = ["hamming_loss", "subset_accuracy", "weighted_f1",
           "macro_f1", "micro_f1"]


# --------------------------------------------------------------------------
# Hard wall-clock timeout for methods with no internal early-stop (ACkEL's
# kernel computation can't be interrupted any other way once it's running).
# Runs the call in a subprocess so it can be force-killed on timeout instead
# of hanging the whole sweep. On Windows this needs `spawn`, which re-imports
# the calling script -- guard your entrypoint with
# `if __name__ == "__main__":` when passing method_timeouts.
# --------------------------------------------------------------------------
def _timeout_worker(fn, args, kwargs, conn):
    """Classify the outcome INSIDE the worker, where the real exception
    object is available, rather than collapsing everything to a string and
    trying to reclassify it after crossing the process boundary (the
    previous version's `raise RuntimeError(payload)` re-raise dance lost
    NotImplementedError/MemoryError's identity and mis-reported both as a
    generic error)."""
    try:
        result = fn(*args, **kwargs)
        conn.send(("ok", result))
    except NotImplementedError as e:
        conn.send(("not_implemented", str(e)))
    except MemoryError as e:
        conn.send(("out_of_memory", str(e)))
    except Exception as e:                       # noqa: BLE001 -- report, don't crash
        conn.send(("error", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def _run_many_with_timeout(jobs):
    """Run several (name, fn, args, kwargs, timeout_sec) jobs CONCURRENTLY,
    each in its own subprocess with its own deadline, yielding (name,
    status, payload) as soon as each one resolves.

    This is what makes independent timed methods (e.g. LP/ACkELD/ACkELO)
    cost max(their timeouts), not sum(their timeouts) -- they don't depend
    on each other, so there's no reason to make one wait for another.

    Yields incrementally (not a single dict returned at the end) so the
    caller can checkpoint each result the moment it's ready, same as the
    single-job path -- a job stuck at its timeout must not delay recording
    one that already finished.

    BUG FIXED 2026-08-02 (also applies here): a Pipe's OS-level buffer is
    small (~64KB on Windows); a worker whose result exceeds that blocks in
    send() until the buffer is drained. The old single-job code blocked on
    proc.join(timeout) BEFORE ever reading the pipe, so a worker that
    finished in seconds with a large result (a numpy prediction array)
    could never deliver it and would falsely read as a full-timeout DNF.
    Fix: poll every pending job's connection in a loop instead of blocking
    on join() first, so the parent is always ready to drain each pipe
    before its buffer fills.
    """
    procs, conns, deadlines = {}, {}, {}
    for name, fn, args, kwargs, timeout_sec in jobs:
        parent_conn, child_conn = mp.Pipe()
        proc = mp.Process(target=_timeout_worker, args=(fn, args, kwargs, child_conn))
        proc.start()
        procs[name] = proc
        conns[name] = parent_conn
        deadlines[name] = time.perf_counter() + timeout_sec

    pending = set(procs)
    while pending:
        now = time.perf_counter()
        for name in list(pending):
            try:
                has_data = conns[name].poll(0.1)
            except (BrokenPipeError, EOFError, OSError):
                has_data = False       # treat a broken pipe like "no data yet"
                                       # -- the is_alive()/deadline checks below
                                       # will correctly classify it as a crash

            if has_data:
                try:
                    status, payload = conns[name].recv()
                except (EOFError, OSError) as e:
                    status, payload = "error", f"failed to receive result: {type(e).__name__}: {e}"
                procs[name].join()
                pending.discard(name)
                yield name, status, payload
            elif not procs[name].is_alive():
                # Child exited without ever sending a result -- a crash
                # (segfault, OS-killed on memory pressure, etc.), not a
                # timeout. Report it immediately rather than waiting out
                # the full deadline and mislabeling it DNF.
                procs[name].join()
                pending.discard(name)
                yield name, "error", "worker process died without sending a result (crashed?)"
            elif now >= deadlines[name]:
                procs[name].terminate()
                procs[name].join()
                pending.discard(name)
                yield name, "DNF", None


def _run_with_timeout(fn, args, kwargs, timeout_sec):
    """Single-job convenience wrapper around _run_many_with_timeout, for
    callers that only need to time-guard one method (e.g. the corel5k
    ACkELO/ACkELD re-verification scripts)."""
    for _, status, payload in _run_many_with_timeout(
            [("_", fn, args, kwargs, timeout_sec)]):
        return status, payload


def run_one_seed(X, Y, seed, cfg: GAConfig, methods=None, k=3,
                 verbose=False, dataset=None, method_timeouts=None,
                 method_seed_limits=None, m_override=None):
    """Run every method once on one train/val/test split, yielding one row
    dict at a time so the caller can checkpoint after every method instead
    of waiting for the whole seed to finish (important once a single method
    can take hours -- see method_timeouts).

    dataset: name used to look up ACkEL's tuned (gamma, beta) in
    gals.baselines.ACKEL_PARAMS; only read by ACkELD/ACkELO.
    method_timeouts: {method_name: seconds}. Exceeding it hard-kills the
    subprocess running that method and records status="DNF" (Did Not
    Finish, following Madjarov et al. 2012's convention) instead of a
    result -- this is a time problem (unbounded O(N^2) kernel cost), not a
    memory problem, so it will not raise MemoryError on its own.
    method_seed_limits: {method_name: max_seed_count}. Seeds >= the limit
    are skipped for that method only (status="skipped_seed_limit"); other
    methods keep running normally for that seed -- e.g. give ACkELO 1 seed
    while everything else on the same dataset gets 3.
    m_override: skip GALS entirely (no GALS row is yielded) and use this
    value as the RAkELO/ACkELO model count instead. GALS is deterministic
    given the same (X, Y, seed, cfg), so if you already know this seed's M
    from a previous run's saved GALS row, recomputing it just to discard
    the row wastes real time -- for mediamill that's 16+ hours. Only use
    this for targeted re-runs of specific baseline methods on a seed whose
    GALS result is already recorded elsewhere.
    """
    method_timeouts = method_timeouts or {}
    method_seed_limits = method_seed_limits or {}

    split = split_data(X, Y, seed)
    Xtr, Ytr = X[split["train"]], Y[split["train"]]
    Xte, Yte = X[split["test"]], Y[split["test"]]

    # ---------------- GALS first: it determines m for RAkEL ----------------
    if m_override is not None:
        m = m_override
    else:
        t0 = time.perf_counter()
        res = run_gals(X, Y, split, cfg, seed, verbose=verbose)
        P = fit_predict_ensemble(X, Y, split, res["subsets"], seed, cfg.rf_params)
        elapsed = time.perf_counter() - t0

        yield dict(method="GALS", seed=seed, runtime_sec=elapsed,
                  n_models=len(res["subsets"]),
                  mean_subset_size=res["mean_subset_size"],
                  threshold=res["threshold"], generations=res["generations"],
                  n_rf_trainings=res["n_rf_trainings"],
                  n_compensated=res["n_missing_compensated"],
                  **evaluate_all(Yte, P))

        m = len(res["subsets"])      # match model count for fair comparison

    def build_call(name):
        fn = BASELINES[name]
        kwargs = {}
        if name == "RAkELD":
            kwargs = dict(k=k)
        elif name == "ACkELD":
            kwargs = dict(k=k, dataset=dataset)
        elif name == "RAkELO":
            kwargs = dict(k=k, m=m)
        elif name == "ACkELO":
            kwargs = dict(k=k, m=m, dataset=dataset)
        return fn, (Xtr, Ytr, Xte, seed, cfg.rf_params), kwargs

    def result_row(name, status, payload, elapsed):
        if status == "DNF":
            return dict(method=name, seed=seed, runtime_sec=elapsed,
                       status=f"DNF (timeout={method_timeouts[name]}s)")
        if status == "not_implemented":
            return dict(method=name, seed=seed, runtime_sec=np.nan,
                       status="not_implemented")
        if status == "out_of_memory":
            return dict(method=name, seed=seed, runtime_sec=np.nan,
                       status="out_of_memory")
        if status == "error":
            return dict(method=name, seed=seed, runtime_sec=np.nan,
                       status=f"error: {payload}")
        try:                                        # status == "ok"
            return dict(method=name, seed=seed, runtime_sec=elapsed,
                       n_models=(m if name in ("RAkELO", "ACkELO") else np.nan),
                       **evaluate_all(Yte, payload))
        except Exception as e:                       # keep the sweep alive
            traceback.print_exc()
            return dict(method=name, seed=seed, runtime_sec=np.nan,
                       status=f"error: {type(e).__name__}")

    # ---------------- baselines ----------------
    methods = methods or ["BR", "CC", "LP", "RAkELD", "RAkELO"]
    runnable = []
    for name in methods:
        if name in method_seed_limits and seed >= method_seed_limits[name]:
            yield dict(method=name, seed=seed, runtime_sec=np.nan,
                      status=f"skipped_seed_limit ({method_seed_limits[name]})")
            continue
        runnable.append(name)

    # untimed methods: no subprocess needed, run sequentially in-process
    for name in [n for n in runnable if n not in method_timeouts]:
        fn, args, kwargs = build_call(name)
        t0 = time.perf_counter()
        try:
            P = fn(*args, **kwargs)
            yield result_row(name, "ok", P, time.perf_counter() - t0)
        except NotImplementedError:
            yield result_row(name, "not_implemented", None, np.nan)
        except MemoryError:
            yield result_row(name, "out_of_memory", None, np.nan)
        except Exception as e:
            traceback.print_exc()
            yield dict(method=name, seed=seed, runtime_sec=np.nan,
                      status=f"error: {type(e).__name__}: {e}")

    # timed methods: launched together, cost = max(their timeouts), not the
    # sum -- they're independent, no reason to make one wait for another
    timed_names = [n for n in runnable if n in method_timeouts]
    if timed_names:
        jobs = []
        for name in timed_names:
            fn, args, kwargs = build_call(name)
            jobs.append((name, fn, args, kwargs, method_timeouts[name]))
        t0 = time.perf_counter()
        for name, status, payload in _run_many_with_timeout(jobs):
            yield result_row(name, status, payload, time.perf_counter() - t0)


def run_dataset(name, X, Y, seeds, cfg: GAConfig, methods=None, k=3,
                out_csv=None, verbose=True, resume=True,
                method_timeouts=None, method_seed_limits=None):
    """Run all seeds. If out_csv exists and resume=True, seeds already present
    are skipped and new results are appended, so a long sweep can be
    interrupted and restarted without losing work. Checkpoints after every
    method (not just every seed), so a slow method (e.g. ACkELO with a long
    method_timeouts entry) can never erase the cheaper methods that already
    finished for the same seed."""
    import os

    done, all_rows = set(), []
    if out_csv and resume and os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        prev = prev[prev["dataset"] == name] if "dataset" in prev else prev
        all_rows = pd.read_csv(out_csv).to_dict("records")
        done = set(prev["seed"].unique())
        if verbose and done:
            print(f"[{name}] resuming; seeds already done: {sorted(done)}")

    for s in seeds:
        if s in done:
            continue
        if verbose:
            print(f"[{name}] seed {s} ...", flush=True)
        for row in run_one_seed(X, Y, s, cfg, methods, k, verbose, dataset=name,
                                method_timeouts=method_timeouts,
                                method_seed_limits=method_seed_limits):
            row["dataset"] = name
            all_rows.append(row)
            if verbose:
                print(f"    {row.get('method')}: "
                     f"{row.get('status', 'ok')} "
                     f"({row.get('runtime_sec', float('nan')):.1f}s)", flush=True)
            if out_csv:                    # checkpoint after every method
                pd.DataFrame(all_rows).to_csv(out_csv, index=False)

    df = pd.DataFrame(all_rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
        if verbose:
            print(f"  -> wrote {out_csv}")
    return df
