# GALS: Genetic Algorithm for Label Selection

Reference implementation for *Effective Label Subset Selection Method for
Multi-Label Classification Based on Genetic Algorithm*.

GALS selects label subsets with a genetic algorithm, builds one Label
Powerset + Random Forest model per subset, and combines their predictions
by majority vote. This repository also includes the baseline methods used
for comparison: Binary Relevance, Classifier Chains, Label Powerset,
RAkEL (disjoint/overlapping), and ACkEL (disjoint/overlapping).

## Installation

```
pip install -r requirements.txt
```

Requires numpy, scipy, scikit-learn, and pandas. `scikit-multilearn` is not
used.

## Quick check

```
python smoke_test.py
```

Runs the full pipeline on synthetic data (~20 seconds) to confirm the
installation works. Its output, `results/smoke_report.md`, is a smoke test on
artificial datasets and is not related to any reported result.

## Reproducing the results

Datasets are not distributed with this repository. Obtain the MULAN `.arff`
/ `.xml` files, put them in one flat directory, and point `GALS_DATA_ROOT` at
it (the scripts default to `../DATA`):

```
export GALS_DATA_ROOT=/path/to/DATA        # Windows: set GALS_DATA_ROOT=...
```

### Step 1 (required first): verify the loaded data

```
python verify_data.py
```

This loads all ten datasets and compares the loaded statistics — number of
instances, number of labels, distinct label sets, cardinality, density —
against the published Table 1 values recorded in `gals/loader.py` (`TABLE1`).
It prints `ALL DATASETS OK` only if every dataset matches.

**Run this before any experiment.** A loading error changes every number
downstream, and none of the experiment scripts check the data for you.

### Step 2: run the experiments

The seed plan, GA parameters, split protocol and wall-clock caps actually
used are recorded in `results/run_config.json`. Raw per-run results are in
`results/*_raw.csv`, one row per (method, seed).

| Reported table | Datasets | Command | Output |
|---|---|---|---|
| Main comparison tables | 7 small datasets, 10 seeds | `python run_batch.py batch1` | `results/<dataset>_raw.csv` |
| Main comparison tables | cal500 (10 seeds) + corel5k (3 seeds) | `python run_batch.py batch2` | `results/cal500_raw.csv`, `results/corel5k_raw.csv` |
| Main comparison tables | mediamill (3 seeds) | `python step3b_mediamill_continue.py` | `results/mediamill_raw.csv` |
| α sensitivity | 7 small datasets, 5 seeds | `python run_alpha_sweep.py` | `results/alpha_sweep.csv`, `results/alpha_sweep_summary.md` |
| L_ideal sensitivity | emotions, Gnegative, Plant, yeast, 5 seeds | `python run_ideal_ratio_sweep.py` | `results/ideal_ratio_sweep.csv`, `results/ideal_ratio_sweep_summary.md` |

Then combine and summarise batch1:

```
python summarize_batch1.py      # -> results/batch1_raw.csv, results/batch1_summary.md
```

### How the corel5k and mediamill results were produced

`run_batch.py` is the driver for the batch plan, but the corel5k and
mediamill rows in `results/` were not produced by a single clean invocation
of it, and re-running `run_batch.py batch2` / `batch3` will not reproduce
them exactly:

- **corel5k** — the batch2 run produced GALS/BR/CC/LP/RAkELD/RAkELO. ACkELD
  and ACkELO were then re-run by `step2b_corel5k_reverify.py`, and ACkELD was
  re-run again by `rerun_corel5k_ackeld.py` under the unified 24h wall-clock
  cap. `run_batch.py batch3` is not involved.
- **mediamill** — produced by `step3b_mediamill_continue.py`, which resumes a
  partially completed seed 0 and runs seeds 1–2 in full.

`results/run_config.json` records the caps and seed counts as actually run,
including where they differ between the two datasets.

## Loading a MULAN dataset

```python
from gals.loader import load_mulan
from gals.core import dataset_stats

X, Y, feature_names, label_names = load_mulan("path/to/yeast.arff", "path/to/yeast.xml")
print(dataset_stats(Y))   # compare against the dataset's published statistics
```

## Running a single experiment

```python
from gals.gals import GAConfig
from gals.runner import run_dataset
from gals.analyze import full_report, to_markdown

cfg = GAConfig()   # defaults documented in gals/gals.py
df = run_dataset("yeast", X, Y, seeds=range(10), cfg=cfg,
                 out_csv="results/yeast_raw.csv")
to_markdown(full_report(df), "results/yeast_report.md")
```

## Repository layout

### Library

- `gals/core.py` -- data loading/splitting, Label Powerset + Random Forest, evaluation metrics
- `gals/gals.py` -- the GA itself (selection, fitness, label compensation)
- `gals/loader.py` -- MULAN `.arff`/`.xml` dataset loader, `TABLE1` reference statistics
- `gals/baselines.py` -- BR, CC, LP, RAkEL, and the ACkEL parameter table
- `gals/ackel.py` -- ACkEL (disjoint/overlapping)
- `gals/runner.py` -- multi-seed experiment runner, wall-clock caps, checkpointing
- `gals/analyze.py` -- result aggregation, ranking, Wilcoxon tests, reporting
- `gals/experiments.py` -- environment capture, run-config dump, parameter sweeps

### Scripts

- `verify_data.py` -- checks all ten datasets against Table 1; run before anything else
- `smoke_test.py` -- end-to-end check on synthetic data
- `run_batch.py` -- batch driver (`batch1` / `batch2` / `batch3`)
- `run_alpha_sweep.py` -- α sweep over {0.8, 0.9, 1.0}, 5 seeds
- `run_ideal_ratio_sweep.py` -- L_ideal sweep over {L/4, L/3, L/2, 2L/3}, 5 seeds
- `resume_ideal_ratio_yeast.py` -- resumes the L_ideal sweep for yeast alone
- `step2b_corel5k_reverify.py` -- re-runs ACkELD/ACkELO on corel5k under a 6h cap
- `step3b_mediamill_continue.py` -- completes mediamill seed 0 and runs seeds 1–2
- `rerun_corel5k_ackeld.py` -- re-runs ACkELD on corel5k under the unified 24h cap
- `rerun_mediamill_ackeld.py` -- the same for mediamill; its docstring fixes the
  outcome criterion in advance, including what a DNF at the cap means
- `measure_corel5k_solo.py` -- times GALS and ACkELD on corel5k seed 0 one
  after the other on an idle machine, so the two are comparable; refuses to
  start if another Python process is running
- `summarize_batch1.py` -- combines the batch1 CSVs and writes the batch1 summary
- `tune_ackel_pseaac.py` -- grid-searches ACkEL (gamma, beta) for the three PseAAC datasets
- `run_quick_tasks.py` -- environment capture and run-config dump. It skips
  any output that already exists, so it will not overwrite the committed
  `results/run_config.json` or `results/environment.json`; those describe
  the runs that produced the reported results and a fresh dump would
  replace them with a description of the current checkout. Pass `--force`
  to regenerate them deliberately.

### Results

- `results/*_raw.csv` -- one row per (method, seed): the four reported metrics plus macro F1, runtime, GALS diagnostics, and a `status` column for runs that did not complete
- `results/run_config.json` -- GA parameters, split protocol, seed plan, wall-clock caps
- `results/run_counts.json` -- per (dataset, method) outcome counts, generated post hoc
- `results/environment.json` -- Python and library versions used for the reported runs
- `results/ackel_tuning.json` -- ACkEL grid-search results for the PseAAC datasets
- `results/*_summary.md` -- aggregated tables derived from the raw CSVs

## License

MIT -- see `LICENSE`.
