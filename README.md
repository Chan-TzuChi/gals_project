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
installation works.

## Loading a MULAN dataset

```python
from gals.loader import load_mulan
from gals.core import dataset_stats

X, Y, feature_names, label_names = load_mulan("path/to/yeast.arff", "path/to/yeast.xml")
print(dataset_stats(Y))   # compare against the dataset's published statistics
```

## Running an experiment

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

- `gals/core.py` -- data loading/splitting, Label Powerset + Random Forest, evaluation metrics
- `gals/gals.py` -- the GA itself (selection, fitness, label compensation)
- `gals/loader.py` -- MULAN `.arff`/`.xml` dataset loader
- `gals/baselines.py` -- BR, CC, LP, RAkEL
- `gals/ackel.py` -- ACkEL (disjoint/overlapping)
- `gals/runner.py` -- multi-seed experiment runner
- `gals/analyze.py` -- result aggregation and reporting
- `gals/experiments.py` -- environment capture and parameter sweeps

## License

MIT -- see `LICENSE`.
