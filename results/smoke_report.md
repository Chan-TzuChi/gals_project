# Smoke test report (synthetic data)

Output of `python smoke_test.py`, which runs the pipeline end to end on three
small synthetic datasets (`syn_a`, `syn_b`, `syn_c`) with planted label
structure. Its purpose is to confirm that an installation works.

These are not results from the paper. The datasets are artificial, only three
seeds are used, and every Wilcoxon test reports `too few datasets`, so the
rankings and values below carry no substantive meaning and should not be
compared with the reported tables.

## hamming_loss

| dataset   | BR              | CC              | LP              | RAkELD          | RAkELO          | GALS            |
|:----------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| syn_a     | 0.194±0.006 (2) | 0.202±0.021 (3) | 0.241±0.012 (6) | 0.224±0.012 (5) | 0.203±0.037 (4) | 0.180±0.012 (1) |
| syn_b     | 0.220±0.023 (3) | 0.224±0.046 (4) | 0.276±0.042 (6) | 0.243±0.024 (5) | 0.210±0.023 (1) | 0.213±0.017 (2) |
| syn_c     | 0.176±0.029 (5) | 0.175±0.007 (4) | 0.191±0.044 (6) | 0.163±0.026 (3) | 0.156±0.019 (2) | 0.149±0.007 (1) |

**Average rank**

| method   |   avg_rank |
|:---------|-----------:|
| GALS     |    1.33333 |
| RAkELO   |    2.33333 |
| BR       |    3.33333 |
| CC       |    3.66667 |
| RAkELD   |    4.33333 |
| LP       |    6       |

**Wilcoxon signed-rank vs GALS**

| method   |   n_datasets |   statistic |   p_value | note             |
|:---------|-------------:|------------:|----------:|:-----------------|
| BR       |            3 |         nan |       nan | too few datasets |
| CC       |            3 |         nan |       nan | too few datasets |
| LP       |            3 |         nan |       nan | too few datasets |
| RAkELD   |            3 |         nan |       nan | too few datasets |
| RAkELO   |            3 |         nan |       nan | too few datasets |

## subset_accuracy

| dataset   | BR              | CC              | LP              | RAkELD          | RAkELO          | GALS            |
|:----------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| syn_a     | 0.278±0.058 (5) | 0.430±0.055 (1) | 0.367±0.040 (2) | 0.267±0.029 (6) | 0.281±0.143 (4) | 0.359±0.083 (3) |
| syn_b     | 0.204±0.042 (5) | 0.319±0.046 (1) | 0.263±0.071 (4) | 0.167±0.073 (6) | 0.281±0.084 (2) | 0.274±0.051 (3) |
| syn_c     | 0.444±0.031 (6) | 0.573±0.023 (1) | 0.569±0.073 (2) | 0.498±0.060 (5) | 0.547±0.027 (3) | 0.547±0.053 (4) |

**Average rank**

| method   |   avg_rank |
|:---------|-----------:|
| CC       |    1       |
| LP       |    2.66667 |
| RAkELO   |    3       |
| GALS     |    3.33333 |
| BR       |    5.33333 |
| RAkELD   |    5.66667 |

**Wilcoxon signed-rank vs GALS**

| method   |   n_datasets |   statistic |   p_value | note             |
|:---------|-------------:|------------:|----------:|:-----------------|
| BR       |            3 |         nan |       nan | too few datasets |
| CC       |            3 |         nan |       nan | too few datasets |
| LP       |            3 |         nan |       nan | too few datasets |
| RAkELD   |            3 |         nan |       nan | too few datasets |
| RAkELO   |            3 |         nan |       nan | too few datasets |

## weighted_f1

| dataset   | BR              | CC              | LP              | RAkELD          | RAkELO          | GALS            |
|:----------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| syn_a     | 0.804±0.003 (2) | 0.787±0.027 (3) | 0.744±0.018 (6) | 0.771±0.014 (5) | 0.773±0.071 (4) | 0.819±0.013 (1) |
| syn_b     | 0.763±0.014 (3) | 0.753±0.050 (4) | 0.703±0.038 (6) | 0.746±0.016 (5) | 0.777±0.017 (1) | 0.772±0.003 (2) |
| syn_c     | 0.831±0.033 (5) | 0.831±0.005 (4) | 0.819±0.046 (6) | 0.847±0.023 (3) | 0.851±0.021 (2) | 0.856±0.016 (1) |

**Average rank**

| method   |   avg_rank |
|:---------|-----------:|
| GALS     |    1.33333 |
| RAkELO   |    2.33333 |
| BR       |    3.33333 |
| CC       |    3.66667 |
| RAkELD   |    4.33333 |
| LP       |    6       |

**Wilcoxon signed-rank vs GALS**

| method   |   n_datasets |   statistic |   p_value | note             |
|:---------|-------------:|------------:|----------:|:-----------------|
| BR       |            3 |         nan |       nan | too few datasets |
| CC       |            3 |         nan |       nan | too few datasets |
| LP       |            3 |         nan |       nan | too few datasets |
| RAkELD   |            3 |         nan |       nan | too few datasets |
| RAkELO   |            3 |         nan |       nan | too few datasets |

## micro_f1

| dataset   | BR              | CC              | LP              | RAkELD          | RAkELO          | GALS            |
|:----------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| syn_a     | 0.804±0.003 (2) | 0.787±0.028 (4) | 0.745±0.017 (6) | 0.771±0.013 (5) | 0.791±0.043 (3) | 0.819±0.013 (1) |
| syn_b     | 0.765±0.013 (3) | 0.754±0.051 (4) | 0.703±0.038 (6) | 0.746±0.015 (5) | 0.778±0.017 (1) | 0.773±0.003 (2) |
| syn_c     | 0.831±0.033 (5) | 0.832±0.005 (4) | 0.819±0.047 (6) | 0.847±0.023 (3) | 0.851±0.021 (2) | 0.856±0.016 (1) |

**Average rank**

| method   |   avg_rank |
|:---------|-----------:|
| GALS     |    1.33333 |
| RAkELO   |    2       |
| BR       |    3.33333 |
| CC       |    4       |
| RAkELD   |    4.33333 |
| LP       |    6       |

**Wilcoxon signed-rank vs GALS**

| method   |   n_datasets |   statistic |   p_value | note             |
|:---------|-------------:|------------:|----------:|:-----------------|
| BR       |            3 |         nan |       nan | too few datasets |
| CC       |            3 |         nan |       nan | too few datasets |
| LP       |            3 |         nan |       nan | too few datasets |
| RAkELD   |            3 |         nan |       nan | too few datasets |
| RAkELO   |            3 |         nan |       nan | too few datasets |

## runtime_sec

| dataset   | BR              | CC              | LP              | RAkELD          | RAkELO          | GALS            |
|:----------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| syn_a     | 0.525±0.031 (3) | 0.581±0.029 (4) | 0.087±0.007 (1) | 0.227±0.009 (2) | 2.061±1.527 (5) | 5.693±3.776 (6) |
| syn_b     | 0.441±0.288 (4) | 0.424±0.256 (3) | 0.069±0.044 (1) | 0.192±0.122 (2) | 2.936±2.879 (5) | 7.725±7.822 (6) |
| syn_c     | 0.162±0.002 (4) | 0.160±0.003 (3) | 0.032±0.002 (1) | 0.060±0.003 (2) | 0.544±0.090 (5) | 1.437±0.181 (6) |

**Average rank**

| method   |   avg_rank |
|:---------|-----------:|
| LP       |    1       |
| RAkELD   |    2       |
| CC       |    3.33333 |
| BR       |    3.66667 |
| RAkELO   |    5       |
| GALS     |    6       |

**Wilcoxon signed-rank vs GALS**

| method   |   n_datasets |   statistic |   p_value | note             |
|:---------|-------------:|------------:|----------:|:-----------------|
| BR       |            3 |         nan |       nan | too few datasets |
| CC       |            3 |         nan |       nan | too few datasets |
| LP       |            3 |         nan |       nan | too few datasets |
| RAkELD   |            3 |         nan |       nan | too few datasets |
| RAkELO   |            3 |         nan |       nan | too few datasets |
