
## hamming_loss

| dataset   | BR              | CC              | LP              | RAkELD          | RAkELO          | GALS            |
|:----------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| syn_a     | 0.194กำ0.006 (2) | 0.202กำ0.021 (3) | 0.241กำ0.012 (6) | 0.224กำ0.012 (5) | 0.203กำ0.037 (4) | 0.180กำ0.012 (1) |
| syn_b     | 0.220กำ0.023 (3) | 0.224กำ0.046 (4) | 0.276กำ0.042 (6) | 0.243กำ0.024 (5) | 0.210กำ0.023 (1) | 0.213กำ0.017 (2) |
| syn_c     | 0.176กำ0.029 (5) | 0.175กำ0.007 (4) | 0.191กำ0.044 (6) | 0.163กำ0.026 (3) | 0.156กำ0.019 (2) | 0.149กำ0.007 (1) |

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
| syn_a     | 0.278กำ0.058 (5) | 0.430กำ0.055 (1) | 0.367กำ0.040 (2) | 0.267กำ0.029 (6) | 0.281กำ0.143 (4) | 0.359กำ0.083 (3) |
| syn_b     | 0.204กำ0.042 (5) | 0.319กำ0.046 (1) | 0.263กำ0.071 (4) | 0.167กำ0.073 (6) | 0.281กำ0.084 (2) | 0.274กำ0.051 (3) |
| syn_c     | 0.444กำ0.031 (6) | 0.573กำ0.023 (1) | 0.569กำ0.073 (2) | 0.498กำ0.060 (5) | 0.547กำ0.027 (3) | 0.547กำ0.053 (4) |

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
| syn_a     | 0.804กำ0.003 (2) | 0.787กำ0.027 (3) | 0.744กำ0.018 (6) | 0.771กำ0.014 (5) | 0.773กำ0.071 (4) | 0.819กำ0.013 (1) |
| syn_b     | 0.763กำ0.014 (3) | 0.753กำ0.050 (4) | 0.703กำ0.038 (6) | 0.746กำ0.016 (5) | 0.777กำ0.017 (1) | 0.772กำ0.003 (2) |
| syn_c     | 0.831กำ0.033 (5) | 0.831กำ0.005 (4) | 0.819กำ0.046 (6) | 0.847กำ0.023 (3) | 0.851กำ0.021 (2) | 0.856กำ0.016 (1) |

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
| syn_a     | 0.804กำ0.003 (2) | 0.787กำ0.028 (4) | 0.745กำ0.017 (6) | 0.771กำ0.013 (5) | 0.791กำ0.043 (3) | 0.819กำ0.013 (1) |
| syn_b     | 0.765กำ0.013 (3) | 0.754กำ0.051 (4) | 0.703กำ0.038 (6) | 0.746กำ0.015 (5) | 0.778กำ0.017 (1) | 0.773กำ0.003 (2) |
| syn_c     | 0.831กำ0.033 (5) | 0.832กำ0.005 (4) | 0.819กำ0.047 (6) | 0.847กำ0.023 (3) | 0.851กำ0.021 (2) | 0.856กำ0.016 (1) |

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
| syn_a     | 0.525กำ0.031 (3) | 0.581กำ0.029 (4) | 0.087กำ0.007 (1) | 0.227กำ0.009 (2) | 2.061กำ1.527 (5) | 5.693กำ3.776 (6) |
| syn_b     | 0.441กำ0.288 (4) | 0.424กำ0.256 (3) | 0.069กำ0.044 (1) | 0.192กำ0.122 (2) | 2.936กำ2.879 (5) | 7.725กำ7.822 (6) |
| syn_c     | 0.162กำ0.002 (4) | 0.160กำ0.003 (3) | 0.032กำ0.002 (1) | 0.060กำ0.003 (2) | 0.544กำ0.090 (5) | 1.437กำ0.181 (6) |

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
