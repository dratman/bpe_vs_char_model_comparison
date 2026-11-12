### A1 final-step repetition bpc (step 61000)

| run | first-copy bpc | second-copy bpc | first - second |
|---|---|---|---|
| B1 | 2.3425 | 1.7960 | 0.5465 |
| B2 | 2.3038 | 1.9012 | 0.4026 |
| B3 | 2.3272 | 1.8627 | 0.4645 |
| C1 | 2.3536 | 0.9415 | 1.4121 |
| C2 | 2.3685 | 0.9326 | 1.4359 |
| C3 | 2.3540 | 1.0242 | 1.3299 |
| C4 | 2.3846 | 0.8323 | 1.5523 |

| group | first (mean [min-max]) | second | diff |
|---|---|---|---|
| BPE B1-3 | 2.325 [2.304-2.342] | 1.853 [1.796-1.901] | 0.471 [0.403-0.546] |
| char C1-3 | 2.359 [2.354-2.368] | 0.966 [0.933-1.024] | 1.393 [1.330-1.436] |

### A1 tokenizer statistics (probe v3)

BPE chars/token (scored spans): prose 4.136 (0.242 tok/char), morphology 4.107 (0.243 tok/char), short_sentence 3.562 (0.281 tok/char), repetition 5.887 (0.170 tok/char)

| text | words | fraction split into 2+ BPE tokens | BPE tokens/word | chars/token (words) |
|---|---|---|---|---|
| repetition (random words, first 512 chars) | 1324 | 0.039 | 1.05 | 5.89 |
| prose | 2836 | 0.159 | 1.23 | 4.53 |
| morphology | 2695 | 0.217 | 1.31 | 4.47 |
| short_sentence | 3044 | 0.174 | 1.26 | 4.00 |
| prose+morphology+short_sentence | 8575 | 0.183 | 1.26 | 4.33 |
| repetition distinct words | 745 | 0.039 | nan | nan |

### A2 induction onsets (head > 0.2 for 5 consecutive log points; steepest rise of 5-pt-smoothed first-second bpc)

| run | onset step | onset chars seen | onset head (layer.head) | steepest-rise step | steepest chars seen | heads>0.2 at final | layers of those heads |
|---|---|---|---|---|---|---|---|
| B1 | 6250 | 200.6M | 7.7 | 4500 | 144.4M | 1 | {7: 1} |
| B2 | 12750 | 409.2M | 7.1 | 18250 | 585.8M | 1 | {3: 1} |
| B3 | 7000 | 224.7M | 7.2 | 3000 | 96.3M | 1 | {7: 1} |
| C1 | 1000 | 32.1M | 5.6 | 1250 | 40.1M | 1 | {5: 1} |
| C2 | 1250 | 40.1M | 4.5 | 1250 | 40.1M | 2 | {4: 1, 5: 1} |
| C3 | 1250 | 40.1M | 6.2 | 1500 | 48.1M | 2 | {4: 1, 6: 1} |
| C4 | 1000 | 32.1M | 5.6 | 1000 | 32.1M | 1 | {5: 1} |

| group | onset chars (mean [min-max]) | steepest-rise chars (mean [min-max]) |
|---|---|---|
| BPE B1-3 | 278.2M [200.6M-409.2M] | 275.5M [96.3M-585.8M] |
| char C1-3 | 37.4M [32.1M-40.1M] | 42.8M [40.1M-48.1M] |

### A3 final-step phase shares per layer (mean [min-max over 3 seeds]; C4 listed separately)

| layer | tok | single | cluster | averaging |
|---|---|---|---|---|
| 0 | BPE | 0.00 [0.00-0.00] | 0.12 [0.12-0.12] | 0.88 [0.88-0.88] |
| 0 | char | 0.25 [0.25-0.25] | 0.00 [0.00-0.00] | 0.75 [0.75-0.75] |
| 0 | C4 | 0.25 | 0.00 | 0.75 |
| 1 | BPE | 0.00 [0.00-0.00] | 0.00 [0.00-0.00] | 1.00 [1.00-1.00] |
| 1 | char | 0.12 [0.12-0.12] | 0.58 [0.12-0.88] | 0.29 [0.00-0.75] |
| 1 | C4 | 0.12 | 0.62 | 0.25 |
| 2 | BPE | 0.00 [0.00-0.00] | 0.12 [0.00-0.25] | 0.88 [0.75-1.00] |
| 2 | char | 0.04 [0.00-0.12] | 0.92 [0.88-1.00] | 0.04 [0.00-0.12] |
| 2 | C4 | 0.00 | 1.00 | 0.00 |
| 3 | BPE | 0.00 [0.00-0.00] | 0.54 [0.50-0.62] | 0.46 [0.38-0.50] |
| 3 | char | 0.00 [0.00-0.00] | 0.92 [0.88-1.00] | 0.08 [0.00-0.12] |
| 3 | C4 | 0.00 | 1.00 | 0.00 |
| 4 | BPE | 0.00 [0.00-0.00] | 0.46 [0.38-0.50] | 0.54 [0.50-0.62] |
| 4 | char | 0.00 [0.00-0.00] | 0.67 [0.50-0.88] | 0.33 [0.12-0.50] |
| 4 | C4 | 0.00 | 0.88 | 0.12 |
| 5 | BPE | 0.00 [0.00-0.00] | 0.46 [0.38-0.50] | 0.54 [0.50-0.62] |
| 5 | char | 0.00 [0.00-0.00] | 0.42 [0.12-0.62] | 0.58 [0.38-0.88] |
| 5 | C4 | 0.00 | 0.25 | 0.75 |
| 6 | BPE | 0.00 [0.00-0.00] | 0.21 [0.12-0.25] | 0.79 [0.75-0.88] |
| 6 | char | 0.00 [0.00-0.00] | 0.12 [0.00-0.25] | 0.88 [0.75-1.00] |
| 6 | C4 | 0.00 | 0.12 | 0.88 |
| 7 | BPE | 0.00 [0.00-0.00] | 0.00 [0.00-0.00] | 1.00 [1.00-1.00] |
| 7 | char | 0.00 [0.00-0.00] | 0.04 [0.00-0.12] | 0.96 [0.88-1.00] |
| 7 | C4 | 0.00 | 0.12 | 0.88 |

Early layers (0-3 pooled, cluster share of heads), per seed: 
- BPE: [0.156, 0.219, 0.219]
- char: [0.5, 0.688, 0.625, 0.656]
- late layers 4-7 cluster share: BPE [0.281, 0.281, 0.281], char [0.312, 0.344, 0.281, 0.344]

### A4 BPE-vs-char comparison by kind (mean over seeds; gap = BPE - char, negative = BPE ahead)

| kind | BPE ahead at first log point (chars seen) | ever behind? | permanently ahead from (chars) | fraction of log points BPE ahead | final gap |
|---|---|---|---|---|---|
| prose | 8.0M | no | start | 1.00 | -0.019 |
| morphology | 8.0M | yes | 1348.1M | 0.99 | -0.008 |
| short_sentence | 8.0M | yes | never | 0.02 | +0.022 |
| repetition | 8.0M | yes | never | 0.01 | +0.418 |

Mean gap (BPE - char) at selected chars-seen points: 
- prose: {'8.0M': -1.502, '40.1M': -0.088, '80.2M': -0.058, '160.5M': -0.044, '321.0M': -0.052, '642.0M': -0.031, '962.9M': -0.023, '1283.9M': -0.032, '1604.9M': -0.024, '1958.0M': -0.019}  (sign changes: 0)
- morphology: {'8.0M': -1.386, '40.1M': -0.071, '80.2M': -0.047, '160.5M': -0.033, '321.0M': -0.026, '642.0M': -0.013, '962.9M': -0.015, '1283.9M': -0.004, '1604.9M': -0.013, '1958.0M': -0.008}  (sign changes: 4)
- short_sentence: {'8.0M': -1.339, '40.1M': 0.035, '80.2M': 0.034, '160.5M': 0.021, '321.0M': 0.011, '642.0M': 0.023, '962.9M': 0.029, '1283.9M': 0.024, '1604.9M': 0.019, '1958.0M': 0.022}  (sign changes: 1)
- repetition: {'8.0M': -1.319, '40.1M': 0.278, '80.2M': 0.409, '160.5M': 0.386, '321.0M': 0.298, '642.0M': 0.342, '962.9M': 0.376, '1283.9M': 0.396, '1604.9M': 0.415, '1958.0M': 0.418}  (sign changes: 1)
