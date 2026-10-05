# Evaluation Results

## Logistic Regression baseline (Section 7.7)

Per-edge infiltration detection (one row per host-pair per window, flattened 49-dim edge vector, `class_weight='balanced'`).

### cic2017 (train rows=1,198,721, pos rate=0.0059)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| LR (val) | 0.0310 | 0.6663 | 0.0592 | 0.8643 | 0.2940 | 0.1447 |
| LR (test) | 0.0122 | 0.3738 | 0.0237 | 0.7892 | 0.3203 | 0.1423 |

### ctu13 (train rows=613,001, pos rate=0.5916)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| LR (val) | 0.4904 | 0.8209 | 0.6140 | 0.9531 | 0.8756 | 0.1195 |
| LR (test) | 0.6291 | 0.5316 | 0.5763 | 0.8461 | 0.6794 | 0.1486 |
