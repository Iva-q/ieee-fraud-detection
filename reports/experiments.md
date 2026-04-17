# Experiments log

Tracking each model iteration with configuration and results.
All scores are ROC-AUC.

## v0 — baseline LightGBM, raw features

**Date**: 2026-04-17
**Notebook**: `notebooks/02_baseline.ipynb`
**Model files**: `data/predictions/sub_lgbm_baseline_v0.csv`, `oof_lgbm_baseline_v0.csv`

### Configuration
- LightGBM, raw features (431 columns), no FE
- learning_rate=0.05, num_leaves=63, min_child_samples=100
- Categorical features: native LightGBM handling (31 columns)
- CV: time-based, 5 folds, expanding window (folds 1-4 as validation)
- Early stopping 100 rounds

### Results
| Metric | Value |
|---|---|
| Fold 1 AUC | 0.88269 |
| Fold 2 AUC | 0.90049 |
| Fold 3 AUC | 0.92602 |
| Fold 4 AUC | 0.92049 |
| CV mean | 0.90742 |
| CV std | 0.01715 |
| **Public LB** | **0.91685** |
| **Private LB** | **0.89546** |

### Notes
- CV-Public gap +0.009 → CV is calibrated.
- Public-Private gap -0.02 → classic IEEE-CIS drift.
- Fold 1 is the weakest (0.883) — steep transition from pre-Christmas (fold 0)
  to post-Christmas period. Future FE should disproportionately help fold 1.
- Fold boundaries overlap on day edges (day 26 in both folds 0 and 1).
  Minor leakage; postponed until it becomes a limiting factor.