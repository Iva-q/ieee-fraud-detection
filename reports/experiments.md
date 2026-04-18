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

## v1 � LightGBM + time/money/card1 features

**Date**: 2026-04-17
**Notebook**: `notebooks/03_baseline_with_fe.ipynb`
**Model files**: `sub_lgbm_fe_v1.csv`, `oof_lgbm_fe_v1.csv`

### What changed vs v0
Added 13 engineered features across three modules:
- `src/features/time_features.py`: day, hour, dayofweek, is_night
- `src/features/money_features.py`: log_amt, amt_cents, amt_has_cents
- `src/features/aggregations.py`: card1_count, card1_amt_{mean,std,max}, card1_nunique_productcd, amt_to_card1_mean_ratio
- Orchestrated via `src/features/build_features.py`

Same LightGBM hyperparameters as v0 to isolate the effect of FE.

### Results
| Metric | Value | Delta vs v0 |
|---|---|---|
| Fold 1 AUC | 0.88107 | -0.002 |
| Fold 2 AUC | 0.90585 | +0.005 |
| Fold 3 AUC | 0.92880 | +0.003 |
| Fold 4 AUC | 0.92209 | +0.002 |
| CV mean | 0.90945 | **+0.002** |
| CV std | 0.01839 | +0.001 |
| **Public LB** | **0.92254** | **+0.006** |
| **Private LB** | **0.90245** | **+0.007** |

### Notes
- **CV underestimates FE gain by ~3x**: +0.002 CV vs +0.006-0.007 LB.
  Hypothesis: expanding-window CV underestimates aggregation features because
  statistics stabilize with data volume; test benefits from full-train aggregates.
- **Fold 1 slightly regressed** (-0.002) � card1 aggregates are noisy on cold-start
  cards with few transactions. D1-based UID (v3) should fix this.
- `card1` is a BIN-level key, not a true client ID (max count 28015).
  Proper UID via card1 + addr1 + D1n is the next major step.
- Jump from top-50% to top-35% on private LB with a single FE iteration.
