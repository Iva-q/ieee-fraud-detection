# Experiments log

Tracking each model iteration with configuration and results.
All scores are ROC-AUC.

## v0 вЂ” baseline LightGBM, raw features

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
- CV-Public gap +0.009 в†’ CV is calibrated.
- Public-Private gap -0.02 в†’ classic IEEE-CIS drift.
- Fold 1 is the weakest (0.883) вЂ” steep transition from pre-Christmas (fold 0)
  to post-Christmas period. Future FE should disproportionately help fold 1.
- Fold boundaries overlap on day edges (day 26 in both folds 0 and 1).
  Minor leakage; postponed until it becomes a limiting factor.

## v1 — LightGBM + time/money/card1 features

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
- **Fold 1 slightly regressed** (-0.002) — card1 aggregates are noisy on cold-start
  cards with few transactions. D1-based UID (v3) should fix this.
- `card1` is a BIN-level key, not a true client ID (max count 28015).
  Proper UID via card1 + addr1 + D1n is the next major step.
- Jump from top-50% to top-35% on private LB with a single FE iteration.


## v2 — LightGBM + encodings

**Date**: 2026-04-17
**Notebook**: `notebooks/04_encodings.ipynb`
**Model files**: `sub_lgbm_enc_v2.csv`, `oof_lgbm_enc_v2.csv`

### What changed vs v1
Added `src/features/encodings.py` with:
- **Frequency encoding** for 12 columns (card1-5, addr1-2, P/R_emaildomain, DeviceInfo, id_30/31/33)
  — target-independent, computed on train + test together
- **Out-of-fold target encoding** for 6 columns (card1, addr1, P/R_emaildomain, ProductCD, DeviceInfo)
  — expanding window, smoothing=10, fold 0 filled with global mean

Same LightGBM hyperparameters as v0 and v1.

### Results
| Metric | Value | Delta vs v1 |
|---|---|---|
| Fold 1 AUC | 0.87698 | -0.004 |
| Fold 2 AUC | 0.90928 | +0.003 |
| Fold 3 AUC | 0.92868 | -0.000 |
| Fold 4 AUC | 0.91967 | -0.002 |
| CV mean | 0.90865 | **-0.001** |
| CV std | 0.01953 | +0.001 |
| **Public LB** | **0.92804** | **+0.006** |
| **Private LB** | **0.90682** | **+0.004** |

### Notes
- **CV regressed by -0.001, but LB improved by +0.006**. Second time CV
  underestimates FE gain — this is a fundamental property of our
  expanding-window scheme combined with target encoding:
  * On fold 1, train = only fold 0, where TE values are constant
    (filled with global mean to avoid self-leakage).
  * Model cannot learn to use TE on fold 1 validation.
  * On real test, model trains on full train where TE is fully populated.
- **Debugging story**: initial implementation used fold-0 itself as bootstrap
  (self-leakage), which caused catastrophic fold 1 AUC drop to 0.822.
  Fix: replace fold-0 TE with global mean. Keeps the feature safe at the
  cost of making it useless on that fold specifically.
- Jump from top-35% to ~top-25% on private LB.
- Potential next step: lower smoothing (5 instead of 10) to let TE signal
  through for mid-frequency categories.


## v3 — LightGBM + UID reconstruction

**Date**: 2026-04-17
**Notebook**: `notebooks/05_uid.ipynb`
**Model files**: `sub_lgbm_uid_v3.csv`, `oof_lgbm_uid_v3.csv`

### What changed vs v2
Added `src/features/uid_features.py`:
- `D1n = D1 - day` — card registration day (per-client invariant)
- `UID = card1_addr1_D1n` — client proxy (199K unique values in train)
- UID aggregations: count, mean/std/max amt, nunique_productcd, amt ratio
- UID also plumbed into frequency + target encoding lists

### Results
| Metric | Value | Delta vs v2 |
|---|---|---|
| Fold 1 AUC | 0.88606 | +0.009 |
| Fold 2 AUC | 0.93196 | +0.023 |
| Fold 3 AUC | 0.95372 | +0.025 |
| Fold 4 AUC | 0.94452 | +0.025 |
| CV mean | 0.92907 | **+0.020** |
| CV std | 0.02600 | +0.006 |
| **Public LB** | **0.95019** | **+0.022** |
| **Private LB** | **0.92573** | **+0.019** |

### Notes
- **The breakthrough iteration.** UID aggregations give 10x the gain of all
  prior FE combined.
- **CV finally calibrated**: +0.020 CV matches +0.022 Public LB. UID
  features don't suffer from the fold-0 dilution problem of target encoding
  because they're target-independent and computed over train + test.
- UID coverage: 88.7% (addr1 is the main source of NaN). 199K unique UIDs
  over 524K valid transactions -> ~2.6 txns per UID on average, close to
  real-world "one client = one UID".
- Jump from top-25% to top-10-12% on private LB with a single iteration.
- Best_iter dropped from 400-500 to 130-280 — model finds signal quickly,
  no overfitting.
- Std grew to 0.026 due to uneven improvement across folds (fold 1 gained
  less because training on only fold 0 still has limited data for stable
  UID statistics). Not a problem — all folds improved, just unevenly.


## v4 — Feature selection via permutation importance

**Date**: 2026-04-17
**Notebook**: `notebooks/06_feature_selection.ipynb`
**Model files**: `sub_lgbm_fs_v4.csv`, `oof_lgbm_fs_v4.csv`
**Selected features**: `reports/selected_features_v4.json`

### What changed vs v3
- Ran permutation importance (n_repeats=3, scoring=roc_auc) on fold-4 model
- Dropped 210 features with importance_mean <= 0 (45% of all features)
- Kept 261 features for training
- Retrained full CV with selected set, same LightGBM hyperparameters

### Top 10 features by permutation importance
| Feature | Importance | Comment |
|---|---|---|
| uid_te | 0.0895 | UID target encoding — #1 by far, 10x the next |
| C1 | 0.0086 | Vesta Counting feature |
| C14, C13, C11 | 0.0053, 0.0045, 0.0034 | More C-features |
| id_31 | 0.0020 | Browser (native categorical) |
| DeviceInfo | 0.0019 | Device model (native categorical) |
| P_emaildomain | 0.0018 | Purchaser email (native categorical) |
| card1_te | 0.0018 | Card TE |
| C9, uid_amt_mean | 0.0018, 0.0017 | |

### Results
| Metric | v3 (471 feat) | v4 (261 feat) | Delta |
|---|---|---|---|
| CV mean | 0.92907 | 0.92933 | +0.00026 |
| CV std | 0.02600 | 0.02773 | +0.00173 |
| Public LB | 0.95019 | 0.95036 | +0.00017 |
| Private LB | 0.92573 | 0.92491 | -0.00082 |
| Fold 4 training time | 31s | 22s | -30% |

### Notes
- All metric deltas are within noise (~+-0.001). Main goal achieved:
  45% fewer features with no accuracy loss.
- `uid_te` alone has 10x the importance of the next feature — confirms
  UID reconstruction was the right call.
- Native LightGBM categoricals (`DeviceInfo`, `id_31`, `P_emaildomain`)
  outperform our explicit `_freq` and `_te` versions — means LightGBM's
  internal cat handling is already very strong.
- CV std grew slightly (0.026 -> 0.028); monitor in future iterations.
- Training speedup will compound during Optuna tuning (50-100 trials).


## Leaderboard calibration (2026-04-17)

Verified actual leaderboard positions via Kaggle (reference points and
neighboring ranks around our scores):

| Private LB | Rank (approx) | Percentile |
|---|---|---|
| 0.9459 (#1, winning team) | 1 | top 0.02% |
| 0.9289 (top-5% reference) | 226 | top 3.5% |
| 0.9249 (our v4) | ~1528 | **top 24%** |
| 0.9068 (our v2) | ~3500 | top 55% |

**Corrected assessment**: v4 reaches approximately top-24%, not top-10% as
initially extrapolated. The leaderboard has a heavy tail between 0.92 and
0.93, with thousands of teams clustered in this zone (public kernels peak
around here).

### Path to a stronger result
Medal zones (silver = top-5%) would require Private ~0.933+. Realistic gains
from remaining items in our plan:
- Hyperparameter tuning via Optuna: +0.002..+0.004
- LightGBM + CatBoost ensemble: +0.003..+0.005
- UID-based post-processing (winning-team technique): +0.010..+0.015

Combined, top-10% (Private ~0.930) is realistic. Top-5% would need the
post-processing trick, which is out of scope for this pet project.


## v5 — Optuna hyperparameter tuning

**Date**: 2026-04-18
**Notebook**: `notebooks/07_optuna.ipynb`
**Model files**: `sub_lgbm_tuned_v5.csv`, `oof_lgbm_tuned_v5.csv`
**Best params**: `reports/best_params_v5.json`

### What changed vs v4
- Ran Optuna study: 50 trials, TPE sampler, MedianPruner (n_warmup=2)
- Optimized 8 LightGBM hyperparameters on full 4-fold CV
- Trained final v5 model with best params, bumped num_boost_round to 3000
  and stopping_rounds to 150 (low LR needs more trees)

### Best params (vs defaults in v0-v4)
| Param | Default | Tuned | Comment |
|---|---|---|---|
| num_leaves | 63 | **127** | Double complexity — model was underfitting |
| learning_rate | 0.05 | **0.011** | 4.4x smaller — slower, more accurate |
| min_child_samples | 100 | **23** | 4x smaller — catches rare UID patterns |
| feature_fraction | 0.8 | **0.52** | Strong per-tree regularization |
| bagging_fraction | 0.8 | 0.82 | Minor tweak |
| bagging_freq | 5 | 8 | Minor tweak |
| reg_alpha | 0 | 0.12 | Small L1 |
| reg_lambda | 0 | 0.02 | Small L2 |

### Results
| Metric | v4 | v5 | Delta |
|---|---|---|---|
| Fold 1 AUC | 0.88317 | **0.89467** | +0.0115 |
| Fold 2 AUC | 0.93302 | 0.93567 | +0.003 |
| Fold 3 AUC | 0.95441 | 0.95608 | +0.002 |
| Fold 4 AUC | 0.94674 | 0.94780 | +0.001 |
| CV mean | 0.92933 | **0.93356** | +0.0042 |
| CV std | 0.02773 | **0.02359** | -0.0042 (more stable) |
| **Public LB** | 0.95036 | **0.95233** | +0.0020 |
| **Private LB** | 0.92491 | **0.92754** | **+0.0026** |

### Notes
- Fold 1 gained the most (+0.012) — the problem fold with small train.
  Larger `num_leaves` + smaller `min_child_samples` extract more from
  limited data.
- **CV std dropped** from 0.028 to 0.024 — model is more stable across folds,
  a desirable property for production.
- LB rank: Private 0.9275 corresponds to rank ~1085 (verified against Kaggle leaderboard), i.e. **top 17%** (up from top-24% at v4, 443-position jump).
- Tuning time: 50 trials = 90 min. TPE pruning skipped ~10 of them early.
- Optuna study object is in memory only; if reproducibility is critical,
  seed is fixed (TPE seed=42, prune on median).



## SHAP analysis (2026-04-18)

**Notebook**: `notebooks/09_shap.ipynb`
**Figures**: `reports/figures/08_shap_summary.png`, `09_shap_bar.png`,
`10_shap_dependence_uid_te.png`, `11_shap_force_tp.png`, `12_shap_force_fp.png`

### Setup
- Model: LightGBM v5 (fold 4, AUC 0.94780, best_iter=816)
- TreeExplainer on a stratified sample of 5000 rows (500 fraud + 4500 non-fraud)
- SHAP values computed in 112s

### Key findings

**1. uid_te dominates.** Mean |SHAP| = 0.70, next feature (C13) = 0.16 — a 4.5x gap.
SHAP values for uid_te span [-2, +7]; no other feature reaches +2.
A single feature can move prediction from 3.5% baseline to 99.7% on its own.

**2. Top-20 feature mix is balanced.**
- 5 engineered (uid_te, card1_te, uid_amt_mean, R_emaildomain_te, card1_amt_mean, D1n)
- 5 Vesta C-features (C13, C1, C11, C14, C5)
- 3 Vesta D-features (D2, D1, D3)
- 3 native categoricals (P_emaildomain, card6, R_emaildomain)
- 2 V-features (V91, V70) + TransactionAmt

**3. S-shaped non-linearity on uid_te.**
Dependence plot reveals the model uses uid_te in three zones:
- uid_te < 0.03 -> SHAP ≈ -2 (clean history)
- 0.05-0.15 -> transition (-1 to +6)
- >= 0.20 -> saturates at +6.5 (near-certain fraud)

**4. TransactionAmt interaction catches card-testing pattern.**
In the high-uid_te zone, low-amount transactions dominate — consistent with
fraudsters running small test charges on a compromised card.

**5. Model is well-calibrated at high confidence.**
Among predictions with P(fraud) > 0.9 in the 5000-sample, 137 are true positives
vs 5 false positives (27:1 ratio). High-confidence errors are rare.

### Notes for CV/interviews
- uid_te being 4.5x stronger than anything else was NOT obvious from EDA.
  It emerged only after engineering UID + applying OOF target encoding
  with smoothing. Key "lessons learned" item.
- The S-curve in dependence plot explains why CatBoost blend didn't help:
  LGBM already extracts most signal from uid_te on its own, leaving little
  for a second model to add.
