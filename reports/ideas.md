# IEEE-CIS Fraud Detection — Ideas & Findings

## 1. Target

- **isFraud = 3.6%** overall, heavy imbalance → use ROC-AUC, PR-AUC; stratify where sensible.
- **Fraud rate drifts over time** (1.1%–7.0% across 182 days): ruling out random KFold, only time-based CV.
- First ~25 days have **lower fraud rate + higher volume** (likely holiday season in anonymized data).

## 2. Time features

- Strongest hour-of-day signal: peak fraud at hour 7 (**10.6%**), minimum at hour 13 (**2.3%**), 4.6× spread.
- Derived features to engineer in day 7:
  - `hour = (TransactionDT % 86400) // 3600`
  - `day = TransactionDT // 86400`
  - `is_night = hour in [0..10]`

## 3. Missing values

- 22 of 436 columns have zero missing; 214 columns have >50%; 12 have >90%.
- **Strong missing-value clustering**: 339 V-columns group into only **14 distinct missing-rate levels**.
  - Largest cluster: 46 V-cols with exactly 77.91% missing — almost certainly a single Vesta subsystem.
  - Implication: within each cluster, V-cols are highly correlated. Reduce via correlation > 0.98 per group.
- Identity columns ~76% missing because `identity` table only covers 24% of transactions.
- Action items:
  - Binary features: `has_identity`, `v_block_77_missing` etc.
  - LightGBM/CatBoost handle NaN natively — no imputation for tree baselines.

## 4. Categorical features

### Low-cardinality (2-5 values) — 22 cols
`ProductCD, card4, card6, DeviceType, M1..M9, most id_*`

- `ProductCD`: W=2% vs C=11.6% — 5× difference, strong signal.
- `card6`: credit 6.6% vs debit 2.5% — 2.5× difference.
- `DeviceType`: mobile 10% vs desktop 6.5% — plus the mere fact of having identity = risk signal.
- Encoding: label encoding or native `categorical_feature` in LightGBM.

### Medium-cardinality (50–130) — `P_emaildomain`, `R_emaildomain`, `id_30`, `id_31`
- `P_emaildomain` fraud rate spans 0.4% (sbcglobal, att) to 9.3% (outlook) — 25× range.
- Encoding strategy:
  - **Frequency encoding** (safe, computed on train+test together).
  - **OOF target encoding** with smoothing α=10..50 and **expanding window** along time to avoid both self-leakage and time leakage.
  - Additional bool features: `is_ms_email`, `is_telecom_email`.
  - Split domain into provider + suffix (`yahoo.com.mx` → `yahoo` + `com.mx`).

### High-cardinality (260+) — `id_33`, `DeviceInfo` (1786 unique)
- `DeviceInfo` needs heavy cleanup: extract vendor prefix (`SAMSUNG SM-G892A ...` → `SAMSUNG`).
- `id_33` (screen resolution `1334x750`): parse width × height → two numeric features; also `aspect_ratio`.
- After cleanup: frequency / target encoding.

## 5. Validation strategy (day 4)

- **TimeSeriesKFold** by `TransactionDT`, 5 folds, each ~36 days.
- Alternative sanity check: `GroupKFold` by `card1` (proxy for user).
- Encoding computations **inside each fold only** (no global leakage).
- Sanity check: CV AUC should match Public LB within ±0.005–0.01. If not — CV is wrong.

## 6. Model iteration plan

- v0 baseline: LightGBM, raw features + native categoricals, time-based CV. Target: CV AUC ~0.91.
- v1: add hour/day features, card1 aggregations. Target: ~0.93.
- v2: frequency + target encoding (expanding-window). Target: ~0.94.
- v3: D1 trick (D1 - day = user registration proxy), device info cleanup. Target: ~0.945.
- v4: LightGBM + CatBoost rank-average ensemble. Target: ~0.95.


## 7. Insights from top solutions (after day 0 reading)

### 7.1 UID reconstruction (Chris Deotte, 1st place technique)

- Dataset does NOT contain a client ID (anonymized by Vesta).
- Reconstructed UID: `card1 + addr1 + D1n`, where `D1n = D1 - day`.
- `D1n` is a per-client constant (proxy for card registration day relative to dataset start).
- Why it works: transactions from the same client share this tuple across time.

### 7.2 Aggregations by UID

Once UID is reconstructed, create per-UID aggregations over transaction history:

- `uid_n_transactions` — total transactions by this UID
- `uid_amt_mean`, `uid_amt_std`, `uid_amt_max` — spending profile
- `uid_n_unique_products` — diversity of product types
- `uid_n_unique_addr2` — geography diversity (multiple countries = risk signal)
- `uid_first_txn_day`, `uid_last_txn_day` — activity window

These are among the strongest features in the competition.

### 7.3 Normalized D-columns

- D-columns (D1-D15) are timedelta-like: days since some event.
- For client-level features, subtract `day` from each D column to get the **event date**, not the delta.
- Example: `D1n = D1 - day` → day the card was issued (constant per client).
- Apply similar normalization to D2-D15 for additional client-invariant features.

### 7.4 Ensemble strategy (1st place)

- Three diverse GBDT models: XGBoost + CatBoost + LightGBM.
- Blended with near-equal weights (avoids LB overfitting).
- Trained on different feature sets to increase diversity.
- Final private LB AUC: 0.9459.

### 7.5 What to realistically target for this project

- v0 baseline (raw features, LightGBM only): CV ~0.91, Public LB ~0.91
- v1-v3 (hour/day, aggregations, encodings, D-normalization): CV ~0.94
- v4 (add UID aggregations + simple XGB+CatBoost ensemble): CV ~0.945, Private LB ~0.92 = top 20%

Top 5-10% requires more aggressive UID reconstruction + client-level feature engineering.
Top 1% adds V-column grouping + custom post-processing.
