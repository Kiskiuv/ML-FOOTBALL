# CLAUDE.md — ML Football Betting System

## Project Overview

Machine learning system for identifying profitable football (soccer) betting opportunities across European leagues. Uses walk-forward validation, FDR correction, and strict anti-leakage measures to find statistically significant edges against bookmaker odds.

**Owner:** Marc — Senior data scientist (ex-bookmaker analytics), based in Mataró, Catalonia.

## Architecture

### Protocol V4 Lean (Active — 21 features, RF+LR+Ensemble)
- Training: `scripts/protocol_lean_v4_no_odds.py`
- Prediction: `scripts/predict_fixtures_1402.py`
- Feature engineering: `scripts/feature_engineering_v4_lean.py` (historical) / `scripts/incremental_features.py` (current season)
- 14 hardcoded hybrid-selected strategies (NO odds leakage)
- Features: BASIC+ tier only (ELO, form, schedule, momentum, H2H)
- Models: RandomForest + LogisticRegression + Ensemble (0.5/0.5 blend)
- Model selection: 3-way by avg test AUC (RF vs LR vs Ensemble)
- Backtest 2024/25: 89 bets, +16.51u, +18.6% ROI

### Protocol V3.4 (Legacy — archived)
- All V3.4 scripts moved to `archive/`
- 77 features, 9 models, odds as features (leakage concern)

## Directory Structure

```
ml-football-betting/
├── scripts/
│   ├── prep_new_season.py              # Step 1: Split raw data into league files
│   ├── incremental_features.py         # Step 2: Build features (current season)
│   ├── feature_engineering_pipeline.py  # Feature engineering (used by incremental_features)
│   ├── feature_engineering_v4_lean.py   # V4 Lean feature engineering (historical, 21 features)
│   ├── protocol_lean_v4_no_odds.py     # Training: V4 Lean (RF+LR+Ensemble)
│   ├── predict_fixtures_1402.py        # Prediction: V4 Lean strategies
│   └── bet_tracker_v2.py              # Track bet results and P&L
├── data/                               # Historical data (Matches_*.xls, 38 leagues)
├── 20252026_season/                    # Current season raw files (*_20252026.xlsx/csv)
├── features/                           # Historical features (Matches_LEAGUE_features.csv)
├── features_20252026/                  # Current season features (LEAGUE_20252026_features.csv)
├── models/                             # Production models (model_LEAGUE_MARKET.joblib)
├── new_fixtures/                       # Upcoming fixtures CSV
├── predictions/                        # Output prediction CSVs
├── config/                             # Strategy config CSVs
├── docs/                               # Documentation
├── archive/                            # Deprecated V3.4 scripts and old files
├── CLAUDE.md
├── requirements.txt
└── .gitignore
```

## Production Workflow (Weekly)

```bash
# Step 1: Update raw season data from football-data.co.uk downloads
python scripts/prep_new_season.py --input all-euro-data-2025-2026.xlsx --output ./20252026_season --latest Latest_Results.xlsx

# Step 2: Build features (rebuilds ELO, form, H2H from scratch — ~15 min for all leagues)
python scripts/incremental_features.py --all --historical ./data --new-season ./20252026_season --output ./features_20252026

# Step 3: Train models (V4 Lean — all leagues)
python scripts/protocol_lean_v4_no_odds.py --batch --data-dir ./features --all --output ./models

# Step 4: Predict upcoming fixtures
python scripts/predict_fixtures_1402.py --fixtures ./new_fixtures/fixtures.csv --models-dir ./models --season-dir ./features_20252026 --output ./predictions/predictions_DDMM.csv --eu-format
```

## V4 Lean Strategies (14 active)

```
League  Market  Strategy          Model                PCT  Edge   MinOdds  p-value
MEX     AWAY    CONSERVATIVE      LogisticRegression   88   0.00   2.5      0.0056 (FDR✅)
FIN     AWAY    SELECTIVE         LogisticRegression   85  -0.01   2.3      0.0078 (FDR✅)
D2      AWAY    CONSERVATIVE      RandomForest         88   0.00   2.5      0.0495
F1      HOME    UPSET             LogisticRegression   85  -0.01   2.0      0.0587
I1      DRAW    CONSERVATIVE      RandomForest         90   0.00   3.0      0.0612
F2      HOME    ULTRA_CONS        RandomForest         90   0.02   2.5      0.0613
SP2     DRAW    SELECTIVE         LogisticRegression   88  -0.01   3.0      0.0637
NOR     HOME    STANDARD          LogisticRegression   80  -0.02   1.9      0.0640
G1      DRAW    LONGSHOT_STRICT   LogisticRegression   92   0.00   3.2      0.0677
F2      AWAY    ULTRA_CONS        LogisticRegression   90   0.02   2.8      0.0697
D1      AWAY    SELECTIVE         LogisticRegression   85  -0.01   2.3      0.0736
ARG     HOME    ULTRA_CONS        LogisticRegression   90   0.02   2.5      0.0880
N1      AWAY    STANDARD_HIGH     LogisticRegression   82   0.00   2.2      0.0888
SWE     HOME    SELECTIVE         LogisticRegression   82  -0.01   2.0      0.0926
```

## Staking Rules

Base stake by edge:
- Edge < 0%: 0.80u (LOW)
- Edge 0-5%: 1.25u (MED)
- Edge > 5%: 1.85u (HIGH)

## Bet Decision Criteria (ALL must pass)

1. `model_probability >= percentile_threshold` (strategy-specific pct)
2. `edge >= strategy.min_edge` (model_prob - implied_prob)
3. `odds >= strategy.min_odds`
4. `0.01 < probability < 0.99` (validity check)

## Key Features (BASIC+ — 21 features, V4 Lean)

```python
FEATURES = [
    # ELO (5)
    "HomeElo", "AwayElo", "elo_diff", "elo_sum", "elo_mismatch",
    # Form (6)
    "Form3Home_ratio", "Form5Home_ratio", "Form3Away_ratio", "Form5Away_ratio",
    "form_diff_3", "form_diff_5",
    # Schedule/Temporal (4)
    "rest_diff", "rest_diff_nonlinear", "winter_period", "season_phase",
    # Form momentum (2)
    "form_momentum_home", "form_momentum_away",
    # H2H (4)
    "h2h_home_wins", "h2h_away_wins", "h2h_draws", "h2h_home_goals_diff"
]
```

## Model Artifacts (.joblib)

Single-model artifact:
```python
{
    "model": trained_sklearn_model,
    "model_type": "RandomForest" | "LogisticRegression",
    "features": list_of_feature_names,
    "scaler": StandardScaler_or_None,  # Critical for LogisticRegression
    "imputation_medians": dict,
    "pct_thresholds": {80: 0.32, 85: 0.35, ...},
}
```

Ensemble artifact (RF+LR blend):
```python
{
    "model": rf_model,              # RandomForest (no scaling needed)
    "model_type": "Ensemble",
    "lr_model": lr_model,           # LogisticRegression
    "lr_scaler": StandardScaler,    # Scaler for LR only
    "ensemble_weight": 0.5,         # RF weight; LR = 1 - weight
    "scaler": None,                 # Ensemble handles scaling internally
    "features": list_of_feature_names,
    "imputation_medians": dict,
    "pct_thresholds": {80: 0.32, 85: 0.35, ...},
}
```

## Data Sources

- **Football-Data.co.uk**: Historical results, odds, match stats
- Files: `all-euro-data-YYYY-YYYY.xlsx` (bulk), `Latest_Results.xlsx` (recent)
- Fixtures: separate CSV with upcoming matches + odds
- Format: `Div, Date, Time, HomeTeam, AwayTeam, AvgH, AvgD, AvgA, ...`

## Known Issues & Gotchas

### League Code Mismatches
- `GI` (incremental_features) vs `G1` (predict script) for Greek league — need consistent mapping
- `USA` maps to `Matches_E0.xls` in historical data
- Some historical files use double underscore: `Matches__E0.xls`

### Column Name Variations
- Raw files: `Date`, `Home`, `Away`, `HG`, `AG`, `Res`
- Engineered files: `MatchDate`, `HomeTeam`, `AwayTeam`, `FTHome`, `FTAway`, `FTResult`
- `prep_new_season.py` handles the mapping

### Odds Columns
- Pre-match/opening: `AvgH`, `AvgD`, `AvgA`, `B365H`, `B365D`, `B365A`
- Closing (FORBIDDEN as features): `AvgCH`, `AvgCD`, `AvgCA`, `PSCH`, `PSCD`, `PSCA`
- V4 Lean does NOT use odds as model features at all — only for edge calculation post-prediction

### Season File Naming
- Feature files may be `LEAGUE_20252026_features.csv` or `LEAGUE_20252026.csv`
- The `find_season_file()` function in predict script must handle both patterns

### sklearn Version
- Models trained with sklearn 1.6.x
- LogisticRegression models may break on sklearn 1.8+ (serialization change)
- RandomForest/ExtraTrees models are stable across versions

## Coding Conventions

- Python 3.10+
- pandas for data manipulation, scikit-learn for ML
- EU CSV format for outputs: semicolon separator, decimal comma (for Google Sheets)
- All dates in dd/mm/yyyy format
- Feature engineering MUST be chronological — no future data leakage
- Walk-forward validation: train on past, test on each season sequentially
- FDR correction (Benjamini-Hochberg, q=0.10) for multiple testing

## Testing Priorities

When modifying code, always verify:
1. No data leakage — features only use information available before the match
2. Chronological ordering is maintained in feature engineering
3. Column mappings work for both raw and engineered file formats
4. Model loading handles missing features gracefully (imputation medians)
5. Scaler is applied when model requires it (LogisticRegression)
6. League code mappings are consistent across all scripts
7. Output directories are created before writing files

## Commands Reference

```bash
# Feature engineering — V4 Lean historical (21 features from raw data)
python scripts/feature_engineering_v4_lean.py --all --data-dir ./data --output-dir ./features

# Feature engineering — current season
python scripts/incremental_features.py --all --historical ./data --new-season ./20252026_season --output ./features_20252026

# Train V4 Lean — single league
python scripts/protocol_lean_v4_no_odds.py --analyze --data features/Matches_SP1_features.csv --output ./models

# Train V4 Lean — all leagues
python scripts/protocol_lean_v4_no_odds.py --batch --data-dir ./features --all --output ./models

# Predict fixtures
python scripts/predict_fixtures_1402.py --fixtures ./new_fixtures/fixtures.csv --models-dir ./models --season-dir ./features_20252026 --output ./predictions/predictions_DDMM.csv --eu-format

# Track bets
python scripts/bet_tracker_v2.py --input tracking/bets.csv --results predictions/
```

## Environment Setup

```bash
pip install pandas numpy scikit-learn joblib tqdm openpyxl xlrd
```

## Critical Rules

1. **NEVER use closing odds as model features** — this is the #1 leakage source
2. **NEVER append new season data to historical training sets** — keep temporal boundaries clean
3. **ALWAYS rebuild team states from scratch** when updating features (ELO is sequential)
4. **ALWAYS check FDR-corrected p-values** before deploying a new strategy
5. **Negative edge bets still get stakes** (0.80u) if strategy passed FDR validation — bookmaker margin means -2% edge might be +3% vs true probability
6. **LogisticRegression models NEED the scaler** — prediction without scaling gives garbage
7. **Ensemble artifacts** contain both RF and LR models — predict script handles blending automatically
