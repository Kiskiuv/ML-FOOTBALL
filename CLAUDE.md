# CLAUDE.md — ML Football Betting System

## Project Overview

Machine learning system for identifying profitable football (soccer) betting opportunities across European leagues. Uses walk-forward validation, FDR correction, and strict anti-leakage measures to find statistically significant edges against bookmaker odds.

**Owner:** Marc — Senior data scientist (ex-bookmaker analytics), based in Mataró, Catalonia.

## Architecture

### Protocol V5 Anti-Cherry-Pick (Active — 21 features, RF+LR+Ensemble)
- Training: `scripts/protocol_lean_v4_no_odds.py` (Protocol V3.0 internally)
- Prediction: `scripts/predict_fixtures_1402.py`
- Feature engineering: `scripts/feature_engineering_v4_lean.py` (historical) / `scripts/incremental_features.py` (current season)
- Comparison: `scripts/compare_hist_vs_season.py` (historical vs live validation)
- 13 hybrid-selected strategies (NO odds leakage, NO cherry-picking)
- 4 markets: HOME, DRAW, AWAY, UNDER25 (Under 2.5 goals)
- Only 3 strategies per market (STRICT/SELECTIVE/STANDARD at pct 95/90/85)
- All strategies require edge >= 0.00 and min_odds >= 1.90
- FDR q=0.05, ~456 total tests (38 leagues x 12 strategies)
- Features: BASIC+ tier only (ELO, form, schedule, momentum, H2H)
- Models: RandomForest + LogisticRegression + Ensemble (0.5/0.5 blend)
- Model selection: 3-way by avg test AUC (RF vs LR vs Ensemble)
- Historical (7-season walk-forward): 813 bets, +189.21u, +23.3% ROI
- 2025/26 season to date: 106 bets, +9.98u, +9.4% ROI

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
│   ├── protocol_lean_v4_no_odds.py     # Training: V5 anti-cherry-pick (RF+LR+Ensemble)
│   ├── predict_fixtures_1402.py        # Prediction: V5 strategies
│   ├── backtest_season.py             # Backtest V5 strategies on 2025/26 season
│   ├── compare_hist_vs_season.py      # Compare historical vs live performance
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

## V5 Strategies (13 active — anti-cherry-pick, 4 markets)

All strategies: edge >= 0.00, pct in {85, 90, 95}, min_odds >= 1.90.

```
League  Market    Strategy   Model                PCT  Edge  MinOdds  p-value    Tier         Hist ROI
MEX     AWAY      SELECTIVE  LogisticRegression   90   0.00  2.3      0.0099     DEPLOY       +28.1%
I2      UNDER25   STANDARD   Ensemble             85   0.00  1.9      0.0134     PAPER_TRADE  +41.6%
POL     DRAW      STRICT     LogisticRegression   95   0.00  3.0      0.0225     PAPER_TRADE  +40.1%
SP2     DRAW      STRICT     Ensemble             95   0.00  3.0      0.0262     PAPER_TRADE  +34.5%
N1      DRAW      STRICT     Ensemble             95   0.00  3.0      0.0288     PAPER_TRADE  +40.8%
RUS     AWAY      SELECTIVE  Ensemble             90   0.00  2.3      0.0345     PAPER_TRADE  +52.5%
FIN     AWAY      STANDARD   LogisticRegression   85   0.00  1.9      0.0378     PAPER_TRADE  +24.6%
P1      UNDER25   STANDARD   LogisticRegression   85   0.00  1.9      0.0420     PAPER_TRADE  +48.6%
SC0     HOME      STANDARD   RandomForest         85   0.00  1.9      0.0460     PAPER_TRADE  +67.8%
F1      HOME      SELECTIVE  LogisticRegression   90   0.00  2.0      0.0574     MONITOR      +44.8%
G1      DRAW      STRICT     Ensemble             95   0.00  3.0      0.0603     MONITOR      +23.8%
I1      DRAW      SELECTIVE  RandomForest         90   0.00  2.5      0.0612     MONITOR      +22.0%
B1      DRAW      STRICT     LogisticRegression   95   0.00  3.0      0.0791     MONITOR      +22.5%
```

### Strategy Search Space (per league)
```
DRAW:    STRICT (pct95, min3.0) | SELECTIVE (pct90, min2.5) | STANDARD (pct85, min2.0)
HOME:    STRICT (pct95, min2.5) | SELECTIVE (pct90, min2.0) | STANDARD (pct85, min1.9)
AWAY:    STRICT (pct95, min2.8) | SELECTIVE (pct90, min2.3) | STANDARD (pct85, min1.9)
UNDER25: STRICT (pct95, min2.3) | SELECTIVE (pct90, min2.0) | STANDARD (pct85, min1.9)
```

## Staking Rules

Quarter-Kelly criterion (DEPLOY tier only; PAPER_TRADE/MONITOR = 0u tracking):
```python
kelly = edge / (odds - 1)
stake = 0.25 * kelly * bankroll  # quarter-Kelly
stake = min(stake, 0.025 * bankroll)  # cap at 2.5% bankroll
stake = max(stake, 0.50)  # floor at 0.50u
```
- LOW: stake <= 0.9u
- MED: 0.9u < stake <= 1.5u
- HIGH: stake > 1.5u

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
    "pct_thresholds": {85: 0.32, 90: 0.38, 95: 0.45},
    "platt_scaler": CalibratedClassifierCV_or_None,
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
    "pct_thresholds": {85: 0.32, 90: 0.38, 95: 0.45},
    "platt_scaler": CalibratedClassifierCV_or_None,
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
- FDR correction (Benjamini-Hochberg, q=0.05) for multiple testing

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

# Train V5 — single league
python scripts/protocol_lean_v4_no_odds.py --analyze --data features/Matches_SP1_features.csv --output ./models

# Train V5 — all leagues
python scripts/protocol_lean_v4_no_odds.py --batch --data-dir ./features --all --output ./models

# Predict fixtures
python scripts/predict_fixtures_1402.py --fixtures ./new_fixtures/fixtures.csv --models-dir ./models --season-dir ./features_20252026 --output ./predictions/predictions_DDMM.csv --eu-format

# Compare historical vs live performance
python scripts/compare_hist_vs_season.py

# Backtest 2025/26 season
python scripts/backtest_season.py --models-dir ./models --season-dir ./features_20252026

# Track bets
python scripts/bet_tracker_v2.py --input tracking/bets.csv --results predictions/
```

## Environment Setup

```bash
pip install pandas numpy scikit-learn joblib tqdm openpyxl xlrd
```

## Strategy Lifecycle Rules

### Promotion (PAPER_TRADE → DEPLOY)
- 50+ live bets accumulated
- p < 0.10 on live data
- Positive live ROI

### Demotion (DEPLOY → PAPER_TRADE)
- Trailing 30-bet Sharpe ratio < -0.3, OR
- Cumulative loss exceeds 10u

### Retirement
- p > 0.15 after retraining on updated historical data

### Drawdown Rules (automatic, no discretion)
- **10u loss**: review strategy, check for market changes
- **15u loss**: reduce stakes by 30%
- **20u loss**: halt all betting for 2 weeks, run full diagnostics
- **30u loss**: stop strategy permanently, require re-evaluation from scratch

### Correlation Caps
- Max 5u total draw exposure per match day
- Max 3 correlated bets (same league or same market type) per match day

## Critical Rules

1. **NEVER use closing odds as model features** — this is the #1 leakage source
2. **NEVER append new season data to historical training sets** — keep temporal boundaries clean
3. **ALWAYS rebuild team states from scratch** when updating features (ELO is sequential)
4. **ALWAYS check FDR-corrected p-values** before deploying a new strategy
5. **All strategies require edge >= 0.00** — no negative-edge bets in V5
6. **LogisticRegression models NEED the scaler** — prediction without scaling gives garbage
7. **Ensemble artifacts** contain both RF and LR models — predict script handles blending automatically
8. **Platt calibration** — only use artifact-based scaler, never season-based (in-sample leakage)
9. **Quarter-Kelly staking** — never exceed 2.5% bankroll per bet
