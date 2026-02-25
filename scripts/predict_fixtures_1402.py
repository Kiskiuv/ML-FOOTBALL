#!/usr/bin/env python3
"""
PROTOCOL V4 — FIXTURE PREDICTOR
================================
Reads a fixtures CSV (all leagues together), applies the 14 hybrid-selected
strategies (NO odds leakage — 21 features), and outputs a CSV with BET / NO BET.

Usage:
    python predict_v3.py --fixtures fixtures.csv --models-dir ./models --season-dir ./seasons

    --fixtures    : CSV with upcoming matches (football-data.co.uk format)
    --models-dir  : Directory containing model_<LEAGUE>_<MARKET>.joblib files
    --season-dir  : Directory containing <LEAGUE>_20252026_features.csv files
    --output      : Output CSV path (default: predictions_v3.csv)
    --eu-format   : Use ; separator and , decimals for Excel EU

Author: Marc | Protocol V3 | February 2026
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from platt import fit_platt_scaler, apply_platt


# =============================================================================
# V4 HYBRID-SELECTED STRATEGIES (14 active — NO odds leakage)
# Trained with 21 features (ELO, form, schedule, momentum, H2H only)
# Backtest 2024/25: 89 bets, +16.51u, +18.6% ROI
# =============================================================================

V3_STRATEGIES = [
    # (league, market, strategy_name, model_type, pct, edge, min_odds, p_value, fdr, hist_roi, tier)
    # DEPLOY — FDR-pass or strong p-value, real stakes
    ("MEX", "AWAY",  "CONSERVATIVE",    "LogisticRegression", 88,  0.00, 2.5,  0.0056, True,  34.1, "DEPLOY"),
    ("FIN", "AWAY",  "SELECTIVE",        "LogisticRegression", 85, -0.01, 2.3,  0.0078, True,  44.2, "DEPLOY"),
    ("D2",  "AWAY",  "CONSERVATIVE",    "RandomForest",       88,  0.00, 2.5,  0.0495, False, 34.1, "DEPLOY"),
    # PAPER_TRADE — p < 0.07, tracked at stake=0
    ("F1",  "HOME",  "UPSET",           "LogisticRegression", 85, -0.01, 2.0,  0.0587, False, 30.2, "PAPER_TRADE"),
    ("I1",  "DRAW",  "CONSERVATIVE",    "RandomForest",       90,  0.00, 3.0,  0.0612, False, 22.0, "PAPER_TRADE"),
    ("F2",  "HOME",  "ULTRA_CONS",      "RandomForest",       90,  0.02, 2.5,  0.0613, False, 77.8, "PAPER_TRADE"),
    ("SP2", "DRAW",  "SELECTIVE",        "LogisticRegression", 88, -0.01, 3.0,  0.0637, False, 17.4, "PAPER_TRADE"),
    ("NOR", "HOME",  "STANDARD",        "LogisticRegression", 80, -0.02, 1.9,  0.0640, False, 26.0, "PAPER_TRADE"),
    ("G1",  "DRAW",  "LONGSHOT_STRICT", "LogisticRegression", 92,  0.00, 3.2,  0.0677, False, 30.4, "PAPER_TRADE"),
    ("F2",  "AWAY",  "ULTRA_CONS",      "LogisticRegression", 90,  0.02, 2.8,  0.0697, False, 35.2, "PAPER_TRADE"),
    # MONITOR — p < 0.10, tracked at stake=0
    ("D1",  "AWAY",  "SELECTIVE",        "LogisticRegression", 85, -0.01, 2.3,  0.0736, False, 27.1, "MONITOR"),
    ("ARG", "HOME",  "ULTRA_CONS",      "LogisticRegression", 90,  0.02, 2.5,  0.0880, False, 71.8, "MONITOR"),
    ("N1",  "AWAY",  "STANDARD_HIGH",   "LogisticRegression", 82,  0.00, 2.2,  0.0888, False, 31.1, "MONITOR"),
    ("SWE", "HOME",  "SELECTIVE",        "LogisticRegression", 82, -0.01, 2.0,  0.0926, False, 27.6, "MONITOR"),
]

# Quick lookup: (league, market) → strategy dict
STRATEGY_LOOKUP = {}
for s in V3_STRATEGIES:
    key = (s[0], s[1])
    STRATEGY_LOOKUP[key] = {
        "league": s[0], "market": s[1], "name": s[2], "model_type": s[3],
        "pct": s[4], "edge": s[5], "min_odds": s[6],
        "p_value": s[7], "fdr": s[8], "hist_roi": s[9], "tier": s[10],
    }

ACTIVE_LEAGUES = sorted(set(s[0] for s in V3_STRATEGIES))

ODDS_COL = {"HOME": "OddHome", "DRAW": "OddDraw", "AWAY": "OddAway"}
MAX_ODDS_COL = {"HOME": "MaxHome", "DRAW": "MaxDraw", "AWAY": "MaxAway"}


# =============================================================================
# STAKING (Protocol V3.4)
# =============================================================================

def get_stake(edge: float) -> Tuple[float, str]:
    """Returns (stake, tier) based on edge."""
    if edge < 0:
        return 0.80, "LOW"
    elif edge < 0.05:
        return 1.25, "MED"
    else:
        return 1.85, "HIGH"


# =============================================================================
# FILE UTILITIES
# =============================================================================

def load_csv(path: str) -> pd.DataFrame:
    for enc in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Cannot read: {path}")


def find_season_file(season_dir: str, league: str) -> Optional[str]:
    d = Path(season_dir)
    if not d.exists():
        return None
    patterns = [
        f"{league}_20252026_features.csv",
        f"{league.upper()}_20252026_features.csv",
        f"{league}_20252026.csv",
        f"{league.upper()}_20252026.csv",
        f"Matches_{league}_features.csv",
        f"{league}_features.csv",
    ]
    for p in patterns:
        if (d / p).exists():
            return str(d / p)
    for f in d.glob(f"*{league}*features*.csv"):
        return str(f)
    return None


def load_model(models_dir: str, league: str, market: str) -> Optional[Dict]:
    for p in [f"model_{league}_{market}.joblib", f"model_{league.upper()}_{market}.joblib"]:
        path = Path(models_dir) / p
        if path.exists():
            return joblib.load(path)
    return None


# =============================================================================
# FEATURE BUILDING (same as predict_all_v8_final.py)
# =============================================================================

def get_team_last_match(season_df: pd.DataFrame, team: str):
    home = season_df[season_df['HomeTeam'] == team].copy()
    away = season_df[season_df['AwayTeam'] == team].copy()
    home['_home'] = True
    away['_home'] = False
    all_m = pd.concat([home, away])
    if len(all_m) == 0:
        return None, None
    all_m['_date'] = pd.to_datetime(all_m['MatchDate'], errors='coerce')
    all_m = all_m.sort_values('_date')
    return all_m.iloc[-1], all_m.iloc[-1]['_home']


def extract_team_features(row, played_home: bool) -> dict:
    if played_home:
        return {
            'elo': row.get('HomeElo', 1500), 'form3': row.get('Form3Home', 4),
            'form5': row.get('Form5Home', 7),
            'form3_ratio': row.get('Form3Home_ratio', 0.44),
            'form5_ratio': row.get('Form5Home_ratio', 0.47),
            'ppg': row.get('home_team_home_ppg', 1.5),
            'goals_scored': row.get('home_team_goals_scored_avg', 1.3),
            'goals_conceded': row.get('home_team_goals_conceded_avg', 1.2),
            'shots': row.get('home_shots_avg', 12),
            'shots_against': row.get('home_shots_against_avg', 11),
            'last_date': row.get('MatchDate', ''),
        }
    else:
        return {
            'elo': row.get('AwayElo', 1500), 'form3': row.get('Form3Away', 4),
            'form5': row.get('Form5Away', 7),
            'form3_ratio': row.get('Form3Away_ratio', 0.44),
            'form5_ratio': row.get('Form5Away_ratio', 0.47),
            'ppg': row.get('away_team_away_ppg', 1.0),
            'goals_scored': row.get('away_team_goals_scored_avg', 1.2),
            'goals_conceded': row.get('away_team_goals_conceded_avg', 1.4),
            'shots': row.get('away_shots_avg', 11),
            'shots_against': row.get('away_shots_against_avg', 12),
            'last_date': row.get('MatchDate', ''),
        }


def compute_rest_days(last_date: str, fixture_date: str) -> int:
    try:
        last = pd.to_datetime(last_date, errors='coerce')
        fix = pd.to_datetime(fixture_date, format='%d/%m/%Y', errors='coerce')
        if pd.isna(fix):
            fix = pd.to_datetime(fixture_date, errors='coerce')
        if pd.notna(last) and pd.notna(fix):
            return max(1, (fix - last).days)
    except Exception:
        pass
    return 7


def build_instance(fixture, season_df: pd.DataFrame, league: str) -> dict:
    """Build prediction feature vector for a single fixture."""
    home_team = fixture['HomeTeam']
    away_team = fixture['AwayTeam']
    fix_date = fixture.get('Date', '')

    home_last, home_was_home = get_team_last_match(season_df, home_team)
    away_last, away_was_home = get_team_last_match(season_df, away_team)

    defaults_home = {
        'elo': 1500, 'form3': 4, 'form5': 7, 'form3_ratio': 0.44,
        'form5_ratio': 0.47, 'ppg': 1.5, 'goals_scored': 1.3,
        'goals_conceded': 1.2, 'shots': 12, 'shots_against': 11, 'last_date': ''
    }
    defaults_away = {
        'elo': 1500, 'form3': 4, 'form5': 7, 'form3_ratio': 0.44,
        'form5_ratio': 0.47, 'ppg': 1.0, 'goals_scored': 1.2,
        'goals_conceded': 1.4, 'shots': 11, 'shots_against': 12, 'last_date': ''
    }

    hf = extract_team_features(home_last, home_was_home) if home_last is not None else defaults_home
    af = extract_team_features(away_last, away_was_home) if away_last is not None else defaults_away

    home_rest = compute_rest_days(hf['last_date'], fix_date)
    away_rest = compute_rest_days(af['last_date'], fix_date)

    oh = float(fixture.get('AvgH') or fixture.get('B365H') or 2.5)
    od = float(fixture.get('AvgD') or fixture.get('B365D') or 3.3)
    oa = float(fixture.get('AvgA') or fixture.get('B365A') or 2.8)

    elo_diff = hf['elo'] - af['elo']

    return {
        'Division': league, 'MatchDate': fix_date,
        'MatchTime': fixture.get('Time', ''),
        'HomeTeam': home_team, 'AwayTeam': away_team,
        # Odds (kept for edge calculation, NOT used as model features)
        'OddHome': oh, 'OddDraw': od, 'OddAway': oa,
        'MaxHome': float(fixture.get('MaxH') or oh),
        'MaxDraw': float(fixture.get('MaxD') or od),
        'MaxAway': float(fixture.get('MaxA') or oa),
        # ELO (5)
        'HomeElo': hf['elo'], 'AwayElo': af['elo'],
        'elo_diff': elo_diff, 'elo_sum': hf['elo'] + af['elo'],
        'elo_mismatch': abs(elo_diff) / (hf['elo'] + af['elo']),
        'elo_mismatch_flag': 1 if abs(elo_diff) > 150 else 0,
        # Form (6)
        'Form3Home': hf['form3'], 'Form5Home': hf['form5'],
        'Form3Away': af['form3'], 'Form5Away': af['form5'],
        'Form3Home_ratio': hf['form3_ratio'], 'Form5Home_ratio': hf['form5_ratio'],
        'Form3Away_ratio': af['form3_ratio'], 'Form5Away_ratio': af['form5_ratio'],
        'form_diff_3': hf['form3'] - af['form3'],
        'form_diff_5': hf['form5'] - af['form5'],
        'form_momentum_home': 0, 'form_momentum_away': 0, 'form_momentum': 0,
        # Rest
        'home_rest_days': home_rest, 'away_rest_days': away_rest,
        'rest_diff': home_rest - away_rest,
        'rest_diff_nonlinear': (home_rest - away_rest)**2 * np.sign(home_rest - away_rest),
        'home_congestion_14d': 2, 'away_congestion_14d': 2,
        'is_midweek': 0, 'is_early_kickoff': 0, 'is_evening_match': 0,
        'winter_period': 1, 'season_phase': 2, 'match_week': 25,
        # H2H
        'h2h_home_wins': 0, 'h2h_away_wins': 0, 'h2h_draws': 0,
        'h2h_home_goals_diff': 0, 'h2h_home_venue_record': 0.5, 'h2h_matches_count': 0,
        # Team stats
        'home_team_home_ppg': hf['ppg'],
        'home_team_goals_scored_avg': hf['goals_scored'],
        'home_team_goals_conceded_avg': hf['goals_conceded'],
        'away_team_away_ppg': af['ppg'],
        'away_team_goals_scored_avg': af['goals_scored'],
        'away_team_goals_conceded_avg': af['goals_conceded'],
        'venue_ppg_diff': hf['ppg'] - af['ppg'],
        # Shots
        'home_shots_avg': hf['shots'], 'home_shots_against_avg': hf['shots_against'],
        'away_shots_avg': af['shots'], 'away_shots_against_avg': af['shots_against'],
        'shots_dominance_home': 0, 'shots_dominance_away': 0,
        'corners_diff': 0, 'discipline_diff': 0,
        # Targets (unused for prediction)
        'target_home': 0, 'target_draw': 0, 'target_away': 0,
    }


# =============================================================================
# PREDICTION ENGINE
# =============================================================================

def _predict_raw_probs(model_data, X, features):
    """Get raw probabilities from model artifact (handles Ensemble and single-model)."""
    model = model_data['model']
    model_type = model_data.get('model_type', '')
    scaler = model_data.get('scaler', None)

    if model_type == 'Ensemble':
        lr_model = model_data['lr_model']
        lr_scaler = model_data.get('lr_scaler')
        w = model_data.get('ensemble_weight', 0.5)
        rf_probs = model.predict_proba(X)[:, 1]
        X_scaled = lr_scaler.transform(X) if lr_scaler is not None else X
        lr_probs = lr_model.predict_proba(X_scaled)[:, 1]
        return w * rf_probs + (1 - w) * lr_probs
    else:
        if scaler is not None:
            X = scaler.transform(X)
        return model.predict_proba(X)[:, 1]


def predict_market(league: str, market: str, strat: dict,
                   fixtures_df: pd.DataFrame, season_df: pd.DataFrame,
                   model_data: dict) -> pd.DataFrame:
    """
    Run prediction for one league-market using saved model.
    Returns DataFrame with one row per fixture and bet decision.

    Platt calibration priority: artifact scaler > season-based scaler > no calibration.
    Raw probs used for percentile threshold (preserves validated strategy behavior).
    Calibrated probs used for Edge calculation and display.
    """
    model = model_data['model']
    features = model_data.get('features', None)
    medians = model_data.get('imputation_medians', {})
    pct_thresholds = model_data.get('pct_thresholds', {})
    model_type = model_data.get('model_type', '')

    if features is None:
        if hasattr(model, 'feature_names_in_'):
            features = list(model.feature_names_in_)
        else:
            raise ValueError(f"No features found in model for {league} {market}")

    # Build feature matrix for fixtures
    X_df = pd.DataFrame()
    for f in features:
        if f in fixtures_df.columns:
            X_df[f] = fixtures_df[f].copy()
        elif f in medians:
            X_df[f] = medians[f]
        else:
            X_df[f] = 0
    for f in features:
        X_df[f] = X_df[f].fillna(medians.get(f, 0))
    X = X_df.values

    # Raw probabilities for fixtures
    probs = _predict_raw_probs(model_data, X, features)

    # --- Always compute season predictions (needed for threshold fallback + Platt) ---
    target_col = f"target_{market.lower()}"
    X_hist = pd.DataFrame()
    for f in features:
        if f in season_df.columns:
            X_hist[f] = season_df[f].fillna(medians.get(f, 0))
        else:
            X_hist[f] = medians.get(f, 0)
    X_hist_vals = X_hist.values
    season_probs = _predict_raw_probs(model_data, X_hist_vals, features)

    # Get percentile threshold (uses raw probs — preserves validated strategy)
    pct = strat['pct']
    if pct in pct_thresholds:
        threshold = pct_thresholds[pct]
        threshold_src = "saved"
    else:
        threshold = np.percentile(season_probs, pct)
        threshold_src = "historical"

    # --- Platt calibration ---
    # Priority: artifact scaler > season-based scaler > no calibration
    platt_model = model_data.get('platt_scaler', None)
    platt_src = 'none'

    if platt_model is not None:
        platt_src = 'artifact'
    elif target_col in season_df.columns:
        season_labels = season_df[target_col].values
        platt_model = fit_platt_scaler(season_probs, season_labels)
        if platt_model is not None:
            platt_src = 'season'

    cal_probs = apply_platt(platt_model, probs)

    # Build results
    odds_col = ODDS_COL[market]
    result = fixtures_df[['Division', 'MatchDate', 'MatchTime', 'HomeTeam', 'AwayTeam',
                           'OddHome', 'OddDraw', 'OddAway', 'HomeElo', 'AwayElo',
                           'elo_diff', 'form_diff_5']].copy()

    result['League'] = league
    result['Market'] = market
    result['Strategy'] = strat['name']
    result['Model'] = model_type or strat['model_type']
    result['Prob'] = probs
    result['Cal_Prob'] = cal_probs
    result['Platt_Src'] = platt_src
    result['Threshold'] = threshold
    result['Threshold_Src'] = threshold_src
    result['Odds'] = fixtures_df[odds_col]
    result['Implied_Prob'] = 1 / result['Odds']
    result['Raw_Edge'] = result['Prob'] - result['Implied_Prob']
    result['Edge'] = result['Cal_Prob'] - result['Implied_Prob']

    # Validity check (on raw probs — same as before)
    result['Prob_Valid'] = (probs > 0.01) & (probs < 0.99)

    # BET criteria — both Pass_Prob and Pass_Edge use RAW values
    # (preserves validated strategy behavior; calibrated edge is for display/staking)
    result['Pass_Prob'] = result['Prob'] >= threshold
    result['Pass_Edge'] = result['Raw_Edge'] >= strat['edge']
    result['Pass_Odds'] = result['Odds'] >= strat['min_odds']
    result['Pass_Valid'] = result['Prob_Valid']

    result['BET'] = (
        result['Pass_Prob'] &
        result['Pass_Edge'] &
        result['Pass_Odds'] &
        result['Pass_Valid']
    )

    # Tier
    strategy_tier = strat.get('tier', 'MONITOR')
    result['Tier'] = strategy_tier

    # Staking — uses calibrated Edge
    result['Stake'] = 0.0
    result['Stake_Tier'] = ''
    for idx in result.index:
        if result.loc[idx, 'BET']:
            stake, tier = get_stake(result.loc[idx, 'Edge'])
            if strategy_tier == 'DEPLOY':
                result.loc[idx, 'Stake'] = stake
                result.loc[idx, 'Stake_Tier'] = tier
            else:
                result.loc[idx, 'Stake'] = 0.0
                result.loc[idx, 'Stake_Tier'] = f'{tier}(PAPER)'

    # Strategy metadata
    result['p_value'] = strat['p_value']
    result['FDR'] = '✅' if strat['fdr'] else '❌'
    result['Hist_ROI'] = strat['hist_roi']

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Protocol V3 — Predict fixtures using 15 hybrid strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_v3.py --fixtures fixtures.csv --models-dir ./models --season-dir ./seasons
  python predict_v3.py --fixtures fixtures.csv --models-dir ./models --season-dir ./seasons --eu-format
        """
    )
    parser.add_argument("--fixtures", required=True, help="Fixtures CSV (football-data.co.uk format)")
    parser.add_argument("--models-dir", required=True, help="Directory with model_<LEAGUE>_<MARKET>.joblib")
    parser.add_argument("--season-dir", required=True, help="Directory with <LEAGUE>_20252026_features.csv")
    parser.add_argument("--output", default="predictions_v3.csv", help="Output CSV path")
    parser.add_argument("--eu-format", action="store_true", help="EU format (;  separator, , decimals)")
    parser.add_argument("--deploy-only", action="store_true", help="Only output DEPLOY tier strategies")

    args = parser.parse_args()

    print("=" * 70)
    print("PROTOCOL V4 — FIXTURE PREDICTOR (NO ODDS LEAKAGE)")
    n_deploy = sum(1 for s in V3_STRATEGIES if s[10] == 'DEPLOY')
    n_paper = sum(1 for s in V3_STRATEGIES if s[10] == 'PAPER_TRADE')
    n_monitor = sum(1 for s in V3_STRATEGIES if s[10] == 'MONITOR')
    print(f"{len(V3_STRATEGIES)} strategies: {n_deploy} DEPLOY | {n_paper} PAPER_TRADE | {n_monitor} MONITOR")
    if args.deploy_only:
        print(">> --deploy-only: only DEPLOY tier strategies will be output")
    print("=" * 70)

    # Load fixtures
    raw = load_csv(args.fixtures)
    print(f"\n📋 Loaded {len(raw)} fixtures from {args.fixtures}")
    print(f"   Leagues in file: {', '.join(sorted(raw['Div'].unique()))}")

    # Find which leagues have active strategies
    fixture_leagues = set(raw['Div'].unique())
    active_in_file = []
    inactive_leagues = []

    for league in ACTIVE_LEAGUES:
        if league in fixture_leagues:
            # Check which markets are active for this league
            markets = [s[1] for s in V3_STRATEGIES if s[0] == league]
            active_in_file.append((league, markets))
        else:
            inactive_leagues.append(league)

    # Also note fixture leagues with NO active strategy
    no_strategy = sorted(fixture_leagues - set(ACTIVE_LEAGUES))

    print(f"\n   ✅ Active V3 leagues in fixtures: {len(active_in_file)}")
    for league, markets in active_in_file:
        n_fix = len(raw[raw['Div'] == league])
        print(f"      {league}: {n_fix} fixtures → {', '.join(markets)}")

    if inactive_leagues:
        print(f"\n   ⏳ V3 leagues NOT in fixtures: {', '.join(inactive_leagues)}")

    if no_strategy:
        print(f"\n   ⊘  Fixture leagues with NO V3 strategy (skipped): {', '.join(no_strategy)}")

    if not active_in_file:
        print("\n❌ No overlap between fixture leagues and V3 strategies. Nothing to predict.")
        print(f"   V3 needs: {', '.join(ACTIVE_LEAGUES)}")
        print(f"   Fixtures have: {', '.join(sorted(fixture_leagues))}")
        sys.exit(0)

    # Load season data
    print(f"\n📂 Loading season data from {args.season_dir}")
    season_data = {}
    for league, _ in active_in_file:
        sf = find_season_file(args.season_dir, league)
        if sf:
            season_data[league] = load_csv(sf)
            print(f"   {league}: {len(season_data[league])} historical matches")
        else:
            print(f"   {league}: ❌ No season file found — skipping")

    # Process predictions
    print(f"\n🔮 Running predictions...")
    all_results = []

    for league, markets in active_in_file:
        if league not in season_data:
            continue

        # Build feature instances for this league's fixtures
        league_fixtures = raw[raw['Div'] == league]
        season_df = season_data[league]

        instances = []
        for _, fix in league_fixtures.iterrows():
            instances.append(build_instance(fix, season_df, league))
        fixtures_df = pd.DataFrame(instances)

        for market in markets:
            strat = STRATEGY_LOOKUP[(league, market)]
            tier = strat.get('tier', 'MONITOR')

            # Skip non-DEPLOY if --deploy-only
            if args.deploy_only and tier != 'DEPLOY':
                continue

            # Load model
            model_data = load_model(args.models_dir, league, market)
            if not model_data:
                print(f"   {league} {market}: No model file")
                continue

            try:
                preds = predict_market(league, market, strat, fixtures_df, season_df, model_data)
                n_bets = preds['BET'].sum()
                n_total = len(preds)

                tier_tag = f"[{tier}]"
                marker = ">>>" if (n_bets > 0 and tier == 'DEPLOY') else ("..." if n_bets > 0 else "   ")
                print(f"   {marker} {league:>4} {market:<5} {strat['name']:<18} {tier_tag:<14} "
                      f"-> {n_total} fixtures, {n_bets} BET(s)"
                      f"  [p={strat['p_value']:.4f} {'FDR+' if strat['fdr'] else ''} "
                      f"hist={strat['hist_roi']:+.1f}%]")

                if n_bets > 0:
                    bets = preds[preds['BET']]
                    for _, b in bets.iterrows():
                        stake_str = f"{b['Stake']:.2f}u" if b['Stake'] > 0 else "PAPER"
                        platt_tag = f"[{b['Platt_Src']}]" if b.get('Platt_Src', 'none') != 'none' else ""
                        print(f"         -> {b['HomeTeam']} vs {b['AwayTeam']}  "
                              f"odds={b['Odds']:.2f}  raw={b['Prob']:.1%}  "
                              f"cal={b['Cal_Prob']:.1%}  "
                              f"edge={b['Edge']:+.1%} (raw={b['Raw_Edge']:+.1%})  "
                              f"stake={stake_str} {platt_tag}")

                all_results.append(preds)

            except Exception as e:
                print(f"   {league} {market}: ERROR -- {e}")

    if not all_results:
        print("\n❌ No predictions generated.")
        print("   Check that model files exist and season files match strategy leagues.")
        sys.exit(0)

    # Combine all results
    df = pd.concat(all_results, ignore_index=True)

    # Sort: BETs first, then by edge descending
    df['_sort'] = df['BET'].astype(int) * -1
    df = df.sort_values(['_sort', 'Edge'], ascending=[True, False]).drop(columns=['_sort'])

    # Output columns
    out_cols = [
        'MatchDate', 'MatchTime', 'League', 'HomeTeam', 'AwayTeam',
        'Market', 'Strategy', 'Model', 'Tier',
        'OddHome', 'OddDraw', 'OddAway', 'Odds',
        'Prob', 'Cal_Prob', 'Platt_Src', 'Threshold', 'Implied_Prob',
        'Edge', 'Raw_Edge',
        'Pass_Prob', 'Pass_Edge', 'Pass_Odds', 'Pass_Valid',
        'BET', 'Stake', 'Stake_Tier',
        'HomeElo', 'AwayElo', 'elo_diff', 'form_diff_5',
        'p_value', 'FDR', 'Hist_ROI',
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    output = df[out_cols].copy()

    # Add human-readable decision
    output.insert(output.columns.get_loc('BET') + 1, 'Decision',
                  output['BET'].map({True: '✅ BET', False: '— SKIP'}))

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.eu_format:
        eu = output.copy()
        for col in eu.select_dtypes(include=[np.number]).columns:
            eu[col] = eu[col].apply(
                lambda x: str(round(x, 4)).replace('.', ',') if pd.notna(x) else ''
            )
        eu.to_csv(out_path, sep=';', index=False)
        print(f"\n   💾 CSV saved (EU format): {out_path}")
    else:
        output.to_csv(out_path, index=False)
        print(f"\n   💾 CSV saved: {out_path}")
    
    print(f"   Rows: {len(output)} | Size: {out_path.stat().st_size / 1024:.1f} KB")

    # Summary
    n_bets = output['BET'].sum()
    n_total = len(output)

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {n_total} fixtures analyzed | {n_bets} BET | {n_total - n_bets} SKIP")
    print(f"{'=' * 70}")

    if n_bets > 0:
        bets_df = output[output['BET'] == True]
        deploy_bets = bets_df[bets_df['Tier'] == 'DEPLOY'] if 'Tier' in bets_df.columns else bets_df
        total_stake = deploy_bets['Stake'].sum()

        print(f"\nBET SUMMARY:")
        print(f"   Total bets: {n_bets} ({len(deploy_bets)} DEPLOY, {n_bets - len(deploy_bets)} PAPER)")
        print(f"   DEPLOY stake: {total_stake:.2f}u")
        if len(deploy_bets) > 0:
            print(f"   DEPLOY avg odds: {deploy_bets['Odds'].mean():.2f}")
            print(f"   DEPLOY avg edge (cal): {deploy_bets['Edge'].mean():+.1%}")
            if 'Raw_Edge' in deploy_bets.columns:
                print(f"   DEPLOY avg edge (raw): {deploy_bets['Raw_Edge'].mean():+.1%}")

        print(f"\n   {'Date':<12} {'Lg':<4} {'Mkt':<5} {'Home':<16} {'Away':<16} "
              f"{'Odds':>5} {'Raw':>6} {'Cal':>6} {'Edge':>6} {'RawE':>6} {'Stake':>6} {'Tier':<12}")
        print(f"   {'-' * 110}")
        for _, b in bets_df.iterrows():
            tier_label = b.get('Tier', '?')
            stake_str = f"{b['Stake']:.2f}u" if b['Stake'] > 0 else "PAPER"
            print(f"   {str(b['MatchDate']):<12} {b['League']:<4} {b['Market']:<5} "
                  f"{str(b['HomeTeam'])[:15]:<16} {str(b['AwayTeam'])[:15]:<16} "
                  f"{b['Odds']:>5.2f} {b['Prob']:>5.1%} {b.get('Cal_Prob', b['Prob']):>5.1%} "
                  f"{b['Edge']:>+5.1%} {b.get('Raw_Edge', b['Edge']):>+5.1%} "
                  f"{stake_str:>6} {tier_label:<12}")
    else:
        print("\n   No bets found for these fixtures.")
        print("   This is normal -- the protocol is highly selective (~8% of fixtures get bets).")

    print(f"\n💾 Saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
