#!/usr/bin/env python3
"""
BACKTEST V4 STRATEGIES — 2025/26 SEASON
========================================
Runs the 16 new V4 strategies against all played matches this season
using pre-computed features from incremental_features.py.
Tracks wins, losses, P&L per strategy and overall.

Usage:
    python scripts/backtest_season.py --models-dir ./models --season-dir ./features_20252026
    python scripts/backtest_season.py --models-dir ./models --season-dir ./features_20252026 --output predictions/backtest_2526.csv --eu-format
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


# =============================================================================
# V4 STRATEGIES (16 active)
# =============================================================================

V4_STRATEGIES = [
    # (league, market, strategy_name, model_type, pct, edge, min_odds, p_value, fdr, hist_roi, tier)
    # DEPLOY — FDR-pass or strong p-value, real stakes
    ("MEX", "AWAY",  "VALUE",            "Ensemble",             80, -0.02, 2.5,  0.0108, True,  24.8, "DEPLOY"),
    ("FIN", "AWAY",  "SELECTIVE",         "LogisticRegression",   85, -0.01, 2.3,  0.0078, True,  44.2, "DEPLOY"),
    ("G1",  "DRAW",  "MODERATE",          "Ensemble",             85, -0.02, 3.0,  0.0300, False, 19.2, "DEPLOY"),
    # PAPER_TRADE — p < 0.07, tracked at stake=0
    ("F2",  "AWAY",  "ULTRA_CONS",        "Ensemble",             90,  0.02, 2.8,  0.0431, False, 47.3, "PAPER_TRADE"),
    ("RUS", "AWAY",  "ULTRA_CONS",        "Ensemble",             90,  0.02, 2.8,  0.0500, False, 75.0, "PAPER_TRADE"),
    ("SP2", "DRAW",  "CONSERVATIVE",      "Ensemble",             90,  0.00, 3.0,  0.0507, False, 20.3, "PAPER_TRADE"),
    ("N1",  "DRAW",  "MODERATE_HIGH",     "Ensemble",             85,  0.00, 2.8,  0.0520, False, 21.7, "PAPER_TRADE"),
    ("F1",  "HOME",  "UPSET",            "LogisticRegression",   85, -0.01, 2.0,  0.0587, False, 30.2, "PAPER_TRADE"),
    ("AUT", "DRAW",  "LONGSHOT_STRICT",   "Ensemble",             92,  0.00, 3.2,  0.0597, False, 31.1, "PAPER_TRADE"),
    ("I1",  "DRAW",  "CONSERVATIVE",     "RandomForest",         90,  0.00, 3.0,  0.0612, False, 22.0, "PAPER_TRADE"),
    # MONITOR — p < 0.10, tracked at stake=0
    ("F2",  "HOME",  "ULTRA_CONS",       "RandomForest",         90,  0.02, 2.5,  0.0613, False, 77.8, "MONITOR"),
    ("NOR", "HOME",  "STANDARD",         "LogisticRegression",   80, -0.02, 1.9,  0.0640, False, 26.0, "MONITOR"),
    ("D1",  "AWAY",  "SELECTIVE",         "LogisticRegression",   85, -0.01, 2.3,  0.0736, False, 27.1, "MONITOR"),
    ("ARG", "HOME",  "ULTRA_CONS",       "LogisticRegression",   90,  0.02, 2.5,  0.0880, False, 71.8, "MONITOR"),
    ("N1",  "AWAY",  "STANDARD_HIGH",    "LogisticRegression",   82,  0.00, 2.2,  0.0888, False, 31.1, "MONITOR"),
    ("SWE", "HOME",  "SELECTIVE",         "LogisticRegression",   82, -0.01, 2.0,  0.0926, False, 27.6, "MONITOR"),
]

TIER_ORDER = {"DEPLOY": 0, "PAPER_TRADE": 1, "MONITOR": 2}

ODDS_COL = {"HOME": "OddHome", "DRAW": "OddDraw", "AWAY": "OddAway"}
RESULT_MAP = {"HOME": "H", "DRAW": "D", "AWAY": "A"}

# Season file name overrides (league code -> filename prefix)
SEASON_FILE_MAP = {"MEX": "MEXICO", "ARG": "ARGENTINA"}


def get_stake(edge: float) -> Tuple[float, str]:
    if edge < 0:
        return 0.80, "LOW"
    elif edge < 0.05:
        return 1.25, "MED"
    else:
        return 1.85, "HIGH"


def load_csv(path: str) -> pd.DataFrame:
    for enc in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"Cannot read: {path}")


def load_model(models_dir: str, league: str, market: str) -> Optional[Dict]:
    for p in [f"model_{league}_{market}.joblib", f"model_{league.upper()}_{market}.joblib"]:
        path = Path(models_dir) / p
        if path.exists():
            return joblib.load(path)
    return None


def backtest_strategy(strat_tuple, season_df, model_data):
    """Run one strategy against all played matches in season data."""
    league, market, name, model_type_strat, pct, min_edge, min_odds, p_val, fdr, hist_roi, tier = strat_tuple

    model = model_data['model']
    features = model_data.get('features', None)
    scaler = model_data.get('scaler', None)
    medians = model_data.get('imputation_medians', {})
    pct_thresholds = model_data.get('pct_thresholds', {})
    model_type = model_data.get('model_type', '')

    if features is None:
        if hasattr(model, 'feature_names_in_'):
            features = list(model.feature_names_in_)
        else:
            return pd.DataFrame()

    # Build feature matrix from season data
    X_df = pd.DataFrame()
    for f in features:
        if f in season_df.columns:
            X_df[f] = season_df[f].copy()
        elif f in medians:
            X_df[f] = medians[f]
        else:
            X_df[f] = 0

    for f in features:
        X_df[f] = X_df[f].fillna(medians.get(f, 0))

    X = X_df.values

    # Predict
    if model_type == 'Ensemble':
        lr_model = model_data['lr_model']
        lr_scaler = model_data.get('lr_scaler')
        w = model_data.get('ensemble_weight', 0.5)
        rf_probs = model.predict_proba(X)[:, 1]
        X_scaled = lr_scaler.transform(X) if lr_scaler is not None else X
        lr_probs = lr_model.predict_proba(X_scaled)[:, 1]
        probs = w * rf_probs + (1 - w) * lr_probs
    else:
        if scaler is not None:
            X = scaler.transform(X)
        probs = model.predict_proba(X)[:, 1]

    # Threshold
    if pct in pct_thresholds:
        threshold = pct_thresholds[pct]
    else:
        threshold = np.percentile(probs, pct)

    # Odds column
    odds_col = ODDS_COL[market]
    odds = season_df[odds_col].values
    implied_prob = 1.0 / odds
    edge = probs - implied_prob

    # Bet criteria
    pass_prob = probs >= threshold
    pass_edge = edge >= min_edge
    pass_odds = odds >= min_odds
    pass_valid = (probs > 0.01) & (probs < 0.99)
    is_bet = pass_prob & pass_edge & pass_odds & pass_valid

    # Actual results
    result_char = RESULT_MAP[market]
    actual_win = (season_df['FTResult'] == result_char).values

    # Build results dataframe
    result = season_df[['MatchDate', 'HomeTeam', 'AwayTeam']].copy()
    result['League'] = league
    result['Market'] = market
    result['Strategy'] = name
    result['Model'] = model_type or model_type_strat
    result['Odds'] = odds
    result['Prob'] = probs
    result['Threshold'] = threshold
    result['Implied_Prob'] = implied_prob
    result['Edge'] = edge
    result['BET'] = is_bet
    result['FTResult'] = season_df['FTResult'].values
    result['Win'] = actual_win

    # Tier
    result['Tier'] = tier

    # Staking and P&L — only DEPLOY gets real stakes
    result['Stake'] = 0.0
    result['Stake_Tier'] = ''
    result['Profit'] = 0.0
    result['Result'] = ''

    for idx in result.index:
        if result.loc[idx, 'BET']:
            stake, stake_tier = get_stake(result.loc[idx, 'Edge'])
            if tier == 'DEPLOY':
                result.loc[idx, 'Stake'] = stake
                result.loc[idx, 'Stake_Tier'] = stake_tier
                if result.loc[idx, 'Win']:
                    result.loc[idx, 'Profit'] = stake * (result.loc[idx, 'Odds'] - 1)
                    result.loc[idx, 'Result'] = 'WIN'
                else:
                    result.loc[idx, 'Profit'] = -stake
                    result.loc[idx, 'Result'] = 'LOSS'
            else:
                result.loc[idx, 'Stake'] = 0.0
                result.loc[idx, 'Stake_Tier'] = f'{stake_tier}(PAPER)'
                result.loc[idx, 'Result'] = 'WIN' if result.loc[idx, 'Win'] else 'LOSS'

    result['p_value'] = p_val
    result['FDR'] = 'YES' if fdr else 'NO'
    result['Hist_ROI'] = hist_roi

    return result


def main():
    parser = argparse.ArgumentParser(description="Backtest V4 strategies on 2025/26 season")
    parser.add_argument("--models-dir", default="./models", help="Directory with model files")
    parser.add_argument("--season-dir", default="./features_20252026", help="Directory with season features")
    parser.add_argument("--output", default="predictions/backtest_2526.csv", help="Output CSV path")
    parser.add_argument("--eu-format", action="store_true", help="EU format output")
    parser.add_argument("--mode", default="all", choices=["deploy", "paper", "all"],
                        help="deploy=DEPLOY only, paper=DEPLOY+PAPER_TRADE, all=everything")
    args = parser.parse_args()

    # Filter strategies by mode
    if args.mode == "deploy":
        active_strategies = [s for s in V4_STRATEGIES if s[10] == "DEPLOY"]
    elif args.mode == "paper":
        active_strategies = [s for s in V4_STRATEGIES if s[10] in ("DEPLOY", "PAPER_TRADE")]
    else:
        active_strategies = V4_STRATEGIES

    n_deploy = sum(1 for s in active_strategies if s[10] == 'DEPLOY')
    n_paper = sum(1 for s in active_strategies if s[10] == 'PAPER_TRADE')
    n_monitor = sum(1 for s in active_strategies if s[10] == 'MONITOR')

    print("=" * 75)
    print("BACKTEST V4 STRATEGIES -- 2025/26 SEASON")
    print(f"{len(active_strategies)} strategies (mode={args.mode}): "
          f"{n_deploy} DEPLOY | {n_paper} PAPER_TRADE | {n_monitor} MONITOR")
    print("=" * 75)

    all_results = []

    for strat in active_strategies:
        league, market = strat[0], strat[1]
        name = strat[2]
        tier = strat[10]

        # Load season data
        fn_prefix = SEASON_FILE_MAP.get(league, league)
        season_path = Path(args.season_dir) / f"{fn_prefix}_20252026_features.csv"
        if not season_path.exists():
            print(f"   {league:>4} {market:<5} {name:<18} -- No season file")
            continue
        season_df = load_csv(str(season_path))

        # Load model
        model_data = load_model(args.models_dir, league, market)
        if not model_data:
            print(f"   {league:>4} {market:<5} {name:<18} -- No model file")
            continue

        # Run backtest
        result = backtest_strategy(strat, season_df, model_data)
        bets = result[result['BET']]
        n_bets = len(bets)

        tier_tag = f"[{tier}]"
        if n_bets > 0:
            wins = (bets['Result'] == 'WIN').sum()
            losses = (bets['Result'] == 'LOSS').sum()
            profit = bets['Profit'].sum()
            staked = bets['Stake'].sum()
            roi = profit / staked * 100 if staked > 0 else 0
            avg_odds = bets['Odds'].mean()
            marker = "+++" if (profit > 0 and tier == 'DEPLOY') else ("..." if profit > 0 else "---")
            profit_str = f"P/L={profit:>+7.2f}u" if tier == 'DEPLOY' else f"P/L=  PAPER"
            print(f"   {marker} {league:>4} {market:<5} {name:<18} {tier_tag:<14} "
                  f"{n_bets:>3} bets  {wins}W/{losses}L  "
                  f"{profit_str}  "
                  f"avgOdds={avg_odds:.2f}  [p={strat[7]:.4f}]")
        else:
            print(f"       {league:>4} {market:<5} {name:<18} {tier_tag:<14}   0 bets")

        all_results.append(result)

    if not all_results:
        print("\nNo results generated.")
        sys.exit(0)

    df = pd.concat(all_results, ignore_index=True)
    bets_df = df[df['BET']].copy()

    # =========================================================================
    # OVERALL SUMMARY
    # =========================================================================
    n_bets = len(bets_df)
    n_wins = (bets_df['Result'] == 'WIN').sum()
    n_losses = (bets_df['Result'] == 'LOSS').sum()
    total_profit = bets_df['Profit'].sum()
    total_staked = bets_df['Stake'].sum()
    overall_roi = total_profit / total_staked * 100 if total_staked > 0 else 0

    print(f"\n{'=' * 75}")
    print(f"OVERALL: {n_bets} bets | {n_wins}W/{n_losses}L ({n_wins/n_bets*100:.1f}% win rate)")
    print(f"         Staked: {total_staked:.2f}u | Profit: {total_profit:+.2f}u | ROI: {overall_roi:+.1f}%")
    print(f"{'=' * 75}")

    # =========================================================================
    # BY TIER
    # =========================================================================
    print(f"\nBY TIER:")
    for tier_name in ['DEPLOY', 'PAPER_TRADE', 'MONITOR']:
        tb = bets_df[bets_df['Tier'] == tier_name] if 'Tier' in bets_df.columns else pd.DataFrame()
        if len(tb) == 0:
            continue
        w = (tb['Result'] == 'WIN').sum()
        l = (tb['Result'] == 'LOSS').sum()
        wr = w / len(tb) * 100 if len(tb) > 0 else 0
        prf = tb['Profit'].sum()
        stk = tb['Stake'].sum()
        roi = prf / stk * 100 if stk > 0 else 0
        ao = tb['Odds'].mean()
        if tier_name == 'DEPLOY':
            print(f"   {tier_name:<12}: {len(tb)} bets, {w}W/{l}L ({wr:.0f}%), "
                  f"staked={stk:.2f}u, profit={prf:+.2f}u, ROI={roi:+.1f}%, avgOdds={ao:.2f}")
        else:
            print(f"   {tier_name:<12}: {len(tb)} bets, {w}W/{l}L ({wr:.0f}%), avgOdds={ao:.2f} (PAPER)")

    # =========================================================================
    # BY STRATEGY
    # =========================================================================
    print(f"\nBY STRATEGY:")
    print(f"   {'League':<5} {'Mkt':<5} {'Strategy':<18} {'Tier':<12} {'Bets':>4} {'W':>3} {'L':>3} {'WR%':>5} "
          f"{'Staked':>7} {'Profit':>8} {'ROI%':>7} {'AvgOdds':>7} {'AvgEdge':>8}")
    print(f"   {'-' * 110}")

    for strat in active_strategies:
        league, market, name = strat[0], strat[1], strat[2]
        tier = strat[10]
        sb = bets_df[(bets_df['League'] == league) & (bets_df['Market'] == market)]
        if len(sb) == 0:
            print(f"   {league:<5} {market:<5} {name:<18} {tier:<12} {'0':>4}")
            continue
        w = (sb['Result'] == 'WIN').sum()
        l = (sb['Result'] == 'LOSS').sum()
        wr = w / len(sb) * 100
        stk = sb['Stake'].sum()
        prf = sb['Profit'].sum()
        roi = prf / stk * 100 if stk > 0 else 0
        ao = sb['Odds'].mean()
        ae = sb['Edge'].mean()
        if tier == 'DEPLOY':
            status = "+++" if prf > 0 else "---"
            print(f"   {league:<5} {market:<5} {name:<18} {tier:<12} {len(sb):>4} {w:>3} {l:>3} {wr:>5.1f} "
                  f"{stk:>7.2f} {prf:>+8.2f} {roi:>+7.1f} {ao:>7.2f} {ae:>+8.1%} {status}")
        else:
            print(f"   {league:<5} {market:<5} {name:<18} {tier:<12} {len(sb):>4} {w:>3} {l:>3} {wr:>5.1f} "
                  f"{'PAPER':>7} {'PAPER':>8} {'':>7} {ao:>7.2f} {ae:>+8.1%}")

    # =========================================================================
    # MONTHLY BREAKDOWN
    # =========================================================================
    bets_df['_date'] = pd.to_datetime(bets_df['MatchDate'], errors='coerce')
    bets_df['_month'] = bets_df['_date'].dt.to_period('M')

    print(f"\nMONTHLY P&L:")
    print(f"   {'Month':<10} {'Bets':>4} {'W':>3} {'L':>3} {'Profit':>8} {'Cumul':>8} {'ROI%':>7}")
    print(f"   {'-' * 50}")
    cumul = 0
    for month, mg in bets_df.groupby('_month', sort=True):
        w = (mg['Result'] == 'WIN').sum()
        l = (mg['Result'] == 'LOSS').sum()
        prf = mg['Profit'].sum()
        stk = mg['Stake'].sum()
        roi = prf / stk * 100 if stk > 0 else 0
        cumul += prf
        print(f"   {str(month):<10} {len(mg):>4} {w:>3} {l:>3} {prf:>+8.2f} {cumul:>+8.2f} {roi:>+7.1f}")

    # =========================================================================
    # TIER BREAKDOWN
    # =========================================================================
    print(f"\nBY STAKE TIER:")
    for tier in ['LOW', 'MED', 'HIGH']:
        tb = bets_df[bets_df['Stake_Tier'] == tier]
        if len(tb) == 0:
            continue
        w = (tb['Result'] == 'WIN').sum()
        l = (tb['Result'] == 'LOSS').sum()
        prf = tb['Profit'].sum()
        stk = tb['Stake'].sum()
        roi = prf / stk * 100 if stk > 0 else 0
        print(f"   {tier:<4}: {len(tb)} bets, {w}W/{l}L, staked={stk:.2f}u, profit={prf:+.2f}u, ROI={roi:+.1f}%")

    # =========================================================================
    # FDR vs NON-FDR
    # =========================================================================
    print(f"\nFDR-PASS vs NON-FDR:")
    for fdr_val in ['YES', 'NO']:
        fb = bets_df[bets_df['FDR'] == fdr_val]
        if len(fb) == 0:
            continue
        w = (fb['Result'] == 'WIN').sum()
        l = (fb['Result'] == 'LOSS').sum()
        prf = fb['Profit'].sum()
        stk = fb['Stake'].sum()
        roi = prf / stk * 100 if stk > 0 else 0
        label = "FDR-pass" if fdr_val == 'YES' else "Non-FDR"
        print(f"   {label:<8}: {len(fb)} bets, {w}W/{l}L, staked={stk:.2f}u, profit={prf:+.2f}u, ROI={roi:+.1f}%")

    # =========================================================================
    # SAVE OUTPUT
    # =========================================================================
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_cols = [
        'MatchDate', 'League', 'HomeTeam', 'AwayTeam', 'Market', 'Strategy', 'Model', 'Tier',
        'Odds', 'Prob', 'Threshold', 'Implied_Prob', 'Edge',
        'BET', 'FTResult', 'Win', 'Result', 'Stake', 'Stake_Tier', 'Profit',
        'p_value', 'FDR', 'Hist_ROI',
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    output = df[out_cols].copy()

    if args.eu_format:
        eu = output.copy()
        for col in eu.select_dtypes(include=[np.number]).columns:
            eu[col] = eu[col].apply(
                lambda x: str(round(x, 4)).replace('.', ',') if pd.notna(x) else ''
            )
        eu.to_csv(out_path, sep=';', index=False)
    else:
        output.to_csv(out_path, index=False)

    print(f"\nSaved full results to: {out_path}")
    print(f"   Total rows: {len(output)} ({n_bets} bets + {len(output)-n_bets} skips)")
    print("=" * 75)


if __name__ == "__main__":
    main()
