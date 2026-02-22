#!/usr/bin/env python3
"""
BET TRACKER V2 - Weekly Bet Tracking System
============================================

Reads bet files from BETS/ folder and appends results to bets_tracker/.
Designed for weekly workflow where you add new bet files each matchweek.

WORKFLOW:
1. Add your bets to BETS/BETS_1.csv, BETS_2.csv, etc.
2. Run this script to process and get results
3. Results are appended to bets_tracker/MASTER_TRACKER.csv
4. Summary updated in bets_tracker/SUMMARY.json

INPUT (BETS folder):
- BETS_1.csv, BETS_2.csv, etc. (or any .csv files)
- Required columns: league, market, HomeTeam, AwayTeam, MatchDate
- Your columns: real_bet, money_bet, actual_odd, bookmaker, betting_date

OUTPUT (bets_tracker folder):
- MASTER_TRACKER.csv: All bets with results (appended each run)
- SUMMARY.json: Performance statistics
- WEEKLY_REPORT.txt: Human-readable summary

USAGE:
------
python bet_tracker_v2.py --bets-dir ./BETS --season-dir ./features_upd --output ./bets_tracker

Author: Marc | February 2026
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import json
import hashlib


# =============================================================================
# DATA CLEANING UTILITIES
# =============================================================================

def parse_eu_number(value) -> float:
    """Parse a number that might be in EU format (comma as decimal)."""
    if pd.isna(value) or value == '' or value is None:
        return 0.0
    
    if isinstance(value, (int, float)):
        return float(value)
    
    s = str(value).strip()
    s = s.replace(',', '.')
    
    try:
        return float(s)
    except:
        return 0.0


def parse_boolean(value) -> bool:
    """Parse a boolean that might be TRUE/FALSE, 1/0, Yes/No, etc."""
    if pd.isna(value) or value == '' or value is None:
        return False
    
    if isinstance(value, bool):
        return value
    
    if isinstance(value, (int, float)):
        return value == 1
    
    s = str(value).strip().upper()
    return s in ['TRUE', '1', 'YES', 'SI', 'SÍ', 'VERDADERO', 'X']


def normalize_team_name(name: str) -> str:
    """Normalize team name for matching."""
    if pd.isna(name):
        return ""
    return str(name).strip().lower().replace("'", "").replace("-", " ")


def normalize_date(date_val) -> str:
    """Normalize date to YYYY-MM-DD format."""
    if pd.isna(date_val):
        return ""
    
    for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d', '%d.%m.%Y']:
        try:
            if isinstance(date_val, str):
                dt = datetime.strptime(date_val.strip(), fmt)
                return dt.strftime('%Y-%m-%d')
        except:
            continue
    
    try:
        return pd.to_datetime(date_val).strftime('%Y-%m-%d')
    except:
        return str(date_val)


def generate_bet_id(row: pd.Series) -> str:
    """Generate unique ID for a bet to avoid duplicates."""
    key = f"{row.get('league', '')}_{row.get('market', '')}_{row.get('HomeTeam', '')}_{row.get('AwayTeam', '')}_{row.get('MatchDate', '')}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


# =============================================================================
# FILE UTILITIES
# =============================================================================

def load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV with multiple encoding attempts."""
    for enc in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            return df
        except:
            pass
        try:
            df = pd.read_csv(filepath, encoding=enc, sep=';', decimal=',')
            return df
        except:
            continue
    raise ValueError(f"Cannot read: {filepath}")


def find_season_file(season_dir: str, league: str) -> Optional[str]:
    """Find the season features file for a league."""
    d = Path(season_dir)
    if not d.exists():
        return None
    
    patterns = [
        f"{league}_20252026_features.csv",
        f"{league.upper()}_20252026_features.csv",
        f"Matches_{league}_features.csv",
        f"Matches__{league}_features.csv",
        f"{league}_features.csv",
    ]
    
    for p in patterns:
        if (d / p).exists():
            return str(d / p)
    
    for f in d.glob(f"*{league}*features*.csv"):
        return str(f)
    
    return None


def load_all_bets(bets_dir: str) -> pd.DataFrame:
    """Load all bet files from BETS directory."""
    d = Path(bets_dir)
    if not d.exists():
        raise ValueError(f"BETS directory not found: {bets_dir}")
    
    all_bets = []
    csv_files = sorted(d.glob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {bets_dir}")
    
    for csv_file in csv_files:
        print(f"  Loading: {csv_file.name}")
        df = load_csv(str(csv_file))
        df['_source_file'] = csv_file.name
        all_bets.append(df)
    
    combined = pd.concat(all_bets, ignore_index=True)
    print(f"  Total rows loaded: {len(combined)}")
    
    return combined


def load_existing_tracker(output_dir: str) -> pd.DataFrame:
    """Load existing master tracker if it exists."""
    tracker_path = Path(output_dir) / "MASTER_TRACKER.csv"
    
    if tracker_path.exists():
        print(f"  Found existing tracker: {tracker_path}")
        df = load_csv(str(tracker_path))
        print(f"  Existing records: {len(df)}")
        return df
    
    return pd.DataFrame()


# =============================================================================
# RESULT MATCHING
# =============================================================================

def load_all_results(season_dir: str, leagues: list) -> pd.DataFrame:
    """Load all match results from season files."""
    all_results = []
    
    for league in leagues:
        sf = find_season_file(season_dir, league)
        if sf is None:
            print(f"  ⚠️  No season file for {league}")
            continue
        
        df = load_csv(sf)
        
        required = ['HomeTeam', 'AwayTeam', 'MatchDate']
        if not all(c in df.columns for c in required):
            print(f"  ⚠️  Missing columns in {league} season file")
            continue
        
        df['league'] = league
        df['_home_norm'] = df['HomeTeam'].apply(normalize_team_name)
        df['_away_norm'] = df['AwayTeam'].apply(normalize_team_name)
        df['_date_norm'] = df['MatchDate'].apply(normalize_date)
        
        if 'FTResult' in df.columns:
            df['_result'] = df['FTResult']
        elif 'FTHome' in df.columns and 'FTAway' in df.columns:
            df['_result'] = df.apply(
                lambda r: 'H' if r['FTHome'] > r['FTAway'] 
                         else ('A' if r['FTHome'] < r['FTAway'] else 'D'),
                axis=1
            )
        else:
            df['_result'] = None
        
        df['_ft_home'] = df.get('FTHome', np.nan)
        df['_ft_away'] = df.get('FTAway', np.nan)
        
        all_results.append(df)
    
    if not all_results:
        return pd.DataFrame()
    
    return pd.concat(all_results, ignore_index=True)


def match_result(bet_row: pd.Series, results_df: pd.DataFrame) -> Dict:
    """Find the match result for a bet."""
    league = bet_row.get('league', '')
    home = normalize_team_name(bet_row.get('HomeTeam', ''))
    away = normalize_team_name(bet_row.get('AwayTeam', ''))
    date = normalize_date(bet_row.get('MatchDate', ''))
    
    league_results = results_df[results_df['league'] == league]
    
    if len(league_results) == 0:
        return {'FTResult': None, 'FTHome': None, 'FTAway': None, 'match_status': 'NO_DATA'}
    
    match = league_results[
        (league_results['_date_norm'] == date) &
        (league_results['_home_norm'] == home) &
        (league_results['_away_norm'] == away)
    ]
    
    if len(match) == 1:
        row = match.iloc[0]
        result = row['_result']
        if pd.isna(result) or result == '':
            return {'FTResult': None, 'FTHome': None, 'FTAway': None, 'match_status': 'PENDING'}
        return {
            'FTResult': result,
            'FTHome': row['_ft_home'],
            'FTAway': row['_ft_away'],
            'match_status': 'PLAYED'
        }
    
    # Fuzzy match
    team_match = league_results[
        (league_results['_home_norm'] == home) &
        (league_results['_away_norm'] == away)
    ]
    
    if len(team_match) > 0:
        team_match = team_match.copy()
        team_match['_date_parsed'] = pd.to_datetime(team_match['_date_norm'], errors='coerce')
        bet_date = pd.to_datetime(date, errors='coerce')
        
        if pd.notna(bet_date):
            team_match['_date_diff'] = abs((team_match['_date_parsed'] - bet_date).dt.days)
            closest = team_match.loc[team_match['_date_diff'].idxmin()]
            
            if closest['_date_diff'] <= 7:
                result = closest['_result']
                if pd.isna(result) or result == '':
                    return {'FTResult': None, 'FTHome': None, 'FTAway': None, 'match_status': 'PENDING'}
                return {
                    'FTResult': result,
                    'FTHome': closest['_ft_home'],
                    'FTAway': closest['_ft_away'],
                    'match_status': 'PLAYED'
                }
    
    return {'FTResult': None, 'FTHome': None, 'FTAway': None, 'match_status': 'NOT_FOUND'}


# =============================================================================
# PROFIT CALCULATION
# =============================================================================

def calculate_profit(row: pd.Series) -> float:
    """Calculate profit/loss for a single bet."""
    real_bet = parse_boolean(row.get('real_bet', False))
    if not real_bet:
        return 0.0
    
    money_bet = parse_eu_number(row.get('money_bet', 0))
    actual_odd = parse_eu_number(row.get('actual_odd', 0))
    
    if actual_odd == 0:
        actual_odd = parse_eu_number(row.get('odds', 0))
    
    market = str(row.get('market', '')).upper()
    result = row.get('FTResult', None)
    
    if money_bet == 0 or actual_odd == 0:
        return 0.0
    
    if pd.isna(result) or result is None:
        return 0.0
    
    won = False
    if market == 'HOME' and result == 'H':
        won = True
    elif market == 'DRAW' and result == 'D':
        won = True
    elif market == 'AWAY' and result == 'A':
        won = True
    
    if won:
        return round(money_bet * (actual_odd - 1), 2)
    else:
        return round(-money_bet, 2)


def determine_bet_outcome(row: pd.Series) -> str:
    """Determine the outcome of a bet."""
    real_bet = parse_boolean(row.get('real_bet', False))
    if not real_bet:
        return 'NOT_BET'
    
    result = row.get('FTResult', None)
    match_status = row.get('match_status', '')
    
    if match_status == 'PENDING' or pd.isna(result):
        return 'PENDING'
    
    if match_status in ['NOT_FOUND', 'NO_DATA']:
        return match_status
    
    market = str(row.get('market', '')).upper()
    
    if market == 'HOME' and result == 'H':
        return 'WON'
    elif market == 'DRAW' and result == 'D':
        return 'WON'
    elif market == 'AWAY' and result == 'A':
        return 'WON'
    else:
        return 'LOST'


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def generate_summary(df: pd.DataFrame) -> Dict:
    """Generate comprehensive summary statistics."""
    df['_real_bet_bool'] = df['real_bet'].apply(parse_boolean)
    bets = df[df['_real_bet_bool'] == True].copy()
    
    bets['_money_bet_num'] = bets['money_bet'].apply(parse_eu_number)
    bets['_actual_odd_num'] = bets['actual_odd'].apply(parse_eu_number)
    
    if len(bets) == 0:
        return {'error': 'No actual bets found'}
    
    played = bets[bets['bet_outcome'].isin(['WON', 'LOST'])].copy()
    pending = bets[bets['bet_outcome'] == 'PENDING']
    
    summary = {
        'generated': datetime.now().isoformat(),
        'totals': {
            'total_bets': len(bets),
            'played_bets': len(played),
            'pending_bets': len(pending),
            'total_invested': round(float(bets['_money_bet_num'].sum()), 2),
            'played_invested': round(float(played['_money_bet_num'].sum()), 2) if len(played) > 0 else 0,
        },
        'performance': {},
        'by_league': {},
        'by_market': {},
        'by_bookmaker': {},
        'by_is_selected': {},
        'by_week': {},
    }
    
    if len(played) > 0:
        won = played[played['bet_outcome'] == 'WON']
        lost = played[played['bet_outcome'] == 'LOST']
        
        total_profit = float(played['profit_loss'].sum())
        total_invested = float(played['_money_bet_num'].sum())
        
        summary['performance'] = {
            'wins': len(won),
            'losses': len(lost),
            'win_rate': round(len(won) / len(played) * 100, 1) if len(played) > 0 else 0,
            'total_profit': round(total_profit, 2),
            'roi': round(total_profit / total_invested * 100, 1) if total_invested > 0 else 0,
            'avg_odds_won': round(float(won['_actual_odd_num'].mean()), 2) if len(won) > 0 else 0,
            'avg_odds_lost': round(float(lost['_actual_odd_num'].mean()), 2) if len(lost) > 0 else 0,
            'best_win': round(float(won['profit_loss'].max()), 2) if len(won) > 0 else 0,
            'worst_loss': round(float(lost['profit_loss'].min()), 2) if len(lost) > 0 else 0,
        }
        
        # By league
        for league in played['league'].unique():
            lg_data = played[played['league'] == league]
            lg_won = lg_data[lg_data['bet_outcome'] == 'WON']
            profit = float(lg_data['profit_loss'].sum())
            invested = float(lg_data['_money_bet_num'].sum())
            
            summary['by_league'][league] = {
                'bets': len(lg_data),
                'wins': len(lg_won),
                'win_rate': round(len(lg_won) / len(lg_data) * 100, 1),
                'profit': round(profit, 2),
                'roi': round(profit / invested * 100, 1) if invested > 0 else 0,
            }
        
        # By market
        for market in played['market'].unique():
            mk_data = played[played['market'] == market]
            mk_won = mk_data[mk_data['bet_outcome'] == 'WON']
            profit = float(mk_data['profit_loss'].sum())
            invested = float(mk_data['_money_bet_num'].sum())
            
            summary['by_market'][market] = {
                'bets': len(mk_data),
                'wins': len(mk_won),
                'win_rate': round(len(mk_won) / len(mk_data) * 100, 1),
                'profit': round(profit, 2),
                'roi': round(profit / invested * 100, 1) if invested > 0 else 0,
            }
        
        # By bookmaker
        if 'bookmaker' in played.columns:
            for bookie in played['bookmaker'].dropna().unique():
                if bookie == '':
                    continue
                bk_data = played[played['bookmaker'] == bookie]
                bk_won = bk_data[bk_data['bet_outcome'] == 'WON']
                profit = float(bk_data['profit_loss'].sum())
                invested = float(bk_data['_money_bet_num'].sum())
                
                summary['by_bookmaker'][str(bookie)] = {
                    'bets': len(bk_data),
                    'wins': len(bk_won),
                    'win_rate': round(len(bk_won) / len(bk_data) * 100, 1),
                    'profit': round(profit, 2),
                    'roi': round(profit / invested * 100, 1) if invested > 0 else 0,
                }
        
        # By is_selected
        if 'is_selected' in played.columns:
            for status in played['is_selected'].unique():
                st_data = played[played['is_selected'] == status]
                st_won = st_data[st_data['bet_outcome'] == 'WON']
                profit = float(st_data['profit_loss'].sum())
                invested = float(st_data['_money_bet_num'].sum())
                
                summary['by_is_selected'][status] = {
                    'bets': len(st_data),
                    'wins': len(st_won),
                    'win_rate': round(len(st_won) / len(st_data) * 100, 1),
                    'profit': round(profit, 2),
                    'roi': round(profit / invested * 100, 1) if invested > 0 else 0,
                }
        
        # By source file (week)
        if '_source_file' in played.columns:
            for src in played['_source_file'].unique():
                wk_data = played[played['_source_file'] == src]
                wk_won = wk_data[wk_data['bet_outcome'] == 'WON']
                profit = float(wk_data['profit_loss'].sum())
                invested = float(wk_data['_money_bet_num'].sum())
                
                summary['by_week'][str(src)] = {
                    'bets': len(wk_data),
                    'wins': len(wk_won),
                    'win_rate': round(len(wk_won) / len(wk_data) * 100, 1),
                    'profit': round(profit, 2),
                    'roi': round(profit / invested * 100, 1) if invested > 0 else 0,
                }
    
    return summary


def generate_report(summary: Dict, output_dir: str):
    """Generate human-readable weekly report."""
    lines = []
    lines.append("=" * 70)
    lines.append("BETTING PERFORMANCE REPORT")
    lines.append(f"Generated: {summary.get('generated', 'N/A')}")
    lines.append("=" * 70)
    
    totals = summary.get('totals', {})
    lines.append(f"\nOVERVIEW")
    lines.append(f"  Total bets placed:  {totals.get('total_bets', 0)}")
    lines.append(f"  Played:             {totals.get('played_bets', 0)}")
    lines.append(f"  Pending:            {totals.get('pending_bets', 0)}")
    lines.append(f"  Total invested:     €{totals.get('total_invested', 0):.2f}")
    
    perf = summary.get('performance', {})
    if perf:
        profit = perf.get('total_profit', 0)
        roi = perf.get('roi', 0)
        sign = '+' if profit >= 0 else ''
        
        lines.append(f"\nPERFORMANCE (played bets)")
        lines.append(f"  Wins:       {perf.get('wins', 0)}")
        lines.append(f"  Losses:     {perf.get('losses', 0)}")
        lines.append(f"  Win Rate:   {perf.get('win_rate', 0):.1f}%")
        lines.append(f"  Profit:     {sign}€{profit:.2f}")
        lines.append(f"  ROI:        {sign}{roi:.1f}%")
    
    by_league = summary.get('by_league', {})
    if by_league:
        lines.append(f"\nBY LEAGUE")
        lines.append(f"  {'League':<8} {'Bets':>6} {'Wins':>6} {'Win%':>7} {'Profit':>10} {'ROI':>8}")
        lines.append(f"  {'-'*47}")
        for lg, data in sorted(by_league.items(), key=lambda x: x[1]['profit'], reverse=True):
            sign = '+' if data['profit'] >= 0 else ''
            lines.append(f"  {lg:<8} {data['bets']:>6} {data['wins']:>6} {data['win_rate']:>6.1f}% {sign}{data['profit']:>9.2f} {sign}{data['roi']:>7.1f}%")
    
    by_market = summary.get('by_market', {})
    if by_market:
        lines.append(f"\nBY MARKET")
        lines.append(f"  {'Market':<8} {'Bets':>6} {'Wins':>6} {'Win%':>7} {'Profit':>10} {'ROI':>8}")
        lines.append(f"  {'-'*47}")
        for mk, data in sorted(by_market.items()):
            sign = '+' if data['profit'] >= 0 else ''
            lines.append(f"  {mk:<8} {data['bets']:>6} {data['wins']:>6} {data['win_rate']:>6.1f}% {sign}{data['profit']:>9.2f} {sign}{data['roi']:>7.1f}%")
    
    by_week = summary.get('by_week', {})
    if by_week:
        lines.append(f"\nBY MATCHWEEK")
        lines.append(f"  {'File':<20} {'Bets':>6} {'Wins':>6} {'Win%':>7} {'Profit':>10} {'ROI':>8}")
        lines.append(f"  {'-'*59}")
        for wk, data in sorted(by_week.items()):
            sign = '+' if data['profit'] >= 0 else ''
            lines.append(f"  {wk:<20} {data['bets']:>6} {data['wins']:>6} {data['win_rate']:>6.1f}% {sign}{data['profit']:>9.2f} {sign}{data['roi']:>7.1f}%")
    
    lines.append("\n" + "=" * 70)
    
    report_path = Path(output_dir) / "WEEKLY_REPORT.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return '\n'.join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bet Tracker V2 - Weekly bet tracking with append mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOW:
  1. Put your bet files in BETS/ folder (BETS_1.csv, BETS_2.csv, etc.)
  2. Run this script weekly
  3. Results are appended to bets_tracker/MASTER_TRACKER.csv

EXAMPLE:
  python bet_tracker_v2.py --bets-dir ./BETS --season-dir ./features_upd --output ./bets_tracker
        """
    )
    
    parser.add_argument("--bets-dir", "-b", default="./BETS",
                        help="Directory containing bet CSV files (default: ./BETS)")
    parser.add_argument("--season-dir", "-s", required=True,
                        help="Path to season features folder (e.g., ./features_upd)")
    parser.add_argument("--output", "-o", default="./bets_tracker",
                        help="Output directory for tracker (default: ./bets_tracker)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress detailed output")
    parser.add_argument("--eu-format", action="store_true",
                        help="Output CSV in EU format")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Ignore existing tracker and rebuild from scratch")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("BET TRACKER V2 - Weekly Tracking System")
    print("=" * 70)
    
    # Create output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all bets from BETS folder
    print(f"\n📂 Loading bets from: {args.bets_dir}")
    try:
        new_bets_df = load_all_bets(args.bets_dir)
    except ValueError as e:
        print(f"  ❌ {e}")
        return
    
    # Generate bet IDs for deduplication
    new_bets_df['bet_id'] = new_bets_df.apply(generate_bet_id, axis=1)
    
    # Load existing tracker
    print(f"\n📂 Checking existing tracker in: {args.output}")
    if args.force_refresh:
        existing_df = pd.DataFrame()
        print("  Force refresh - starting fresh")
    else:
        existing_df = load_existing_tracker(args.output)
    
    # Filter out duplicates
    if len(existing_df) > 0 and 'bet_id' in existing_df.columns:
        existing_ids = set(existing_df['bet_id'].tolist())
        new_bets_df = new_bets_df[~new_bets_df['bet_id'].isin(existing_ids)]
        print(f"  New bets to process: {len(new_bets_df)}")
    
    if len(new_bets_df) == 0:
        print("\n✅ No new bets to process. Tracker is up to date.")
        # Still update results for pending bets
        if len(existing_df) > 0:
            print("\n🔄 Checking for updated results on pending bets...")
            pending_mask = existing_df['bet_outcome'] == 'PENDING'
            if pending_mask.sum() > 0:
                leagues = existing_df['league'].unique().tolist()
                results_df = load_all_results(args.season_dir, leagues)
                
                for idx in existing_df[pending_mask].index:
                    row = existing_df.loc[idx]
                    match_info = match_result(row, results_df)
                    
                    if match_info['match_status'] == 'PLAYED':
                        existing_df.loc[idx, 'FTResult'] = match_info['FTResult']
                        existing_df.loc[idx, 'FTHome'] = match_info['FTHome']
                        existing_df.loc[idx, 'FTAway'] = match_info['FTAway']
                        existing_df.loc[idx, 'match_status'] = match_info['match_status']
                        existing_df.loc[idx, 'bet_outcome'] = determine_bet_outcome(existing_df.loc[idx])
                        existing_df.loc[idx, 'profit_loss'] = calculate_profit(existing_df.loc[idx])
                
                # Recalculate cumulative P&L
                existing_df['_real_bet_bool'] = existing_df['real_bet'].apply(parse_boolean)
                actual_bets = existing_df[existing_df['_real_bet_bool'] == True].copy()
                if len(actual_bets) > 0:
                    actual_bets = actual_bets.sort_values('MatchDate')
                    actual_bets['cumulative_pnl'] = actual_bets['profit_loss'].cumsum()
                    existing_df['cumulative_pnl'] = existing_df['bet_id'].map(
                        actual_bets.set_index('bet_id')['cumulative_pnl']
                    ).fillna(0)
                
                new_played = (existing_df['bet_outcome'].isin(['WON', 'LOST'])).sum()
                print(f"  Updated results: {new_played} total played")
                
                # Save updated tracker
                tracker_path = out_dir / "MASTER_TRACKER.csv"
                if args.eu_format:
                    existing_df.to_csv(tracker_path, sep=';', decimal=',', index=False)
                else:
                    existing_df.to_csv(tracker_path, index=False)
                print(f"\n✅ Saved: {tracker_path}")
                
                # Generate summary
                summary = generate_summary(existing_df)
                json_path = out_dir / "SUMMARY.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, default=str)
                
                report = generate_report(summary, args.output)
                if not args.quiet:
                    print(report)
        return
    
    # Get unique leagues
    leagues = new_bets_df['league'].unique().tolist()
    if len(existing_df) > 0:
        leagues = list(set(leagues + existing_df['league'].unique().tolist()))
    print(f"\nLeagues: {', '.join(leagues)}")
    
    # Load match results
    print(f"\n📂 Loading match results from: {args.season_dir}")
    results_df = load_all_results(args.season_dir, leagues)
    print(f"  Loaded {len(results_df)} match records")
    
    # Match results for new bets
    print("\n🔄 Matching results...")
    result_data = []
    for idx, row in new_bets_df.iterrows():
        match_info = match_result(row, results_df)
        result_data.append(match_info)
    
    results_matched = pd.DataFrame(result_data)
    new_bets_df = pd.concat([new_bets_df.reset_index(drop=True), results_matched], axis=1)
    
    # Calculate outcomes and profit
    new_bets_df['bet_outcome'] = new_bets_df.apply(determine_bet_outcome, axis=1)
    new_bets_df['profit_loss'] = new_bets_df.apply(calculate_profit, axis=1)
    new_bets_df['processed_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # Combine with existing
    if len(existing_df) > 0:
        # Also update pending bets in existing
        pending_mask = existing_df['bet_outcome'] == 'PENDING'
        for idx in existing_df[pending_mask].index:
            row = existing_df.loc[idx]
            match_info = match_result(row, results_df)
            
            if match_info['match_status'] == 'PLAYED':
                existing_df.loc[idx, 'FTResult'] = match_info['FTResult']
                existing_df.loc[idx, 'FTHome'] = match_info['FTHome']
                existing_df.loc[idx, 'FTAway'] = match_info['FTAway']
                existing_df.loc[idx, 'match_status'] = match_info['match_status']
                existing_df.loc[idx, 'bet_outcome'] = determine_bet_outcome(existing_df.loc[idx])
                existing_df.loc[idx, 'profit_loss'] = calculate_profit(existing_df.loc[idx])
        
        combined_df = pd.concat([existing_df, new_bets_df], ignore_index=True)
    else:
        combined_df = new_bets_df
    
    # Calculate cumulative P&L
    combined_df['_real_bet_bool'] = combined_df['real_bet'].apply(parse_boolean)
    actual_bets = combined_df[combined_df['_real_bet_bool'] == True].copy()
    if len(actual_bets) > 0:
        actual_bets = actual_bets.sort_values('MatchDate')
        actual_bets['cumulative_pnl'] = actual_bets['profit_loss'].cumsum()
        combined_df['cumulative_pnl'] = combined_df['bet_id'].map(
            actual_bets.set_index('bet_id')['cumulative_pnl']
        ).fillna(0)
    else:
        combined_df['cumulative_pnl'] = 0
    
    # Summary
    n_actual = combined_df['_real_bet_bool'].sum()
    n_played = combined_df['bet_outcome'].isin(['WON', 'LOST']).sum()
    n_pending = (combined_df['bet_outcome'] == 'PENDING').sum()
    n_won = (combined_df['bet_outcome'] == 'WON').sum()
    
    print(f"\n📊 Results:")
    print(f"  Total bets in tracker: {len(combined_df)}")
    print(f"  Actual bets (real_bet=TRUE): {n_actual}")
    print(f"  Played: {n_played} | Won: {n_won} | Pending: {n_pending}")
    
    # Output columns
    out_cols = [
        'bet_id', 'league', 'market', 'strategy', 'is_selected',
        'MatchDate', 'HomeTeam', 'AwayTeam',
        'odds', 'probability', 'edge', 'stake', 'stake_tier',
        'real_bet', 'money_bet', 'actual_odd', 'bookmaker', 'betting_date',
        'FTResult', 'FTHome', 'FTAway', 'match_status',
        'bet_outcome', 'profit_loss', 'cumulative_pnl',
        '_source_file', 'processed_date'
    ]
    out_cols = [c for c in out_cols if c in combined_df.columns]
    
    final_df = combined_df[out_cols].copy()
    
    # Save master tracker
    tracker_path = out_dir / "MASTER_TRACKER.csv"
    if args.eu_format:
        final_df.to_csv(tracker_path, sep=';', decimal=',', index=False)
    else:
        final_df.to_csv(tracker_path, index=False)
    print(f"\n✅ Saved: {tracker_path}")
    
    # Generate summary
    summary = generate_summary(combined_df)
    json_path = out_dir / "SUMMARY.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"✅ Saved: {json_path}")
    
    # Generate report
    report = generate_report(summary, args.output)
    print(f"✅ Saved: {out_dir}/WEEKLY_REPORT.txt")
    
    if not args.quiet:
        print(report)
    
    print(f"\n{'='*70}")
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
