"""
FEATURE ENGINEERING PIPELINE - Protocol V3.4
=============================================

Production script for processing ANY league CSV from Football-Data.co.uk
Computes all features with robust NULL handling.

ODDS SPECIFICATION (Protocol V3.3):
- Uses MARKET AVERAGE odds: AvgH, AvgD, AvgA (pre-closing, 1-3 days before match)
- Uses MAXIMUM odds: MaxH, MaxD, MaxA (pre-closing)
- FORBIDDEN: Any closing odds (columns with 'C' like AvgCH, B365CH, etc.)

NULL HANDLING:
- Missing odds: Uses median imputation
- Missing stats: Uses league averages
- Missing form: Uses neutral values
- Missing ELO: Computes from scratch

Author: Marc | January 2026 | Protocol V3.4
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import warnings
import sys
import os
import json
from pathlib import Path

warnings.filterwarnings('ignore')


# =============================================================================
# FORBIDDEN COLUMNS - CLV/CLOSING ODDS
# =============================================================================
FORBIDDEN_PATTERNS = [
    'CH', 'CD', 'CA',  # Any closing odds pattern
    'C>2.5', 'C<2.5',  # Closing O/U
    'CAHH', 'CAHA',    # Closing Asian Handicap
]


class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline following Protocol V3.4.
    Handles ANY league with robust NULL handling.
    """
    
    # ELO parameters (ClubElo methodology)
    ELO_K_FACTOR = 20
    ELO_DEFAULT = 1500.0
    ELO_HFA_INITIAL = 65.0
    
    # Rolling windows
    FORM_WINDOW_3 = 3
    FORM_WINDOW_5 = 5
    ROLLING_WINDOW = 10
    H2H_LOOKBACK = 10
    
    # Default values for missing data
    DEFAULTS = {
        'shots': 12.0,
        'shots_target': 4.0,
        'corners': 5.0,
        'fouls': 12.0,
        'yellows': 1.5,
        'reds': 0.05,
        'goals': 1.3,
        'rest_days': 7,
        'overround': 1.06,
    }
    
    def __init__(self, verbose: bool = True):
        """Initialize pipeline with empty state."""
        self.verbose = verbose
        self._reset_state()
    
    def _reset_state(self):
        """Reset all historical tracking dictionaries."""
        # ELO ratings
        self.team_elo: Dict[str, float] = {}
        self.league_hfa: Dict[str, float] = defaultdict(lambda: self.ELO_HFA_INITIAL)
        
        # Form tracking (last N results)
        self.team_results: Dict[str, List[int]] = defaultdict(list)
        
        # Match history for rest days
        self.team_last_match_date: Dict[str, pd.Timestamp] = {}
        
        # Team matches for venue-specific stats
        self.team_home_matches: Dict[str, List[Dict]] = defaultdict(list)
        self.team_away_matches: Dict[str, List[Dict]] = defaultdict(list)
        self.team_matches: Dict[str, List[Dict]] = defaultdict(list)
        
        # H2H history
        self.h2h_matches: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        
        # Match statistics rolling
        self.team_shots_for: Dict[str, List[float]] = defaultdict(list)
        self.team_shots_against: Dict[str, List[float]] = defaultdict(list)
        self.team_corners: Dict[str, List[float]] = defaultdict(list)
        self.team_yellows: Dict[str, List[float]] = defaultdict(list)
        self.team_reds: Dict[str, List[float]] = defaultdict(list)
        
        # League statistics for imputation
        self.league_stats = {
            'shots': [], 'corners': [], 'yellows': [], 'reds': [], 'goals': []
        }
    
    def _log(self, msg: str):
        """Print if verbose mode."""
        if self.verbose:
            print(msg)
    
    # =========================================================================
    # SAFE VALUE EXTRACTION WITH NULL HANDLING
    # =========================================================================
    
    def _safe_float(self, value, default: float = 0.0) -> float:
        """Safely convert to float, handling nulls."""
        if pd.isna(value):
            return default
        try:
            return float(value)
        except:
            return default
    
    def _safe_int(self, value, default: int = 0) -> int:
        """Safely convert to int, handling nulls."""
        if pd.isna(value):
            return default
        try:
            return int(float(value))
        except:
            return default
    
    def _get_league_avg(self, stat: str) -> float:
        """Get league average for a statistic, or default if not enough data."""
        values = self.league_stats.get(stat, [])
        if len(values) >= 10:
            return np.mean(values[-100:])  # Use last 100 for stability
        return self.DEFAULTS.get(stat, 0)
    
    # =========================================================================
    # COLUMN MAPPING
    # =========================================================================
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map Football-Data.co.uk columns to standard names."""
        column_map = {
            # Core
            'Div': 'Division',
            'Date': 'MatchDate',
            'Time': 'MatchTime',
            'FTHG': 'FTHome',
            'FTAG': 'FTAway',
            'FTR': 'FTResult',
            'HTHG': 'HTHome',
            'HTAG': 'HTAway',
            'HTR': 'HTResult',
            
            # Match stats
            'HS': 'HomeShots',
            'AS': 'AwayShots',
            'HST': 'HomeTarget',
            'AST': 'AwayTarget',
            'HF': 'HomeFouls',
            'AF': 'AwayFouls',
            'HC': 'HomeCorners',
            'AC': 'AwayCorners',
            'HY': 'HomeYellow',
            'AY': 'AwayYellow',
            'HR': 'HomeRed',
            'AR': 'AwayRed',
            
            # PRE-CLOSING ODDS (ALLOWED) ✅
            'AvgH': 'OddHome',
            'AvgD': 'OddDraw',
            'AvgA': 'OddAway',
            'MaxH': 'MaxHome',
            'MaxD': 'MaxDraw',
            'MaxA': 'MaxAway',
            
            # Fallback to B365 if Avg not available
            'B365H': 'B365Home',
            'B365D': 'B365Draw',
            'B365A': 'B365Away',
            
            # Over/Under (pre-closing)
            'Avg>2.5': 'OddOver25',
            'Avg<2.5': 'OddUnder25',
            'Max>2.5': 'MaxOver25',
            'Max<2.5': 'MaxUnder25',
            'Over25': 'OddOver25',
            'Under25': 'OddUnder25',
            
            # Asian Handicap (pre-closing)
            'AHh': 'HandiSize',
            'AvgAHH': 'AvgHandiHome',
            'AvgAHA': 'AvgHandiAway',
        }
        
        df = df.rename(columns=column_map)
        
        # Drop FORBIDDEN closing odds columns
        cols_to_drop = [col for col in df.columns 
                        if any(p in col for p in FORBIDDEN_PATTERNS) 
                        and col not in ['HomeCorners', 'AwayCorners', 'HC', 'AC']]
        
        if cols_to_drop and self.verbose:
            self._log(f"  ⚠️  Dropping {len(cols_to_drop)} FORBIDDEN closing odds columns")
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Use B365 as fallback if Avg not available
        if 'OddHome' not in df.columns and 'B365Home' in df.columns:
            df['OddHome'] = df['B365Home']
            df['OddDraw'] = df['B365Draw']
            df['OddAway'] = df['B365Away']
            self._log("  ℹ️  Using B365 odds (Avg not available)")
        
        return df
    
    # =========================================================================
    # ELO CALCULATIONS (ClubElo methodology)
    # =========================================================================
    
    def _get_elo(self, team: str) -> float:
        """Get current ELO rating for team."""
        if team not in self.team_elo:
            self.team_elo[team] = self.ELO_DEFAULT
        return self.team_elo[team]
    
    def _expected_score(self, elo_diff: float) -> float:
        """Calculate expected score from ELO difference."""
        return 1.0 / (10 ** (-elo_diff / 400) + 1)
    
    def _update_elo(self, home_team: str, away_team: str, 
                    home_goals: int, away_goals: int, league: str):
        """Update ELO ratings after a match."""
        home_elo = self._get_elo(home_team)
        away_elo = self._get_elo(away_team)
        
        hfa = self.league_hfa[league]
        elo_diff = home_elo - away_elo + hfa
        expected_home = self._expected_score(elo_diff)
        
        if home_goals > away_goals:
            actual_home = 1.0
            margin = home_goals - away_goals
        elif home_goals < away_goals:
            actual_home = 0.0
            margin = away_goals - home_goals
        else:
            actual_home = 0.5
            margin = 0
        
        base_change = self.ELO_K_FACTOR * (actual_home - expected_home)
        elo_change = base_change * np.sqrt(max(margin, 1))
        
        self.team_elo[home_team] = home_elo + elo_change
        self.team_elo[away_team] = away_elo - elo_change
        
        self.league_hfa[league] += elo_change * 0.075
        self.league_hfa[league] = np.clip(self.league_hfa[league], 30, 120)
    
    # =========================================================================
    # FORM CALCULATIONS
    # =========================================================================
    
    def _get_form(self, team: str, n_matches: int) -> int:
        """Get form points from last N matches."""
        results = self.team_results.get(team, [])
        if not results:
            return n_matches  # Neutral form (1 point per match avg)
        recent = results[-n_matches:]
        return sum(recent)
    
    def _update_form(self, team: str, points: int):
        """Add match result to team's form history."""
        self.team_results[team].append(points)
    
    # =========================================================================
    # REST DAYS
    # =========================================================================
    
    def _get_rest_days(self, team: str, current_date: pd.Timestamp) -> int:
        """Get days since team's last match."""
        if team in self.team_last_match_date:
            days = (current_date - self.team_last_match_date[team]).days
            return min(max(days, 1), 30)  # Clamp to reasonable range
        return self.DEFAULTS['rest_days']
    
    # =========================================================================
    # H2H STATS
    # =========================================================================
    
    def _get_h2h_stats(self, home_team: str, away_team: str) -> Dict:
        """Get head-to-head statistics from PAST matches only."""
        h2h_key = tuple(sorted([home_team, away_team]))
        h2h_history = self.h2h_matches.get(h2h_key, [])[-self.H2H_LOOKBACK:]
        
        if not h2h_history:
            return {
                'h2h_home_wins': 0, 'h2h_away_wins': 0, 'h2h_draws': 0,
                'h2h_home_goals_diff': 0, 'h2h_home_venue_record': 0.5, 
                'h2h_matches_count': 0
            }
        
        home_wins = away_wins = draws = home_goals = away_goals = 0
        home_venue_wins = home_venue_matches = 0
        
        for m in h2h_history:
            if m['home_team'] == home_team:
                home_venue_matches += 1
                if m['result'] == 'H':
                    home_wins += 1
                    home_venue_wins += 1
                elif m['result'] == 'A':
                    away_wins += 1
                else:
                    draws += 1
                home_goals += m['home_goals']
                away_goals += m['away_goals']
            else:
                if m['result'] == 'A':
                    home_wins += 1
                elif m['result'] == 'H':
                    away_wins += 1
                else:
                    draws += 1
                home_goals += m['away_goals']
                away_goals += m['home_goals']
        
        return {
            'h2h_home_wins': home_wins,
            'h2h_away_wins': away_wins,
            'h2h_draws': draws,
            'h2h_home_goals_diff': home_goals - away_goals,
            'h2h_home_venue_record': home_venue_wins / home_venue_matches if home_venue_matches > 0 else 0.5,
            'h2h_matches_count': len(h2h_history)
        }
    
    def _update_h2h(self, home_team: str, away_team: str, 
                    home_goals: int, away_goals: int, result: str, match_date: pd.Timestamp):
        """Update H2H history AFTER computing features."""
        h2h_key = tuple(sorted([home_team, away_team]))
        self.h2h_matches[h2h_key].append({
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'result': result,
            'date': match_date
        })
    
    # =========================================================================
    # TEAM TENDENCY STATS
    # =========================================================================
    
    def _get_team_venue_stats(self, team: str, venue: str) -> Dict:
        """Get rolling stats for team at home or away."""
        matches = (self.team_home_matches if venue == 'home' else self.team_away_matches).get(team, [])
        matches = matches[-self.ROLLING_WINDOW:]
        
        if not matches:
            if venue == 'home':
                return {'ppg': 1.5, 'goals_scored': 1.3, 'goals_conceded': 1.1}
            else:
                return {'ppg': 1.0, 'goals_scored': 1.0, 'goals_conceded': 1.4}
        
        return {
            'ppg': np.mean([m['points'] for m in matches]),
            'goals_scored': np.mean([m['goals_for'] for m in matches]),
            'goals_conceded': np.mean([m['goals_against'] for m in matches])
        }
    
    def _update_team_venue_stats(self, team: str, venue: str, 
                                  goals_for: int, goals_against: int, points: int,
                                  match_date: pd.Timestamp):
        """Update team venue stats AFTER computing features."""
        record = {
            'goals_for': goals_for,
            'goals_against': goals_against,
            'points': points,
            'date': match_date
        }
        if venue == 'home':
            self.team_home_matches[team].append(record)
        else:
            self.team_away_matches[team].append(record)
    
    # =========================================================================
    # MATCH STATS (Rolling)
    # =========================================================================
    
    def _get_rolling_stats(self, team: str) -> Dict:
        """Get rolling match statistics for team."""
        shots_for = self.team_shots_for.get(team, [])[-self.ROLLING_WINDOW:]
        shots_against = self.team_shots_against.get(team, [])[-self.ROLLING_WINDOW:]
        corners = self.team_corners.get(team, [])[-self.ROLLING_WINDOW:]
        yellows = self.team_yellows.get(team, [])[-self.ROLLING_WINDOW:]
        reds = self.team_reds.get(team, [])[-self.ROLLING_WINDOW:]
        
        return {
            'shots_avg': np.mean(shots_for) if shots_for else self._get_league_avg('shots'),
            'shots_against_avg': np.mean(shots_against) if shots_against else self._get_league_avg('shots'),
            'corners_avg': np.mean(corners) if corners else self._get_league_avg('corners'),
            'yellows_avg': np.mean(yellows) if yellows else self._get_league_avg('yellows'),
            'reds_avg': np.mean(reds) if reds else self._get_league_avg('reds')
        }
    
    def _update_rolling_stats(self, team: str, shots_for: float, shots_against: float,
                               corners: float, yellows: float, reds: float):
        """Update rolling stats AFTER computing features."""
        self.team_shots_for[team].append(shots_for)
        self.team_shots_against[team].append(shots_against)
        self.team_corners[team].append(corners)
        self.team_yellows[team].append(yellows)
        self.team_reds[team].append(reds)
    
    def _update_league_stats(self, shots: float, corners: float, yellows: float, 
                              reds: float, goals: float):
        """Track league averages for imputation."""
        self.league_stats['shots'].append(shots)
        self.league_stats['corners'].append(corners)
        self.league_stats['yellows'].append(yellows)
        self.league_stats['reds'].append(reds)
        self.league_stats['goals'].append(goals)
    
    # =========================================================================
    # CONGESTION
    # =========================================================================
    
    def _get_congestion(self, team: str, current_date: pd.Timestamp, days: int) -> int:
        """Count team's matches in last N days."""
        matches = self.team_matches.get(team, [])
        if not matches:
            return 0
        cutoff = current_date - pd.Timedelta(days=days)
        return sum(1 for m in matches if cutoff <= m['date'] < current_date)
    
    # =========================================================================
    # MAIN FEATURE COMPUTATION
    # =========================================================================
    
    def compute_features(self, row: pd.Series) -> Dict:
        """
        Compute ALL features for a single match.
        Uses ONLY historical data available BEFORE the match.
        Handles NULL values robustly.
        """
        features = {}
        
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        match_date = pd.to_datetime(row['MatchDate'])
        league = row.get('Division', 'DEFAULT')
        
        # === ELO FEATURES (6) ===
        home_elo = self._get_elo(home_team)
        away_elo = self._get_elo(away_team)
        elo_sum = home_elo + away_elo
        
        features['HomeElo'] = round(home_elo, 1)
        features['AwayElo'] = round(away_elo, 1)
        features['elo_diff'] = round(home_elo - away_elo, 1)
        features['elo_sum'] = round(elo_sum, 1)
        features['elo_mismatch'] = round(abs(home_elo - away_elo) / elo_sum, 4) if elo_sum > 0 else 0
        features['elo_mismatch_flag'] = 1 if abs(home_elo - away_elo) > 200 else 0
        
        # === FORM FEATURES (13) ===
        form3_home = self._get_form(home_team, 3)
        form5_home = self._get_form(home_team, 5)
        form3_away = self._get_form(away_team, 3)
        form5_away = self._get_form(away_team, 5)
        
        features['Form3Home'] = form3_home
        features['Form5Home'] = form5_home
        features['Form3Away'] = form3_away
        features['Form5Away'] = form5_away
        features['Form3Home_ratio'] = round(form3_home / 9.0, 4)
        features['Form5Home_ratio'] = round(form5_home / 15.0, 4)
        features['Form3Away_ratio'] = round(form3_away / 9.0, 4)
        features['Form5Away_ratio'] = round(form5_away / 15.0, 4)
        features['form_diff_3'] = form3_home - form3_away
        features['form_diff_5'] = form5_home - form5_away
        features['form_momentum_home'] = form3_home - (form5_home - form3_home)
        features['form_momentum_away'] = form3_away - (form5_away - form3_away)
        features['form_momentum'] = features['form_momentum_home'] - features['form_momentum_away']
        
        # === REST & SCHEDULE FEATURES (10) ===
        home_rest = self._get_rest_days(home_team, match_date)
        away_rest = self._get_rest_days(away_team, match_date)
        
        features['home_rest_days'] = home_rest
        features['away_rest_days'] = away_rest
        features['rest_diff'] = home_rest - away_rest
        
        rd = features['rest_diff']
        features['rest_diff_nonlinear'] = -2 if rd <= -4 else (-1 if rd <= -2 else (0 if rd <= 2 else (1 if rd <= 4 else 2)))
        
        features['home_congestion_14d'] = self._get_congestion(home_team, match_date, 14)
        features['away_congestion_14d'] = self._get_congestion(away_team, match_date, 14)
        features['is_midweek'] = 1 if match_date.dayofweek in [1, 2] else 0
        
        try:
            time_str = row.get('MatchTime', '15:00')
            if pd.notna(time_str):
                hour = pd.to_datetime(str(time_str), format='%H:%M').hour
            else:
                hour = 15
            features['is_early_kickoff'] = 1 if hour < 14 else 0
            features['is_evening_match'] = 1 if hour >= 19 else 0
        except:
            features['is_early_kickoff'] = 0
            features['is_evening_match'] = 0
        
        month = match_date.month
        features['winter_period'] = 1 if month in [11, 12, 1, 2] else 0
        features['season_phase'] = 1 if month in [8, 9, 10] else (2 if month in [11, 12, 1, 2] else 3)
        
        season_start = pd.Timestamp(year=match_date.year if month >= 8 else match_date.year - 1, month=8, day=1)
        features['match_week'] = max(1, min(46, (match_date - season_start).days // 7 + 1))
        
        # === H2H FEATURES (6) ===
        h2h_stats = self._get_h2h_stats(home_team, away_team)
        features.update(h2h_stats)
        
        # === TEAM TENDENCY FEATURES (7) ===
        home_stats = self._get_team_venue_stats(home_team, 'home')
        away_stats = self._get_team_venue_stats(away_team, 'away')
        
        features['home_team_home_ppg'] = round(home_stats['ppg'], 3)
        features['home_team_goals_scored_avg'] = round(home_stats['goals_scored'], 3)
        features['home_team_goals_conceded_avg'] = round(home_stats['goals_conceded'], 3)
        features['away_team_away_ppg'] = round(away_stats['ppg'], 3)
        features['away_team_goals_scored_avg'] = round(away_stats['goals_scored'], 3)
        features['away_team_goals_conceded_avg'] = round(away_stats['goals_conceded'], 3)
        features['venue_ppg_diff'] = round(home_stats['ppg'] - away_stats['ppg'], 3)
        
        # === MARKET FEATURES (5) - PRE-CLOSING ONLY ✅ ===
        odd_home = self._safe_float(row.get('OddHome'), 0)
        odd_draw = self._safe_float(row.get('OddDraw'), 0)
        odd_away = self._safe_float(row.get('OddAway'), 0)
        max_home = self._safe_float(row.get('MaxHome'), odd_home)
        max_draw = self._safe_float(row.get('MaxDraw'), odd_draw)
        max_away = self._safe_float(row.get('MaxAway'), odd_away)
        
        if odd_home > 1 and odd_draw > 1 and odd_away > 1:
            impl_home = 1 / odd_home
            impl_draw = 1 / odd_draw
            impl_away = 1 / odd_away
            overround = impl_home + impl_draw + impl_away
            
            features['overround'] = round(overround, 4)
            
            # Sharp money signals
            gap_home = (max_home - odd_home) / odd_home if max_home > odd_home else 0
            gap_draw = (max_draw - odd_draw) / odd_draw if max_draw > odd_draw else 0
            gap_away = (max_away - odd_away) / odd_away if max_away > odd_away else 0
            
            features['odds_disagreement'] = round(max(gap_home, gap_draw, gap_away), 4)
            features['sharp_money_home'] = round(gap_home, 4)
            features['sharp_money_away'] = round(gap_away, 4)
            
            # ELO vs market
            elo_impl_home = self._expected_score(home_elo - away_elo + 70)
            features['elo_vs_odds_home'] = round(elo_impl_home - (impl_home / overround), 4)
        else:
            features['overround'] = self.DEFAULTS['overround']
            features['odds_disagreement'] = 0
            features['sharp_money_home'] = 0
            features['sharp_money_away'] = 0
            features['elo_vs_odds_home'] = 0
        
        # === ROLLING MATCH STATS FEATURES (8) ===
        home_rolling = self._get_rolling_stats(home_team)
        away_rolling = self._get_rolling_stats(away_team)
        
        features['home_shots_avg'] = round(home_rolling['shots_avg'], 2)
        features['home_shots_against_avg'] = round(home_rolling['shots_against_avg'], 2)
        features['away_shots_avg'] = round(away_rolling['shots_avg'], 2)
        features['away_shots_against_avg'] = round(away_rolling['shots_against_avg'], 2)
        features['shots_dominance_home'] = round(home_rolling['shots_avg'] - home_rolling['shots_against_avg'], 2)
        features['shots_dominance_away'] = round(away_rolling['shots_avg'] - away_rolling['shots_against_avg'], 2)
        features['corners_diff'] = round(home_rolling['corners_avg'] - away_rolling['corners_avg'], 2)
        features['discipline_diff'] = round(
            (home_rolling['yellows_avg'] + 2*home_rolling['reds_avg']) - 
            (away_rolling['yellows_avg'] + 2*away_rolling['reds_avg']), 2
        )
        
        return features
    
    # =========================================================================
    # UPDATE HISTORIES (AFTER feature computation)
    # =========================================================================
    
    def update_histories(self, row: pd.Series):
        """
        Update all historical data AFTER computing features.
        Handles NULL values robustly.
        """
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        match_date = pd.to_datetime(row['MatchDate'])
        league = row.get('Division', 'DEFAULT')
        
        home_goals = self._safe_int(row.get('FTHome'), 0)
        away_goals = self._safe_int(row.get('FTAway'), 0)
        
        # Determine result
        if home_goals > away_goals:
            result = 'H'
            home_points, away_points = 3, 0
        elif home_goals < away_goals:
            result = 'A'
            home_points, away_points = 0, 3
        else:
            result = 'D'
            home_points, away_points = 1, 1
        
        # Update ELO
        self._update_elo(home_team, away_team, home_goals, away_goals, league)
        
        # Update form
        self._update_form(home_team, home_points)
        self._update_form(away_team, away_points)
        
        # Update last match date
        self.team_last_match_date[home_team] = match_date
        self.team_last_match_date[away_team] = match_date
        
        # Update H2H
        self._update_h2h(home_team, away_team, home_goals, away_goals, result, match_date)
        
        # Update venue stats
        self._update_team_venue_stats(home_team, 'home', home_goals, away_goals, home_points, match_date)
        self._update_team_venue_stats(away_team, 'away', away_goals, home_goals, away_points, match_date)
        
        # Update match tracking
        self.team_matches[home_team].append({'date': match_date})
        self.team_matches[away_team].append({'date': match_date})
        
        # Get match stats with defaults
        home_shots = self._safe_float(row.get('HomeShots'), self._get_league_avg('shots'))
        away_shots = self._safe_float(row.get('AwayShots'), self._get_league_avg('shots'))
        home_corners = self._safe_float(row.get('HomeCorners'), self._get_league_avg('corners'))
        away_corners = self._safe_float(row.get('AwayCorners'), self._get_league_avg('corners'))
        home_yellows = self._safe_float(row.get('HomeYellow'), self._get_league_avg('yellows'))
        away_yellows = self._safe_float(row.get('AwayYellow'), self._get_league_avg('yellows'))
        home_reds = self._safe_float(row.get('HomeRed'), self._get_league_avg('reds'))
        away_reds = self._safe_float(row.get('AwayRed'), self._get_league_avg('reds'))
        
        # Update rolling stats
        self._update_rolling_stats(home_team, home_shots, away_shots, home_corners, home_yellows, home_reds)
        self._update_rolling_stats(away_team, away_shots, home_shots, away_corners, away_yellows, away_reds)
        
        # Update league stats for imputation
        self._update_league_stats(
            (home_shots + away_shots) / 2,
            (home_corners + away_corners) / 2,
            (home_yellows + away_yellows) / 2,
            (home_reds + away_reds) / 2,
            (home_goals + away_goals) / 2
        )
    
    # =========================================================================
    # MAIN PROCESSING FUNCTION
    # =========================================================================
    
    def process_league(self, filepath: str, output_path: str = None) -> pd.DataFrame:
        """
        Process a complete league CSV file.
        
        Args:
            filepath: Path to Football-Data.co.uk CSV file
            output_path: Optional path to save output CSV
            
        Returns:
            DataFrame with all features computed
        """
        self._log(f"\n{'='*70}")
        self._log(f"Processing: {Path(filepath).name}")
        self._log(f"{'='*70}")
        
        # Reset state for new league
        self._reset_state()
        
        # Load data - handle CSV, XLS, XLSX with multiple fallbacks
        filepath_lower = filepath.lower()
        df = None
        
        # Try multiple loading methods
        load_methods = []
        
        if filepath_lower.endswith('.xlsx'):
            load_methods = [
                ('openpyxl', lambda: pd.read_excel(filepath, engine='openpyxl')),
            ]
        elif filepath_lower.endswith('.xls'):
            load_methods = [
                ('xlrd', lambda: pd.read_excel(filepath, engine='xlrd')),
                ('openpyxl', lambda: pd.read_excel(filepath, engine='openpyxl')),
                ('csv_fallback', lambda: pd.read_csv(filepath, encoding='utf-8-sig')),
                ('csv_latin1', lambda: pd.read_csv(filepath, encoding='latin-1')),
                ('html_table', lambda: pd.read_html(filepath)[0]),
            ]
        else:
            load_methods = [
                ('csv_utf8', lambda: pd.read_csv(filepath, encoding='utf-8-sig')),
                ('csv_latin1', lambda: pd.read_csv(filepath, encoding='latin-1')),
            ]
        
        for method_name, method in load_methods:
            try:
                df = method()
                if df is not None and len(df) > 0:
                    break
            except Exception as e:
                continue
        
        if df is None or len(df) == 0:
            self._log(f"  ❌ Could not load file with any method")
            return pd.DataFrame()
        
        self._log(f"  Loaded {len(df):,} matches")
        
        # Standardize columns
        df = self._standardize_columns(df)
        
        # Parse dates with multiple format attempts
        date_parsed = False
        for date_format in ['%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d', '%m/%d/%Y']:
            try:
                df['MatchDate'] = pd.to_datetime(df['MatchDate'], format=date_format)
                date_parsed = True
                break
            except:
                continue
        
        if not date_parsed:
            df['MatchDate'] = pd.to_datetime(df['MatchDate'], dayfirst=True)
        
        # Sort chronologically
        df = df.sort_values('MatchDate').reset_index(drop=True)
        self._log(f"  Date range: {df['MatchDate'].min().date()} to {df['MatchDate'].max().date()}")
        
        # Check data coverage
        odds_coverage = df['OddHome'].notna().mean() * 100 if 'OddHome' in df.columns else 0
        stats_coverage = df['HomeShots'].notna().mean() * 100 if 'HomeShots' in df.columns else 0
        self._log(f"  Odds coverage: {odds_coverage:.1f}%")
        self._log(f"  Stats coverage: {stats_coverage:.1f}%")
        
        # Process each match
        all_features = []
        
        for idx, row in df.iterrows():
            # Skip rows without basic data
            if pd.isna(row.get('HomeTeam')) or pd.isna(row.get('AwayTeam')):
                all_features.append({})
                continue
            
            # Step 1: Compute features using ONLY past data
            features = self.compute_features(row)
            all_features.append(features)
            
            # Step 2: Update histories AFTER computing features
            if pd.notna(row.get('FTHome')) and pd.notna(row.get('FTAway')):
                self.update_histories(row)
        
        # Combine features with original data
        features_df = pd.DataFrame(all_features)
        
        # Select key columns to keep
        keep_cols = ['Division', 'MatchDate', 'MatchTime', 'HomeTeam', 'AwayTeam',
                     'FTHome', 'FTAway', 'FTResult', 'HTHome', 'HTAway', 'HTResult',
                     'OddHome', 'OddDraw', 'OddAway', 'MaxHome', 'MaxDraw', 'MaxAway',
                     'OddUnder25', 'OddOver25', 'MaxUnder25', 'MaxOver25']
        keep_cols = [c for c in keep_cols if c in df.columns]
        
        result_df = pd.concat([df[keep_cols].reset_index(drop=True), features_df], axis=1)
        
        # Add target variables
        if 'FTResult' in result_df.columns:
            result_df['target_home'] = (result_df['FTResult'] == 'H').astype(int)
            result_df['target_draw'] = (result_df['FTResult'] == 'D').astype(int)
            result_df['target_away'] = (result_df['FTResult'] == 'A').astype(int)
        if 'FTHome' in result_df.columns and 'FTAway' in result_df.columns:
            result_df['target_UNDER25'] = ((result_df['FTHome'] + result_df['FTAway']) < 3).astype(int)
        
        # Handle any remaining NaN in features (median imputation)
        feature_cols = [c for c in features_df.columns if c in result_df.columns]
        for col in feature_cols:
            if result_df[col].isna().any():
                median_val = result_df[col].median()
                result_df[col] = result_df[col].fillna(median_val)
        
        self._log(f"  ✅ Complete: {len(result_df):,} matches, {len(feature_cols)} features")
        
        # Build league statistics for JSON export
        league_code = result_df['Division'].iloc[0] if 'Division' in result_df.columns else Path(filepath).stem
        
        # Season breakdown
        result_df['Season'] = result_df['MatchDate'].apply(
            lambda x: f"{x.year}/{x.year+1}" if x.month >= 7 else f"{x.year-1}/{x.year}"
        )
        seasons = result_df.groupby('Season').agg({
            'HomeTeam': 'count',
            'target_home': 'mean',
            'target_draw': 'mean', 
            'target_away': 'mean'
        }).rename(columns={'HomeTeam': 'matches'}).to_dict('index')
        
        # Top teams by ELO
        top_teams = sorted(self.team_elo.items(), key=lambda x: -x[1])[:10]
        
        # Build JSON summary
        league_summary = {
            "league": league_code,
            "generated_at": datetime.now().isoformat(),
            "protocol_version": "3.4.3",
            "source_file": Path(filepath).name,
            "overview": {
                "total_matches": len(result_df),
                "total_teams": len(self.team_elo),
                "date_range": {
                    "start": str(result_df['MatchDate'].min().date()),
                    "end": str(result_df['MatchDate'].max().date())
                },
                "seasons": len(seasons)
            },
            "outcome_distribution": {
                "home_wins": int(result_df['target_home'].sum()),
                "home_win_pct": round(result_df['target_home'].mean() * 100, 1),
                "draws": int(result_df['target_draw'].sum()),
                "draw_pct": round(result_df['target_draw'].mean() * 100, 1),
                "away_wins": int(result_df['target_away'].sum()),
                "away_win_pct": round(result_df['target_away'].mean() * 100, 1)
            },
            "data_coverage": {
                "odds_coverage_pct": round(odds_coverage, 1),
                "stats_coverage_pct": round(stats_coverage, 1),
                "features_computed": len(feature_cols),
                "null_values_remaining": int(result_df[feature_cols].isna().sum().sum())
            },
            "elo_rankings": {
                "top_10": [{"team": t, "elo": round(e, 1)} for t, e in top_teams],
                "league_hfa": round(self.league_hfa[league_code], 1)
            },
            "seasons": {
                season: {
                    "matches": int(data['matches']),
                    "home_win_pct": round(data['target_home'] * 100, 1),
                    "draw_pct": round(data['target_draw'] * 100, 1),
                    "away_win_pct": round(data['target_away'] * 100, 1)
                }
                for season, data in seasons.items()
            },
            "features": {
                "basic_15": [
                    "HomeElo", "AwayElo", "elo_diff", "elo_sum", "elo_mismatch",
                    "Form3Home_ratio", "Form5Home_ratio", "Form3Away_ratio", "Form5Away_ratio",
                    "form_diff_3", "form_diff_5", "rest_diff", "rest_diff_nonlinear",
                    "winter_period", "season_phase"
                ],
                "basic_plus_26": [
                    "form_momentum_home", "form_momentum_away",
                    "overround", "odds_disagreement", "sharp_money_home", "sharp_money_away",
                    "elo_vs_odds_home", "h2h_home_wins", "h2h_away_wins", "h2h_draws",
                    "h2h_home_goals_diff"
                ],
                "all_computed": feature_cols
            }
        }
        
        # Remove Season column (was just for grouping)
        result_df = result_df.drop(columns=['Season'])
        
        # Save if output path provided
        if output_path:
            result_df.to_csv(output_path, index=False)
            self._log(f"  💾 Saved: {output_path}")
            
            # Save JSON summary
            json_path = output_path.replace('.csv', '_summary.json')
            with open(json_path, 'w') as f:
                json.dump(league_summary, f, indent=2)
            self._log(f"  📊 Summary: {json_path}")
        
        # Store summary for return
        self.last_summary = league_summary
        
        return result_df


# =============================================================================
# BATCH PROCESSING FUNCTION
# =============================================================================

def process_all_leagues(input_dir: str, output_dir: str, verbose: bool = True):
    """
    Process all CSV files in a directory.
    
    Args:
        input_dir: Directory containing league CSV files
        output_dir: Directory to save output files
        verbose: Print progress
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(input_path.glob('*.csv'))
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"BATCH FEATURE ENGINEERING - Protocol V3.4")
        print(f"{'='*70}")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        print(f"Files found: {len(csv_files)}")
        print(f"{'='*70}")
    
    pipeline = FeatureEngineeringPipeline(verbose=verbose)
    results = {}
    
    for csv_file in sorted(csv_files):
        league_code = csv_file.stem
        output_file = output_path / f"{league_code}_features.csv"
        
        try:
            df = pipeline.process_league(str(csv_file), str(output_file))
            results[league_code] = {
                'matches': len(df),
                'status': 'SUCCESS'
            }
        except Exception as e:
            results[league_code] = {
                'matches': 0,
                'status': f'ERROR: {str(e)}'
            }
            if verbose:
                print(f"  ❌ Error processing {league_code}: {e}")
    
    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*70}")
        success = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
        total_matches = sum(r['matches'] for r in results.values())
        print(f"  Successful: {success}/{len(results)} leagues")
        print(f"  Total matches: {total_matches:,}")
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Process league files from command line."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Feature Engineering Pipeline - Protocol V3.4.3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Process ALL leagues in current directory
  python feature_engineering_pipeline.py --all
  
  # Process ALL leagues in specific directory
  python feature_engineering_pipeline.py --all --data-dir ./data --output ./features
  
  # Single league
  python feature_engineering_pipeline.py Matches_E0.xls
  
  # Multiple leagues
  python feature_engineering_pipeline.py Matches_E0.xls Matches_E1.xls Matches_SP1.xls
  
  # Custom output directory
  python feature_engineering_pipeline.py Matches_E0.xls --output ./features
  
  # Quiet mode (no verbose output)
  python feature_engineering_pipeline.py --all --quiet

SUPPORTED FORMATS:
  .csv, .xls, .xlsx

OUTPUTS:
  {league}_features.csv         - Feature data for ML training
  {league}_features_summary.json - League statistics and metadata
        """
    )
    
    parser.add_argument('files', nargs='*', help='Input CSV/XLS files from Football-Data.co.uk')
    parser.add_argument('--output', '-o', type=str, default='./output',
                        help='Output directory (default: ./output)')
    parser.add_argument('--data-dir', '-d', type=str, default='.',
                        help='Directory containing data files (default: current directory)')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Process ALL data files in data-dir')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("\n" + "="*70)
        print("FEATURE ENGINEERING PIPELINE - Protocol V3.4.3")
        print("="*70)
        print("\nODDS SPECIFICATION:")
        print("  ✅ ALLOWED: AvgH, AvgD, AvgA (Market Average Pre-Closing)")
        print("  ✅ ALLOWED: MaxH, MaxD, MaxA (Maximum Pre-Closing)")
        print("  ❌ FORBIDDEN: Any closing odds (AvgCH, B365CH, etc.)")
        print("\nNULL HANDLING:")
        print("  • Missing odds: Median imputation")
        print("  • Missing stats: League average")
        print("  • Missing form: Neutral values")
        print("="*70)
    
    pipeline = FeatureEngineeringPipeline(verbose=not args.quiet)
    
    # Get input files
    input_files = args.files if args.files else []
    
    # If --all flag, find all data files in data-dir
    if args.all or not input_files:
        data_dir = Path(args.data_dir)
        input_files = []
        for pattern in ['Matches_*.xls', 'Matches_*.xlsx', 'Matches_*.csv', '*.xls', '*.xlsx', '*.csv']:
            input_files += [str(f) for f in data_dir.glob(pattern)]
        # Remove duplicates and sort
        input_files = sorted(set(input_files))
    
    if not input_files:
        print("\n❌ No data files found!")
        print(f"   Searched in: {Path(args.data_dir).absolute()}")
        print("\nUsage:")
        print("  python feature_engineering_pipeline.py --all                    # All files in current dir")
        print("  python feature_engineering_pipeline.py --all --data-dir ./data  # All files in ./data")
        print("  python feature_engineering_pipeline.py Matches_E0.xls           # Single file")
        print("\nSupported formats: .csv, .xls, .xlsx")
        return
    
    if not args.quiet:
        print(f"\n📁 Found {len(input_files)} data files to process")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    successful = 0
    for filepath in input_files:
        if not os.path.exists(filepath):
            print(f"\n❌ File not found: {filepath}")
            continue
        
        filename = Path(filepath).stem
        output_path = output_dir / f"{filename}_features.csv"
        
        df = pipeline.process_league(filepath, str(output_path))
        
        if len(df) > 0:
            successful += 1
        
        # Summary
        if not args.quiet and len(df) > 0:
            print(f"\n📊 {filename} Summary:")
            print(f"   Matches: {len(df):,}")
            if 'target_home' in df.columns:
                print(f"   Home wins: {df['target_home'].sum():,} ({df['target_home'].mean()*100:.1f}%)")
                print(f"   Draws: {df['target_draw'].sum():,} ({df['target_draw'].mean()*100:.1f}%)")
                print(f"   Away wins: {df['target_away'].sum():,} ({df['target_away'].mean()*100:.1f}%)")
            
            # Top teams
            print(f"\n   Top 5 teams by ELO:")
            elo_rankings = sorted(pipeline.team_elo.items(), key=lambda x: -x[1])[:5]
            for team, elo in elo_rankings:
                print(f"      {team}: {elo:.0f}")
    
    if not args.quiet:
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE")
        print(f"   Processed: {successful}/{len(input_files)} leagues")
        print(f"   Output directory: {output_dir.absolute()}")
        print("="*70)
        
        # List output files
        print("\n📁 Output files:")
        for f in sorted(output_dir.glob('*_features.csv')):
            print(f"   {f.name}")


if __name__ == "__main__":
    main()
