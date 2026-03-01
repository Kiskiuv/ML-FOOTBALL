#!/usr/bin/env python3
"""
V4 Lean Feature Engineering — 26 BASIC+ features from raw data.

Reads pre-processed Matches_*.xls/csv files from data/ and computes
exactly the 26 features used by the V4 Lean protocol (RF + LR models).
ELO is rebuilt from scratch per league. No odds leakage.

Usage:
    python scripts/feature_engineering_v4_lean.py --all --data-dir ./data --output-dir ./features_v4
    python scripts/feature_engineering_v4_lean.py --league SP1 --data-dir ./data --output-dir ./features_v4
"""

import argparse
import sys
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
    "h2h_home_wins", "h2h_away_wins", "h2h_draws", "h2h_home_goals_diff",
    # Goals/Venue (5) — rolling 10-match venue-specific averages
    "home_team_goals_scored_avg", "home_team_goals_conceded_avg",
    "away_team_goals_scored_avg", "away_team_goals_conceded_avg",
    "venue_ppg_diff",
]

# ELO parameters (matching feature_engineering_pipeline.py)
ELO_K = 20
ELO_DEFAULT = 1500.0
ELO_HFA_INITIAL = 65.0
ELO_HFA_MIN = 30.0
ELO_HFA_MAX = 120.0
ELO_HFA_ADAPT = 0.075

CONTEXT_COLS = [
    "Division", "MatchDate", "HomeTeam", "AwayTeam",
    "FTHome", "FTAway", "FTResult",
]


# ---------------------------------------------------------------------------
# Per-league mutable state
# ---------------------------------------------------------------------------

VENUE_ROLLING_WINDOW = 10

@dataclass
class LeagueState:
    elo: Dict[str, float] = field(default_factory=dict)
    hfa: float = ELO_HFA_INITIAL
    form: Dict[str, List[int]] = field(default_factory=dict)
    last_match_date: Dict[str, pd.Timestamp] = field(default_factory=dict)
    h2h: Dict[Tuple[str, str], List[dict]] = field(default_factory=dict)
    home_matches: Dict[str, List[dict]] = field(default_factory=dict)
    away_matches: Dict[str, List[dict]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature engineer
# ---------------------------------------------------------------------------

class V4LeanFeatureEngineer:
    """Computes 26 BASIC+ features from raw match data."""

    def __init__(self):
        self.state: Optional[LeagueState] = None

    # --- Data loading -------------------------------------------------------

    @staticmethod
    def _load_file(path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, encoding="latin-1")
        df = df.dropna(how="all").reset_index(drop=True)
        return df

    @staticmethod
    def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
        col = df["MatchDate"]
        for fmt in ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%m/%d/%Y"]:
            try:
                df["MatchDate"] = pd.to_datetime(col, format=fmt, errors="raise")
                return df
            except (ValueError, TypeError):
                continue
        df["MatchDate"] = pd.to_datetime(col, dayfirst=True, errors="coerce")
        return df

    # --- Feature reads (no state mutation) ----------------------------------

    def _get_elo(self, team: str) -> float:
        return self.state.elo.get(team, ELO_DEFAULT)

    @staticmethod
    def _expected(elo_diff_with_hfa: float) -> float:
        return 1.0 / (10 ** (-elo_diff_with_hfa / 400) + 1)

    def _get_form(self, team: str, n: int) -> int:
        results = self.state.form.get(team, [])
        if not results:
            return n  # neutral: 1 pt per match
        return sum(results[-n:])

    def _get_rest_days(self, team: str, match_date: pd.Timestamp) -> int:
        last = self.state.last_match_date.get(team)
        if last is None or pd.isna(last):
            return 7
        return max(int((match_date - last).days), 1)

    def _get_h2h(self, home: str, away: str) -> dict:
        key = tuple(sorted([home, away]))
        records = self.state.h2h.get(key, [])[-10:]

        home_wins = away_wins = draws = 0
        goals_diff = 0

        for rec in records:
            if rec["result"] == "H":
                if rec["home"] == home:
                    home_wins += 1
                else:
                    away_wins += 1
            elif rec["result"] == "A":
                if rec["away"] == away:
                    away_wins += 1
                else:
                    home_wins += 1
            else:
                draws += 1
            # goal diff from current home team's perspective
            if rec["home"] == home:
                goals_diff += rec["hg"] - rec["ag"]
            else:
                goals_diff += rec["ag"] - rec["hg"]

        return {
            "h2h_home_wins": home_wins,
            "h2h_away_wins": away_wins,
            "h2h_draws": draws,
            "h2h_home_goals_diff": goals_diff,
        }

    def _get_venue_stats(self, team: str, venue: str) -> dict:
        """Get rolling stats for team at home or away (last VENUE_ROLLING_WINDOW matches)."""
        matches = (self.state.home_matches if venue == "home" else self.state.away_matches).get(team, [])
        matches = matches[-VENUE_ROLLING_WINDOW:]
        if not matches:
            if venue == "home":
                return {"ppg": 1.5, "goals_scored": 1.3, "goals_conceded": 1.1}
            else:
                return {"ppg": 1.0, "goals_scored": 1.0, "goals_conceded": 1.4}
        return {
            "ppg": np.mean([m["points"] for m in matches]),
            "goals_scored": np.mean([m["goals_for"] for m in matches]),
            "goals_conceded": np.mean([m["goals_against"] for m in matches]),
        }

    def _compute_features(self, row) -> dict:
        home, away = row["HomeTeam"], row["AwayTeam"]
        match_date = row["MatchDate"]

        # ELO (5)
        home_elo = self._get_elo(home)
        away_elo = self._get_elo(away)
        elo_diff = home_elo - away_elo
        elo_sum = home_elo + away_elo
        elo_mismatch = abs(elo_diff) / elo_sum if elo_sum > 0 else 0.0

        # Form (6)
        f3h = self._get_form(home, 3)
        f5h = self._get_form(home, 5)
        f3a = self._get_form(away, 3)
        f5a = self._get_form(away, 5)

        # Momentum (2) — 2*form3 - form5
        mom_h = 2 * f3h - f5h
        mom_a = 2 * f3a - f5a

        # Schedule (2)
        rd = self._get_rest_days(home, match_date) - self._get_rest_days(away, match_date)
        # Binned at ±2 / ±4 (matching feature_engineering_pipeline.py)
        rd_nl = -2 if rd <= -4 else (-1 if rd <= -2 else (0 if rd <= 2 else (1 if rd <= 4 else 2)))

        # Temporal (2)
        month = match_date.month
        winter = 1 if month in {11, 12, 1, 2} else 0
        if month in {8, 9, 10}:
            phase = 1
        elif month in {11, 12, 1, 2}:
            phase = 2
        else:
            phase = 3

        # H2H (4)
        h2h = self._get_h2h(home, away)

        # Goals/Venue (5)
        home_stats = self._get_venue_stats(home, "home")
        away_stats = self._get_venue_stats(away, "away")

        return {
            "HomeElo": round(home_elo, 1),
            "AwayElo": round(away_elo, 1),
            "elo_diff": round(elo_diff, 1),
            "elo_sum": round(elo_sum, 1),
            "elo_mismatch": round(elo_mismatch, 4),
            "Form3Home_ratio": round(f3h / 9.0, 4),
            "Form5Home_ratio": round(f5h / 15.0, 4),
            "Form3Away_ratio": round(f3a / 9.0, 4),
            "Form5Away_ratio": round(f5a / 15.0, 4),
            "form_diff_3": f3h - f3a,
            "form_diff_5": f5h - f5a,
            "rest_diff": rd,
            "rest_diff_nonlinear": rd_nl,
            "winter_period": winter,
            "season_phase": phase,
            "form_momentum_home": mom_h,
            "form_momentum_away": mom_a,
            **h2h,
            "home_team_goals_scored_avg": round(home_stats["goals_scored"], 3),
            "home_team_goals_conceded_avg": round(home_stats["goals_conceded"], 3),
            "away_team_goals_scored_avg": round(away_stats["goals_scored"], 3),
            "away_team_goals_conceded_avg": round(away_stats["goals_conceded"], 3),
            "venue_ppg_diff": round(home_stats["ppg"] - away_stats["ppg"], 3),
        }

    # --- State updates (mutate AFTER feature computation) -------------------

    def _update_elo(self, home: str, away: str, hg: int, ag: int):
        home_elo = self.state.elo.get(home, ELO_DEFAULT)
        away_elo = self.state.elo.get(away, ELO_DEFAULT)

        expected_home = self._expected(home_elo - away_elo + self.state.hfa)

        if hg > ag:
            actual, margin = 1.0, hg - ag
        elif hg < ag:
            actual, margin = 0.0, ag - hg
        else:
            actual, margin = 0.5, 0

        change = ELO_K * (actual - expected_home) * sqrt(max(margin, 1))

        self.state.elo[home] = home_elo + change
        self.state.elo[away] = away_elo - change
        self.state.hfa = max(ELO_HFA_MIN, min(ELO_HFA_MAX,
                                               self.state.hfa + change * ELO_HFA_ADAPT))

    def _update_state(self, row):
        home, away = row["HomeTeam"], row["AwayTeam"]
        hg, ag = int(row["FTHome"]), int(row["FTAway"])
        match_date = row["MatchDate"]

        if hg > ag:
            result, hp, ap = "H", 3, 0
        elif hg < ag:
            result, hp, ap = "A", 0, 3
        else:
            result, hp, ap = "D", 1, 1

        self._update_elo(home, away, hg, ag)

        self.state.form.setdefault(home, []).append(hp)
        self.state.form.setdefault(away, []).append(ap)

        # Venue stats (goals + points per venue)
        self.state.home_matches.setdefault(home, []).append(
            {"goals_for": hg, "goals_against": ag, "points": hp}
        )
        self.state.away_matches.setdefault(away, []).append(
            {"goals_for": ag, "goals_against": hg, "points": ap}
        )

        key = tuple(sorted([home, away]))
        self.state.h2h.setdefault(key, []).append(
            {"home": home, "away": away, "hg": hg, "ag": ag, "result": result}
        )

        self.state.last_match_date[home] = match_date
        self.state.last_match_date[away] = match_date

    # --- Main processing ----------------------------------------------------

    def process_league(self, filepath: Path, output_dir: Path) -> pd.DataFrame:
        self.state = LeagueState()
        df = self._load_file(filepath)

        required = ["MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  SKIP {filepath.name}: missing columns {missing}")
            return pd.DataFrame()

        df = self._parse_dates(df)
        df = df.dropna(subset=["MatchDate", "FTHome", "FTAway"]).reset_index(drop=True)
        df["FTHome"] = df["FTHome"].astype(int)
        df["FTAway"] = df["FTAway"].astype(int)
        df = df.sort_values("MatchDate").reset_index(drop=True)

        league_code = str(df["Division"].iloc[0]).strip() if "Division" in df.columns else filepath.stem.replace("Matches_", "")
        print(f"  {league_code} ({filepath.name}): {len(df)} matches")

        # Compute features row-by-row (chronological)
        feat_rows = []
        for _, row in df.iterrows():
            try:
                feats = self._compute_features(row)
                self._update_state(row)
            except Exception as e:
                print(f"    Error: {e}")
                feats = {f: np.nan for f in FEATURES}
                try:
                    self._update_state(row)
                except Exception:
                    pass
            feat_rows.append(feats)

        feat_df = pd.DataFrame(feat_rows)

        # Build output dataframe — include odds columns if available
        ctx_cols = ["MatchDate", "HomeTeam", "AwayTeam", "FTHome", "FTAway"]
        # Carry through odds for edge calculation (not used as model features)
        odds_cols = ["OddHome", "OddDraw", "OddAway", "MaxHome", "MaxDraw", "MaxAway",
                     "OddUnder25", "OddOver25", "MaxUnder25", "MaxOver25"]
        # Map raw column names to standardized names
        raw_odds_map = {"Avg>2.5": "OddOver25", "Avg<2.5": "OddUnder25",
                        "Max>2.5": "MaxOver25", "Max<2.5": "MaxUnder25",
                        "Over25": "OddOver25", "Under25": "OddUnder25",
                        "AvgH": "OddHome", "AvgD": "OddDraw", "AvgA": "OddAway",
                        "MaxH": "MaxHome", "MaxD": "MaxDraw", "MaxA": "MaxAway"}
        for raw_name, std_name in raw_odds_map.items():
            if raw_name in df.columns and std_name not in df.columns:
                df[std_name] = df[raw_name]
        for oc in odds_cols:
            if oc in df.columns:
                ctx_cols.append(oc)
        ctx = df[ctx_cols].copy()
        ctx.insert(0, "Division", league_code)

        # Compute FTResult from scores (don't trust raw column)
        ctx["FTResult"] = np.where(
            df["FTHome"] > df["FTAway"], "H",
            np.where(df["FTHome"] < df["FTAway"], "A", "D")
        )

        # Targets after features
        out = pd.concat([ctx.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
        out["target_home"] = (out["FTResult"] == "H").astype(int)
        out["target_draw"] = (out["FTResult"] == "D").astype(int)
        out["target_away"] = (out["FTResult"] == "A").astype(int)
        out["target_UNDER25"] = ((out["FTHome"] + out["FTAway"]) < 3).astype(int)

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{league_code}_features.csv"
        out.to_csv(out_path, index=False)
        print(f"  -> {out_path} ({len(out)} rows, {len(FEATURES)} features)")
        return out

    def process_all(self, data_dir: Path, output_dir: Path):
        files = sorted(
            list(data_dir.glob("Matches_*.xls"))
            + list(data_dir.glob("Matches_*.csv"))
        )
        if not files:
            print(f"No Matches_* files found in {data_dir}")
            sys.exit(1)

        print(f"Found {len(files)} league files in {data_dir}\n")
        for f in tqdm(files, desc="Leagues"):
            try:
                self.process_league(f, output_dir)
            except Exception as e:
                print(f"  FAILED {f.name}: {e}")
        print(f"\nDone. Output in {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="V4 Lean Feature Engineering (26 BASIC+ features)"
    )
    parser.add_argument("--all", action="store_true", help="Process all leagues")
    parser.add_argument("--league", type=str, help="Single league code (e.g. SP1)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory with Matches_*.xls files")
    parser.add_argument("--output-dir", type=str, default="./features_v4",
                        help="Output directory for feature CSVs")
    args = parser.parse_args()

    if not args.all and not args.league:
        parser.error("Specify --all or --league LEAGUE")

    eng = V4LeanFeatureEngineer()

    if args.all:
        eng.process_all(Path(args.data_dir), Path(args.output_dir))
    else:
        data_dir = Path(args.data_dir)
        # Try several filename patterns
        candidates = [
            data_dir / f"Matches_{args.league}.xls",
            data_dir / f"Matches__{args.league}.xls",
            data_dir / f"Matches_{args.league}.csv",
        ]
        found = next((c for c in candidates if c.exists()), None)
        if not found:
            hits = list(data_dir.glob(f"Matches_*{args.league}*"))
            found = hits[0] if hits else None
        if not found:
            print(f"Could not find file for league {args.league} in {data_dir}")
            sys.exit(1)
        eng.process_league(found, Path(args.output_dir))


if __name__ == "__main__":
    main()
