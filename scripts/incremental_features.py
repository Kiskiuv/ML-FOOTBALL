"""
INCREMENTAL FEATURE ENGINEERING - New Season Update
====================================================

Uses historical data (all previous seasons) to build team states
(ELO, form, H2H, rolling stats), then applies that state to
compute features for new season matches.

STRATEGY: COMBINE -> BUILD STATE -> DATE CUTOFF -> OUTPUT
  1. Load raw historical data (Matches_*.xls)
  2. Load new season file (*_20252026.xlsx) -> remap columns
  3. COMBINE both datasets, deduplicate, sort chronologically
  4. Find cutoff date from existing features file
  5. Process ALL matches chronologically (build state + compute features)
  6. Output ONLY features for matches AFTER the cutoff date
  7. Optionally append to existing historical features file

WHY THIS APPROACH:
  The historical file and new season file often come from DIFFERENT sources
  with different coverage for the same seasons. Key-based dedup fails because
  source A may have 169 matches for 2012/13 while source B has 180.
  By COMBINING both and using a DATE CUTOFF, we:
  - Build the most complete historical state possible
  - Avoid "phantom new" matches from old seasons leaking through
  - Only output truly new matches (post-cutoff)

HANDLES:
  - Standard format files (HomeTeam, AwayTeam, MatchDate, OddHome, etc.)
  - Alternative format files (Home, Away, Date, AvgCH, etc.)
  - New season files that contain ENTIRE history (auto-handled by combine)
  - Closing-only odds (AvgCH -> OddHome with warning)
  - Weekly updates: just re-download the season file and re-run

USAGE:
  # Process all leagues
  python incremental_features.py --all

  # Process specific league
  python incremental_features.py --league AUT

  # Custom directories
  python incremental_features.py --all --historical ./data --new-season ./20252026_season --output ./features

  # Append new matches to existing features file
  python incremental_features.py --league AUT --append

Author: Marc | February 2026 | Protocol V3.4.3
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
import argparse
from pathlib import Path

warnings.filterwarnings('ignore')


# =============================================================================
# LEAGUE MAPPING: Maps new season filenames to historical filenames
# =============================================================================
LEAGUE_MAP = {
    'ARG':       ('Matches_ARGENTINA',  'ARGENTINA'),
    'AUT':       ('Matches_AUT',        'AUT'),
    'B1':        ('Matches_BELGIUM',    'B1'),
    'BRA':       ('Matches_BRASIL',     'BRASIL'),
    'CHN':       ('Matches_CHN',        'CHN'),
    'D1':        ('Matches_D1',         'D1'),
    'D2':        ('Matches_D2',         'D2'),
    'DEN':       ('Matches_DEN',        'DEN'),
    'DNK':       ('Matches_DEN',        'DEN'),
    'E0':        ('Matches__E0',        'E0'),
    'E1':        ('Matches_E1',         'E1'),
    'E2':        ('Matches_E2',         'E2'),
    'E3':        ('Matches_E3',         'E3'),
    'EC':        ('Matches_EC',         'EC'),
    'F1':        ('Matches_F1',         'F1'),
    'F2':        ('Matches_F2',         'F2'),
    'FIN':       ('Matches_FIN',        'FIN'),
    'GI':        ('Matches_GI',         'GI'),
    'G1':        ('Matches_GI',         'GI'),
    'I1':        ('Matches_I1',         'I1'),
    'I2':        ('Matches_I2',         'I2'),
    'IRL':       ('Matches_IRL',        'IRL'),
    'JAP':       ('Matches_JAP',        'JAP'),
    'JPN':       ('Matches_JAP',        'JAP'),
    'MEX':       ('Matches_MEXICO',     'MEXICO'),
    'N1':        ('Matches_N1',         'N1'),
    'NOR':       ('Matches_NOR',        'NOR'),
    'P1':        ('Matches_P1',         'P1'),
    'POL':       ('Matches_POL',        'POL'),
    'ROM':       ('Matches_ROM',        'ROM'),
    'ROU':       ('Matches_ROM',        'ROM'),
    'RUS':       ('Matches_RUS',        'RUS'),
    'SC0':       ('Matches_SC0',        'SC0'),
    'SC1':       ('Matches_SC1',        'SC1'),
    'SC2':       ('Matches_SC2',        'SC2'),
    'SC3':       ('Matches_SC3',        'SC3'),
    'SP1':       ('Matches_SP1',        'SP1'),
    'SP2':       ('Matches_SP2',        'SP2'),
    'SUI':       ('Matches_SUI',        'SUI'),
    'SWZ':       ('Matches_SUI',        'SUI'),
    'SWE':       ('Matches_SWE',        'SWE'),
    'T1':        ('Matches_T1',         'T1'),
    'USA':       ('Matches_USA',        'USA'),
}


# =============================================================================
# COLUMN REMAPPING: Alternative data source formats -> standard format
# =============================================================================
NEW_FORMAT_COLUMN_MAP = {
    'Home':     'HomeTeam',
    'Away':     'AwayTeam',
    'HG':       'FTHG',
    'AG':       'FTAG',
    'Res':      'FTR',
}

CLOSING_TO_OPENING_MAP = {
    'AvgCH':    'AvgH',
    'AvgCD':    'AvgD',
    'AvgCA':    'AvgA',
    'MaxCH':    'MaxH',
    'MaxCD':    'MaxD',
    'MaxCA':    'MaxA',
    'B365CH':   'B365H',
    'B365CD':   'B365D',
    'B365CA':   'B365A',
    'B36CA':    'B365A',
    'PSCH':     'PSH',
    'PSCD':     'PSD',
    'PSCA':     'PSA',
    'BFECH':    'BFEH',
    'BFECD':    'BFED',
    'BFECA':    'BFEA',
    'AvgC>2.5': 'Avg>2.5',
    'AvgC<2.5': 'Avg<2.5',
    'MaxC>2.5': 'Max>2.5',
    'MaxC<2.5': 'Max<2.5',
    'AvgCAHH':  'AvgAHH',
    'AvgCAHA':  'AvgAHA',
    'AHCh':     'AHh',
}

NEW_FORMAT_DROP_COLS = ['Country', 'League']


# =============================================================================
# FILE I/O HELPERS
# =============================================================================

def load_file(filepath: str) -> pd.DataFrame:
    """Load a data file with multiple fallback methods."""
    filepath_lower = filepath.lower()
    load_methods = []

    if filepath_lower.endswith('.xlsx'):
        load_methods = [
            ('openpyxl', lambda: pd.read_excel(filepath, engine='openpyxl')),
        ]
    elif filepath_lower.endswith('.xls'):
        load_methods = [
            ('xlrd', lambda: pd.read_excel(filepath, engine='xlrd')),
            ('openpyxl', lambda: pd.read_excel(filepath, engine='openpyxl')),
            ('csv_utf8', lambda: pd.read_csv(filepath, encoding='utf-8-sig')),
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
                return df
        except:
            continue

    return None


def find_file(directory: str, base_name: str) -> Optional[str]:
    """Find a file with any supported extension in a directory."""
    directory = Path(directory)
    for ext in ['.csv', '.xls', '.xlsx']:
        path = directory / f"{base_name}{ext}"
        if path.exists():
            return str(path)
    return None


def detect_league_from_filename(filename: str) -> Optional[str]:
    """Extract league code from new season filename."""
    stem = Path(filename).stem
    if '_2025' in stem:
        league = stem.split('_2025')[0]
        return league.upper()
    if '_features' in stem:
        league = stem.replace('Matches_', '').replace('Matches__', '').split('_features')[0]
        return league.upper()
    return stem.upper()


# =============================================================================
# COLUMN REMAPPING & FORMAT DETECTION
# =============================================================================

def detect_and_remap_columns(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, bool, bool]:
    """
    Detect alternative column format and remap to standard.
    Returns: (remapped_df, was_remapped, used_closing_as_fallback)
    """
    cols = set(df.columns)
    was_remapped = False

    is_new_format = ('Home' in cols and 'HomeTeam' not in cols)

    CLOSING_PATTERNS = ['CH', 'CD', 'CA', 'C>2.5', 'C<2.5', 'CAHH', 'CAHA']
    closing_cols = [c for c in cols if any(c.endswith(p) for p in CLOSING_PATTERNS)]
    has_closing_odds = len(closing_cols) > 0
    has_opening_odds = any(c in cols and df[c].notna().any() for c in ['AvgH', 'AvgD', 'AvgA', 'OddHome', 'OddDraw', 'OddAway'])

    if not is_new_format and not has_closing_odds:
        return df, False, False

    # Structural remapping
    remap = {k: v for k, v in NEW_FORMAT_COLUMN_MAP.items() if k in cols}
    if remap:
        if verbose:
            print(f"  Column remapping: {len(remap)} structural columns")
            for old, new in remap.items():
                print(f"     {old:10s} -> {new}")
        df = df.rename(columns=remap)
        was_remapped = True

    # Drop metadata
    drop_cols = [c for c in NEW_FORMAT_DROP_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        if verbose:
            print(f"  Dropped metadata columns: {drop_cols}")

    # Handle closing odds
    used_closing = False
    if has_closing_odds:
        if has_opening_odds:
            # Fill any NaN opening odds from closing odds before dropping
            for closing_col, opening_col in CLOSING_TO_OPENING_MAP.items():
                if closing_col in df.columns and opening_col in df.columns:
                    n_filled = df[opening_col].isna().sum()
                    if n_filled > 0:
                        df[opening_col] = df[opening_col].fillna(df[closing_col])
                        if verbose and n_filled > 0:
                            print(f"  Filled {n_filled} NaN in {opening_col} from {closing_col}")
                            used_closing = True
            df = df.drop(columns=[c for c in closing_cols if c in df.columns], errors='ignore')
            if verbose:
                if used_closing:
                    print(f"  Dropped {len(closing_cols)} closing odds columns (after filling gaps)")
                else:
                    print(f"  Dropped {len(closing_cols)} closing odds columns (pre-match available)")
        else:
            closing_remap = {k: v for k, v in CLOSING_TO_OPENING_MAP.items() if k in df.columns}
            if closing_remap:
                df = df.rename(columns=closing_remap)
                used_closing = True
                if verbose:
                    print(f"  WARNING: No pre-match odds -- using {len(closing_remap)} CLOSING odds as fallback")
                    print(f"     Acceptable for feature engineering & predictions,")
                    print(f"     but for actual bet placement always use real pre-match odds.")

    return df, was_remapped, used_closing


# =============================================================================
# DATE PARSING
# =============================================================================

def parse_dates(df: pd.DataFrame, col: str = 'MatchDate') -> pd.DataFrame:
    """Parse MatchDate column with multiple format fallbacks."""
    if col not in df.columns:
        return df
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return df

    for date_format in ['%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d', '%m/%d/%Y']:
        try:
            df[col] = pd.to_datetime(df[col], format=date_format)
            return df
        except (ValueError, TypeError):
            continue

    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
    return df


# =============================================================================
# COMBINE & DEDUPLICATE TWO DATA SOURCES
# =============================================================================

def combine_data_sources(
    df_hist: pd.DataFrame,
    df_new: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Combine historical and new data into a single chronological timeline.
    Removes exact duplicate matches (same date + home + away), keeping
    the row with more complete data (more non-null columns).

    This handles the common case where the two files have overlapping but
    different coverage for the same seasons.
    """
    if verbose:
        print(f"\n  Combining data sources:")
        print(f"     Historical: {len(df_hist):,} matches")
        print(f"     New file:   {len(df_new):,} matches")

    # Build composite match key
    def add_match_key(df):
        dates = pd.to_datetime(df['MatchDate'], errors='coerce').dt.strftime('%Y-%m-%d')
        homes = df['HomeTeam'].astype(str).str.strip()
        aways = df['AwayTeam'].astype(str).str.strip()
        df = df.copy()
        df['_match_key'] = dates + '|' + homes + '|' + aways
        return df

    df_hist = add_match_key(df_hist)
    df_new = add_match_key(df_new)

    # Count non-null columns per row (to keep the most complete row)
    df_hist['_data_completeness'] = df_hist.notna().sum(axis=1)
    df_new['_data_completeness'] = df_new.notna().sum(axis=1)

    # Tag source
    df_hist['_source'] = 'historical'
    df_new['_source'] = 'new_file'

    # Concatenate
    df_combined = pd.concat([df_hist, df_new], ignore_index=True)

    # Sort by match key + completeness (most complete first), then drop dupes
    df_combined = df_combined.sort_values(
        ['_match_key', '_data_completeness'], ascending=[True, False]
    )
    n_before = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset='_match_key', keep='first')
    n_dupes = n_before - len(df_combined)

    # Sort chronologically
    df_combined = parse_dates(df_combined, 'MatchDate')
    df_combined = df_combined.sort_values('MatchDate').reset_index(drop=True)

    # Stats
    from_hist = (df_combined['_source'] == 'historical').sum()
    from_new = (df_combined['_source'] == 'new_file').sum()

    if verbose:
        print(f"     Combined:   {len(df_combined):,} unique matches")
        print(f"     Duplicates: {n_dupes:,} removed")
        print(f"     Kept from historical: {from_hist:,}")
        print(f"     Kept from new file:   {from_new:,}")
        date_range_start = df_combined['MatchDate'].min()
        date_range_end = df_combined['MatchDate'].max()
        if pd.notna(date_range_start):
            print(f"     Date range: {date_range_start.date()} -> {date_range_end.date()}")

    # Clean up temp columns
    df_combined = df_combined.drop(columns=['_match_key', '_data_completeness', '_source'])

    return df_combined


# =============================================================================
# FIND CUTOFF DATE FROM EXISTING FEATURES FILE
# =============================================================================

def find_cutoff_date(
    features_path: Optional[str],
    verbose: bool = True
) -> Optional[pd.Timestamp]:
    """
    Find the last date in the existing features file.
    Matches AFTER this date are considered 'new' and will be output.
    """
    if features_path is None:
        return None

    try:
        df = pd.read_csv(features_path, usecols=['MatchDate'])
        df['MatchDate'] = pd.to_datetime(df['MatchDate'], errors='coerce')
        cutoff = df['MatchDate'].max()
        if verbose and pd.notna(cutoff):
            print(f"  Cutoff date from features: {cutoff.date()}")
            print(f"  Will output features for matches AFTER this date")
        return cutoff
    except Exception as e:
        if verbose:
            print(f"  Could not determine cutoff from features file: {e}")
        return None


# =============================================================================
# MAIN INCREMENTAL PROCESSING
# =============================================================================

def process_league_incremental(
    historical_path: str,
    new_season_path: str,
    output_path: str,
    existing_features_path: Optional[str] = None,
    division_code: str = None,
    verbose: bool = True
) -> Optional[pd.DataFrame]:
    """
    Process a league incrementally using COMBINE + DATE CUTOFF strategy.

    1. Load both data sources -> remap columns -> combine
    2. Find cutoff date from existing features
    3. Process ALL matches chronologically (state building + feature computation)
    4. Output ONLY features for matches AFTER the cutoff
    """
    # Import the pipeline
    sys.path.insert(0, str(Path(__file__).parent))
    from feature_engineering_pipeline import FeatureEngineeringPipeline

    pipeline = FeatureEngineeringPipeline(verbose=False)

    if verbose:
        print(f"\n{'='*70}")
        print(f"INCREMENTAL FEATURE ENGINEERING")
        print(f"{'='*70}")
        print(f"  Historical: {Path(historical_path).name}")
        print(f"  New season: {Path(new_season_path).name}")
        if existing_features_path:
            print(f"  Features:   {Path(existing_features_path).name}")
        if division_code:
            print(f"  Division:   {division_code}")
        print(f"{'='*70}")

    # =========================================================================
    # STEP 1: Load historical data
    # =========================================================================
    if verbose:
        print(f"\n--- Step 1: Loading historical data...")

    df_hist = load_file(historical_path)
    if df_hist is None or len(df_hist) == 0:
        print(f"  ERROR: Could not load historical file: {historical_path}")
        return None

    # Detect if this is already a features file (has standardized columns)
    hist_is_features = 'HomeElo' in df_hist.columns or '_features' in historical_path

    if hist_is_features:
        if verbose:
            print(f"  [Using pre-engineered features file as historical source]")
            print(f"  Note: Rolling stats (shots/corners/discipline) will use league averages")
        # Already has standardized column names - just parse dates
        df_hist = parse_dates(df_hist, 'MatchDate')
    else:
        df_hist, hist_remapped, _ = detect_and_remap_columns(df_hist, verbose=verbose)
        df_hist = pipeline._standardize_columns(df_hist)
        df_hist = parse_dates(df_hist, 'MatchDate')

    if 'Division' not in df_hist.columns or df_hist['Division'].isna().all():
        if division_code:
            df_hist['Division'] = division_code

    if verbose:
        print(f"  Loaded {len(df_hist):,} historical matches")
        print(f"  Date range: {df_hist['MatchDate'].min().date()} to {df_hist['MatchDate'].max().date()}")

    # =========================================================================
    # STEP 2: Load new season data
    # =========================================================================
    if verbose:
        print(f"\n--- Step 2: Loading new season data...")

    df_new = load_file(new_season_path)
    if df_new is None or len(df_new) == 0:
        print(f"  ERROR: Could not load new season file: {new_season_path}")
        return None

    if verbose:
        print(f"  Loaded {len(df_new):,} rows from new file")

    df_new, was_remapped, closing_odds_only = detect_and_remap_columns(df_new, verbose=verbose)
    df_new = pipeline._standardize_columns(df_new)
    df_new = parse_dates(df_new, 'MatchDate')

    if 'Division' not in df_new.columns or df_new['Division'].isna().all():
        if division_code:
            df_new['Division'] = division_code

    if verbose:
        print(f"  Date range: {df_new['MatchDate'].min().date()} to {df_new['MatchDate'].max().date()}")

    # =========================================================================
    # STEP 3: Combine both sources + find cutoff
    # =========================================================================
    if verbose:
        print(f"\n--- Step 3: Combining data sources...")

    df_combined = combine_data_sources(df_hist, df_new, verbose=verbose)

    # Find cutoff date
    cutoff_date = find_cutoff_date(existing_features_path, verbose=verbose)

    if cutoff_date is None:
        # Fallback: use the last date from the historical raw file
        cutoff_date = df_hist['MatchDate'].max()
        if verbose:
            print(f"  No features file found -- using historical max date as cutoff: {cutoff_date.date()}")

    # Count how many matches will be new
    n_new = (df_combined['MatchDate'] > cutoff_date).sum()
    n_state = (df_combined['MatchDate'] <= cutoff_date).sum()

    if verbose:
        print(f"\n  Timeline split:")
        print(f"     State-building matches (before cutoff): {n_state:,}")
        print(f"     New matches (after cutoff):             {n_new:,}")

    if n_new == 0:
        print(f"\n  WARNING: No new matches found after cutoff date {cutoff_date.date()}!")
        print(f"      Check that your new season file contains data beyond this date.")
        empty_df = pd.DataFrame(columns=['Division', 'MatchDate', 'MatchTime',
                                          'HomeTeam', 'AwayTeam'])
        empty_df.to_csv(output_path, index=False)
        return empty_df

    # =========================================================================
    # STEP 4: Process ALL matches chronologically
    # =========================================================================
    if verbose:
        print(f"\n--- Step 4: Processing {len(df_combined):,} matches chronologically...")
        print(f"     Building state from {n_state:,} historical matches")
        print(f"     Computing features for {n_new:,} new matches")

    all_features = []
    new_match_indices = []

    for idx, row in df_combined.iterrows():
        if pd.isna(row.get('HomeTeam')) or pd.isna(row.get('AwayTeam')):
            continue

        match_date = pd.to_datetime(row['MatchDate'])
        is_new_match = match_date > cutoff_date

        if is_new_match:
            # Compute features for this new match
            features = pipeline.compute_features(row)
            all_features.append(features)
            new_match_indices.append(idx)

        # Update state AFTER feature computation (for ALL played matches)
        if pd.notna(row.get('FTHome')) and pd.notna(row.get('FTAway')):
            pipeline.update_histories(row)

    if verbose:
        print(f"  State built: {len(pipeline.team_elo)} teams tracked")
        top_teams = sorted(pipeline.team_elo.items(), key=lambda x: -x[1])[:5]
        for team, elo in top_teams:
            print(f"     {team}: {elo:.0f}")

    # =========================================================================
    # STEP 5: Build output DataFrame
    # =========================================================================
    if verbose:
        print(f"\n--- Step 5: Building output ({len(all_features):,} matches)...")

    features_df = pd.DataFrame(all_features)

    # Get the new match rows from combined data
    df_new_matches = df_combined.loc[new_match_indices].reset_index(drop=True)
    features_df = features_df.reset_index(drop=True)

    # Select key columns
    keep_cols = ['Division', 'MatchDate', 'MatchTime', 'HomeTeam', 'AwayTeam',
                 'FTHome', 'FTAway', 'FTResult', 'HTHome', 'HTAway', 'HTResult',
                 'OddHome', 'OddDraw', 'OddAway', 'MaxHome', 'MaxDraw', 'MaxAway',
                 'OddUnder25', 'OddOver25', 'MaxUnder25', 'MaxOver25']
    keep_cols = [c for c in keep_cols if c in df_new_matches.columns]

    result_df = pd.concat([df_new_matches[keep_cols].reset_index(drop=True), features_df], axis=1)

    # Add target variables
    if 'FTResult' in result_df.columns:
        result_df['target_home'] = (result_df['FTResult'] == 'H').astype(int)
        result_df['target_draw'] = (result_df['FTResult'] == 'D').astype(int)
        result_df['target_away'] = (result_df['FTResult'] == 'A').astype(int)
    if 'FTHome' in result_df.columns and 'FTAway' in result_df.columns:
        result_df['target_UNDER25'] = ((result_df['FTHome'] + result_df['FTAway']) < 3).astype(int)

    # Handle remaining NaN (median imputation)
    feature_cols = [c for c in features_df.columns if c in result_df.columns]
    for col in feature_cols:
        if result_df[col].isna().any():
            median_val = result_df[col].median()
            result_df[col] = result_df[col].fillna(median_val if pd.notna(median_val) else 0)

    null_count = result_df[feature_cols].isna().sum().sum()

    # Check data coverage
    odds_coverage = float(result_df['OddHome'].notna().mean() * 100) if 'OddHome' in result_df.columns else 0.0

    if verbose:
        print(f"\n  Output: {len(result_df):,} new matches, {len(feature_cols)} features")
        print(f"  Null values: {null_count}")
        print(f"  Odds coverage:  {odds_coverage:.1f}%")
        if 'target_home' in result_df.columns:
            played = result_df['FTResult'].notna().sum()
            fixtures = result_df['FTResult'].isna().sum()
            print(f"  Played matches:  {played:,}")
            print(f"  Future fixtures: {fixtures:,}")
            if played > 0:
                mask_p = result_df['FTResult'].notna()
                print(f"  Home wins: {result_df.loc[mask_p, 'target_home'].sum():,} "
                      f"({result_df.loc[mask_p, 'target_home'].mean()*100:.1f}%)")
                print(f"  Draws:     {result_df.loc[mask_p, 'target_draw'].sum():,} "
                      f"({result_df.loc[mask_p, 'target_draw'].mean()*100:.1f}%)")
                print(f"  Away wins: {result_df.loc[mask_p, 'target_away'].sum():,} "
                      f"({result_df.loc[mask_p, 'target_away'].mean()*100:.1f}%)")

    # =========================================================================
    # STEP 6: Save output
    # =========================================================================
    result_df.to_csv(output_path, index=False)
    if verbose:
        print(f"\n  Saved: {output_path}")

    # Save JSON summary
    league_code = result_df['Division'].iloc[0] if 'Division' in result_df.columns else 'UNKNOWN'
    result_df_tmp = result_df.copy()
    result_df_tmp['Season'] = result_df_tmp['MatchDate'].apply(
        lambda x: f"{x.year}/{x.year+1}" if pd.notna(x) and x.month >= 7 else
                  (f"{x.year-1}/{x.year}" if pd.notna(x) else 'UNKNOWN')
    )

    top_teams = sorted(pipeline.team_elo.items(), key=lambda x: -x[1])[:10]
    played_mask = result_df['FTResult'].notna() if 'FTResult' in result_df.columns else pd.Series([False]*len(result_df))

    summary = {
        "league": league_code,
        "generated_at": datetime.now().isoformat(),
        "protocol_version": "3.4.3",
        "mode": "incremental_v2_date_cutoff",
        "historical_source": Path(historical_path).name,
        "new_season_source": Path(new_season_path).name,
        "cutoff_date": str(cutoff_date.date()),
        "column_remapping_applied": was_remapped,
        "closing_odds_only": closing_odds_only,
        "data_combination": {
            "historical_matches": len(df_hist),
            "new_file_matches": len(df_new),
            "combined_unique": len(df_combined),
            "state_building_matches": int(n_state),
            "new_output_matches": int(n_new)
        },
        "overview": {
            "total_new_matches": len(result_df),
            "played_matches": int(played_mask.sum()),
            "future_fixtures": int((~played_mask).sum()),
            "total_teams": len(set(result_df['HomeTeam'].unique()) | set(result_df['AwayTeam'].unique())),
            "date_range": {
                "start": str(result_df['MatchDate'].min().date()) if pd.notna(result_df['MatchDate'].min()) else None,
                "end": str(result_df['MatchDate'].max().date()) if pd.notna(result_df['MatchDate'].max()) else None
            },
            "seasons": result_df_tmp['Season'].value_counts().sort_index().to_dict()
        },
        "outcome_distribution": {
            "home_wins": int(result_df.loc[played_mask, 'target_home'].sum()) if 'target_home' in result_df.columns and played_mask.any() else 0,
            "home_win_pct": round(result_df.loc[played_mask, 'target_home'].mean() * 100, 1) if 'target_home' in result_df.columns and played_mask.any() else 0,
            "draws": int(result_df.loc[played_mask, 'target_draw'].sum()) if 'target_draw' in result_df.columns and played_mask.any() else 0,
            "draw_pct": round(result_df.loc[played_mask, 'target_draw'].mean() * 100, 1) if 'target_draw' in result_df.columns and played_mask.any() else 0,
            "away_wins": int(result_df.loc[played_mask, 'target_away'].sum()) if 'target_away' in result_df.columns and played_mask.any() else 0,
            "away_win_pct": round(result_df.loc[played_mask, 'target_away'].mean() * 100, 1) if 'target_away' in result_df.columns and played_mask.any() else 0
        },
        "data_coverage": {
            "odds_coverage_pct": round(odds_coverage, 1),
            "features_computed": len(feature_cols),
            "null_values_remaining": int(null_count)
        },
        "elo_rankings": {
            "top_10": [{"team": t, "elo": round(e, 1)} for t, e in top_teams]
        },
        "state_info": {
            "combined_matches_processed": len(df_combined),
            "teams_in_state": len(pipeline.team_elo),
            "hfa": round(pipeline.league_hfa.get(league_code, 65.0), 1)
        }
    }

    json_path = output_path.replace('.csv', '_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    if verbose:
        print(f"  Summary: {json_path}")

    return result_df


# =============================================================================
# APPEND: Merge new features into existing historical features file
# =============================================================================

def append_to_historical(
    new_features_path: str,
    historical_features_path: str,
    output_path: str = None,
    verbose: bool = True
) -> Optional[pd.DataFrame]:
    """
    Append new features to the existing historical features file.
    Uses date-based dedup to avoid duplicates.
    """
    df_new = pd.read_csv(new_features_path)
    df_hist = pd.read_csv(historical_features_path)

    if output_path is None:
        output_path = historical_features_path

    if verbose:
        print(f"\n  Appending new features to historical...")
        print(f"   Historical: {len(df_hist):,} matches")
        print(f"   New:        {len(df_new):,} matches")

    # Parse dates
    df_new = parse_dates(df_new, 'MatchDate')
    df_hist = parse_dates(df_hist, 'MatchDate')

    # Date-based dedup: only keep new matches after the historical max date
    hist_max_date = df_hist['MatchDate'].max()
    df_to_append = df_new[df_new['MatchDate'] > hist_max_date].copy()

    # Safety: also check key-based for any stragglers ON the cutoff date
    if len(df_to_append) < len(df_new):
        cutoff_matches = df_new[df_new['MatchDate'] == hist_max_date]
        if len(cutoff_matches) > 0:
            def make_keys(df):
                dates = pd.to_datetime(df['MatchDate'], errors='coerce').dt.strftime('%Y-%m-%d')
                homes = df['HomeTeam'].astype(str).str.strip()
                aways = df['AwayTeam'].astype(str).str.strip()
                return dates + '|' + homes + '|' + aways

            hist_keys = set(make_keys(df_hist))
            cutoff_keys = make_keys(cutoff_matches)
            extra = cutoff_matches[~cutoff_keys.isin(hist_keys)]
            if len(extra) > 0:
                df_to_append = pd.concat([extra, df_to_append], ignore_index=True)

    n_skipped = len(df_new) - len(df_to_append)

    if verbose:
        print(f"   Already covered: {n_skipped:,}")
        print(f"   Appending:       {len(df_to_append):,}")

    if len(df_to_append) == 0:
        if verbose:
            print(f"   Nothing new to append!")
        return df_hist

    # Align columns
    missing_cols = [c for c in df_hist.columns if c not in df_to_append.columns]
    if missing_cols and verbose:
        print(f"   Columns in historical but not in new: {missing_cols[:5]}...")

    for col in missing_cols:
        df_to_append[col] = np.nan

    # Append
    df_merged = pd.concat([df_hist, df_to_append[df_hist.columns]], ignore_index=True)
    df_merged = parse_dates(df_merged, 'MatchDate')
    df_merged = df_merged.sort_values('MatchDate').reset_index(drop=True)

    # Save
    df_merged.to_csv(output_path, index=False)

    if verbose:
        print(f"   Merged: {len(df_merged):,} total matches")
        print(f"   Saved: {output_path}")

    return df_merged


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Incremental Feature Engineering - New Season Update (Protocol V3.4.3)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Process ALL leagues (auto-detect from new season folder)
  python incremental_features.py --all

  # Process specific league
  python incremental_features.py --league AUT

  # Custom directories
  python incremental_features.py --all --historical ./data --new-season ./20252026_season --output ./features

  # Append new matches to existing features file
  python incremental_features.py --league AUT --append

  # Quiet mode
  python incremental_features.py --all --quiet

STRATEGY (COMBINE + DATE CUTOFF):
  1. Loads BOTH historical and new file
  2. Combines into single timeline (deduplicating overlaps)
  3. Finds cutoff date from existing features file
  4. Processes ALL matches chronologically (complete state)
  5. Outputs features ONLY for matches AFTER cutoff

WEEKLY UPDATE WORKFLOW:
  1. Download updated season file from Football-Data.co.uk
  2. Replace the file in 20252026_season/ folder
  3. Run: python incremental_features.py --league AUT --append
  4. Output in features/ is complete and up-to-date

DIRECTORY STRUCTURE:
  project/
  |-- Matches_AUT.xls                 # Historical data
  |-- 20252026_season/
  |   |-- AUT_20252026.xlsx           # New season data (may contain full history)
  |   +-- ...
  |-- features/
  |   |-- Matches_AUT_features.csv    # Historical features (appended to)
  |   |-- AUT_20252026_features.csv   # New season features only
  |   +-- ...
  |-- feature_engineering_pipeline.py
  +-- incremental_features.py
        """
    )

    parser.add_argument('--league', '-l', type=str, default=None,
                        help='Process specific league (e.g., AUT, E0, D1, SP1)')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Process ALL leagues found in new season folder')
    parser.add_argument('--historical', type=str, default='.',
                        help='Directory containing historical Matches_*.xls files (default: current dir)')
    parser.add_argument('--new-season', type=str, default='./20252026_season',
                        help='Directory containing new season data (default: ./20252026_season)')
    parser.add_argument('--output', '-o', type=str, default='./features',
                        help='Output directory (default: ./features)')
    parser.add_argument('--append', action='store_true',
                        help='After computing, append new features to existing historical features file')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    if not args.league and not args.all:
        parser.error("Please specify --league or --all")
        return

    if not args.quiet:
        print("\n" + "="*70)
        print("INCREMENTAL FEATURE ENGINEERING - Protocol V3.4.3")
        print("New Season Update: 2025/2026")
        print("Strategy: COMBINE + DATE CUTOFF")
        print("="*70)
        print(f"\n  Historical data:  {Path(args.historical).absolute()}")
        print(f"  New season data:  {Path(args.new_season).absolute()}")
        print(f"  Output directory: {Path(args.output).absolute()}")
        if args.append:
            print(f"  Mode: APPEND (will merge into historical features)")
        print("="*70)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find new season files
    new_season_dir = Path(args.new_season)
    if not new_season_dir.exists():
        print(f"\nERROR: New season directory not found: {new_season_dir.absolute()}")
        print(f"   Create it and add your 2025/2026 season files there.")
        return

    # Discover new season files
    new_files = []
    for ext in ['*.xls', '*.xlsx', '*.csv']:
        new_files.extend(new_season_dir.glob(ext))
    new_files = sorted(set(new_files))

    if not new_files:
        print(f"\nERROR: No data files found in {new_season_dir.absolute()}")
        return

    if not args.quiet:
        print(f"\n  Found {len(new_files)} new season files")

    # Filter by league if specified
    if args.league:
        target = args.league.upper()
        new_files = [f for f in new_files if target in f.stem.upper()]
        if not new_files:
            print(f"\nERROR: No files found for league '{args.league}'")
            return

    # Process each league
    successful = 0
    failed = 0

    for new_file in new_files:
        league_code = detect_league_from_filename(new_file.name)

        if not args.quiet:
            print(f"\n{'='*70}")
            print(f"Processing: {new_file.name} (League: {league_code})")
            print(f"{'='*70}")

        # Get Division code
        division_code = LEAGUE_MAP[league_code][1] if league_code in LEAGUE_MAP else league_code

        # ── Find historical file (raw OR features) ──
        hist_path = None
        candidates_raw = []
        candidates_features = []

        if league_code in LEAGUE_MAP:
            hist_name = LEAGUE_MAP[league_code][0]
            candidates_raw.append(hist_name)
            candidates_features.append(f"{hist_name}_features")

        candidates_raw.extend([f'Matches__{league_code}', f'Matches_{league_code}'])
        candidates_features.extend([
            f'Matches__{league_code}_features',
            f'Matches_{league_code}_features'
        ])

        # Priority 1: Raw files in historical directory
        for candidate in candidates_raw:
            hist_path = find_file(args.historical, candidate)
            if hist_path:
                break

        # Priority 2: Features files in output directory
        if hist_path is None:
            for candidate in candidates_features:
                hist_path = find_file(args.output, candidate)
                if hist_path:
                    break

        # Priority 3: Features files in historical directory
        if hist_path is None:
            for candidate in candidates_features:
                hist_path = find_file(args.historical, candidate)
                if hist_path:
                    break

        # Priority 4: Glob fallback for unusual filenames
        if hist_path is None:
            for search_dir in [Path(args.historical), Path(args.output)]:
                if not search_dir.exists():
                    continue
                for pattern in [f'*{division_code}*features*.csv', f'*{league_code}*features*.csv',
                                f'*{division_code}*.xls', f'*{league_code}*.xls']:
                    matches = list(search_dir.glob(pattern))
                    # Exclude the 20252026 output files
                    matches = [m for m in matches if '20252026' not in m.name]
                    if matches:
                        hist_path = str(matches[0])
                        break
                if hist_path:
                    break

        if hist_path is None:
            print(f"  WARNING: No historical data found for {league_code}")
            print(f"      Tried raw: {', '.join(candidates_raw[:3])}[.xls/.xlsx/.csv]")
            print(f"      Tried features: {', '.join(candidates_features[:2])}.csv")
            print(f"      In: {Path(args.historical).absolute()}")
            print(f"          {Path(args.output).absolute()}")
            failed += 1
            continue

        if not args.quiet:
            print(f"  Historical: {Path(hist_path).name}")

        # ── Find existing features file (for cutoff date) ──
        existing_features_path = None
        features_candidates = []
        if league_code in LEAGUE_MAP:
            features_candidates.append(f"{LEAGUE_MAP[league_code][0]}_features")
        features_candidates.extend([
            f'Matches__{league_code}_features',
            f'Matches_{league_code}_features'
        ])

        for candidate in features_candidates:
            existing_features_path = find_file(args.output, candidate)
            if existing_features_path:
                break
        if existing_features_path is None:
            for candidate in features_candidates:
                existing_features_path = find_file(args.historical, candidate)
                if existing_features_path:
                    break

        if not args.quiet and existing_features_path:
            print(f"  Features:   {Path(existing_features_path).name}")

        # If historical IS a features file and no separate features found, use it as cutoff source
        if existing_features_path is None and hist_path and '_features' in hist_path:
            existing_features_path = hist_path

        # ── Process ──
        output_path = str(output_dir / f"{division_code}_20252026_features.csv")

        try:
            result = process_league_incremental(
                historical_path=hist_path,
                new_season_path=str(new_file),
                output_path=output_path,
                existing_features_path=existing_features_path,
                division_code=division_code,
                verbose=not args.quiet
            )
            if result is not None and len(result) > 0:
                successful += 1

                # Append mode
                if args.append and existing_features_path:
                    append_to_historical(
                        new_features_path=output_path,
                        historical_features_path=existing_features_path,
                        verbose=not args.quiet
                    )
                elif args.append and not existing_features_path:
                    if not args.quiet:
                        print(f"  WARNING: No historical features file found for append")
                        print(f"      Looked for: {', '.join(features_candidates[:2])}.csv")
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR processing {league_code}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Final summary
    if not args.quiet:
        print("\n" + "="*70)
        print("INCREMENTAL UPDATE COMPLETE")
        print(f"   Successful: {successful}/{successful + failed}")
        print(f"   Failed: {failed}/{successful + failed}")
        print(f"   Output: {output_dir.absolute()}")
        print("="*70)

        print("\n   Output files:")
        for f in sorted(output_dir.glob('*20252026*_features.csv')):
            size_kb = f.stat().st_size / 1024
            print(f"   {f.name} ({size_kb:.0f} KB)")
        if args.append:
            print("\n   Updated historical features:")
            for f in sorted(output_dir.glob('Matches_*_features.csv')):
                size_kb = f.stat().st_size / 1024
                print(f"   {f.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
