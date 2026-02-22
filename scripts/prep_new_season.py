#!/usr/bin/env python3
"""
PREP SCRIPT — Split all-euro-data-2025-2026.xlsx into individual league files
for use with incremental_features.py.

Reads multi-sheet Excel → saves individual {LEAGUE}_20252026.xlsx files.

USAGE:
  python prep_new_season.py --input all-euro-data-2025-2026.xlsx --output ./20252026_season
  python prep_new_season.py --input all-euro-data-2025-2026.xlsx --output ./20252026_season --latest Latest_Results.xlsx

Author: Marc | February 2026
"""

import pandas as pd
import argparse
from pathlib import Path


# Map Latest_Results league names → Football-Data league codes
LATEST_RESULTS_MAP = {
    'Premier League': 'E0',
    'Championship': 'E1',
    'League One': 'E2',
    'League Two': 'E3',
    'National League': 'EC',
    'Premiership': 'SC0',
    'Championship': 'SC1',
    'Bundesliga': 'D1',
    'Bundesliga 2': 'D2',
    'La Liga': 'SP1',
    'La Liga 2': 'SP2',
    'Serie A': 'I1',
    'Serie B': 'I2',
    'Ligue 1': 'F1',
    'Ligue 2': 'F2',
    'Jupiler League': 'B1',
    'Eredivisie': 'N1',
    'Primeira Liga': 'P1',
    'Super Lig': 'T1',
    'Super League': 'G1',
    'Superliga': 'D1',  # Could be Denmark too - ambiguous
    'Premier Division': 'IRL',
    'Ekstraklasa': 'POL',
    'Liga Profesional': 'ARG',
    'Liga MX': 'MEX',
}


def main():
    parser = argparse.ArgumentParser(description='Prep new season data for incremental_features.py')
    parser.add_argument('--input', '-i', required=True, help='Path to all-euro-data-2025-2026.xlsx')
    parser.add_argument('--output', '-o', default='./20252026_season', help='Output directory')
    parser.add_argument('--latest', '-l', default=None,
                        help='Path to Latest_Results.xlsx (optional, appends recent results)')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Split multi-sheet Excel
    # =========================================================================
    print(f"\n{'='*60}")
    print("PREP NEW SEASON DATA")
    print(f"{'='*60}")

    xls = pd.ExcelFile(args.input)
    print(f"\nSource: {args.input}")
    print(f"Sheets: {len(xls.sheet_names)} leagues: {', '.join(xls.sheet_names)}")

    league_dfs = {}
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet)
        if len(df) == 0:
            print(f"  ⚠️  {sheet}: empty, skipping")
            continue
        league_dfs[sheet] = df
        print(f"  ✅ {sheet}: {len(df)} matches, {df['Date'].min()} to {df['Date'].max()}")

    # =========================================================================
    # Step 2: Merge Latest_Results if provided
    # =========================================================================
    if args.latest:
        print(f"\n📥 Loading Latest_Results from: {args.latest}")
        lr = pd.read_excel(args.latest)
        print(f"   {len(lr)} results, date range: {lr['Date'].min()} to {lr['Date'].max()}")

        # WARNING: Latest_Results only has CLOSING odds
        print("\n   ⚠️  WARNING: Latest_Results only contains CLOSING odds (AvgCH, not AvgH)")
        print("   Matches from Latest_Results will have closing odds mapped to AvgH columns.")
        print("   This means odds-based features for these matches will use closing odds.")
        print("   For maximum accuracy, wait for all-euro-data to be updated instead.\n")

        # Map to league codes and merge
        merged_count = 0
        for _, row in lr.iterrows():
            league_name = row.get('League', '')
            league_code = LATEST_RESULTS_MAP.get(league_name)

            if league_code is None or league_code not in league_dfs:
                continue

            # Convert to Football-Data format
            new_row = {
                'Div': league_code,
                'Date': row['Date'],
                'Time': row.get('Time', ''),
                'HomeTeam': row['Home'],
                'AwayTeam': row['Away'],
                'FTHG': row['HG'],
                'FTAG': row['AG'],
                'FTR': row['Res'],
                # Map closing odds as pre-match (imperfect but best available)
                'AvgH': row.get('AvgCH'),
                'AvgD': row.get('AvgCD'),
                'AvgA': row.get('AvgCA'),
                'MaxH': row.get('MaxCH'),
                'MaxD': row.get('MaxCD'),
                'MaxA': row.get('MaxCA'),
                'B365H': row.get('B365CH'),
                'B365D': row.get('B365CD'),
                'B365A': row.get('B36CA'),  # Note: typo in source 'B36CA'
            }

            target_df = league_dfs[league_code]

            # Check if match already exists (by date + teams)
            match_date = pd.to_datetime(row['Date'])
            existing = target_df[
                (pd.to_datetime(target_df['Date']) == match_date) &
                (target_df['HomeTeam'] == row['Home']) &
                (target_df['AwayTeam'] == row['Away'])
            ]

            if len(existing) == 0:
                new_row_df = pd.DataFrame([new_row])
                league_dfs[league_code] = pd.concat([target_df, new_row_df], ignore_index=True)
                merged_count += 1

        print(f"   Merged {merged_count} new results into league files")

    # =========================================================================
    # Step 3: Save individual files
    # =========================================================================
    print(f"\n💾 Saving to: {output_dir.absolute()}")

    for league_code, df in league_dfs.items():
        # Sort by date
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        out_path = output_dir / f"{league_code}_20252026.xlsx"
        df.to_excel(out_path, index=False, engine='openpyxl')
        print(f"  ✅ {out_path.name}: {len(df)} matches")

    print(f"\n{'='*60}")
    print(f"✅ DONE — {len(league_dfs)} league files in {output_dir}")
    print(f"\nNext step:")
    print(f"  python incremental_features.py --all --new-season {output_dir} --historical . --output ./features_upd")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
