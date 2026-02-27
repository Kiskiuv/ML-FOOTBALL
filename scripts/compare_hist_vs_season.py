#!/usr/bin/env python3
"""Compare historical walk-forward performance vs 2025/26 season for all V5 strategies."""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURES = [
    'HomeElo', 'AwayElo', 'elo_diff', 'elo_sum', 'elo_mismatch',
    'Form3Home_ratio', 'Form5Home_ratio', 'Form3Away_ratio', 'Form5Away_ratio',
    'form_diff_3', 'form_diff_5',
    'rest_diff', 'rest_diff_nonlinear', 'winter_period', 'season_phase',
    'form_momentum_home', 'form_momentum_away',
    'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'h2h_home_goals_diff'
]

ODDS_COL = {'HOME': 'OddHome', 'DRAW': 'OddDraw', 'AWAY': 'OddAway'}
RESULT_MAP = {'HOME': 'H', 'DRAW': 'D', 'AWAY': 'A'}

FILE_MAP = {
    'MEX': 'Matches_MEXICO', 'FIN': 'Matches_FIN', 'SC0': 'Matches_SC0',
    'POL': 'Matches_POL', 'SP2': 'Matches_SP2', 'N1': 'Matches_N1',
    'RUS': 'Matches_RUS', 'F1': 'Matches_F1', 'G1': 'Matches_GI',
    'I1': 'Matches_I1', 'B1': 'Matches_BELGIUM',
}

SEASON_FILE_MAP = {'MEX': 'MEXICO'}

V4_STRATEGIES = [
    # (league, market, strategy_name, model_type, pct, edge, min_odds, p_value, fdr, hist_roi, tier)
    # DEPLOY — FDR-pass, real stakes
    ('MEX', 'AWAY', 'SELECTIVE',  'LogisticRegression', 90, 0.00, 2.3, 0.0099, True,  28.1, 'DEPLOY'),
    ('SC0', 'HOME', 'STANDARD',  'RandomForest',       85, 0.00, 1.9, 0.0460, True,  67.8, 'DEPLOY'),
    # PAPER_TRADE — p < 0.05, FDR-fail
    ('POL', 'DRAW', 'STRICT',    'LogisticRegression', 95, 0.00, 3.0, 0.0225, False, 40.1, 'PAPER_TRADE'),
    ('SP2', 'DRAW', 'STRICT',    'Ensemble',           95, 0.00, 3.0, 0.0262, False, 34.5, 'PAPER_TRADE'),
    ('N1',  'DRAW', 'STRICT',    'Ensemble',           95, 0.00, 3.0, 0.0328, False, 38.1, 'PAPER_TRADE'),
    ('RUS', 'AWAY', 'SELECTIVE', 'Ensemble',           90, 0.00, 2.3, 0.0345, False, 52.5, 'PAPER_TRADE'),
    ('FIN', 'AWAY', 'STANDARD',  'LogisticRegression', 85, 0.00, 1.9, 0.0378, False, 24.6, 'PAPER_TRADE'),
    # MONITOR — p < 0.10
    ('F1',  'HOME', 'SELECTIVE', 'LogisticRegression', 90, 0.00, 2.0, 0.0574, False, 44.8, 'MONITOR'),
    ('G1',  'DRAW', 'STRICT',    'Ensemble',           95, 0.00, 3.0, 0.0603, False, 23.8, 'MONITOR'),
    ('I1',  'DRAW', 'SELECTIVE', 'RandomForest',       90, 0.00, 2.5, 0.0612, False, 22.0, 'MONITOR'),
    ('B1',  'DRAW', 'STRICT',    'LogisticRegression', 95, 0.00, 3.0, 0.0791, False, 22.5, 'MONITOR'),
]


def walk_forward_hist(league, market, model_type_str, pct, min_edge, min_odds):
    feat_file = f'features/{FILE_MAP[league]}_features.csv'
    try:
        df = pd.read_csv(feat_file)
    except Exception:
        return None
    df['MatchDate'] = pd.to_datetime(df['MatchDate'])
    df = df.sort_values('MatchDate').reset_index(drop=True)

    target_col = f'is_{market.lower()}'
    if target_col not in df.columns:
        df[target_col] = (df['FTResult'] == RESULT_MAP[market]).astype(int)

    odds_col = ODDS_COL[market]
    df['season'] = df['MatchDate'].dt.year.where(df['MatchDate'].dt.month >= 7, df['MatchDate'].dt.year - 1)
    seasons = sorted(df['season'].unique())
    eval_seasons = seasons[-7:] if len(seasons) >= 8 else seasons[1:]

    total_bets = 0
    total_wins = 0
    total_profit = 0.0
    total_odds = []
    avail_feats = [f for f in FEATURES if f in df.columns]

    for eval_s in eval_seasons:
        train = df[df['season'] < eval_s]
        test = df[df['season'] == eval_s]
        if len(train) < 200 or len(test) < 20:
            continue

        train_v = train.dropna(subset=avail_feats + [target_col])
        test_v = test.dropna(subset=avail_feats)
        test_v = test_v[test_v[odds_col].notna()]
        if len(train_v) < 200 or len(test_v) < 10:
            continue

        X_tr = train_v[avail_feats].fillna(0).values
        y_tr = train_v[target_col].values
        X_te = test_v[avail_feats].fillna(0).values

        if model_type_str == 'Ensemble':
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)
            rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
            rf.fit(X_tr, y_tr)
            lr = LogisticRegression(max_iter=1000, C=1.0)
            lr.fit(X_tr_s, y_tr)
            probs = 0.5 * rf.predict_proba(X_te)[:, 1] + 0.5 * lr.predict_proba(X_te_s)[:, 1]
            tr_probs = 0.5 * rf.predict_proba(X_tr)[:, 1] + 0.5 * lr.predict_proba(X_tr_s)[:, 1]
        elif model_type_str == 'LogisticRegression':
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)
            m = LogisticRegression(max_iter=1000, C=1.0)
            m.fit(X_tr_s, y_tr)
            probs = m.predict_proba(X_te_s)[:, 1]
            tr_probs = m.predict_proba(X_tr_s)[:, 1]
        else:
            m = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
            m.fit(X_tr, y_tr)
            probs = m.predict_proba(X_te)[:, 1]
            tr_probs = m.predict_proba(X_tr)[:, 1]

        threshold = np.percentile(tr_probs, pct)
        odds = test_v[odds_col].values
        implied = 1.0 / odds
        edge = probs - implied
        actual_win = (test_v['FTResult'] == RESULT_MAP[market]).values

        is_bet = ((probs >= threshold) & (edge >= min_edge) & (odds >= min_odds)
                  & (probs > 0.01) & (probs < 0.99))

        n = is_bet.sum()
        w = (is_bet & actual_win).sum()
        if n > 0:
            profit = sum(o - 1 if win else -1 for o, win in zip(odds[is_bet], actual_win[is_bet]))
            total_bets += n
            total_wins += w
            total_profit += profit
            total_odds.extend(odds[is_bet].tolist())

    n_seasons = len(eval_seasons)
    return {
        'bets': total_bets, 'wins': total_wins, 'losses': total_bets - total_wins,
        'profit': total_profit, 'staked': total_bets,
        'roi': total_profit / total_bets * 100 if total_bets > 0 else 0,
        'wr': total_wins / total_bets * 100 if total_bets > 0 else 0,
        'avg_bets_season': total_bets / n_seasons if n_seasons > 0 else 0,
        'avg_odds': np.mean(total_odds) if total_odds else 0,
        'n_seasons': n_seasons,
    }


def current_season(league, market, model_type_str, pct, min_edge, min_odds):
    fn = SEASON_FILE_MAP.get(league, league)
    feat_file = f'features_20252026/{fn}_20252026_features.csv'
    try:
        df = pd.read_csv(feat_file)
    except Exception:
        return None

    model_data = joblib.load(f'models/model_{league}_{market}.joblib')
    features = model_data['features']
    medians = model_data.get('imputation_medians', {})
    scaler = model_data.get('scaler')
    pct_thresholds = model_data.get('pct_thresholds', {})
    mt = model_data.get('model_type', '')

    X_df = pd.DataFrame()
    for f in features:
        if f in df.columns:
            X_df[f] = df[f].copy()
        elif f in medians:
            X_df[f] = medians[f]
        else:
            X_df[f] = 0
    for f in features:
        X_df[f] = X_df[f].fillna(medians.get(f, 0))
    X = X_df.values

    if mt == 'Ensemble':
        lr_m = model_data['lr_model']
        lr_sc = model_data.get('lr_scaler')
        w = model_data.get('ensemble_weight', 0.5)
        rf_p = model_data['model'].predict_proba(X)[:, 1]
        X_s = lr_sc.transform(X) if lr_sc else X
        lr_p = lr_m.predict_proba(X_s)[:, 1]
        probs = w * rf_p + (1 - w) * lr_p
    else:
        if scaler:
            X = scaler.transform(X)
        probs = model_data['model'].predict_proba(X)[:, 1]

    threshold = pct_thresholds.get(pct, np.percentile(probs, pct))
    odds_col = ODDS_COL[market]
    odds = df[odds_col].values
    valid = ~np.isnan(odds)
    implied = np.where(valid, 1.0 / odds, np.nan)
    edge = probs - implied
    actual_win = (df['FTResult'] == RESULT_MAP[market]).values

    is_bet = (valid & (probs >= threshold) & (edge >= min_edge) & (odds >= min_odds)
              & (probs > 0.01) & (probs < 0.99))

    n = is_bet.sum()
    w = (is_bet & actual_win).sum()
    if n > 0:
        profit = sum(o - 1 if win else -1 for o, win in zip(odds[is_bet], actual_win[is_bet]))
        avg_odds = odds[is_bet].mean()
    else:
        profit = 0
        avg_odds = 0

    return {
        'bets': n, 'wins': w, 'losses': n - w, 'profit': profit,
        'roi': profit / n * 100 if n > 0 else 0,
        'wr': w / n * 100 if n > 0 else 0, 'avg_odds': avg_odds,
    }


def main():
    print("=" * 140)
    print("V4 LEAN STRATEGIES: HISTORICAL WALK-FORWARD vs 2025/26 SEASON")
    print("=" * 140)
    print()

    header_hist = "HISTORICAL (7-season walk-forward)"
    header_curr = "2025/26 SEASON (to date)"

    print(f"{'Lg':<5} {'Mkt':<5} {'Strategy':<16} {'Tier':<12} {'p-val':>6} "
          f"| {'Bets':>4} {'W/L':>7} {'WR%':>5} {'P/L':>8} {'ROI%':>7} {'Avg/Sz':>6} {'AvgOdd':>6} "
          f"| {'Bets':>4} {'W/L':>7} {'WR%':>5} {'P/L':>8} {'ROI%':>7} {'AvgOdd':>6} {'':>3}")
    print(f"{'':5} {'':5} {'':16} {'':12} {'':>6} "
          f"| {header_hist:^52} "
          f"| {header_curr:^46}")
    print("-" * 140)

    tot_h = {'bets': 0, 'wins': 0, 'losses': 0, 'profit': 0}
    tot_c = {'bets': 0, 'wins': 0, 'losses': 0, 'profit': 0}

    for strat in V4_STRATEGIES:
        lg, mkt, name, mt, pct, me, mo, pv, fdr, hr, tier = strat

        h = walk_forward_hist(lg, mkt, mt, pct, me, mo)
        c = current_season(lg, mkt, mt, pct, me, mo)

        # Format historical
        if h and h['bets'] > 0:
            h_str = (f"{h['bets']:>4} {h['wins']}W/{h['losses']}L {h['wr']:>5.0f} "
                     f"{h['profit']:>+8.2f} {h['roi']:>+7.1f} {h['avg_bets_season']:>6.1f} {h['avg_odds']:>6.2f}")
            tot_h['bets'] += h['bets']
            tot_h['wins'] += h['wins']
            tot_h['losses'] += h['losses']
            tot_h['profit'] += h['profit']
        else:
            h_str = f"{'0':>4} {'--':>7} {'--':>5} {'--':>8} {'--':>7} {'--':>6} {'--':>6}"

        # Format current
        if c and c['bets'] > 0:
            delta_roi = c['roi'] - (h['roi'] if h and h['bets'] > 0 else 0)
            status = '+++' if c['profit'] > 0 else '---'
            c_str = (f"{c['bets']:>4} {c['wins']}W/{c['losses']}L {c['wr']:>5.0f} "
                     f"{c['profit']:>+8.2f} {c['roi']:>+7.1f} {c['avg_odds']:>6.2f} {status}")
            tot_c['bets'] += c['bets']
            tot_c['wins'] += c['wins']
            tot_c['losses'] += c['losses']
            tot_c['profit'] += c['profit']
        else:
            c_str = f"{'0':>4} {'--':>7} {'--':>5} {'--':>8} {'--':>7} {'--':>6}    "

        print(f"{lg:<5} {mkt:<5} {name:<16} {tier:<12} {pv:>6.4f} | {h_str} | {c_str}")

    print("-" * 140)

    # Totals
    h_roi = tot_h['profit'] / tot_h['bets'] * 100 if tot_h['bets'] > 0 else 0
    h_wr = tot_h['wins'] / tot_h['bets'] * 100 if tot_h['bets'] > 0 else 0
    c_roi = tot_c['profit'] / tot_c['bets'] * 100 if tot_c['bets'] > 0 else 0
    c_wr = tot_c['wins'] / tot_c['bets'] * 100 if tot_c['bets'] > 0 else 0

    print(f"{'TOTAL':<5} {'':5} {'':16} {'':12} {'':>6} "
          f"| {tot_h['bets']:>4} {tot_h['wins']}W/{tot_h['losses']}L {h_wr:>5.0f} "
          f"{tot_h['profit']:>+8.2f} {h_roi:>+7.1f} {'':>6} {'':>6} "
          f"| {tot_c['bets']:>4} {tot_c['wins']}W/{tot_c['losses']}L {c_wr:>5.0f} "
          f"{tot_c['profit']:>+8.2f} {c_roi:>+7.1f}")

    print("=" * 140)


if __name__ == "__main__":
    main()
