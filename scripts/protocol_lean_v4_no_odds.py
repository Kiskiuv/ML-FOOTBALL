#!/usr/bin/env python3
"""
================================================================================
PROTOCOL LEAN V2.0 — HYBRID SELECTION + RANDOMFOREST
================================================================================

Stripped-down protocol with drastically reduced search space:
  - 2 base models + ensemble: RandomForest, LogisticRegression, RF+LR blend (0.5/0.5)
  - 1 feature tier: BASIC+ (21 features)
  - 7 evaluation seasons (2018/19 through 2024/25)
  - ~8 strategies × 3 markets = ~24 strategy evaluations (pct >= 75 only)
  - Model selection: 3-way (RF vs LR vs Ensemble, by avg test AUC)
  - TOTAL: ~72 combinations per league (vs 810 in V3.4.5 = 11× reduction)

Strategy selection: HYBRID (default)
  - Only strategies with pct >= 80 AND p < 0.10 AND auc_gap < 0.15
  - Picks lowest p-value among qualifying strategies
  - Non-qualifying markets get __SKIP__ (zero bets)

Changes from V1.0:
  - XGBoost/GradientBoosting replaced with RandomForest (lower overfitting)
  - Removed high-volume strategies: MAX_VOLUME (65), VOLUME (70), AGGRESSIVE (70)
  - Added hybrid selection method (default)
  - Removed GPU code (RandomForest + LR are CPU-only)

WHY RANDOMFOREST:
  XGBoost (via sklearn GradientBoosting fallback) showed 0.35-0.44 AUC gap
  (train ~0.95 vs test ~0.53). Boosting memorizes training data. RandomForest
  uses bagging (independent trees on random subsets) with natural regularization
  via averaging — typical AUC gap 0.02-0.08 on football data.

WHY HYBRID:
  sel_simple favored high-volume strategies (41% picks were pct 65-75) that
  showed -9% to -11% live ROI. Hybrid enforces pct >= 80 floor and requires
  p < 0.10 statistical significance, only deploying strategies with evidence.

Author: Marc | February 2026
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import Counter
import warnings
import argparse
import json
import joblib
from platt import fit_platt_scaler

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def tqdm_wrapper(iterable, **kwargs):
    if HAS_TQDM:
        return tqdm(iterable, **kwargs)
    return iterable


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Lean Protocol V2.0 — RF + LR, BASIC+, hybrid selection."""
    
    EVAL_SEASONS: List[int] = field(default_factory=lambda: [2018, 2019, 2020, 2021, 2022, 2023, 2024])
    MIN_TRAIN_MATCHES: int = 500
    MIN_TEST_MATCHES: int = 50
    
    RANDOM_STATE: int = 42
    
    # RandomForest — bagged trees, naturally regularized
    RF_N_ESTIMATORS: int = 200
    RF_MAX_DEPTH: int = 6           # shallow trees
    RF_MIN_SAMPLES_LEAF: int = 20   # won't split tiny groups
    RF_MAX_FEATURES: str = "sqrt"   # random feature subsets (bagging)
    
    # LogisticRegression — linear baseline
    LR_C: float = 1.0
    LR_MAX_ITER: int = 1000
    
    # Evaluation
    MIN_BETS_FOR_EVAL: int = 20
    MIN_BETS_FOR_DEPLOY: int = 100
    MIN_PROFITABLE_PCT: float = 0.60
    
    # Monte Carlo
    N_SIMULATIONS: int = 50000
    MC_SEED: int = 42
    
    # FDR
    FDR_Q: float = 0.10
    
    # Rolling validation
    LOOKBACK_SIMPLE: int = 4
    LOOKBACK_STRICT: int = 6
    MIN_BETS_STRICT: int = 100
    MIN_HISTORY_SEASONS: int = 2
    
    # Hybrid selection thresholds
    HYBRID_MIN_PCT: int = 80        # minimum percentile threshold
    HYBRID_MAX_P: float = 0.10      # maximum p-value
    HYBRID_MAX_AUC_GAP: float = 0.15  # maximum train-test AUC gap
    
    # Export
    EXPORT_MODELS: bool = True
    EXPORT_JSON: bool = True
    OUTPUT_DIR: str = "./output"
    
    # Selection — default hybrid
    TWO_LAYER_SELECTION: str = "hybrid"
    
    # BASIC+ features (26) — FROZEN
    FEATURES: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.FEATURES = [
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


CONFIG = Config()

# Only 2 model types
MODEL_TYPES = ["RandomForest", "LogisticRegression"]
MODELS_NEEDING_SCALING = {"LogisticRegression"}
ENSEMBLE_KEY = "Ensemble"
ENSEMBLE_WEIGHT = 0.5  # RF weight; LR weight = 1 - this


# =============================================================================
# STRATEGIES (FROZEN — identical to V3.4.5)
# =============================================================================

DRAW_STRATEGIES = [
    {"name": "LONGSHOT_STRICT", "pct": 92, "edge": 0.00, "min_odds": 3.2},
    {"name": "LONGSHOT", "pct": 90, "edge": -0.02, "min_odds": 2.8},
    {"name": "CONSERVATIVE", "pct": 90, "edge": 0.00, "min_odds": 3.0},
    {"name": "SELECTIVE", "pct": 88, "edge": -0.01, "min_odds": 3.0},
    {"name": "MODERATE_HIGH", "pct": 85, "edge": 0.00, "min_odds": 2.8},
    {"name": "MODERATE", "pct": 85, "edge": -0.02, "min_odds": 3.0},
    {"name": "BALANCED", "pct": 80, "edge": -0.02, "min_odds": 2.8},
    # REMOVED: VOLUME (75), AGGRESSIVE (70), MAX_VOLUME (65) — live ROI -9% to -11%
]

HOME_STRATEGIES = [
    {"name": "ULTRA_CONS", "pct": 90, "edge": 0.02, "min_odds": 2.5},
    {"name": "CONSERVATIVE", "pct": 88, "edge": 0.00, "min_odds": 2.3},
    {"name": "UPSET_HIGH", "pct": 85, "edge": 0.00, "min_odds": 2.2},
    {"name": "UPSET", "pct": 85, "edge": -0.01, "min_odds": 2.0},
    {"name": "SELECTIVE", "pct": 82, "edge": -0.01, "min_odds": 2.0},
    {"name": "STANDARD", "pct": 80, "edge": -0.02, "min_odds": 1.9},
    {"name": "VALUE", "pct": 80, "edge": -0.02, "min_odds": 2.0},
    {"name": "FAVORITE", "pct": 75, "edge": 0.00, "min_odds": 1.8},
    # REMOVED: VOLUME (70), MAX_VOLUME (65) — live ROI -9% to -11%
]

AWAY_STRATEGIES = [
    {"name": "ULTRA_CONS", "pct": 90, "edge": 0.02, "min_odds": 2.8},
    {"name": "CONSERVATIVE", "pct": 88, "edge": 0.00, "min_odds": 2.5},
    {"name": "SELECTIVE", "pct": 85, "edge": -0.01, "min_odds": 2.3},
    {"name": "STANDARD_HIGH", "pct": 82, "edge": 0.00, "min_odds": 2.2},
    {"name": "STANDARD", "pct": 80, "edge": -0.02, "min_odds": 2.0},
    {"name": "VALUE", "pct": 80, "edge": -0.02, "min_odds": 2.5},
    {"name": "BALANCED", "pct": 78, "edge": -0.02, "min_odds": 2.2},
    # REMOVED: VOLUME (75), AGGRESSIVE (70), MAX_VOLUME (65) — live ROI -9% to -11%
]

MARKET_STRATEGIES = {"HOME": HOME_STRATEGIES, "DRAW": DRAW_STRATEGIES, "AWAY": AWAY_STRATEGIES}
MARKET_ODDS_COL = {"HOME": "OddHome", "DRAW": "OddDraw", "AWAY": "OddAway"}
DEFAULT_STRATEGY_IDX = 5


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(filepath: str, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print(f"Loading data from {filepath}...")
    
    df = pd.read_csv(filepath, low_memory=False)
    df["MatchDate"] = pd.to_datetime(df["MatchDate"])
    df = df.sort_values("MatchDate").reset_index(drop=True)
    
    df["SeasonYear"] = df["MatchDate"].apply(lambda x: x.year if x.month >= 7 else x.year - 1)
    df["Season"] = df["MatchDate"].apply(
        lambda x: f"{x.year}/{x.year + 1}" if x.month >= 7 else f"{x.year - 1}/{x.year}"
    )
    
    df["target_HOME"] = (df["FTResult"] == "H").astype(int)
    df["target_DRAW"] = (df["FTResult"] == "D").astype(int)
    df["target_AWAY"] = (df["FTResult"] == "A").astype(int)
    
    mask = df["OddHome"].notna() & df["OddDraw"].notna() & df["OddAway"].notna()
    df_valid = df[mask].copy()
    
    if verbose:
        print(f"  Total: {len(df):,} | With odds: {len(df_valid):,}")
    
    return df_valid


def get_available_features(df: pd.DataFrame, config: Config = CONFIG) -> List[str]:
    """Get BASIC+ features that exist and have >50% coverage."""
    available = []
    for feat in config.FEATURES:
        if feat in df.columns:
            coverage = df[feat].notna().mean()
            if coverage > 0.5:
                available.append(feat)
    return available


# =============================================================================
# MODELS — ONLY 2
# =============================================================================

def create_model(model_type: str, config: Config = CONFIG):
    """Create one of the 2 allowed models."""
    if model_type == "RandomForest":
        return RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS,
            max_depth=config.RF_MAX_DEPTH,
            min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
            max_features=config.RF_MAX_FEATURES,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
    elif model_type == "LogisticRegression":
        return LogisticRegression(
            C=config.LR_C,
            max_iter=config.LR_MAX_ITER,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model: {model_type}. Allowed: {MODEL_TYPES}")


def impute_features(train_df, test_df, features) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    medians = X_train.median()
    X_train = X_train.fillna(medians).values
    X_test = X_test.fillna(medians).values
    return X_train, X_test, medians.to_dict()


def compute_ml_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    metrics = {}
    if len(np.unique(y_true)) >= 2:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["brier"] = float(brier_score_loss(y_true, y_prob))
        metrics["logloss"] = float(log_loss(y_true, y_prob))
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
            bin_counts = np.histogram(y_prob, bins=10, range=(0, 1))[0]
            weights = bin_counts[:len(prob_true)] / len(y_true)
            metrics["calibration_ece"] = float(np.sum(weights * np.abs(prob_true - prob_pred)))
        except:
            pass
    return metrics


def train_and_evaluate(
    model_type: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    target: str,
    config: Config = CONFIG
) -> Tuple[np.ndarray, np.ndarray, Dict, Dict, Any]:
    """Train model, return (test_probs, train_probs, train_metrics, test_metrics, model)."""
    X_train, X_test, medians = impute_features(train_df, test_df, features)
    y_train = train_df[target].values
    y_test = test_df[target].values
    
    scaler = None
    if model_type in MODELS_NEEDING_SCALING:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    model = create_model(model_type, config)
    model.fit(X_train, y_train)
    
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    
    train_metrics = compute_ml_metrics(y_train, train_probs)
    test_metrics = compute_ml_metrics(y_test, test_probs)
    
    return test_probs, train_probs, train_metrics, test_metrics, model


def select_best_model(model_results: Dict[str, Dict]) -> str:
    """Select best model by average TEST AUC, tiebreak by Brier."""
    best_key = None
    best_auc = -1
    best_brier = 999
    
    for key, results in model_results.items():
        test_aucs = [r["test_metrics"].get("auc", 0) for r in results["seasons"]]
        test_briers = [r["test_metrics"].get("brier", 1) for r in results["seasons"]]
        
        if not test_aucs:
            continue
        
        avg_auc = np.mean(test_aucs)
        avg_brier = np.mean(test_briers)
        
        if avg_auc > best_auc or (avg_auc == best_auc and avg_brier < best_brier):
            best_auc = avg_auc
            best_brier = avg_brier
            best_key = key
    
    return best_key or "RandomForest"


# =============================================================================
# STATISTICS
# =============================================================================

def monte_carlo_pvalue(odds_vector: np.ndarray, wins_vector: np.ndarray,
                       n_sims: int = 50000, seed: int = 42) -> float:
    n_bets = len(odds_vector)
    if n_bets == 0:
        return 1.0
    
    actual_profit = np.sum(wins_vector * odds_vector) - n_bets
    implied_probs = 1.0 / odds_vector
    
    rng = np.random.default_rng(seed)
    random_matrix = rng.random((n_sims, n_bets))
    sim_wins = (random_matrix < implied_probs).astype(float)
    sim_profits = np.sum(sim_wins * odds_vector, axis=1) - n_bets
    
    p_value = np.mean(sim_profits >= actual_profit)
    return max(p_value, 1.0 / n_sims)


def apply_fdr_correction(p_values: List[float], q: float = 0.10) -> List[bool]:
    n = len(p_values)
    if n == 0:
        return []
    
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    k_max = 0
    for k in range(1, n + 1):
        threshold = (k / n) * q
        if indexed[k - 1][1] <= threshold:
            k_max = k
    
    passes = [False] * n
    for k in range(k_max):
        passes[indexed[k][0]] = True
    
    return passes


def compute_risk_metrics(profits: List[float], odds: List[float]) -> Dict:
    if not profits or len(profits) < 2:
        return {}
    
    cumulative = np.cumsum(profits)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0
    
    volatility = float(np.std(profits))
    mean_profit = np.mean(profits)
    sharpe = float(mean_profit / volatility) if volatility > 0 else 0
    
    wins = [1 if p > 0 else 0 for p in profits]
    max_win_streak = max_loss_streak = current_win = current_loss = 0
    
    for w in wins:
        if w == 1:
            current_win += 1
            current_loss = 0
            max_win_streak = max(max_win_streak, current_win)
        else:
            current_loss += 1
            current_win = 0
            max_loss_streak = max(max_loss_streak, current_loss)
    
    return {
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "avg_odds": float(np.mean(odds)) if odds else 0,
        "median_odds": float(np.median(odds)) if odds else 0
    }


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SeasonResult:
    season: str
    season_year: int
    n_bets: int
    n_wins: int
    profit: float
    roi: float
    odds_list: List[float]
    wins_list: List[int]
    pct_threshold: float = 0.0


@dataclass
class StrategyResult:
    name: str
    market: str
    n_bets: int
    n_wins: int
    profit: float
    roi: float
    avg_odds: float
    p_value: float
    seasons: List[SeasonResult]
    profitable_seasons: int
    total_seasons: int
    fdr_pass: bool = False
    model_metrics: Dict = field(default_factory=dict)
    strategy_params: Dict = field(default_factory=dict)
    risk_metrics: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name, "market": self.market,
            "n_bets": self.n_bets, "n_wins": self.n_wins,
            "profit": round(self.profit, 2), "roi": round(self.roi, 2),
            "avg_odds": round(self.avg_odds, 3), "p_value": round(self.p_value, 4),
            "profitable_seasons": self.profitable_seasons,
            "total_seasons": self.total_seasons, "fdr_pass": self.fdr_pass,
            "model_metrics": self.model_metrics,
            "strategy_params": self.strategy_params,
            "risk_metrics": self.risk_metrics
        }


@dataclass
class TwoLayerResult:
    eval_season: int
    selected_strategy: str
    selection_method: str
    n_bets: int
    n_wins: int
    profit: float
    roi: float


# =============================================================================
# STRATEGY SELECTION
# =============================================================================

def compute_score(seasons: List[SeasonResult], lookback: int) -> float:
    """Composite: 50% median ROI + 30% mean ROI + 20% profitable%."""
    if not seasons:
        return -999.0
    recent = seasons[-lookback:] if len(seasons) >= lookback else seasons
    if not recent:
        return -999.0
    rois = [s.roi for s in recent]
    median_roi = np.median(rois)
    mean_roi = np.mean(rois)
    prof_pct = sum(1 for s in recent if s.roi > 0) / len(recent)
    return 0.50 * median_roi + 0.30 * mean_roi + 0.20 * (prof_pct * 100)


def select_simple(history: Dict[str, List[SeasonResult]], lookback: int = 4, min_bets: int = 10) -> Optional[str]:
    best_name = None
    best_score = -999.0
    for name, seasons in history.items():
        total = sum(s.n_bets for s in seasons)
        if total < min_bets:
            continue
        score = compute_score(seasons, lookback)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def select_strict(history: Dict, lookback: int = 6, min_bets: int = 100, min_prof_pct: float = 0.60) -> Optional[str]:
    best_name = None
    best_score = -999.0
    for name, seasons in history.items():
        total = sum(s.n_bets for s in seasons)
        if total < min_bets or len(seasons) < 3:
            continue
        recent = seasons[-lookback:] if len(seasons) >= lookback else seasons
        prof_pct = sum(1 for s in recent if s.roi > 0) / len(recent)
        if prof_pct < min_prof_pct:
            continue
        score = compute_score(seasons, lookback)
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def select_fdr(results: List[StrategyResult], history: Dict) -> Optional[str]:
    fdr_pass = [r for r in results if r.fdr_pass]
    if not fdr_pass:
        return None
    best = min(fdr_pass, key=lambda x: x.p_value)
    return best.name


def select_ensemble(results: List[StrategyResult], history: Dict, config: Config) -> Optional[str]:
    simple = select_simple(history, config.LOOKBACK_SIMPLE)
    strict = select_strict(history, config.LOOKBACK_STRICT, config.MIN_BETS_STRICT, config.MIN_PROFITABLE_PCT)
    fdr = select_fdr(results, history)
    votes = [v for v in [simple, strict, fdr] if v is not None]
    if not votes:
        return simple
    counts = Counter(votes)
    top = counts.most_common(1)[0]
    if top[1] >= 2:
        return top[0]
    return simple


def select_hybrid(
    results: List[StrategyResult],
    model_metrics: Dict,
    config: Config
) -> Optional[str]:
    """
    Hybrid selection: pct >= HYBRID_MIN_PCT AND p < HYBRID_MAX_P AND auc_gap < HYBRID_MAX_AUC_GAP.
    Among qualifying strategies, pick lowest p-value (tiebreak by higher ROI).
    Returns None if nothing qualifies → market gets __SKIP__.
    """
    # Compute AUC gap from model metrics
    auc_gap = 0.0
    if model_metrics and isinstance(model_metrics, dict):
        train_auc = model_metrics.get("train", {}).get("auc", 0)
        test_auc = model_metrics.get("test", {}).get("auc", 0)
        auc_gap = abs(train_auc - test_auc)
    
    # If model itself is overfitting, skip entire market
    if auc_gap >= config.HYBRID_MAX_AUC_GAP:
        return None
    
    qualifying = []
    for r in results:
        pct = r.strategy_params.get("pct", 0)
        if pct >= config.HYBRID_MIN_PCT and r.p_value < config.HYBRID_MAX_P:
            qualifying.append(r)
    
    if not qualifying:
        return None
    
    # Pick lowest p-value, tiebreak by highest ROI
    best = min(qualifying, key=lambda x: (x.p_value, -x.roi))
    return best.name


def rebuild_for_fallback_model(
    model_results: Dict,
    fallback_model: str,
    strategies: List[Dict],
    market: str,
    config: Config
) -> Tuple[List, Dict, Dict]:
    """
    Rebuild strategy results and metrics for a fallback model.
    Used when the winning model (typically RF) fails the AUC gap check
    and we want to try LR instead.
    
    Returns: (strat_results, aggregated_metrics, model_comparison_entry)
    """
    fallback_data = model_results.get(fallback_model)
    if not fallback_data or not fallback_data["seasons"]:
        return [], {}, {}
    
    # Aggregated metrics for fallback model
    seasons = fallback_data["seasons"]
    train_aucs = [s["train_metrics"].get("auc", 0) for s in seasons]
    test_aucs = [s["test_metrics"].get("auc", 0) for s in seasons]
    
    aggregated_metrics = {
        "train": {
            "auc": np.mean(train_aucs),
            "brier": np.mean([s["train_metrics"].get("brier", 1) for s in seasons]),
            "auc_std": np.std(train_aucs)
        },
        "test": {
            "auc": np.mean(test_aucs),
            "brier": np.mean([s["test_metrics"].get("brier", 1) for s in seasons]),
            "auc_std": np.std(test_aucs)
        },
        "n_seasons": len(seasons),
        "model_type": fallback_model,
        "feature_tier": "BASIC+",
        "n_features": int(np.mean([s.get("n_features", 0) for s in seasons]))
    }
    
    comparison_entry = {
        "train_auc_mean": round(np.mean(train_aucs), 4),
        "test_auc_mean": round(np.mean(test_aucs), 4),
        "auc_gap": round(np.mean(train_aucs) - np.mean(test_aucs), 4),
        "n_seasons": len(seasons)
    }
    
    # Aggregate strategy results
    final_results = []
    for strat in strategies:
        strat_seasons = [s for s in fallback_data["all_results"][strat["name"]] if s is not None]
        if strat_seasons:
            result = aggregate_results(strat_seasons, strat, market, config, aggregated_metrics)
            if result:
                final_results.append(result)
    
    # Apply FDR
    if final_results:
        pvals = [r.p_value for r in final_results]
        fdr_pass = apply_fdr_correction(pvals, config.FDR_Q)
        for i, r in enumerate(final_results):
            r.fdr_pass = fdr_pass[i]
    
    return final_results, aggregated_metrics, comparison_entry


def do_selection(method: str, history: Dict, results: List[StrategyResult], config: Config,
                 model_metrics: Dict = None) -> Tuple[Optional[str], str]:
    if method == "hybrid":
        sel = select_hybrid(results, model_metrics, config)
        if sel is None:
            return "__SKIP__", "HYBRID_SKIP"
        return sel, "HYBRID"
    elif method == "simple":
        return select_simple(history, config.LOOKBACK_SIMPLE), "ROLLING_SIMPLE"
    elif method == "strict":
        sel = select_strict(history, config.LOOKBACK_STRICT, config.MIN_BETS_STRICT, config.MIN_PROFITABLE_PCT)
        if sel is None:
            sel = select_simple(history, config.LOOKBACK_SIMPLE)
            return sel, "FALLBACK_SIMPLE"
        return sel, "ROLLING_STRICT"
    elif method == "fdr":
        sel = select_fdr(results, history)
        return sel, "ROLLING_FDR" if sel else "FALLBACK"
    elif method == "ensemble":
        return select_ensemble(results, history, config), "ENSEMBLE"
    else:
        return select_simple(history, config.LOOKBACK_SIMPLE), "ROLLING_SIMPLE"


# =============================================================================
# STRATEGY TESTING
# =============================================================================

def test_strategy(df: pd.DataFrame, strategy: Dict, market: str, pct_threshold: float) -> Optional[SeasonResult]:
    odds_col = MARKET_ODDS_COL[market]
    target_col = f"target_{market}"
    
    mask = (
        (df["prob"] >= pct_threshold) &
        (df["edge"] >= strategy["edge"]) &
        (df[odds_col] >= strategy["min_odds"])
    )
    bets = df[mask]
    
    if len(bets) == 0:
        return None
    
    n_bets = len(bets)
    n_wins = int(bets[target_col].sum())
    odds_list = bets[odds_col].tolist()
    wins_list = bets[target_col].tolist()
    
    profit = sum(w * o for w, o in zip(wins_list, odds_list)) - n_bets
    roi = (profit / n_bets) * 100 if n_bets > 0 else 0
    
    return SeasonResult(
        season=df["Season"].iloc[0], season_year=int(df["SeasonYear"].iloc[0]),
        n_bets=n_bets, n_wins=n_wins, profit=profit, roi=roi,
        odds_list=odds_list, wins_list=wins_list, pct_threshold=pct_threshold
    )


def aggregate_results(season_results: List[SeasonResult], strategy: Dict, market: str,
                      config: Config = CONFIG, model_metrics: Dict = None) -> Optional[StrategyResult]:
    valid = [s for s in season_results if s is not None]
    if not valid:
        return None
    
    total_bets = sum(s.n_bets for s in valid)
    total_wins = sum(s.n_wins for s in valid)
    total_profit = sum(s.profit for s in valid)
    
    if total_bets == 0:
        return None
    
    roi = (total_profit / total_bets) * 100
    
    all_odds, all_wins = [], []
    for s in valid:
        all_odds.extend(s.odds_list)
        all_wins.extend(s.wins_list)
    
    avg_odds = np.mean(all_odds)
    p_value = monte_carlo_pvalue(np.array(all_odds), np.array(all_wins),
                                  config.N_SIMULATIONS, config.MC_SEED)
    
    profitable = sum(1 for s in valid if s.roi > 0)
    all_profits = [w * o - 1 for w, o in zip(all_wins, all_odds)]
    computed_risk = compute_risk_metrics(all_profits, all_odds)
    
    return StrategyResult(
        name=strategy["name"], market=market,
        n_bets=total_bets, n_wins=total_wins,
        profit=total_profit, roi=roi, avg_odds=avg_odds, p_value=p_value,
        seasons=valid, profitable_seasons=profitable, total_seasons=len(valid),
        model_metrics=model_metrics or {},
        strategy_params={"pct": strategy["pct"], "edge": strategy["edge"], "min_odds": strategy["min_odds"]},
        risk_metrics=computed_risk
    )


# =============================================================================
# WALK-FORWARD — 2 MODELS ONLY
# =============================================================================

def run_walk_forward(
    df: pd.DataFrame,
    market: str,
    config: Config = CONFIG,
    verbose: bool = False
) -> Tuple[List[TwoLayerResult], List[StrategyResult], str, Dict]:
    """
    Walk-forward with 2 models (RandomForest + LR), single feature tier (BASIC+).
    
    Returns: (two_layer_results, final_results, best_model, model_comparison)
    """
    target = f"target_{market}"
    odds_col = MARKET_ODDS_COL[market]
    strategies = MARKET_STRATEGIES[market]
    
    # Initialize results for base models + ensemble
    model_results = {}
    for m in MODEL_TYPES + [ENSEMBLE_KEY]:
        model_results[m] = {"seasons": [], "all_results": {s["name"]: [] for s in strategies}}

    for eval_season in config.EVAL_SEASONS:
        if verbose:
            print(f"    Season {eval_season}...")

        train_df = df[df["SeasonYear"] < eval_season].copy()
        test_df = df[df["SeasonYear"] == eval_season].copy()

        if len(train_df) < config.MIN_TRAIN_MATCHES or len(test_df) < config.MIN_TEST_MATCHES:
            continue

        features = get_available_features(train_df, config)
        if len(features) < 5:
            continue

        y_train = train_df[target].values
        y_test = test_df[target].values

        # Train base models, collect probs
        season_probs = {}
        for model_type in MODEL_TYPES:
            try:
                test_probs, train_probs, train_metrics, test_metrics, _ = train_and_evaluate(
                    model_type, train_df, test_df, features, target, config
                )
                season_probs[model_type] = {
                    "test": test_probs, "train": train_probs,
                    "train_metrics": train_metrics, "test_metrics": test_metrics
                }
                model_results[model_type]["seasons"].append({
                    "season": eval_season,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "n_features": len(features)
                })
            except Exception as e:
                if verbose:
                    print(f"      {model_type} failed: {e}")

        # Compute ensemble if both base models succeeded
        if len(season_probs) == len(MODEL_TYPES):
            w = ENSEMBLE_WEIGHT
            ens_test = (w * season_probs["RandomForest"]["test"]
                        + (1 - w) * season_probs["LogisticRegression"]["test"])
            ens_train = (w * season_probs["RandomForest"]["train"]
                         + (1 - w) * season_probs["LogisticRegression"]["train"])
            ens_train_m = compute_ml_metrics(y_train, ens_train)
            ens_test_m = compute_ml_metrics(y_test, ens_test)
            season_probs[ENSEMBLE_KEY] = {
                "test": ens_test, "train": ens_train,
                "train_metrics": ens_train_m, "test_metrics": ens_test_m
            }
            model_results[ENSEMBLE_KEY]["seasons"].append({
                "season": eval_season,
                "train_metrics": ens_train_m,
                "test_metrics": ens_test_m,
                "n_features": len(features)
            })

        # Test strategies for all available models (RF, LR, Ensemble)
        for model_key, probs_data in season_probs.items():
            pct_thresholds = {
                pct: np.percentile(probs_data["train"], pct)
                for pct in [75, 78, 80, 82, 85, 88, 90, 92]
            }
            test_copy = test_df.copy()
            test_copy["prob"] = probs_data["test"]
            test_copy["implied"] = 1.0 / test_copy[odds_col]
            test_copy["edge"] = test_copy["prob"] - test_copy["implied"]

            for strat in strategies:
                thresh = pct_thresholds.get(strat["pct"], 0)
                sr = test_strategy(test_copy, strat, market, thresh)
                model_results[model_key]["all_results"][strat["name"]].append(sr)
    
    # Select best model by TEST AUC
    best_model = select_best_model(model_results)
    
    if verbose:
        print(f"    Best model: {best_model}")
    
    # Model comparison
    model_comparison = {}
    for key, data in model_results.items():
        seasons = data["seasons"]
        if seasons:
            train_aucs = [s["train_metrics"].get("auc", 0) for s in seasons]
            test_aucs = [s["test_metrics"].get("auc", 0) for s in seasons]
            model_comparison[key] = {
                "train_auc_mean": round(np.mean(train_aucs), 4),
                "test_auc_mean": round(np.mean(test_aucs), 4),
                "auc_gap": round(np.mean(train_aucs) - np.mean(test_aucs), 4),
                "n_seasons": len(seasons)
            }
    
    # Aggregated metrics for best model
    best_seasons = model_results[best_model]["seasons"]
    aggregated_metrics = {
        "train": {
            "auc": np.mean([s["train_metrics"].get("auc", 0) for s in best_seasons]),
            "brier": np.mean([s["train_metrics"].get("brier", 1) for s in best_seasons]),
            "auc_std": np.std([s["train_metrics"].get("auc", 0) for s in best_seasons])
        },
        "test": {
            "auc": np.mean([s["test_metrics"].get("auc", 0) for s in best_seasons]),
            "brier": np.mean([s["test_metrics"].get("brier", 1) for s in best_seasons]),
            "auc_std": np.std([s["test_metrics"].get("auc", 0) for s in best_seasons])
        },
        "n_seasons": len(best_seasons),
        "model_type": best_model,
        "feature_tier": "BASIC+",
        "n_features": int(np.mean([s.get("n_features", 0) for s in best_seasons]))
    }
    
    # Aggregate strategy results for best model
    final_results = []
    for strat in strategies:
        seasons = [s for s in model_results[best_model]["all_results"][strat["name"]] if s is not None]
        if seasons:
            result = aggregate_results(seasons, strat, market, config, aggregated_metrics)
            if result:
                final_results.append(result)
    
    # Apply FDR
    if final_results:
        pvals = [r.p_value for r in final_results]
        fdr_pass = apply_fdr_correction(pvals, config.FDR_Q)
        for i, r in enumerate(final_results):
            r.fdr_pass = fdr_pass[i]
    
    # Two-layer simulation
    two_layer_results = []
    history = {s["name"]: [] for s in strategies}
    
    for eval_season in config.EVAL_SEASONS:
        train_df = df[df["SeasonYear"] < eval_season].copy()
        test_df = df[df["SeasonYear"] == eval_season].copy()
        
        if len(train_df) < config.MIN_TRAIN_MATCHES or len(test_df) < config.MIN_TEST_MATCHES:
            continue
        
        features = get_available_features(train_df, config)
        if len(features) < 5:
            continue
        
        try:
            if best_model == ENSEMBLE_KEY:
                rf_test, rf_train, _, _, _ = train_and_evaluate(
                    "RandomForest", train_df, test_df, features, target, config
                )
                lr_test, lr_train, _, _, _ = train_and_evaluate(
                    "LogisticRegression", train_df, test_df, features, target, config
                )
                test_probs = ENSEMBLE_WEIGHT * rf_test + (1 - ENSEMBLE_WEIGHT) * lr_test
                train_probs = ENSEMBLE_WEIGHT * rf_train + (1 - ENSEMBLE_WEIGHT) * lr_train
            else:
                test_probs, train_probs, _, _, _ = train_and_evaluate(
                    best_model, train_df, test_df, features, target, config
                )
        except Exception:
            continue
        
        pct_thresholds = {
            pct: np.percentile(train_probs, pct)
            for pct in [75, 78, 80, 82, 85, 88, 90, 92]
        }
        
        test_df = test_df.copy()
        test_df["prob"] = test_probs
        test_df["implied"] = 1.0 / test_df[odds_col]
        test_df["edge"] = test_df["prob"] - test_df["implied"]
        
        # Build history
        for strat in strategies:
            past = [sr for sr in history[strat["name"]] if sr is not None and sr.season_year < eval_season]
            history[strat["name"]] = past
        
        max_history = max(len(v) for v in history.values()) if history else 0
        
        if max_history < config.MIN_HISTORY_SEASONS:
            selected_name = strategies[DEFAULT_STRATEGY_IDX]["name"]
            method_used = "DEFAULT"
        else:
            temp_results = []
            for strat in strategies:
                seasons = history[strat["name"]]
                if seasons:
                    result = aggregate_results(seasons, strat, market, config)
                    if result:
                        temp_results.append(result)
            
            if temp_results:
                pvals = [r.p_value for r in temp_results]
                fdr_pass_temp = apply_fdr_correction(pvals, config.FDR_Q)
                for i, r in enumerate(temp_results):
                    r.fdr_pass = fdr_pass_temp[i]
            
            selected_name, method_used = do_selection(
                config.TWO_LAYER_SELECTION, history, temp_results, config,
                model_metrics=aggregated_metrics
            )
            
            if selected_name is None:
                selected_name = strategies[DEFAULT_STRATEGY_IDX]["name"]
                method_used = "FALLBACK"
        
        # Hybrid returns __SKIP__ when nothing qualifies
        if selected_name == "__SKIP__":
            # Update history but produce no bets
            for strat in strategies:
                seasons_list = model_results[best_model]["all_results"][strat["name"]]
                matching = [s for s in seasons_list if s is not None and s.season_year == eval_season]
                if matching:
                    history[strat["name"]].append(matching[0])
            continue
        
        selected_strat = next(s for s in strategies if s["name"] == selected_name)
        pct_thresh = pct_thresholds.get(selected_strat["pct"], 0)
        result = test_strategy(test_df, selected_strat, market, pct_thresh)
        
        if result:
            two_layer_results.append(TwoLayerResult(
                eval_season=eval_season, selected_strategy=selected_name,
                selection_method=method_used, n_bets=result.n_bets,
                n_wins=result.n_wins, profit=result.profit, roi=result.roi
            ))
        
        # Update history
        for strat in strategies:
            seasons_list = model_results[best_model]["all_results"][strat["name"]]
            matching = [s for s in seasons_list if s is not None and s.season_year == eval_season]
            if matching:
                history[strat["name"]].append(matching[0])
    
    return two_layer_results, final_results, best_model, model_comparison, model_results


# =============================================================================
# MODEL EXPORT
# =============================================================================

def export_final_model(
    df: pd.DataFrame,
    league_code: str,
    market: str,
    best_model_type: str,
    config: Config = CONFIG,
    verbose: bool = True,
    platt_calibration_data: dict = None
) -> str:
    """
    Train and export final model using BASIC+ features.

    platt_calibration_data: optional dict with 'probs' and 'labels' arrays
        (out-of-sample predictions from last eval season) for Platt scaling.
    """
    target = f"target_{market}"
    features = get_available_features(df, config)

    if len(features) < 5:
        return None

    X_df = df[features].copy()
    medians = X_df.median()
    X = X_df.fillna(medians).values
    y = df[target].values

    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"model_{league_code}_{market}.joblib"

    # Fit Platt scaler on out-of-sample calibration data (if provided)
    platt_scaler = None
    if platt_calibration_data is not None:
        cal_probs = platt_calibration_data.get('probs')
        cal_labels = platt_calibration_data.get('labels')
        if cal_probs is not None and cal_labels is not None:
            platt_scaler = fit_platt_scaler(cal_probs, cal_labels)
            if platt_scaler is not None and verbose:
                print(f"    Platt scaler fitted on {len(cal_probs)} OOS samples")

    if best_model_type == ENSEMBLE_KEY:
        # Train both base models
        rf_model = create_model("RandomForest", config)
        rf_model.fit(X, y)

        lr_scaler = StandardScaler()
        X_scaled = lr_scaler.fit_transform(X)
        lr_model = create_model("LogisticRegression", config)
        lr_model.fit(X_scaled, y)

        # Ensemble probs for thresholds
        rf_probs = rf_model.predict_proba(X)[:, 1]
        lr_probs = lr_model.predict_proba(X_scaled)[:, 1]
        probs = ENSEMBLE_WEIGHT * rf_probs + (1 - ENSEMBLE_WEIGHT) * lr_probs

        pct_thresholds = {
            pct: float(np.percentile(probs, pct))
            for pct in [75, 78, 80, 82, 85, 88, 90, 92]
        }
        metrics = compute_ml_metrics(y, probs)
        importance = dict(zip(features, rf_model.feature_importances_.tolist()))

        joblib.dump({
            "model": rf_model,
            "model_type": ENSEMBLE_KEY,
            "lr_model": lr_model,
            "lr_scaler": lr_scaler,
            "ensemble_weight": ENSEMBLE_WEIGHT,
            "feature_tier": "BASIC+",
            "features": features,
            "target": target,
            "market": market,
            "league": league_code,
            "n_samples": len(df),
            "n_features": len(features),
            "metrics": metrics,
            "imputation_medians": medians.to_dict(),
            "pct_thresholds": pct_thresholds,
            "scaler": None,
            "platt_scaler": platt_scaler,
            "feature_importance": importance,
            "config": {
                "model_type": ENSEMBLE_KEY,
                "feature_tier": "BASIC+",
                "random_state": config.RANDOM_STATE,
                "protocol": "LEAN_V2.0"
            },
            "created": datetime.now().isoformat()
        }, model_path)
    else:
        scaler = None
        if best_model_type in MODELS_NEEDING_SCALING:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        model = create_model(best_model_type, config)
        model.fit(X, y)

        probs = model.predict_proba(X)[:, 1]
        pct_thresholds = {
            pct: float(np.percentile(probs, pct))
            for pct in [75, 78, 80, 82, 85, 88, 90, 92]
        }
        metrics = compute_ml_metrics(y, probs)

        importance = {}
        if hasattr(model, 'feature_importances_'):
            importance = dict(zip(features, model.feature_importances_.tolist()))

        joblib.dump({
            "model": model,
            "model_type": best_model_type,
            "feature_tier": "BASIC+",
            "features": features,
            "target": target,
            "market": market,
            "league": league_code,
            "n_samples": len(df),
            "n_features": len(features),
            "metrics": metrics,
            "imputation_medians": medians.to_dict(),
            "pct_thresholds": pct_thresholds,
            "scaler": scaler,
            "platt_scaler": platt_scaler,
            "feature_importance": importance,
            "config": {
                "model_type": best_model_type,
                "feature_tier": "BASIC+",
                "random_state": config.RANDOM_STATE,
                "protocol": "LEAN_V2.0"
            },
            "created": datetime.now().isoformat()
        }, model_path)

    if verbose:
        platt_tag = " +Platt" if platt_scaler is not None else ""
        print(f"  Saved: {model_path.name} ({best_model_type}+BASIC+{platt_tag}, {len(features)} feats, AUC={metrics.get('auc', 0):.3f})")

    return str(model_path)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_league(
    filepath: str,
    league_code: str = None,
    config: Config = CONFIG,
    verbose: bool = True
) -> Dict:
    """Complete league analysis with 2 models, BASIC+ features."""
    df = load_data(filepath, verbose)
    
    if league_code is None:
        league_code = df["Division"].iloc[0] if "Division" in df.columns else "UNKNOWN"
    
    if verbose:
        n_strats = sum(len(s) for s in MARKET_STRATEGIES.values())
        n_combos = len(MODEL_TYPES) * n_strats
        print(f"\n{'=' * 80}")
        print(f"PROTOCOL LEAN V2.0: {league_code}")
        print(f"{'=' * 80}")
        print(f"Models: {', '.join(MODEL_TYPES)}")
        print(f"Features: BASIC+ ({len(config.FEATURES)} features)")
        print(f"Selection: {config.TWO_LAYER_SELECTION}")
        print(f"Total search space: {n_combos} combinations (vs 810 in V3.4.5)")
    
    results = {
        "league": league_code,
        "timestamp": datetime.now().isoformat(),
        "version": "LEAN_V2.0",
        "config": {
            "two_layer_selection": config.TWO_LAYER_SELECTION,
            "fdr_q": config.FDR_Q,
            "models": MODEL_TYPES,
            "feature_tier": "BASIC+"
        },
        "data": {
            "total": len(df),
            "eval": len(df[df["SeasonYear"].isin(config.EVAL_SEASONS)])
        },
        "markets": {}
    }
    
    all_strategy_rows = []
    best_overall = None
    best_pvalue = 1.0
    
    for market in ["HOME", "DRAW", "AWAY"]:
        if verbose:
            print(f"\n  {market}...")
        
        two_layer, strat_results, best_model, model_comparison, model_results = run_walk_forward(
            df, market, config, verbose=False
        )
        
        if not strat_results:
            results["markets"][market] = {"best": None}
            continue
        
        # Selection methods
        history = {sr.name: sr.seasons for sr in strat_results}
        simple_sel = select_simple(history, config.LOOKBACK_SIMPLE)
        strict_sel = select_strict(history, config.LOOKBACK_STRICT, config.MIN_BETS_STRICT, config.MIN_PROFITABLE_PCT)
        fdr_sel = select_fdr(strat_results, history)
        ensemble_sel = select_ensemble(strat_results, history, config)
        
        # Hybrid selection — uses model metrics for AUC gap check
        market_model_metrics = strat_results[0].model_metrics if strat_results else {}
        hybrid_sel = select_hybrid(strat_results, market_model_metrics, config)
        
        # RF→LR FALLBACK: if hybrid failed due to AUC gap and best model was RF,
        # try LR instead (LR typically has much lower AUC gap)
        fallback_used = False
        if hybrid_sel is None and best_model != "LogisticRegression":
            fb_model = "LogisticRegression"
            fb_comp = model_comparison.get(fb_model, {})
            fb_gap = abs(fb_comp.get("auc_gap", 999))
            
            if fb_gap < config.HYBRID_MAX_AUC_GAP:
                # LR passes AUC gap — rebuild results with LR
                strategies = MARKET_STRATEGIES[market]
                fb_results, fb_metrics, fb_comp_entry = rebuild_for_fallback_model(
                    model_results, fb_model, strategies, market, config
                )
                
                if fb_results:
                    fb_market_metrics = fb_results[0].model_metrics if fb_results else {}
                    fb_hybrid_sel = select_hybrid(fb_results, fb_market_metrics, config)
                    
                    if fb_hybrid_sel:
                        # Fallback succeeded — replace everything with LR results
                        hybrid_sel = fb_hybrid_sel
                        strat_results = fb_results
                        best_model = fb_model
                        model_comparison[fb_model] = fb_comp_entry
                        fallback_used = True
                        if verbose:
                            print(f"    RF->LR fallback: RF auc_gap too high, using LR (gap={fb_gap:.4f})")
        
        # Best by p-value
        best = min(strat_results, key=lambda x: x.p_value)
        
        if best.p_value < best_pvalue:
            best_pvalue = best.p_value
            best_overall = (market, best)
        
        tl_bets = sum(t.n_bets for t in two_layer) if two_layer else 0
        tl_profit = sum(t.profit for t in two_layer) if two_layer else 0
        tl_roi = (tl_profit / tl_bets * 100) if tl_bets > 0 else 0
        
        # Recommendation
        fdr_pass = best.fdr_pass
        prof_ratio = best.profitable_seasons / best.total_seasons if best.total_seasons > 0 else 0
        
        if fdr_pass and best.n_bets >= config.MIN_BETS_FOR_DEPLOY and prof_ratio >= config.MIN_PROFITABLE_PCT:
            rec_status = "DEPLOY"
        elif fdr_pass or best.p_value < 0.05:
            rec_status = "PAPER_TRADE"
        elif best.p_value < 0.10:
            rec_status = "MONITOR"
        else:
            rec_status = "SKIP"
        
        results["markets"][market] = {
            "best": best.to_dict(),
            "best_model": best_model,
            "best_feature_tier": "BASIC+",
            "model_comparison": model_comparison,
            "all": [sr.to_dict() for sr in strat_results],
            "selections": {
                "simple": simple_sel, "strict": strict_sel,
                "fdr": fdr_sel, "ensemble": ensemble_sel,
                "hybrid": hybrid_sel if hybrid_sel else "__SKIP__",
                "hybrid_fallback": fallback_used
            },
            "two_layer": {
                "bets": tl_bets, "profit": round(tl_profit, 2), "roi": round(tl_roi, 1),
                "details": [{"season": t.eval_season, "strategy": t.selected_strategy,
                             "method": t.selection_method, "bets": t.n_bets, "roi": round(t.roi, 1)}
                            for t in two_layer]
            },
            "recommendation": rec_status
        }
        
        # CSV rows
        for sr in strat_results:
            is_best = (sr.name == best.name)
            season_rois = [s.roi for s in sr.seasons]
            mm = sr.model_metrics
            
            row = {
                "league": league_code,
                "recommendation": rec_status,
                "market": market,
                "strategy": sr.name,
                "is_best": is_best,
                "ml_model": best_model,
                "feature_tier": "BASIC+",
                "n_features": aggregated_metrics_count(mm),
                "ml_model_comparison": json.dumps(model_comparison),
                "strat_pct": sr.strategy_params.get("pct", ""),
                "strat_edge": sr.strategy_params.get("edge", ""),
                "strat_min_odds": sr.strategy_params.get("min_odds", ""),
                "n_bets": sr.n_bets,
                "n_wins": sr.n_wins,
                "win_rate": round(sr.n_wins / sr.n_bets * 100, 2) if sr.n_bets > 0 else 0,
                "profit": round(sr.profit, 2),
                "roi": round(sr.roi, 2),
                "roi_mean": round(np.mean(season_rois), 2) if season_rois else 0,
                "roi_median": round(np.median(season_rois), 2) if season_rois else 0,
                "roi_std": round(np.std(season_rois), 2) if season_rois else 0,
                "avg_odds": round(sr.avg_odds, 3),
                "p_value": round(sr.p_value, 4),
                "fdr_pass": sr.fdr_pass,
                "profitable_seasons": sr.profitable_seasons,
                "total_seasons": sr.total_seasons,
                "profitable_pct": round(sr.profitable_seasons / sr.total_seasons * 100, 1) if sr.total_seasons > 0 else 0,
                "max_drawdown": round(sr.risk_metrics.get("max_drawdown", 0), 2),
                "sharpe_ratio": round(sr.risk_metrics.get("sharpe_ratio", 0), 4),
                "train_auc": round(mm.get("train", {}).get("auc", 0), 4) if mm else "",
                "test_auc": round(mm.get("test", {}).get("auc", 0), 4) if mm else "",
                "auc_gap": round(mm.get("train", {}).get("auc", 0) - mm.get("test", {}).get("auc", 0), 4) if mm else "",
                "two_layer_bets": tl_bets,
                "two_layer_profit": round(tl_profit, 2),
                "two_layer_roi": round(tl_roi, 1),
                "sel_simple": simple_sel if is_best else "",
                "sel_strict": strict_sel if is_best else "",
                "sel_fdr": fdr_sel if is_best else "",
                "sel_ensemble": ensemble_sel if is_best else "",
                "sel_hybrid": (hybrid_sel if hybrid_sel else "__SKIP__") if is_best else "",
            }
            all_strategy_rows.append(row)
        
        # Export model (with Platt calibration from last eval season OOS probs)
        if config.EXPORT_MODELS:
            platt_cal_data = None
            last_eval = max(config.EVAL_SEASONS)
            train_cal = df[df["SeasonYear"] < last_eval].copy()
            test_cal = df[df["SeasonYear"] == last_eval].copy()
            target_col = f"target_{market}"

            if len(train_cal) >= config.MIN_TRAIN_MATCHES and len(test_cal) >= config.MIN_TEST_MATCHES:
                cal_features = get_available_features(train_cal, config)
                if len(cal_features) >= 5:
                    try:
                        if best_model == ENSEMBLE_KEY:
                            # Train both models on all-but-last, predict last season
                            rf_test, rf_train, _, _, _ = train_and_evaluate(
                                "RandomForest", train_cal, test_cal, cal_features, target_col, config)
                            lr_test, lr_train, _, _, _ = train_and_evaluate(
                                "LogisticRegression", train_cal, test_cal, cal_features, target_col, config)
                            cal_probs = ENSEMBLE_WEIGHT * rf_test + (1 - ENSEMBLE_WEIGHT) * lr_test
                        else:
                            cal_probs, _, _, _, _ = train_and_evaluate(
                                best_model, train_cal, test_cal, cal_features, target_col, config)
                        platt_cal_data = {
                            'probs': cal_probs,
                            'labels': test_cal[target_col].values
                        }
                    except Exception as e:
                        if verbose:
                            print(f"    Platt calibration data failed: {e}")

            export_final_model(df, league_code, market, best_model, config, verbose,
                               platt_calibration_data=platt_cal_data)
        
        if verbose:
            print(f"    Best model: {best_model}{' (via RF->LR fallback)' if fallback_used else ''}")
            print(f"    Best strategy: {best.name} | {best.n_bets} bets | {best.roi:+.1f}% ROI | p={best.p_value:.4f}")
            print(f"    Recommendation: {rec_status}")
            if hybrid_sel:
                # Find hybrid strategy stats
                hybrid_strat = next((r for r in strat_results if r.name == hybrid_sel), None)
                if hybrid_strat:
                    print(f"    Hybrid selection: {hybrid_sel} | {hybrid_strat.n_bets} bets | {hybrid_strat.roi:+.1f}% ROI | p={hybrid_strat.p_value:.4f}")
            else:
                print(f"    Hybrid selection: __SKIP__")
    
    # Overall
    if best_overall:
        market, best_strat = best_overall
        results["recommendation"] = {
            "status": results["markets"][market]["recommendation"],
            "market": market,
            "strategy": best_strat.name,
            "roi": round(best_strat.roi, 1),
            "p_value": round(best_strat.p_value, 4),
            "fdr_pass": best_strat.fdr_pass,
            "model": results["markets"][market]["best_model"],
            "feature_tier": "BASIC+"
        }
    else:
        results["recommendation"] = {"status": "SKIP"}
    
    results["strategy_rows"] = all_strategy_rows
    
    if config.EXPORT_JSON:
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_path = output_dir / f"results_{league_code}.json"
        with open(results_path, "w") as f:
            json_results = {k: v for k, v in results.items() if k != "strategy_rows"}
            json.dump(json_results, f, indent=2, default=str)
        if verbose:
            print(f"\n  Saved: {results_path.name}")
    
    return results


def aggregated_metrics_count(mm):
    """Extract feature count from model metrics."""
    if mm and isinstance(mm, dict):
        return mm.get("n_features", "")
    return ""


# =============================================================================
# BATCH ANALYSIS
# =============================================================================

def batch_analyze(
    data_dir: str,
    leagues: List[str] = None,
    output_dir: str = None,
    config: Config = CONFIG
) -> Dict[str, Dict]:
    """Batch analyze all leagues."""
    data_path = Path(data_dir)
    csv_files = [f for f in data_path.glob("*_features*.csv") if "20252026" not in f.stem]
    
    if not csv_files:
        print(f"No feature CSV files found in {data_dir}")
        return {}
    
    if leagues:
        leagues_set = set(leagues)
        csv_files = [f for f in csv_files if any(l in f.stem for l in leagues_set)]
    
    print(f"Found {len(csv_files)} league files")
    
    out_path = Path(output_dir) if output_dir else data_path
    out_path.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR = str(out_path)
    
    all_results = {}
    all_strategy_rows = []
    
    for filepath in tqdm_wrapper(sorted(csv_files), desc="Analyzing leagues"):
        print(f"\n{'#' * 60}")
        print(f"# {filepath.name}")
        print(f"{'#' * 60}")
        
        try:
            results = analyze_league(str(filepath), None, config, verbose=True)
            league = results["league"]
            all_results[league] = results
            all_strategy_rows.extend(results.get("strategy_rows", []))
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Save CSV outputs
    if all_strategy_rows:
        detailed_df = pd.DataFrame(all_strategy_rows)
        detailed_path = out_path / "BATCH_DETAILED.csv"
        detailed_df.to_csv(detailed_path, index=False)
        print(f"\nSaved: {detailed_path} ({len(detailed_df)} rows)")
        
        best_df = pd.DataFrame([r for r in all_strategy_rows if r.get("is_best")])
        summary_path = out_path / "BATCH_SUMMARY.csv"
        best_df.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")
        
        # Comparison summary
        print(f"\n{'=' * 100}")
        print("LEAN PROTOCOL — BATCH SUMMARY")
        print(f"{'=' * 100}")
        
        summary_cols = ["league", "market", "ml_model", "strategy", "recommendation",
                        "n_bets", "roi", "p_value", "fdr_pass", "test_auc", "auc_gap"]
        avail_cols = [c for c in summary_cols if c in best_df.columns]
        print(best_df[avail_cols].to_string(index=False))
        
        # V3.4.5 comparison header
        n_deploy = len(best_df[best_df["recommendation"] == "DEPLOY"])
        n_paper = len(best_df[best_df["recommendation"] == "PAPER_TRADE"])
        n_monitor = len(best_df[best_df["recommendation"] == "MONITOR"])
        n_skip = len(best_df[best_df["recommendation"] == "SKIP"])
        
        print(f"\n  DEPLOY: {n_deploy} | PAPER_TRADE: {n_paper} | MONITOR: {n_monitor} | SKIP: {n_skip}")
        n_strats = max(len(s) for s in MARKET_STRATEGIES.values())
        print(f"  Search space: {len(MODEL_TYPES)} models × 1 tier × {n_strats} strategies × 3 markets = {len(MODEL_TYPES) * n_strats * 3} combos/league")
    
    return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Protocol Lean V2.0 — RF + LR, BASIC+ Features, Hybrid Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single league
  python protocol_lean.py --analyze --data SP1_features.csv --output ./output_lean

  # All leagues
  python protocol_lean.py --batch --data-dir ./features --all --output ./output_lean
  
  # Specific leagues
  python protocol_lean.py --batch --data-dir ./features --leagues SP1,I1,E0 --output ./output_lean
  
  # Use simple selection instead of hybrid (for comparison)
  python protocol_lean.py --batch --data-dir ./features --all --output ./output_lean --selection simple
        """
    )
    
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--analyze", action="store_true", help="Analyze single league")
    mode.add_argument("--batch", action="store_true", help="Batch analyze multiple leagues")
    
    parser.add_argument("--data", type=str, help="Features CSV path")
    parser.add_argument("--data-dir", type=str, help="Features directory")
    parser.add_argument("--league", type=str, help="League code")
    parser.add_argument("--leagues", type=str, help="Comma-separated leagues")
    parser.add_argument("--all", action="store_true", help="All leagues")
    parser.add_argument("--output", type=str, default="./output_lean", help="Output directory")
    parser.add_argument("--selection", type=str, default="hybrid",
                        choices=["hybrid", "simple", "strict", "fdr", "ensemble"])
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    config = Config()
    config.OUTPUT_DIR = args.output
    config.TWO_LAYER_SELECTION = args.selection
    
    if args.analyze:
        if not args.data:
            parser.error("--data required")
        
        results = analyze_league(args.data, args.league, config, verbose=not args.quiet)
        
        if results.get("strategy_rows"):
            out_path = Path(config.OUTPUT_DIR)
            out_path.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(results["strategy_rows"])
            csv_path = out_path / f"detailed_{results['league']}.csv"
            df.to_csv(csv_path, index=False)
            print(f"\nSaved: {csv_path}")
        
        print("\nDone!")
    
    elif args.batch:
        if not args.data_dir:
            parser.error("--data-dir required")
        
        leagues = args.leagues.split(",") if args.leagues else None
        if args.all:
            leagues = None
        
        batch_analyze(args.data_dir, leagues, args.output, config)
        print(f"\nDone! Results in {args.output}")


if __name__ == "__main__":
    main()
