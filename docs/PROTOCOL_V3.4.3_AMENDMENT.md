# PROTOCOL V3.4.3 AMENDMENT

## Changes from V3.4.2

**Date:** January 2026  
**Author:** Marc  
**Status:** Production Ready

---

## SUMMARY OF CHANGES

| Section | V3.4.2 | V3.4.3 | Reason |
|---------|--------|--------|--------|
| 9.3 (NEW) | - | Model Metrics Tracking | Track AUC/Brier/LogLoss per season |
| 5.x Imputation | fillna(0) | Median imputation | Avoid artificial signals |
| Monte Carlo | Loop (slow) | Vectorized | 10-50x faster |
| CLI | --output ambiguous | --output as directory | Consistent behavior |

---

## AMENDMENT 1: Section 11.1 - Monte Carlo P-Value (CLARIFICATION)

```
## 11.1 Monte Carlo P-Value

The p-value answers: "Could random bettors achieve my profit by luck?"

**How it works:**
1. You bet 100 matches at real odds [3.2, 2.9, 3.5, ...]
2. You won 36 times → profit = +12 units

3. Simulate 50,000 random bettors:
   - Same 100 matches, same odds
   - Each bet wins at implied probability (1/odds)
   - Calculate their profit

4. p-value = % of random bettors who got ≥ +12 units

**Key insight:**
- Random bettors win at 1/odds → break even on average
- You vs break-even baseline = fair comparison
- Your actual win rate vs implied win rate = your edge

**Implementation (vectorized for speed):**
    def monte_carlo_pvalue(odds_vector, wins_vector, n_sims=50000):
        actual_profit = np.sum(wins_vector * odds_vector) - n_bets
        implied_probs = 1.0 / odds_vector  # Break-even baseline
        
        rng = np.random.default_rng(seed)
        random_matrix = rng.random((n_sims, n_bets))
        sim_wins = (random_matrix < implied_probs).astype(float)
        sim_profits = np.sum(sim_wins * odds_vector, axis=1) - n_bets
        
        return np.mean(sim_profits >= actual_profit)
```

---

## AMENDMENT 2: Section 9.3 - Model Metrics Tracking (NEW)

```
## 9.3 Model Metrics Tracking (NEW in V3.4.3)

Model evaluation metrics (AUC, Brier Score, Log Loss) are now tracked
and reported per season and per market.

### Why Track Metrics
- Diagnose model quality independent of betting outcomes
- Identify seasons where model performed poorly
- Compare predictive power across markets
- Detect concept drift over time

### Metrics Stored

Per Season:
    {
        "season": 2023,
        "auc": 0.584,
        "brier": 0.218,
        "logloss": 0.612
    }

Per Market (averaged):
    {
        "auc": 0.581,      # Mean across seasons
        "brier": 0.220,
        "logloss": 0.615,
        "n_seasons": 5,
        "per_season": [...]  # Detail per season
    }

### Output Display
Results now show metrics inline:
    Best: AGGRESSIVE | 156 bets | ROI=+18.2% | p=0.0234 | Prof=4/5 | FDR=3 | AUC=0.581, Brier=0.220

### JSON Export
The results_<league>.json now includes:
    {
        "markets": {
            "DRAW": {
                "model_metrics": {
                    "auc": 0.581,
                    "brier": 0.220,
                    "logloss": 0.615,
                    "n_seasons": 5,
                    "per_season": [...]
                },
                "best": {...}
            }
        }
    }

### Using Metrics in Model Selection
While ROI-based selection is prohibited (hindsight bias), you MAY use
metrics for tie-breaking between otherwise equivalent strategies:
- Prefer strategy backed by model with higher AUC
- Prefer strategy backed by model with lower Brier score
```

---

## AMENDMENT 3: Section 5.x - Feature Imputation (REVISED)

### V3.4.2 (Old)
```python
# Dangerous: fillna(0) can create artificial signals
X_train = train_df[features].fillna(0).values
X_test = test_df[features].fillna(0).values
```

### V3.4.3 (New)
```
## 5.x Feature Imputation

Missing values MUST be imputed using training data median, NOT zero.

**Why fillna(0) is dangerous:**
- For Elo features: 0 is far below any real team (artificial signal)
- For form ratios: 0 suggests terrible form when data is just missing
- For odds-derived: 0 is impossible and creates outliers

**Correct Method:**
    # Compute medians from training data ONLY
    medians = X_train_df.median()
    
    # Apply to both train and test
    X_train = X_train_df.fillna(medians).values
    X_test = X_test_df.fillna(medians).values

**Model Export:**
Saved models now include imputation_medians for prediction:
    data = joblib.load("model_SP1_DRAW.joblib")
    medians = data["imputation_medians"]
    X_new = new_df[features].fillna(medians).values
```

---

## AMENDMENT 4: Performance Improvements

### Monte Carlo Vectorization
```
## 11.1.1 Vectorized Monte Carlo

The Monte Carlo simulation is now fully vectorized using NumPy.

**Before (V3.4.2):** Python loop, ~50,000 iterations
    for i in range(n_sims):
        sim_wins = np.random.random(n_bets) < implied_probs
        sim_profits[i] = ...

**After (V3.4.3):** Single NumPy operation
    random_matrix = rng.random((n_sims, n_bets))  # Shape: (50000, n_bets)
    sim_wins = (random_matrix < true_probs)       # Vectorized comparison
    sim_profits = np.sum(sim_wins * odds_vector, axis=1) - n_bets

**Performance:** 10-50x faster depending on number of bets

**RNG Change:**
- Old: np.random.seed(seed) - mutates global state
- New: np.random.default_rng(seed) - isolated RNG instance
```

---

## UPDATED DATA STRUCTURES

### SeasonResult
```python
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
```

### StrategyResult
```python
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
    model_metrics: Dict = field(default_factory=dict)  # NEW in V3.4.3
```

---

## IMPACT ANALYSIS

### Expected Changes in Results

| Metric | V3.4.2 | V3.4.3 | Direction |
|--------|--------|--------|-----------|
| p-values | Same | Same | No change (same logic) |
| FDR pass rate | Same | Same | No change |
| ROI | Same | Same | No change |
| Execution time | Slow | Fast | 10-50x improvement |
| Missing data handling | fillna(0) | Median | More robust |

### Migration Notes

1. **No change to p-values** - Monte Carlo uses same break-even baseline
2. **Faster execution** - vectorized Monte Carlo
3. **Better imputation** - median instead of zero
4. **Model metrics visible** - see AUC/Brier per market

---

## VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 3.1 | Dec 2025 | Initial protocol with Hindsight approach |
| 3.2 | Jan 2026 | Removed Hindsight, added Ensemble Voting |
| 3.3 | Jan 2026 | Three feature tiers, detailed strategy explanation |
| 3.4 | Jan 2026 | Removed mandatory calibration |
| 3.4.2 | Jan 2026 | Model frozen, two-layer fixes, exports |
| **3.4.3** | **Jan 2026** | **Vectorized Monte Carlo, metrics tracking, median imputation** |

---

## QUICK REFERENCE: What Changed

```
FIX #1: --output now always used as directory (consistent)
FIX #2: Monte Carlo vectorized (10-50x faster)
FIX #3: Missing values use median imputation (not zero)
FIX #4: Missing odds column validated before prediction
FIX #5: Model metrics tracked and reported per season/market
```

---

**END OF AMENDMENT**

*Protocol Version: 3.4.3 | January 2026*
