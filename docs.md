# ⚽ FIFA World Cup 2026 ML Forecasting System

> A professional-grade probabilistic forecasting platform for FIFA World Cup 2026 — built with XGBoost, Dixon-Coles Poisson modeling, dynamic Elo ratings, and Monte Carlo tournament simulation.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Sources](#data-sources)
- [Data Preprocessing](#data-preprocessing)
- [Rating System](#rating-system)
- [Feature Engineering](#feature-engineering)
- [Expected Goals Model](#expected-goals-model)
- [Dixon-Coles Model](#dixon-coles-model)
- [Tournament Simulation](#tournament-simulation)
- [Monte Carlo Engine](#monte-carlo-engine)
- [Ensemble Forecast Layer](#ensemble-forecast-layer)
- [Validation Framework](#validation-framework)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Output Format](#output-format)
- [Roadmap](#roadmap)

---

## Project Overview

This system predicts match outcomes, scoreline distributions, and tournament progression probabilities for all 48 teams in FIFA World Cup 2026. It combines:

- **Historical match data** (1950–2026, 54,000+ matches)
- **Dynamic Elo rating system** updated after every match
- **Dixon-Coles Poisson model** with ρ low-score correction
- **XGBoost match outcome classifier**
- **Monte Carlo tournament simulation** (100,000 iterations)
- **Ensemble forecast layer** combining all model outputs

The 2026 tournament uses the new **48-team format**: 12 groups → Round of 32 (top 2 from each group + 8 best third-place finishers) → Round of 16 → Quarterfinals → Semifinals → Final.

---

## Features

| Feature | Description |
|---|---|
| Match outcome probabilities | Win / Draw / Loss % per team |
| Scoreline probability matrix | Full Poisson grid (0-0 through 5-5+) |
| Expected goals (xG) | λ values per team per match |
| Group stage standings | Simulated qualification probabilities |
| Knockout progression | R32 → R16 → QF → SF → Final → Champion % |
| Dynamic Elo updates | Ratings update live during tournament simulation |
| Ensemble model | Elo + Dixon-Coles + XGBoost + Market calibration |
| Streamlit dashboard | Interactive match and tournament forecasts |

---

## System Architecture

```
Historical Match Data (1950–2026)
          │
          ▼
  Data Preprocessing
  (name normalization, alias resolution)
          │
          ▼
  Temporal & Competition Weighting
  (exponential decay × competition importance)
          │
          ▼
  Dynamic Elo Rating System
  (updated per match, stored pre-match)
          │
          ▼
  Feature Engineering
  (Elo diff, form, xG, home advantage, confederation)
          │
          ▼
  Expected Goals Model
  Dixon-Coles Poisson + XGBoost → λ_A, λ_B
          │
          ▼
  Scoreline Probability Matrix
  (Poisson PMF grid with ρ correction)
          │
          ▼
  Match Probability Engine
  (Win / Draw / Loss aggregation)
          │
          ▼
  Monte Carlo Tournament Simulation
  (100,000 iterations × full bracket)
          │
          ▼
  Ensemble Forecast Layer
  (Elo 35% + Dixon-Coles 35% + XGBoost 20% + Market 10%)
          │
          ▼
  Streamlit Dashboard & Forecast Reports
```

---

## Project Structure

```
wc2026-forecasting/
├── data/
│   ├── raw/
│   │   ├── results.csv                  # martj42 international results dataset
│   │   └── wc2026_fixtures.csv          # 2026 official fixtures
│   ├── processed/
│   │   ├── matches_weighted.csv         # time + competition weighted matches
│   │   ├── elo_ratings.csv              # historical Elo per team per match
│   │   └── features.csv                 # engineered feature matrix
│   └── external/
│       └── market_odds.csv              # betting market calibration data
│
├── src/
│   ├── preprocessing/
│   │   ├── normalize.py                 # team name & competition standardization
│   │   └── weighting.py                 # temporal decay + competition weights
│   │
│   ├── ratings/
│   │   └── elo.py                       # dynamic Elo rating engine
│   │
│   ├── features/
│   │   └── engineer.py                  # feature extraction pipeline
│   │
│   ├── models/
│   │   ├── dixon_coles.py               # Dixon-Coles Poisson model with ρ correction
│   │   ├── xgboost_model.py             # XGBoost outcome classifier
│   │   └── ensemble.py                  # weighted ensemble combiner
│   │
│   ├── simulation/
│   │   ├── match_engine.py              # single match simulator
│   │   ├── group_stage.py               # group stage + FIFA tiebreakers
│   │   ├── knockout.py                  # knockout bracket simulator
│   │   ├── third_place.py               # 2026 best third-place slot logic
│   │   └── monte_carlo.py               # full tournament Monte Carlo runner
│   │
│   └── dashboard/
│       └── app.py                       # Streamlit dashboard
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_elo_validation.ipynb
│   ├── 03_dixon_coles_fit.ipynb
│   ├── 04_xgboost_training.ipynb
│   └── 05_backtest_results.ipynb
│
├── models/
│   ├── dixon_coles_params.pkl
│   └── xgboost_model.pkl
│
├── outputs/
│   ├── match_forecasts/
│   ├── group_probabilities/
│   └── tournament_probabilities/
│
├── requirements.txt
└── README.md
```

---

## Installation

### Requirements

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/mosesamwoma/wc2026-forecasting.git
cd wc2026-forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
pandas>=2.0
numpy>=1.26
scipy>=1.11
scikit-learn>=1.3
xgboost>=2.0
statsmodels>=0.14
streamlit>=1.35
matplotlib>=3.8
seaborn>=0.13
plotly>=5.18
joblib>=1.3
tqdm>=4.66
```

---

## Data Sources

### Primary Training Dataset

**martj42 — International Football Results (1872–2026)**
- Source: [Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- File: `results.csv`
- Fields: `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`, `city`, `country`, `neutral`
- Coverage: 54,000+ matches across all international competitions

### 2026 Fixtures Dataset

**areezvisram12 — FIFA World Cup 2026 Fixtures**
- Source: Kaggle
- File: `wc2026_fixtures.csv`
- Fields: match ID, group, team A, team B, date, venue, city

### Data Download

```bash
# Using Kaggle CLI
kaggle datasets download martj42/international-football-results-from-1872-to-2017 -p data/raw/
kaggle datasets download areezvisram12/fifa-world-cup-2026 -p data/raw/
```

---

## Data Preprocessing

### Team Name Normalization

Historical team names are standardized to current FIFA names:

```python
# src/preprocessing/normalize.py

TEAM_ALIASES = {
    "FR Germany": "Germany",
    "West Germany": "Germany",
    "Yugoslavia": "Serbia",           # treated as historical entity
    "Czechoslovakia": "Czech Republic",
    "Soviet Union": "Russia",
    "Dutch East Indies": "Indonesia",
    "Zaire": "DR Congo",
    "Republic of Ireland": "Ireland",
    "China PR": "China",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "USA": "United States",
}

def normalize_team(name: str) -> str:
    return TEAM_ALIASES.get(name.strip(), name.strip())
```

### Competition Normalization

```python
COMPETITION_MAP = {
    "FIFA World Cup": "World Cup",
    "FIFA World Cup qualification": "WC Qualifier",
    "UEFA Euro": "Continental Championship",
    "Copa América": "Continental Championship",
    "Africa Cup of Nations": "Continental Championship",
    "AFC Asian Cup": "Continental Championship",
    "CONCACAF Gold Cup": "Continental Championship",
    "UEFA Nations League": "Nations League",
    "Confederations Cup": "Confederations Cup",
    "Friendly": "Friendly",
}
```

---

## Rating System

### Elo Rating Engine

Every team starts with a base Elo of **1500**. Ratings update after every match in chronological order.

```python
# src/ratings/elo.py

BASE_ELO = 1500
HOME_ADVANTAGE = 65      # Elo points added for home team (neutral = 0)
K_FACTOR_BASE = 40

def expected_score(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def goal_index(gd: int) -> float:
    """Goal difference multiplier for K-factor scaling."""
    if gd <= 1:
        return 1.0
    elif gd == 2:
        return 1.5
    else:
        return 1.75 + (gd - 3) / 8

def update_elo(
    elo_a: float,
    elo_b: float,
    goals_a: int,
    goals_b: int,
    competition_weight: float,
    neutral: bool
) -> tuple[float, float]:

    home_bonus = 0 if neutral else HOME_ADVANTAGE
    adj_elo_a = elo_a + home_bonus

    exp_a = expected_score(adj_elo_a, elo_b)
    actual_a = 1.0 if goals_a > goals_b else (0.5 if goals_a == goals_b else 0.0)

    gd = abs(goals_a - goals_b)
    k = K_FACTOR_BASE * goal_index(gd) * competition_weight

    delta = k * (actual_a - exp_a)
    return elo_a + delta, elo_b - delta
```

---

## Feature Engineering

For every match in the training set, the following features are computed:

```python
# src/features/engineer.py

FEATURES = [
    # Elo
    "elo_a",                    # Team A Elo before match
    "elo_b",                    # Team B Elo before match
    "elo_diff",                 # elo_a - elo_b

    # Recent form — last 5 matches
    "form5_pts_a",              # points per game (W=3, D=1, L=0)
    "form5_gf_a",               # goals scored per game
    "form5_ga_a",               # goals conceded per game
    "form5_pts_b",
    "form5_gf_b",
    "form5_ga_b",

    # Recent form — last 10 matches
    "form10_pts_a",
    "form10_gd_a",              # goal difference per game
    "form10_pts_b",
    "form10_gd_b",

    # Context
    "neutral",                  # 1 if neutral venue
    "is_host_a",                # 1 if team A is host nation
    "competition_weight",       # competition importance weight
    "confederation_a",          # one-hot encoded
    "confederation_b",
]
```

---

## Expected Goals Model

### Competition Importance Weights

```python
# src/preprocessing/weighting.py

COMPETITION_WEIGHTS = {
    "World Cup":                4.0,
    "Continental Championship": 3.0,
    "Nations League":           2.5,
    "WC Qualifier":             2.0,
    "Continental Qualifier":    1.5,
    "Confederations Cup":       1.5,
    "Friendly":                 0.5,
}
```

### Temporal Decay

```python
import numpy as np
from datetime import date

LAMBDA_DECAY = 0.25

def time_weight(match_date: date, reference_date: date = None) -> float:
    if reference_date is None:
        reference_date = date.today()
    years_ago = (reference_date - match_date).days / 365.25
    return np.exp(-LAMBDA_DECAY * years_ago)

def final_weight(match_date: date, competition: str) -> float:
    return time_weight(match_date) * COMPETITION_WEIGHTS.get(competition, 1.0)
```

**Example decay values:**

| Year | Time Weight |
|---|---|
| 2025 | 0.95 |
| 2023 | 0.86 |
| 2020 | 0.55 |
| 2015 | 0.29 |
| 2010 | 0.08 |
| 2000 | 0.01 |

---

## Dixon-Coles Model

The Dixon-Coles model corrects raw Poisson for systematic underestimation of low-scoring results (0-0, 1-0, 0-1, 1-1).

```python
# src/models/dixon_coles.py

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

def rho_correction(x: int, y: int, lambda_a: float, lambda_b: float, rho: float) -> float:
    """Low-score correction factor."""
    if x == 0 and y == 0:
        return 1 - lambda_a * lambda_b * rho
    elif x == 1 and y == 0:
        return 1 + lambda_b * rho
    elif x == 0 and y == 1:
        return 1 + lambda_a * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0

def score_probability(
    goals_a: int,
    goals_b: int,
    lambda_a: float,
    lambda_b: float,
    rho: float
) -> float:
    p = (poisson.pmf(goals_a, lambda_a)
         * poisson.pmf(goals_b, lambda_b)
         * rho_correction(goals_a, goals_b, lambda_a, lambda_b, rho))
    return max(p, 0.0)

def score_matrix(lambda_a: float, lambda_b: float, rho: float, max_goals: int = 8):
    """Full score probability matrix."""
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i, j] = score_probability(i, j, lambda_a, lambda_b, rho)
    return matrix

def match_probabilities(matrix: np.ndarray) -> dict:
    """Aggregate score matrix into win/draw/loss."""
    win_a = np.sum(np.tril(matrix, -1))
    draw  = np.sum(np.diag(matrix))
    win_b = np.sum(np.triu(matrix, 1))
    total = win_a + draw + win_b
    return {
        "win_a": win_a / total,
        "draw":  draw  / total,
        "win_b": win_b / total,
    }
```

---

## Tournament Simulation

### 2026 Format

The 2026 World Cup uses a **48-team format**:

- **12 groups** of 4 teams each
- **Top 2 from each group** advance automatically (24 teams)
- **8 best third-place finishers** advance (from 12 third-place teams)
- **Round of 32** → Round of 16 → Quarterfinals → Semifinals → Final

### FIFA Tiebreakers (Group Stage)

Teams level on points are separated by the following criteria in order:

```python
# src/simulation/group_stage.py

TIEBREAKER_ORDER = [
    "points",
    "head_to_head_points",
    "head_to_head_gd",
    "head_to_head_gf",
    "overall_gd",
    "overall_gf",
    "fair_play_points",     # yellow/red card deductions
    "drawing_of_lots",      # random if all else equal
]
```

### Best Third-Place Slot Logic

The 8 best third-place teams are ranked by:
1. Points
2. Goal difference
3. Goals scored
4. Fair play
5. Drawing of lots

```python
# src/simulation/third_place.py

def select_best_third_place(third_place_teams: list[dict]) -> list[dict]:
    """
    Rank all 12 third-place teams and return top 8.
    Each dict: {team, group, points, gd, gf, fair_play}
    """
    sorted_teams = sorted(
        third_place_teams,
        key=lambda t: (
            t["points"],
            t["gd"],
            t["gf"],
            -t["fair_play"],    # fewer cards is better
        ),
        reverse=True
    )
    return sorted_teams[:8]
```

### Knockout Match Logic

```python
# src/simulation/match_engine.py

def simulate_knockout_match(
    team_a: str,
    team_b: str,
    elo_ratings: dict,
    model_params: dict
) -> str:
    """Returns winner after extra time / penalty shootout if needed."""
    matrix = score_matrix(
        lambda_a=model_params["lambda"][team_a],
        lambda_b=model_params["lambda"][team_b],
        rho=model_params["rho"]
    )
    probs = match_probabilities(matrix)

    roll = np.random.random()
    if roll < probs["win_a"]:
        return team_a
    elif roll < probs["win_a"] + probs["draw"]:
        # Extra time + penalties
        return simulate_penalties(team_a, team_b, elo_ratings)
    else:
        return team_b

def simulate_penalties(team_a: str, team_b: str, elo_ratings: dict) -> str:
    """Penalty win probability estimated from Elo difference."""
    elo_diff = elo_ratings[team_a] - elo_ratings[team_b]
    p_a = 1 / (1 + 10 ** (-elo_diff / 800))   # dampened for penalties
    return team_a if np.random.random() < p_a else team_b
```

---

## Monte Carlo Engine

```python
# src/simulation/monte_carlo.py

from tqdm import tqdm
import numpy as np

def run_monte_carlo(
    fixtures: list[dict],
    elo_ratings: dict,
    model_params: dict,
    n_simulations: int = 100_000
) -> dict:
    """
    Run n_simulations full tournament simulations.
    Returns probability dict for each team × stage.
    """
    stages = ["group", "r32", "r16", "qf", "sf", "final", "champion"]
    counts = {team: {s: 0 for s in stages} for team in elo_ratings}

    for _ in tqdm(range(n_simulations)):
        result = simulate_full_tournament(fixtures, elo_ratings.copy(), model_params)
        for team, reached_stages in result.items():
            for stage in reached_stages:
                counts[team][stage] += 1

    # Convert counts to probabilities
    probs = {
        team: {s: counts[team][s] / n_simulations for s in stages}
        for team in counts
    }
    return probs
```

**Recommended iteration counts:**

| Use Case | Iterations | Runtime (approx.) |
|---|---|---|
| Development / testing | 10,000 | ~30 seconds |
| Standard run | 100,000 | ~5 minutes |
| High precision | 1,000,000 | ~50 minutes |

---

## Ensemble Forecast Layer

```python
# src/models/ensemble.py

ENSEMBLE_WEIGHTS = {
    "elo":          0.35,
    "dixon_coles":  0.35,
    "xgboost":      0.20,
    "market":       0.10,
}

def ensemble_probabilities(
    elo_probs: dict,
    dc_probs: dict,
    xgb_probs: dict,
    market_probs: dict = None
) -> dict:
    """
    Weighted combination of model outputs.
    Each input: {"win_a": float, "draw": float, "win_b": float}
    """
    w = ENSEMBLE_WEIGHTS
    if market_probs is None:
        # Redistribute market weight to Elo and DC equally
        w = {"elo": 0.40, "dixon_coles": 0.40, "xgboost": 0.20, "market": 0.0}

    result = {}
    for key in ["win_a", "draw", "win_b"]:
        result[key] = (
            w["elo"]         * elo_probs[key] +
            w["dixon_coles"] * dc_probs[key] +
            w["xgboost"]     * xgb_probs[key] +
            w["market"]      * (market_probs[key] if market_probs else 0)
        )
    return result
```

---

## Validation Framework

The system is backtested on four World Cups with held-out data:

```python
BACKTEST_TOURNAMENTS = [
    {"year": 2010, "host": "South Africa"},
    {"year": 2014, "host": "Brazil"},
    {"year": 2018, "host": "Russia"},
    {"year": 2022, "host": "Qatar"},
]
```

### Metrics

```python
from sklearn.metrics import log_loss, brier_score_loss
import numpy as np

def evaluate_forecasts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    y_true: one-hot encoded [win_a, draw, win_b] per match
    y_pred: predicted probabilities per match
    """
    return {
        "log_loss":         log_loss(y_true, y_pred),
        "brier_score":      brier_score_loss(y_true[:, 0], y_pred[:, 0]),
        "accuracy":         np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)),
        "calibration_error": calibration_error(y_true, y_pred),
    }
```

> **Important:** Use **temporal cross-validation** — never random train/test split. Train only on matches before each tournament's start date.

```python
def temporal_split(df, cutoff_date):
    train = df[df["date"] < cutoff_date]
    test  = df[df["date"] >= cutoff_date]
    return train, test
```

---

## Streamlit Dashboard

```bash
# Run the dashboard
streamlit run src/dashboard/app.py
```

### Dashboard Pages

| Page | Content |
|---|---|
| Match Forecast | Select any two teams → win %, scoreline grid, λ values |
| Team Profile | Elo rating, form, offensive/defensive metrics |
| Group Stage | Group tables with qualification probabilities |
| Tournament Odds | All 48 teams × all stages, sorted by champion % |
| Simulation Runner | Run Monte Carlo with configurable iterations |

---

## Output Format

### Match Forecast

```
Spain vs Morocco  |  Group C  |  June 18, 2026

Spain Win:    57.3%   ████████████████░░░░
Draw:         24.8%   ███████░░░░░░░░░░░░░
Morocco Win:  17.9%   █████░░░░░░░░░░░░░░░

Expected Goals:  Spain λ=1.74  |  Morocco λ=0.92

Top Scorelines:
  1-0  11.4%      2-0  10.1%
  1-1   9.8%      2-1   9.3%
  0-0   7.1%      3-0   5.2%
  0-1   5.0%      3-1   4.8%
```

### Tournament Probabilities

```
Team          | Group | R32  | R16  | QF   | SF   | Final | Champion
France        | 99.1% | 82.4%| 68.1%| 52.3%| 38.0%| 23.5% | 14.2%
Brazil        | 98.7% | 80.2%| 64.9%| 49.7%| 36.1%| 21.8% | 12.1%
England       | 98.2% | 77.8%| 62.1%| 47.0%| 33.4%| 20.1% | 10.8%
Argentina     | 97.9% | 76.1%| 60.0%| 45.2%| 31.8%| 19.0% |  9.5%
Spain         | 97.5% | 74.0%| 57.8%| 43.5%| 30.6%| 17.9% |  9.1%
```

---

## Roadmap

- [x] Historical data pipeline
- [x] Elo rating engine
- [x] Feature engineering
- [x] Dixon-Coles model with ρ correction
- [x] XGBoost classifier
- [x] Monte Carlo tournament engine
- [x] 2026 third-place slot logic
- [x] Streamlit dashboard v1
- [ ] Live match updates during tournament
- [ ] Player-level absence/injury flags
- [ ] Confederation-specific Elo inflation correction
- [ ] Betting market odds integration
- [ ] Daily forecast report generation (PDF/HTML)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Musa Amwoma**
Data Science · Meru University of Science and Technology
GitHub: [@mosesamwoma](https://github.com/mosesamwoma)
Portfolio: [iammoses.vercel.app](https://iammoses.vercel.app)