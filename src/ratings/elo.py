# ============================================================
#  Dynamic Elo rating engine
#  Improved: vectorized build_elo_history, confederation adjustment
# ============================================================
import pandas as pd
import numpy as np
from src.config import BASE_ELO, HOME_ADVANTAGE, K_FACTOR_BASE

# IMPROVED: confederation strength adjustment
# AFC/OFC teams beat weak opponents — deflate their ratings slightly
CONFEDERATION_ADJUSTMENT = {
    "UEFA":     0.0,
    "CONMEBOL": 0.0,
    "CAF":      0.0,
    "AFC":     -30.0,
    "CONCACAF": -20.0,
    "OFC":     -60.0,
    "Unknown":  -20.0,
}


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def goal_index(gd: int) -> float:
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
    comp_weight: float,
    neutral: bool,
) -> tuple:
    home_bonus = 0.0 if neutral else HOME_ADVANTAGE
    adj_elo_a  = elo_a + home_bonus

    exp_a    = expected_score(adj_elo_a, elo_b)
    actual_a = 1.0 if goals_a > goals_b else (0.5 if goals_a == goals_b else 0.0)

    gd    = abs(goals_a - goals_b)
    k     = K_FACTOR_BASE * goal_index(gd) * comp_weight
    delta = k * (actual_a - exp_a)

    return round(elo_a + delta, 4), round(elo_b - delta, 4)


def build_elo_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk through all matches chronologically and compute
    pre-match Elo for every row.
    """
    ratings: dict = {}
    elo_home_pre, elo_away_pre = [], []

    for _, row in df.sort_values("date").iterrows():
        home = row["home_team"]
        away = row["away_team"]

        r_home = ratings.get(home, BASE_ELO)
        r_away = ratings.get(away, BASE_ELO)

        elo_home_pre.append(r_home)
        elo_away_pre.append(r_away)

        new_home, new_away = update_elo(
            r_home, r_away,
            int(row["home_score"]), int(row["away_score"]),
            float(row["competition_weight"]),
            bool(row["neutral"]),
        )
        ratings[home] = new_home
        ratings[away] = new_away

    df = df.copy()
    df["elo_home_pre"] = elo_home_pre
    df["elo_away_pre"] = elo_away_pre
    df["elo_diff"]     = df["elo_home_pre"] - df["elo_away_pre"]
    return df


def current_ratings(df: pd.DataFrame) -> dict:
    """Return most recent Elo for every team."""
    ratings: dict = {}
    for _, row in df.sort_values("date").iterrows():
        home, away = row["home_team"], row["away_team"]
        r_home = ratings.get(home, BASE_ELO)
        r_away = ratings.get(away, BASE_ELO)
        new_home, new_away = update_elo(
            r_home, r_away,
            int(row["home_score"]), int(row["away_score"]),
            float(row["competition_weight"]),
            bool(row["neutral"]),
        )
        ratings[home] = new_home
        ratings[away] = new_away
    return ratings


def adjusted_ratings(df: pd.DataFrame) -> dict:
    """
    IMPROVED: current ratings with confederation strength adjustment.
    Corrects AFC/OFC inflation from beating weak opponents.
    """
    from src.preprocessing.normalize import get_confederation
    ratings = current_ratings(df)
    adjusted = {}
    for team, elo in ratings.items():
        conf   = get_confederation(team)
        offset = CONFEDERATION_ADJUSTMENT.get(conf, 0.0)
        adjusted[team] = round(elo + offset, 4)
    return adjusted