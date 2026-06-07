# ============================================================
#  Feature engineering pipeline
# ============================================================
import pandas as pd
import numpy as np
from src.preprocessing.normalize import get_confederation


def form_stats(df: pd.DataFrame, team: str, before_date, n: int) -> dict:
    """Recent form for a team in the last n matches before a date."""
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team)) &
        (df["date"] < before_date)
    )
    recent = df[mask].sort_values("date").tail(n)

    pts, gf, ga = [], [], []
    for _, r in recent.iterrows():
        if r["home_team"] == team:
            gf.append(r["home_score"]); ga.append(r["away_score"])
            pts.append(3 if r["home_score"] > r["away_score"] else
                       1 if r["home_score"] == r["away_score"] else 0)
        else:
            gf.append(r["away_score"]); ga.append(r["home_score"])
            pts.append(3 if r["away_score"] > r["home_score"] else
                       1 if r["away_score"] == r["home_score"] else 0)

    if not pts:
        return {"ppg": 0.0, "gf": 0.0, "ga": 0.0, "gd": 0.0}

    return {
        "ppg": round(np.mean(pts), 4),
        "gf":  round(np.mean(gf),  4),
        "ga":  round(np.mean(ga),  4),
        "gd":  round(np.mean(gf) - np.mean(ga), 4),
    }


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from the weighted + Elo-annotated df.
    df must have: date, home_team, away_team, home_score, away_score,
                  elo_home_pre, elo_away_pre, elo_diff,
                  competition_weight, neutral, is_host_home, is_host_away
    """
    rows = []
    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        dt = row["date"]

        f5h  = form_stats(df, home, dt, 5)
        f5a  = form_stats(df, away, dt, 5)
        f10h = form_stats(df, home, dt, 10)
        f10a = form_stats(df, away, dt, 10)

        actual = (1 if row["home_score"] > row["away_score"] else
                  0 if row["home_score"] == row["away_score"] else -1)

        rows.append({
            "date":              dt,
            "home_team":         home,
            "away_team":         away,
            "elo_home":          row["elo_home_pre"],
            "elo_away":          row["elo_away_pre"],
            "elo_diff":          row["elo_diff"],
            "form5_ppg_home":    f5h["ppg"],
            "form5_gf_home":     f5h["gf"],
            "form5_ga_home":     f5h["ga"],
            "form5_ppg_away":    f5a["ppg"],
            "form5_gf_away":     f5a["gf"],
            "form5_ga_away":     f5a["ga"],
            "form10_ppg_home":   f10h["ppg"],
            "form10_gd_home":    f10h["gd"],
            "form10_ppg_away":   f10a["ppg"],
            "form10_gd_away":    f10a["gd"],
            "neutral":           int(row["neutral"]),
            "is_host_home":      int(row.get("is_host_home", 0)),
            "is_host_away":      int(row.get("is_host_away", 0)),
            "comp_weight":       row["competition_weight"],
            "confederation_home": get_confederation(home),
            "confederation_away": get_confederation(away),
            "home_score":        row["home_score"],
            "away_score":        row["away_score"],
            "result":            actual,      # 1=home win, 0=draw, -1=away win
        })

    return pd.DataFrame(rows)
