# ============================================================
#  Feature engineering pipeline — vectorized (fast)
#  Fixed: form assignment bug + added stronger features
# ============================================================
import pandas as pd
import numpy as np
from src.preprocessing.normalize import get_confederation


def _compute_form(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Vectorized form stats for every team before every match.
    FIXED: removed incorrect isin filter that mixed home/away form.
    """
    df = df.sort_values("date").reset_index(drop=True)

    home = df[["date", "home_team", "away_team", "home_score", "away_score"]].copy()
    home.columns = ["date", "team", "opp", "gf", "ga"]
    home["pts"] = np.where(home.gf > home.ga, 3, np.where(home.gf == home.ga, 1, 0))

    away = df[["date", "away_team", "home_team", "away_score", "home_score"]].copy()
    away.columns = ["date", "team", "opp", "gf", "ga"]
    away["pts"] = np.where(away.gf > away.ga, 3, np.where(away.gf == away.ga, 1, 0))

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)
    grp  = long.groupby("team")

    roll_pts = grp["pts"].transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    roll_gf  = grp["gf"].transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    roll_ga  = grp["ga"].transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    roll_gd  = roll_gf - roll_ga

    # IMPROVED: add clean sheet rate and goals scored ratio
    roll_cs  = grp["ga"].transform(
        lambda x: (x.shift(1) == 0).rolling(n, min_periods=1).mean()
    )
    roll_gsr = (roll_gf / (roll_gf + roll_ga + 1e-6))

    long[f"form{n}_ppg"] = roll_pts.fillna(0)
    long[f"form{n}_gf"]  = roll_gf.fillna(0)
    long[f"form{n}_ga"]  = roll_ga.fillna(0)
    long[f"form{n}_gd"]  = roll_gd.fillna(0)
    long[f"form{n}_cs"]  = roll_cs.fillna(0)
    long[f"form{n}_gsr"] = roll_gsr.fillna(0.5)

    # FIXED: merge by team identity, not isin filter
    # Get last record per (team, date) to avoid duplicates
    long_deduped = long.drop_duplicates(["team", "date"], keep="last")

    form_cols = ["date", "team",
                 f"form{n}_ppg", f"form{n}_gf", f"form{n}_ga",
                 f"form{n}_gd", f"form{n}_cs", f"form{n}_gsr"]

    home_form = long_deduped[form_cols].copy()
    home_form.columns = ["date", "home_team",
                         f"form{n}_ppg_home", f"form{n}_gf_home",
                         f"form{n}_ga_home",  f"form{n}_gd_home",
                         f"form{n}_cs_home",  f"form{n}_gsr_home"]

    away_form = long_deduped[form_cols].copy()
    away_form.columns = ["date", "away_team",
                         f"form{n}_ppg_away", f"form{n}_gf_away",
                         f"form{n}_ga_away",  f"form{n}_gd_away",
                         f"form{n}_cs_away",  f"form{n}_gsr_away"]

    df = df.merge(home_form, on=["date", "home_team"], how="left")
    df = df.merge(away_form, on=["date", "away_team"], how="left")

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build full feature matrix from weighted + Elo-annotated dataframe.
    """
    df = df.sort_values("date").reset_index(drop=True)

    print("  Computing form (last 5)...")
    df = _compute_form(df, 5)
    print("  Computing form (last 10)...")
    df = _compute_form(df, 10)

    # IMPROVED: days since last match (rest/fatigue signal)
    df = df.sort_values(["home_team", "date"])
    df["days_rest_home"] = (
        df.groupby("home_team")["date"]
        .transform(lambda x: x.diff().dt.days.shift(0))
        .fillna(30)
        .clip(1, 60)
    )
    df = df.sort_values(["away_team", "date"])
    df["days_rest_away"] = (
        df.groupby("away_team")["date"]
        .transform(lambda x: x.diff().dt.days.shift(0))
        .fillna(30)
        .clip(1, 60)
    )
    df = df.sort_values("date").reset_index(drop=True)

    # Confederation
    df["confederation_home"] = df["home_team"].apply(get_confederation)
    df["confederation_away"] = df["away_team"].apply(get_confederation)

    # Result label
    df["result"] = np.where(
        df["home_score"] > df["away_score"], 1,
        np.where(df["home_score"] == df["away_score"], 0, -1)
    )

    # Rename to match FEATURE_COLS in xgboost_model.py
    df = df.rename(columns={
        "elo_home_pre":       "elo_home",
        "elo_away_pre":       "elo_away",
        "competition_weight": "comp_weight",
    })

    feature_cols = [
        "date", "home_team", "away_team",
        "elo_home", "elo_away", "elo_diff",
        "form5_ppg_home",  "form5_gf_home",  "form5_ga_home",
        "form5_gsr_home",  "form5_cs_home",
        "form5_ppg_away",  "form5_gf_away",  "form5_ga_away",
        "form5_gsr_away",  "form5_cs_away",
        "form10_ppg_home", "form10_gd_home", "form10_gsr_home",
        "form10_ppg_away", "form10_gd_away", "form10_gsr_away",
        "days_rest_home",  "days_rest_away",
        "neutral", "is_host_home", "is_host_away",
        "comp_weight",
        "confederation_home", "confederation_away",
        "home_score", "away_score", "result",
    ]

    available = [c for c in feature_cols if c in df.columns]
    result_df = df[available].copy()

    form_cols = [c for c in result_df.columns if "form" in c or "days_rest" in c]
    result_df[form_cols] = result_df[form_cols].fillna(0)

    print(f"  Feature matrix built: {result_df.shape}")
    return result_df