# ============================================================
#  Feature engineering pipeline — vectorized (fast)
# ============================================================
import pandas as pd
import numpy as np
from src.preprocessing.normalize import get_confederation


def _compute_form(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Vectorized form stats for every team before every match.
    Returns df with columns: form{n}_ppg_{home|away}, form{n}_gf_{home|away},
                              form{n}_ga_{home|away}, form{n}_gd_{home|away}
    """
    df = df.sort_values("date").reset_index(drop=True)

    # Build long format: one row per team per match
    home = df[["date", "home_team", "away_team", "home_score", "away_score"]].copy()
    home.columns = ["date", "team", "opp", "gf", "ga"]
    home["pts"] = np.where(home.gf > home.ga, 3, np.where(home.gf == home.ga, 1, 0))

    away = df[["date", "away_team", "home_team", "away_score", "home_score"]].copy()
    away.columns = ["date", "team", "opp", "gf", "ga"]
    away["pts"] = np.where(away.gf > away.ga, 3, np.where(away.gf == away.ga, 1, 0))

    long = pd.concat([home, away], ignore_index=True).sort_values("date")

    # Rolling stats per team — shift(1) so we never include the current match
    long = long.sort_values(["team", "date"])
    grp  = long.groupby("team")

    roll_pts = grp["pts"].transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    roll_gf  = grp["gf"].transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    roll_ga  = grp["ga"].transform(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    roll_gd  = roll_gf - roll_ga

    long[f"form{n}_ppg"] = roll_pts.fillna(0)
    long[f"form{n}_gf"]  = roll_gf.fillna(0)
    long[f"form{n}_ga"]  = roll_ga.fillna(0)
    long[f"form{n}_gd"]  = roll_gd.fillna(0)

    # Split back to home and away
    home_form = long[long["team"].isin(df["home_team"])].copy()
    away_form = long[long["team"].isin(df["away_team"])].copy()

    # Merge home form
    home_cols = home_form[["date", "team", f"form{n}_ppg", f"form{n}_gf",
                            f"form{n}_ga", f"form{n}_gd"]].copy()
    home_cols.columns = ["date", "home_team",
                         f"form{n}_ppg_home", f"form{n}_gf_home",
                         f"form{n}_ga_home",  f"form{n}_gd_home"]
    home_cols = home_cols.drop_duplicates(["date", "home_team"])

    away_cols = away_form[["date", "team", f"form{n}_ppg", f"form{n}_gf",
                            f"form{n}_ga", f"form{n}_gd"]].copy()
    away_cols.columns = ["date", "away_team",
                         f"form{n}_ppg_away", f"form{n}_gf_away",
                         f"form{n}_ga_away",  f"form{n}_gd_away"]
    away_cols = away_cols.drop_duplicates(["date", "away_team"])

    df = df.merge(home_cols, on=["date", "home_team"], how="left")
    df = df.merge(away_cols, on=["date", "away_team"], how="left")

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build full feature matrix from weighted + Elo-annotated dataframe.
    Vectorized — runs in seconds instead of hours.
    """
    df = df.sort_values("date").reset_index(drop=True)

    # Compute form for last 5 and last 10 matches
    print("  Computing form (last 5)...")
    df = _compute_form(df, 5)
    print("  Computing form (last 10)...")
    df = _compute_form(df, 10)

    # Confederation one-hot
    df["confederation_home"] = df["home_team"].apply(get_confederation)
    df["confederation_away"] = df["away_team"].apply(get_confederation)

    # Result label
    df["result"] = np.where(
        df["home_score"] > df["away_score"], 1,
        np.where(df["home_score"] == df["away_score"], 0, -1)
    )

    # Select and rename final feature columns
    feature_cols = [
        "date", "home_team", "away_team",
        "elo_home_pre", "elo_away_pre", "elo_diff",
        "form5_ppg_home",  "form5_gf_home",  "form5_ga_home",
        "form5_ppg_away",  "form5_gf_away",  "form5_ga_away",
        "form10_ppg_home", "form10_gd_home",
        "form10_ppg_away", "form10_gd_away",
        "neutral", "is_host_home", "is_host_away",
        "competition_weight",
        "confederation_home", "confederation_away",
        "home_score", "away_score", "result",
    ]

    # Rename elo cols to match FEATURE_COLS in xgboost_model.py
    df = df.rename(columns={
        "elo_home_pre":      "elo_home",
        "elo_away_pre":      "elo_away",
        "competition_weight": "comp_weight",
    })

    feature_cols = [c.replace("elo_home_pre", "elo_home")
                     .replace("elo_away_pre", "elo_away")
                     .replace("competition_weight", "comp_weight")
                    for c in feature_cols]

    available = [c for c in feature_cols if c in df.columns]
    result_df = df[available].copy()

    # Fill any remaining NaN form values with 0
    form_cols = [c for c in result_df.columns if "form" in c]
    result_df[form_cols] = result_df[form_cols].fillna(0)

    print(f"  Feature matrix built: {result_df.shape}")
    return result_df