# ============================================================
#  End-to-end pipeline: raw data → trained models → simulation
# ============================================================
import numpy as np
import pandas as pd
import joblib
from src.config import DATA_RAW, DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR
from src.preprocessing.normalize import normalize_team, normalize_competition
from src.preprocessing.weighting import final_weight
from src.ratings.elo import build_elo_history, current_ratings
from src.features.engineer import build_feature_matrix
from src.models import dixon_coles, xgboost_model


def load_raw() -> pd.DataFrame:
    path = DATA_RAW / "results.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df["home_team"] = df["home_team"].apply(normalize_team)
    df["away_team"] = df["away_team"].apply(normalize_team)
    df["competition_norm"] = df["tournament"].apply(normalize_competition)
    df["competition_weight"] = df.apply(
        lambda r: final_weight(r["date"].date(), r["competition_norm"]), axis=1
    )
    df["neutral"] = df["neutral"].astype(bool)
    df["is_host_home"] = 0
    df["is_host_away"] = 0

    # FIXED: drop rows with no scores (future fixtures like WC 2026 group games)
    # keeping them would corrupt Elo history and feature engineering
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    print(f"Loaded {len(df):,} matches from {df['date'].min().year} to {df['date'].max().year}")
    return df


def get_2026_groups() -> dict:
    """
    Derive WC 2026 groups directly from results.csv fixtures.
    Uses round-robin structure — 4 teams that all play each other = 1 group.
    This avoids hardcoding and stays correct if fixture data updates.
    """
    df = pd.read_csv(DATA_RAW / "results.csv", parse_dates=["date"])

    # Future WC 2026 group stage fixtures = FIFA World Cup 2026 with no scores yet
    fixtures = df[
        (df["tournament"] == "FIFA World Cup") &
        (df["date"].dt.year == 2026) &
        (df["home_score"].isna())
    ][["date", "home_team", "away_team"]].copy()

    fixtures["home_team"] = fixtures["home_team"].apply(normalize_team)
    fixtures["away_team"] = fixtures["away_team"].apply(normalize_team)

    # Build set of all matchups
    played = set()
    for _, r in fixtures.iterrows():
        played.add(frozenset([r.home_team, r.away_team]))

    all_teams = sorted(set(fixtures.home_team) | set(fixtures.away_team))
    groups, assigned = {}, set()
    letter = "A"

    for t1 in all_teams:
        if t1 in assigned:
            continue
        group = [t1]
        for t2 in all_teams:
            if t2 == t1 or t2 in assigned:
                continue
            if frozenset([t1, t2]) in played:
                if all(frozenset([t2, gm]) in played for gm in group):
                    group.append(t2)
            if len(group) == 4:
                break
        if len(group) == 4:
            groups[letter] = sorted(group)
            assigned.update(group)
            letter = chr(ord(letter) + 1)

    print(f"Groups loaded from fixtures: {len(groups)} groups, {len(assigned)} teams")
    for g, teams in groups.items():
        print(f"  Group {g}: {teams}")

    return groups


def run_pipeline():
    print("── Step 1: Loading raw data ──")
    df = load_raw()

    print("\n── Step 2: Building Elo history ──")
    df = build_elo_history(df)
    df.to_csv(DATA_PROCESSED / "matches_with_elo.csv", index=False)
    print(f"Saved matches_with_elo.csv ({len(df):,} rows)")

    print("\n── Step 3: Building feature matrix ──")
    features = build_feature_matrix(df)
    features.to_csv(DATA_PROCESSED / "features.csv", index=False)
    print(f"Feature matrix: {features.shape}")

    print("\n── Step 4: Fitting Dixon-Coles model ──")
    # Use matches from 2000 onwards — older data adds noise, low relevance weight anyway
    df_fit = df[df["date"] >= "2000-01-01"].copy()
    df_fit["final_weight"] = df_fit.apply(
        lambda r: final_weight(r["date"].date(), r["competition_norm"]), axis=1
    )
    dc_params = dixon_coles.fit(df_fit, weight_col="final_weight")
    joblib.dump(dc_params, MODELS_DIR / "dixon_coles_params.pkl")
    print(f"Dixon-Coles fitted — rho={dc_params['rho']:.4f}  home_adv={dc_params['home_adv']:.4f}")

    print("\n── Step 5: Training XGBoost model ──")
    feature_cols = xgboost_model.FEATURE_COLS + ["result", "date"]
    clean = features[feature_cols].dropna()
    model = xgboost_model.train(clean)
    xgboost_model.save(model)
    print(f"XGBoost trained on {len(clean):,} matches")

    print("\n── Step 6: Saving current Elo ratings ──")
    ratings = current_ratings(df)
    ratings_df = pd.DataFrame(
        sorted(ratings.items(), key=lambda x: -x[1]),
        columns=["team", "elo"]
    )
    ratings_df.to_csv(DATA_PROCESSED / "elo_ratings.csv", index=False)
    print(f"Saved Elo ratings for {len(ratings_df)} teams")
    print(ratings_df.head(10).to_string(index=False))

    print("\n✅ Pipeline complete. Models saved to models/")


def run_simulation(n: int = 10_000):
    """Full tournament Monte Carlo simulation."""
    print(f"Running Monte Carlo simulation ({n:,} iterations)...")

    dc_params  = joblib.load(MODELS_DIR / "dixon_coles_params.pkl")
    ratings_df = pd.read_csv(DATA_PROCESSED / "elo_ratings.csv")
    elo_ratings = dict(zip(ratings_df["team"], ratings_df["elo"]))

    # FIXED: groups derived from fixture data — not hardcoded
    groups = get_2026_groups()

    # Build lambda lookup from fitted Dixon-Coles attack/defence params
    lambdas = {}
    all_teams = [t for g in groups.values() for t in g]
    for ta in all_teams:
        for tb in all_teams:
            if ta != tb:
                lam_a = np.exp(
                    dc_params["attack"].get(ta, 0) +
                    dc_params["defence"].get(tb, 0)
                )
                lam_b = np.exp(
                    dc_params["attack"].get(tb, 0) +
                    dc_params["defence"].get(ta, 0)
                )
                lambdas[(ta, tb)] = (lam_a, lam_b)

    from src.simulation.monte_carlo import run as mc_run
    results = mc_run(
        groups, lambdas, elo_ratings,
        dc_params["rho"],
        n_simulations=n,
        verbose=True,
    )

    out = pd.DataFrame([
        {"team": team, **probs}
        for team, probs in results.items()
    ]).sort_values("champion", ascending=False)

    out_path = OUTPUTS_DIR / "tournament_probabilities" / "latest.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(out.head(12).to_string(index=False))