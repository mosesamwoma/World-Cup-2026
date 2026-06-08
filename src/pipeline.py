# ============================================================
#  End-to-end pipeline: raw data → trained models → simulation
# ============================================================
import pandas as pd
import joblib
from datetime import date
from src.config import DATA_RAW, DATA_PROCESSED, MODELS_DIR
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
    # neutral column is already bool in martj42 dataset
    df["neutral"] = df["neutral"].astype(bool)
    df["is_host_home"] = 0
    df["is_host_away"] = 0
    print(f"Loaded {len(df):,} matches from {df['date'].min().year} to {df['date'].max().year}")
    return df


def run_pipeline():
    print("── Step 1: Loading raw data ──")
    df = load_raw()

    print("── Step 2: Building Elo history ──")
    df = build_elo_history(df)
    df.to_csv(DATA_PROCESSED / "matches_with_elo.csv", index=False)
    print(f"Saved matches_with_elo.csv")

    print("── Step 3: Building feature matrix ──")
    features = build_feature_matrix(df)
    features.to_csv(DATA_PROCESSED / "features.csv", index=False)
    print(f"Feature matrix: {features.shape}")

    print("── Step 4: Fitting Dixon-Coles model ──")
    # Use only matches from 2000 onwards for fitting — older data too noisy
    df_fit = df[df["date"] >= "2000-01-01"].copy()
    df_fit["final_weight"] = df_fit.apply(
        lambda r: final_weight(r["date"].date(), r["competition_norm"]), axis=1
    )
    dc_params = dixon_coles.fit(df_fit, weight_col="final_weight")
    joblib.dump(dc_params, MODELS_DIR / "dixon_coles_params.pkl")
    print(f"Dixon-Coles fitted — rho={dc_params['rho']:.4f}, home_adv={dc_params['home_adv']:.4f}")

    print("── Step 5: Training XGBoost model ──")
    # Drop rows with missing features
    feature_cols = xgboost_model.FEATURE_COLS + ["result", "date"]
    clean = features[feature_cols].dropna()
    model = xgboost_model.train(clean)
    xgboost_model.save(model)
    print(f"XGBoost trained on {len(clean):,} matches")

    print("── Step 6: Saving current Elo ratings ──")
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
    """Quick simulation demo using current Elo ratings."""
    import pandas as pd
    from src.simulation.monte_carlo import run as mc_run
    from src.config import OUTPUTS_DIR

    print(f"Running Monte Carlo simulation ({n:,} iterations)...")

    dc_params = joblib.load(MODELS_DIR / "dixon_coles_params.pkl")
    ratings_df = pd.read_csv(DATA_PROCESSED / "elo_ratings.csv")
    elo_ratings = dict(zip(ratings_df["team"], ratings_df["elo"]))

    # 2026 WC groups — hardcoded from official draw
    groups = {
        "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
        "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
        "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
        "D": ["United States", "Paraguay", "Australia", "Turkey"],
        "E": ["Germany", "Ivory Coast", "Ecuador", "Curacao"],
        "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
        "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
        "H": ["Spain", "Saudi Arabia", "Uruguay", "Cape Verde Islands"],
        "I": ["France", "Senegal", "Iraq", "Norway"],
        "J": ["Argentina", "Algeria", "Austria", "Jordan"],
        "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
        "L": ["England", "Croatia", "Ghana", "Panama"],
    }

    # Build lambda lookup from Dixon-Coles params
    lambdas = {}
    all_teams = [t for g in groups.values() for t in g]
    for ta in all_teams:
        for tb in all_teams:
            if ta != tb:
                import numpy as np
                lam_a = np.exp(
                    dc_params["attack"].get(ta, 0) +
                    dc_params["defence"].get(tb, 0)
                )
                lam_b = np.exp(
                    dc_params["attack"].get(tb, 0) +
                    dc_params["defence"].get(ta, 0)
                )
                lambdas[(ta, tb)] = (lam_a, lam_b)

    results = mc_run(groups, lambdas, elo_ratings, dc_params["rho"],
                     n_simulations=n, verbose=True)

    out = pd.DataFrame([
        {"team": team, **probs}
        for team, probs in results.items()
    ]).sort_values("champion", ascending=False)

    out_path = OUTPUTS_DIR / "tournament_probabilities" / "latest.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(out.head(12).to_string(index=False))