# ============================================================
#  Dixon-Coles Poisson model with rho low-score correction
#  Fixed: use time-decayed weights so all data contributes
#         but recent matches dominate — solves convergence
#         by making old obscure matches near-zero weight
# ============================================================
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from src.config import MAX_GOALS


def rho_correction(x: int, y: int, lam_a: float, lam_b: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1 - lam_a * lam_b * rho
    elif x == 1 and y == 0:
        return 1 + lam_b * rho
    elif x == 0 and y == 1:
        return 1 + lam_a * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def score_prob(ga: int, gb: int, lam_a: float, lam_b: float, rho: float) -> float:
    p = (poisson.pmf(ga, lam_a)
         * poisson.pmf(gb, lam_b)
         * rho_correction(ga, gb, lam_a, lam_b, rho))
    return max(float(p), 0.0)


def score_matrix(lam_a: float, lam_b: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    m = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            m[i, j] = score_prob(i, j, lam_a, lam_b, rho)
    return m


def match_probs(matrix: np.ndarray) -> dict:
    win_a = float(np.sum(np.tril(matrix, -1)))
    draw  = float(np.sum(np.diag(matrix)))
    win_b = float(np.sum(np.triu(matrix, 1)))
    total = win_a + draw + win_b
    return {
        "win_a": win_a / total,
        "draw":  draw  / total,
        "win_b": win_b / total,
    }


def neg_log_likelihood(params, home_teams, away_teams, home_goals,
                       away_goals, weights, teams):
    """Vectorized negative log-likelihood."""
    n        = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    attack   = params[:n]
    defence  = params[n:2*n]
    home_adv = params[2*n]
    rho      = params[2*n + 1]

    hi = np.array([team_idx[t] for t in home_teams])
    ai = np.array([team_idx[t] for t in away_teams])

    lam_h = np.exp(attack[hi] + defence[ai] + home_adv)
    lam_a = np.exp(attack[ai] + defence[hi])

    hg = np.array(home_goals)
    ag = np.array(away_goals)
    w  = np.array(weights)

    from scipy.special import gammaln
    def poisson_pmf(k, lam):
        return np.exp(
            k * np.log(np.clip(lam, 1e-10, None)) - lam - gammaln(k + 1)
        )

    p_h = poisson_pmf(hg, lam_h)
    p_a = poisson_pmf(ag, lam_a)

    rho_corr = np.ones(len(hg))
    m00 = (hg == 0) & (ag == 0)
    m10 = (hg == 1) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m11 = (hg == 1) & (ag == 1)
    rho_corr[m00] = 1 - lam_h[m00] * lam_a[m00] * rho
    rho_corr[m10] = 1 + lam_a[m10] * rho
    rho_corr[m01] = 1 + lam_h[m01] * rho
    rho_corr[m11] = 1 - rho

    p  = p_h * p_a * np.clip(rho_corr, 1e-10, None)
    ll = np.sum(w * np.log(np.clip(p, 1e-10, None)))
    return -ll


def fit(df, weight_col: str = "final_weight"):
    """
    Fit Dixon-Coles using ALL data with time-decayed weights.

    Key insight: with λ=0.15 decay, a match from 2000 has weight ~0.10
    and a match from 1990 has weight ~0.02. This means obscure old teams
    have near-zero effective influence on the optimization, allowing
    convergence without hard filtering.

    We still apply a soft filter: teams with effective weight sum < 0.5
    (equivalent to <1 recent match) are excluded to keep parameters finite.
    """
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Soft filter: exclude teams whose total weighted matches < threshold
    # This naturally keeps only teams with meaningful recent history
    home_w = df.groupby("home_team")[weight_col].sum()
    away_w = df.groupby("away_team")[weight_col].sum()
    total_w = home_w.add(away_w, fill_value=0)

    # Threshold = 1.0 effective match weight
    # With λ=0.15: a 2020 WC qualifier has weight ~0.67
    # So threshold=1.0 means ~2 recent matches minimum
    WEIGHT_THRESHOLD = 1.0
    active_teams = set(total_w[total_w >= WEIGHT_THRESHOLD].index)

    df = df[
        df["home_team"].isin(active_teams) &
        df["away_team"].isin(active_teams)
    ].copy()

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    n     = len(teams)

    print(f"  Fitting Dixon-Coles on {len(df):,} matches, {n} teams...")
    print(f"  (Using time-decayed weights — old matches have near-zero influence)")

    x0          = np.zeros(2 * n + 2)
    x0[2*n]     =  0.1
    x0[2*n + 1] = -0.1

    bounds = (
        [(-3, 3)] * n +
        [(-3, 3)] * n +
        [(0, 1)]  +
        [(-1, 0)]
    )

    # Two-phase optimization:
    # Phase 1 — optimize attack/defence with fixed home_adv and rho
    # This is a simpler landscape (2n params instead of 2n+2)
    def nll_phase1(ad_params):
        full = np.concatenate([ad_params, [0.1, -0.1]])
        return neg_log_likelihood(
            full, df["home_team"].tolist(), df["away_team"].tolist(),
            df["home_score"].tolist(), df["away_score"].tolist(),
            df[weight_col].tolist(), teams
        )

    print("  Phase 1: optimizing attack/defence parameters...")
    r1 = minimize(
        nll_phase1,
        x0[:2*n],
        method="L-BFGS-B",
        bounds=[(-3, 3)] * (2 * n),
        options={"maxiter": 500, "ftol": 1e-6, "gtol": 1e-5},
    )

    # Phase 2 — optimize all params with warm start from phase 1
    print("  Phase 2: fine-tuning all parameters...")
    x0_warm = np.concatenate([r1.x, [0.1, -0.1]])
    result = minimize(
        neg_log_likelihood,
        x0_warm,
        args=(
            df["home_team"].tolist(),
            df["away_team"].tolist(),
            df["home_score"].tolist(),
            df["away_score"].tolist(),
            df[weight_col].tolist(),
            teams,
        ),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-7, "gtol": 1e-6},
    )

    params = result.x
    print(f"  Converged: {result.success} — {result.message}")
    print(f"  Teams fitted: {n}  |  Matches used: {len(df):,}")

    return {
        "teams":    teams,
        "attack":   dict(zip(teams, params[:n])),
        "defence":  dict(zip(teams, params[n:2*n])),
        "home_adv": float(params[2*n]),
        "rho":      float(params[2*n + 1]),
    }


def predict(team_a: str, team_b: str, params: dict, neutral: bool = True) -> dict:
    """Predict match outcome using fitted Dixon-Coles params."""
    home_adv = 0.0 if neutral else params["home_adv"]
    lam_a = np.exp(
        params["attack"].get(team_a, 0) +
        params["defence"].get(team_b, 0) +
        home_adv
    )
    lam_b = np.exp(
        params["attack"].get(team_b, 0) +
        params["defence"].get(team_a, 0)
    )
    matrix = score_matrix(lam_a, lam_b, params["rho"])
    probs  = match_probs(matrix)
    probs.update({"lambda_a": round(lam_a, 4), "lambda_b": round(lam_b, 4)})
    return probs