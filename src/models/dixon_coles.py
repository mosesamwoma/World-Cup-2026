# ============================================================
#  Dixon-Coles Poisson model with rho low-score correction
#  Fixed: 20+ match filter + maxiter 1000 for convergence
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


def neg_log_likelihood(params, home_teams, away_teams, home_goals, away_goals, weights, teams):
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
        return np.exp(k * np.log(np.clip(lam, 1e-10, None)) - lam - gammaln(k + 1))

    p_h = poisson_pmf(hg, lam_h)
    p_a = poisson_pmf(ag, lam_a)

    rho_corr = np.ones(len(hg))
    rho_corr[(hg == 0) & (ag == 0)] = 1 - lam_h[(hg == 0) & (ag == 0)] * lam_a[(hg == 0) & (ag == 0)] * rho
    rho_corr[(hg == 1) & (ag == 0)] = 1 + lam_a[(hg == 1) & (ag == 0)] * rho
    rho_corr[(hg == 0) & (ag == 1)] = 1 + lam_h[(hg == 0) & (ag == 1)] * rho
    rho_corr[(hg == 1) & (ag == 1)] = 1 - rho

    p  = p_h * p_a * np.clip(rho_corr, 1e-10, None)
    ll = np.sum(w * np.log(np.clip(p, 1e-10, None)))
    return -ll


def fit(df, weight_col: str = "final_weight"):
    """
    Fit Dixon-Coles model.
    FIXED: filter to 20+ matches (was 10) — reduces to ~150 teams
           maxiter increased to 1000 for convergence
    """
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # FIXED: 20+ matches threshold — reduces parameter count further
    match_counts = (
        pd.concat([df["home_team"], df["away_team"]])
        .value_counts()
    )
    active_teams = set(match_counts[match_counts >= 20].index)
    df = df[
        df["home_team"].isin(active_teams) &
        df["away_team"].isin(active_teams)
    ].copy()

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    n     = len(teams)

    print(f"  Fitting Dixon-Coles on {len(df):,} matches, {n} teams...")

    x0          = np.zeros(2 * n + 2)
    x0[2*n]     =  0.1
    x0[2*n + 1] = -0.1

    bounds = (
        [(-3, 3)] * n +
        [(-3, 3)] * n +
        [(0, 1)]  +
        [(-1, 0)]
    )

    result = minimize(
        neg_log_likelihood,
        x0,
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
        # FIXED: maxiter 1000, relaxed tolerances
        options={"maxiter": 1000, "ftol": 1e-6, "gtol": 1e-5},
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