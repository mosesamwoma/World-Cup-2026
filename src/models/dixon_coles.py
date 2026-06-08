# ============================================================
#  Dixon-Coles Poisson model with rho low-score correction
#  Vectorized for speed
# ============================================================
import numpy as np
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
    """
    Vectorized negative log-likelihood — no Python loop over matches.
    """
    n = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    attack   = params[:n]
    defence  = params[n:2*n]
    home_adv = params[2*n]
    rho      = params[2*n + 1]

    # Vectorized lambda computation
    hi = np.array([team_idx[t] for t in home_teams])
    ai = np.array([team_idx[t] for t in away_teams])

    lam_h = np.exp(attack[hi] + defence[ai] + home_adv)
    lam_a = np.exp(attack[ai] + defence[hi])

    hg = np.array(home_goals)
    ag = np.array(away_goals)
    w  = np.array(weights)

    # Vectorized Poisson PMF
    from scipy.special import gammaln
    def poisson_pmf(k, lam):
        return np.exp(k * np.log(np.clip(lam, 1e-10, None)) - lam - gammaln(k + 1))

    p_h = poisson_pmf(hg, lam_h)
    p_a = poisson_pmf(ag, lam_a)

    # Rho correction — only affects scores 0-0, 1-0, 0-1, 1-1
    rho_corr = np.ones(len(hg))
    rho_corr[(hg == 0) & (ag == 0)] = 1 - lam_h[(hg == 0) & (ag == 0)] * lam_a[(hg == 0) & (ag == 0)] * rho
    rho_corr[(hg == 1) & (ag == 0)] = 1 + lam_a[(hg == 1) & (ag == 0)] * rho
    rho_corr[(hg == 0) & (ag == 1)] = 1 + lam_h[(hg == 0) & (ag == 1)] * rho
    rho_corr[(hg == 1) & (ag == 1)] = 1 - rho

    p = p_h * p_a * np.clip(rho_corr, 1e-10, None)
    ll = np.sum(w * np.log(np.clip(p, 1e-10, None)))
    return -ll


def fit(df, weight_col: str = "final_weight"):
    """
    Fit Dixon-Coles model to match dataframe.
    df needs: home_team, away_team, home_score, away_score, <weight_col>
    """
    # Filter to valid completed matches only
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    n = len(teams)

    print(f"  Fitting Dixon-Coles on {len(df):,} matches, {n} teams...")

    x0 = np.zeros(2 * n + 2)
    x0[2*n]     =  0.1    # home advantage
    x0[2*n + 1] = -0.1    # rho

    bounds = (
        [(-3, 3)] * n +   # attack
        [(-3, 3)] * n +   # defence
        [(0, 1)]  +       # home advantage
        [(-1, 0)]         # rho
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
        options={"maxiter": 200, "ftol": 1e-9},
    )

    params = result.x
    print(f"  Converged: {result.success} — {result.message}")

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
    lam_a = np.exp(params["attack"].get(team_a, 0)
                   + params["defence"].get(team_b, 0)
                   + home_adv)
    lam_b = np.exp(params["attack"].get(team_b, 0)
                   + params["defence"].get(team_a, 0))

    matrix = score_matrix(lam_a, lam_b, params["rho"])
    probs  = match_probs(matrix)
    probs.update({"lambda_a": round(lam_a, 4), "lambda_b": round(lam_b, 4)})
    return probs