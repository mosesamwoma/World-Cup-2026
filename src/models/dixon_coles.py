# ============================================================
#  Dixon-Coles Poisson model with rho low-score correction
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
    """Full (max_goals+1) × (max_goals+1) score probability matrix."""
    m = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            m[i, j] = score_prob(i, j, lam_a, lam_b, rho)
    return m


def match_probs(matrix: np.ndarray) -> dict:
    """Aggregate score matrix → win/draw/loss probabilities."""
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
    """Negative log-likelihood for Dixon-Coles parameter fitting."""
    n = len(teams)
    attack  = dict(zip(teams, params[:n]))
    defence = dict(zip(teams, params[n:2*n]))
    home_adv = params[2*n]
    rho      = params[2*n + 1]

    ll = 0.0
    for i in range(len(home_teams)):
        ht, at = home_teams[i], away_teams[i]
        lam_h = np.exp(attack[ht] + defence[at] + home_adv)
        lam_a = np.exp(attack[at] + defence[ht])
        p = (score_prob(home_goals[i], away_goals[i], lam_h, lam_a, rho)
             * weights[i])
        if p > 0:
            ll += np.log(p)
    return -ll


def fit(df, weight_col: str = "final_weight"):
    """
    Fit Dixon-Coles model to match dataframe.
    df needs: home_team, away_team, home_score, away_score, <weight_col>
    Returns dict of fitted params.
    """
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    n = len(teams)

    x0 = np.zeros(2 * n + 2)   # attack, defence, home_adv, rho
    x0[2*n] = 0.1               # home advantage init
    x0[2*n+1] = -0.1            # rho init (negative for low-score correction)

    bounds = (
        [(-3, 3)] * n +         # attack
        [(-3, 3)] * n +         # defence
        [(0, 1)] +              # home advantage
        [(-1, 0)]               # rho (must be negative for valid correction)
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
    )

    params = result.x
    return {
        "teams":     teams,
        "attack":    dict(zip(teams, params[:n])),
        "defence":   dict(zip(teams, params[n:2*n])),
        "home_adv":  params[2*n],
        "rho":       params[2*n+1],
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
