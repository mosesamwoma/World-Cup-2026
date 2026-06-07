# ============================================================
#  Single match simulator
# ============================================================
import numpy as np
from src.models.dixon_coles import score_matrix, match_probs
from src.config import MAX_GOALS


def simulate_match(
    team_a: str,
    team_b: str,
    lam_a: float,
    lam_b: float,
    rho: float,
) -> tuple[int, int]:
    """Sample a scoreline from the Dixon-Coles probability matrix."""
    matrix = score_matrix(lam_a, lam_b, rho, MAX_GOALS)
    flat   = matrix.flatten()
    flat  /= flat.sum()          # renormalise
    idx    = np.random.choice(len(flat), p=flat)
    ga, gb = divmod(idx, MAX_GOALS + 1)
    return int(ga), int(gb)


def simulate_penalties(elo_a: float, elo_b: float) -> str:
    """
    Penalty shootout: win probability dampened by elo diff.
    Returns "a" or "b".
    """
    p_a = 1 / (1 + 10 ** (-(elo_a - elo_b) / 800))
    return "a" if np.random.random() < p_a else "b"


def simulate_knockout(
    team_a: str,
    team_b: str,
    lam_a: float,
    lam_b: float,
    rho: float,
    elo_a: float,
    elo_b: float,
) -> str:
    """
    Simulate a knockout match.
    If draw after 90min → penalty shootout (extra time modelled as 50/50 extension).
    Returns winning team name.
    """
    ga, gb = simulate_match(team_a, team_b, lam_a, lam_b, rho)
    if ga > gb:
        return team_a
    elif gb > ga:
        return team_b
    else:
        winner_side = simulate_penalties(elo_a, elo_b)
        return team_a if winner_side == "a" else team_b
