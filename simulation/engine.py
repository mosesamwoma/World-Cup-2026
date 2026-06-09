# ============================================================
#  Vectorized simulation engine
#  Precomputes all score matrices once, then batch-samples
#  100K tournament iterations in under 2 minutes
# ============================================================
import numpy as np
from scipy.special import gammaln

MAX_GOALS = 8
_GOALS = np.arange(MAX_GOALS + 1)


def _poisson_pmf(lam: float) -> np.ndarray:
    """Vectorized Poisson PMF for goals 0..MAX_GOALS."""
    lam = max(lam, 1e-10)
    return np.exp(_GOALS * np.log(lam) - lam - gammaln(_GOALS + 1))


def build_score_cache(lambdas: dict, rho: float) -> dict:
    """
    Precompute flattened score probability arrays for every matchup.
    Call ONCE before simulation — O(n_teams^2) not O(n_simulations).
    Returns {(team_a, team_b): np.ndarray of shape (81,)}
    """
    cache = {}
    for (ta, tb), (la, lb) in lambdas.items():
        pa = _poisson_pmf(la)
        pb = _poisson_pmf(lb)
        m  = np.outer(pa, pb)
        # Dixon-Coles rho correction (4 cells only)
        m[0, 0] *= max(1 - la * lb * rho, 0)
        m[1, 0] *= 1 + lb * rho
        m[0, 1] *= 1 + la * rho
        m[1, 1] *= 1 - rho
        np.clip(m, 0, None, out=m)
        flat = m.flatten()
        flat /= flat.sum()
        cache[(ta, tb)] = flat
    return cache


def batch_sample(flat: np.ndarray, n: int) -> tuple:
    """
    Sample n match outcomes from a precomputed flat score distribution.
    Returns (goals_a array, goals_b array) both shape (n,).
    """
    idxs   = np.random.choice(len(flat), size=n, p=flat)
    goals_a = idxs // (MAX_GOALS + 1)
    goals_b = idxs %  (MAX_GOALS + 1)
    return goals_a, goals_b


def elo_win_prob(elo_a: float, elo_b: float) -> float:
    """Win probability from Elo difference."""
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def penalty_win_prob(elo_a: float, elo_b: float) -> float:
    """Dampened win probability for penalty shootout."""
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 800))