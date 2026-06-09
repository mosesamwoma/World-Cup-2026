# ============================================================
#  Vectorized group stage simulator
# ============================================================
import itertools
import numpy as np
from simulation.engine import batch_sample


def simulate_groups_batch(
    groups: dict,
    score_cache: dict,
    n_sims: int,
) -> dict:
    """
    Simulate all group stage matches for ALL n_sims at once.

    Returns per-team per-simulation standings:
    {
      "A": {
        "teams": [t1,t2,t3,t4],
        "pts":   np.ndarray (4, n_sims),
        "gf":    np.ndarray (4, n_sims),
        "ga":    np.ndarray (4, n_sims),
        "gd":    np.ndarray (4, n_sims),
      }, ...
    }
    """
    group_results = {}

    for group_name, teams in groups.items():
        n = len(teams)
        pts = np.zeros((n, n_sims), dtype=np.int32)
        gf  = np.zeros((n, n_sims), dtype=np.int32)
        ga  = np.zeros((n, n_sims), dtype=np.int32)

        idx = {t: i for i, t in enumerate(teams)}

        for ta, tb in itertools.combinations(teams, 2):
            flat        = score_cache[(ta, tb)]
            goals_a, goals_b = batch_sample(flat, n_sims)

            i, j = idx[ta], idx[tb]
            gf[i] += goals_a;  ga[i] += goals_b
            gf[j] += goals_b;  ga[j] += goals_a

            home_win = goals_a > goals_b
            draw     = goals_a == goals_b
            away_win = goals_a < goals_b

            pts[i] += 3 * home_win + draw
            pts[j] += 3 * away_win + draw

        group_results[group_name] = {
            "teams": teams,
            "pts":   pts,
            "gf":    gf,
            "ga":    ga,
            "gd":    gf - ga,
        }

    return group_results


def get_standings(group_result: dict, n_sims: int) -> np.ndarray:
    """
    Sort teams by pts→gd→gf for each simulation.
    Returns rank array shape (n_teams, n_sims) where value = rank (0=first).
    """
    n = len(group_result["teams"])
    pts = group_result["pts"]   # (n, n_sims)
    gd  = group_result["gd"]
    gf  = group_result["gf"]

    # Stack into (n_sims, n, 3) then argsort
    scores = np.stack([pts, gd, gf], axis=2)  # (n, n_sims, 3)
    scores = scores.transpose(1, 0, 2)         # (n_sims, n, 3)

    # Lexsort descending: negate for descending
    ranks = np.zeros((n_sims, n), dtype=np.int8)
    for sim in range(n_sims):
        order = np.lexsort((-scores[sim, :, 2],
                            -scores[sim, :, 1],
                            -scores[sim, :, 0]))
        ranks[sim, order] = np.arange(n)

    return ranks.T  # (n_teams, n_sims)