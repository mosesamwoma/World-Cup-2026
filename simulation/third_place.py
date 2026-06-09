# ============================================================
#  Best third-place team selection — vectorized
# ============================================================
import numpy as np
from src.config import THIRD_PLACE_SLOTS


def select_best_third_batch(
    third_place_data: dict,
    n_sims: int,
) -> np.ndarray:
    """
    Select best 8 third-place teams across all simulations at once.

    third_place_data: {group_name: {"teams": [...], "pts": (n,n_sims),
                                    "gd": ..., "gf": ...}}
                      Only the 3rd-place finisher (rank==2) per group.

    Returns np.ndarray shape (8, n_sims) — indices into all_third_teams list.
    """
    all_groups = sorted(third_place_data.keys())
    n_groups   = len(all_groups)   # 12

    # Shape: (12, n_sims) for each stat
    pts_matrix = np.zeros((n_groups, n_sims), dtype=np.int32)
    gd_matrix  = np.zeros((n_groups, n_sims), dtype=np.int32)
    gf_matrix  = np.zeros((n_groups, n_sims), dtype=np.int32)
    team_matrix = []   # list of (n_sims,) arrays of team indices

    for i, g in enumerate(all_groups):
        d = third_place_data[g]
        pts_matrix[i] = d["pts"]
        gd_matrix[i]  = d["gd"]
        gf_matrix[i]  = d["gf"]
        team_matrix.append(d["team_idx"])

    team_matrix = np.array(team_matrix)  # (12, n_sims)

    # For each sim, rank 12 third-place teams by pts→gd→gf, take top 8
    # Stack scores: (n_sims, 12, 3)
    scores = np.stack([pts_matrix, gd_matrix, gf_matrix], axis=2).transpose(1, 0, 2)

    best8 = np.zeros((THIRD_PLACE_SLOTS, n_sims), dtype=np.int32)
    for sim in range(n_sims):
        order = np.lexsort((-scores[sim, :, 2],
                            -scores[sim, :, 1],
                            -scores[sim, :, 0]))
        best8[:, sim] = team_matrix[order[:THIRD_PLACE_SLOTS], sim]

    return best8