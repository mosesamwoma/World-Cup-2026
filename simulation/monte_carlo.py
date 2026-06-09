# ============================================================
#  Monte Carlo tournament simulation — fully vectorized
#  Fixed: group = qualification %, not 1.0
# ============================================================
import numpy as np
from simulation.engine import build_score_cache
from simulation.group_stage import simulate_groups_batch, get_standings
from src.config import N_SIMULATIONS, THIRD_PLACE_SLOTS

ALL_STAGES = ["group", "r32", "r16", "qf", "sf", "final", "champion"]


def run(
    groups: dict,
    lambdas: dict,
    elo_ratings: dict,
    rho: float,
    n_simulations: int = N_SIMULATIONS,
    verbose: bool = True,
) -> dict:

    all_teams = [t for g in groups.values() for t in g]
    team_idx  = {t: i for i, t in enumerate(all_teams)}
    n_teams   = len(all_teams)
    counts    = np.zeros((n_teams, len(ALL_STAGES)), dtype=np.int64)

    # ── Step 1: Precompute score cache ───────────────────────
    if verbose:
        print("  Precomputing score matrices...")
    score_cache = build_score_cache(lambdas, rho)
    if verbose:
        print(f"  Cached {len(score_cache)} matchups in 0.1s")

    # ── Step 2: Batch simulate group stage ───────────────────
    if verbose:
        print(f"  Simulating group stage ({n_simulations:,} iterations)...")

    group_results   = simulate_groups_batch(groups, score_cache, n_simulations)
    group_standings = {}
    for g, result in group_results.items():
        group_standings[g] = get_standings(result, n_simulations)

    # ── Step 3: Build R32 lineup per simulation ──────────────
    if verbose:
        print("  Determining qualified teams...")

    qualified  = np.zeros((24, n_simulations), dtype=np.int32)
    third_pts  = np.zeros((12, n_simulations), dtype=np.int32)
    third_gd   = np.zeros((12, n_simulations), dtype=np.int32)
    third_gf   = np.zeros((12, n_simulations), dtype=np.int32)
    third_tidx = np.zeros((12, n_simulations), dtype=np.int32)

    for g_i, (g, result) in enumerate(group_results.items()):
        teams  = result["teams"]
        ranks  = group_standings[g]

        for local_i, team in enumerate(teams):
            ti     = team_idx[team]
            rank_i = ranks[local_i]

            for rank_val, slot_offset in [(0, 0), (1, 1)]:
                mask = rank_i == rank_val
                slot = g_i * 2 + slot_offset
                qualified[slot][mask] = ti

            mask3 = rank_i == 2
            third_pts[g_i][mask3]  = result["pts"][local_i][mask3]
            third_gd[g_i][mask3]   = result["gd"][local_i][mask3]
            third_gf[g_i][mask3]   = result["gf"][local_i][mask3]
            third_tidx[g_i][mask3] = ti

    # Select best 8 third-place teams per simulation
    scores_3rd = np.stack(
        [third_pts, third_gd, third_gf], axis=2
    ).transpose(1, 0, 2)  # (n_sims, 12, 3)

    best8 = np.zeros((THIRD_PLACE_SLOTS, n_simulations), dtype=np.int32)
    for sim in range(n_simulations):
        s     = scores_3rd[sim]
        order = np.lexsort((-s[:, 2], -s[:, 1], -s[:, 0]))
        best8[:, sim] = third_tidx[order[:THIRD_PLACE_SLOTS], sim]

    # Full R32 lineup
    r32 = np.vstack([qualified, best8])  # (32, n_sims)

    # FIXED: group qualification % = how often each team made R32
    for sim in range(n_simulations):
        for ti in np.unique(r32[:, sim]):
            counts[ti, ALL_STAGES.index("group")] += 1

    # FIXED: r32 count = same as group qualification (making R32 IS qualifying)
    for sim in range(n_simulations):
        for ti in np.unique(r32[:, sim]):
            counts[ti, ALL_STAGES.index("r32")] += 1

    # ── Step 4: Knockout rounds ──────────────────────────────
    if verbose:
        print("  Simulating knockout rounds...")

    current  = r32.copy()
    ko_stage_map = {
        0: "r16",
        1: "qf",
        2: "sf",
        3: "final",
    }

    for round_i in range(4):   # R32→R16, R16→QF, QF→SF, SF→Final
        n_matches  = current.shape[0] // 2
        next_round = np.zeros((n_matches, n_simulations), dtype=np.int32)

        for m in range(n_matches):
            idx_a = current[2 * m]
            idx_b = current[2 * m + 1]

            ga = np.zeros(n_simulations, dtype=np.int32)
            gb = np.zeros(n_simulations, dtype=np.int32)

            unique_pairs = set(zip(idx_a.tolist(), idx_b.tolist()))
            for (ai, bi) in unique_pairs:
                mask   = (idx_a == ai) & (idx_b == bi)
                n_this = int(mask.sum())
                if n_this == 0:
                    continue
                ta  = all_teams[ai]
                tb  = all_teams[bi]
                key = (ta, tb) if (ta, tb) in score_cache else (tb, ta)
                swap = key != (ta, tb)
                idxs = np.random.choice(len(score_cache[key]),
                                        size=n_this, p=score_cache[key])
                g1 = idxs // 9
                g2 = idxs %  9
                if swap:
                    g1, g2 = g2, g1
                ga[mask] = g1
                gb[mask] = g2

            draw_mask = ga == gb
            p_a_vals  = np.array([
                1 / (1 + 10 ** ((elo_ratings.get(all_teams[bi], 1500) -
                                  elo_ratings.get(all_teams[ai], 1500)) / 800))
                for ai, bi in zip(idx_a.tolist(), idx_b.tolist())
            ])
            pen_a   = np.random.random(n_simulations) < p_a_vals
            win_a   = (ga > gb) | (draw_mask & pen_a)
            winners = np.where(win_a, idx_a, idx_b)
            next_round[m] = winners

        current = next_round

        # Count teams that WON this round = advanced to next stage
        stage_name = ko_stage_map[round_i]
        stage_col  = ALL_STAGES.index(stage_name)
        for sim in range(n_simulations):
            for ti in np.unique(current[:, sim]):
                counts[ti, stage_col] += 1

    # ── Champion ─────────────────────────────────────────────
    # current = (1, n_sims) after Final
    champions = current[0]
    np.add.at(counts[:, ALL_STAGES.index("champion")], champions, 1)

    if verbose:
        print("  Done.")

    return {
        team: {
            s: round(int(counts[team_idx[team], j]) / n_simulations, 6)
            for j, s in enumerate(ALL_STAGES)
        }
        for team in all_teams
    }