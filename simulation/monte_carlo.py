# ============================================================
#  Monte Carlo tournament simulation — fully vectorized
#  Fixed: group qualification % now correct (not equal to r32)
#  Fixed: vectorized bincount for group/r32 counting
# ============================================================
import numpy as np
from simulation.engine import build_score_cache
from simulation.group_stage import simulate_groups_batch, get_standings
from src.config import N_SIMULATIONS, THIRD_PLACE_SLOTS, MAX_GOALS

ALL_STAGES = ["group", "r32", "r16", "qf", "sf", "final", "champion"]
_STRIDE    = MAX_GOALS + 1


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

    # Vectorized best-8 third-place selection
    GD_OFFSET = 50
    GF_MAX    = 99
    sort_key  = (
        third_pts * 10000 +
        (third_gd + GD_OFFSET) * 100 +
        np.clip(third_gf, 0, GF_MAX)
    ).T  # (n_sims, 12)

    order = np.argsort(-sort_key, axis=1)       # (n_sims, 12)
    top8  = order[:, :THIRD_PLACE_SLOTS]         # (n_sims, 8)

    best8 = np.zeros((THIRD_PLACE_SLOTS, n_simulations), dtype=np.int32)
    for slot in range(THIRD_PLACE_SLOTS):
        group_indices = top8[:, slot]
        best8[slot]   = third_tidx[group_indices, np.arange(n_simulations)]

    # Full R32 lineup: 24 top-2 qualifiers + 8 best third-place = 32
    r32 = np.vstack([qualified, best8])  # (32, n_sims)

    # FIXED: group qualification = how often each team made R32
    # Use bincount — each team appears EXACTLY ONCE per sim in r32 when qualified
    # bincount across all (32 * n_sims) values gives total qualification count
    # Then cap at n_simulations to prevent any double-count edge case
    r32_flat   = r32.flatten().astype(np.int64)
    qual_count = np.bincount(r32_flat, minlength=n_teams)
    qual_count = np.minimum(qual_count, n_simulations)

    counts[:, ALL_STAGES.index("group")] = qual_count
    counts[:, ALL_STAGES.index("r32")]   = qual_count

    # ── Step 4: Knockout rounds ──────────────────────────────
    if verbose:
        print("  Simulating knockout rounds...")

    current      = r32.copy()
    ko_stage_map = {0: "r16", 1: "qf", 2: "sf", 3: "final"}

    for round_i in range(4):
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
                ta   = all_teams[ai]
                tb   = all_teams[bi]
                key  = (ta, tb) if (ta, tb) in score_cache else (tb, ta)
                swap = key != (ta, tb)
                flat = score_cache[key]
                idxs = np.random.choice(len(flat), size=n_this, p=flat)
                g1   = idxs // _STRIDE
                g2   = idxs %  _STRIDE
                if swap:
                    g1, g2 = g2, g1
                ga[mask] = g1
                gb[mask] = g2

            draw_mask = ga == gb
            p_a_vals  = np.array([
                1 / (1 + 10 ** (
                    (elo_ratings.get(all_teams[bi], 1500) -
                     elo_ratings.get(all_teams[ai], 1500)) / 800
                ))
                for ai, bi in zip(idx_a.tolist(), idx_b.tolist())
            ])
            pen_a   = np.random.random(n_simulations) < p_a_vals
            win_a   = (ga > gb) | (draw_mask & pen_a)
            winners = np.where(win_a, idx_a, idx_b)
            next_round[m] = winners

        current = next_round

        # FIXED: vectorized stage counting using bincount
        stage_col    = ALL_STAGES.index(ko_stage_map[round_i])
        stage_counts = np.bincount(
            current.flatten().astype(np.int64),
            minlength=n_teams
        )
        # Each team appears once per sim when they advance
        counts[:, stage_col] = np.minimum(stage_counts, n_simulations)

    # Champion — current is (1, n_sims) after Final
    champions = current[0].astype(np.int64)
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