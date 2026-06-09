# ============================================================
#  Vectorized knockout bracket simulator
# ============================================================
import numpy as np
from simulation.engine import batch_sample, penalty_win_prob

ROUNDS = ["R32", "R16", "QF", "SF", "Final"]


def simulate_knockout_batch(
    bracket: list,
    score_cache: dict,
    elo_ratings: dict,
    n_sims: int,
) -> tuple:
    """
    Simulate one knockout round for all n_sims at once.

    bracket: list of (team_a, team_b) — same bracket for all sims
    Returns (winners list of length n_sims//2... )

    NOTE: For knockout, bracket changes per simulation so we
    simulate round by round but each round is fully vectorized.
    """
    # Group matchups that share the same pair
    # In knockout each position has a specific matchup per sim
    winners = np.empty(len(bracket), dtype=object)

    for match_idx, (ta, tb) in enumerate(bracket):
        key = (ta, tb)
        if key not in score_cache:
            key = (tb, ta)
            swap = True
        else:
            swap = False

        flat = score_cache[key]
        ga, gb = batch_sample(flat, n_sims)

        if swap:
            ga, gb = gb, ga

        # Determine winner per sim
        home_win  = ga > gb
        away_win  = gb > ga
        draw_mask = ga == gb

        # Penalties for draws
        p_a = penalty_win_prob(
            elo_ratings.get(ta, 1500),
            elo_ratings.get(tb, 1500)
        )
        penalty_win_a = np.random.random(n_sims) < p_a

        win_a = home_win | (draw_mask & penalty_win_a)
        # win_a[i] = True means ta wins simulation i
        winners[match_idx] = (ta, tb, win_a)

    return winners


def simulate_full_knockout_batch(
    r32_teams_per_sim: np.ndarray,
    all_teams: list,
    score_cache: dict,
    elo_ratings: dict,
    n_sims: int,
) -> tuple:
    """
    Simulate full knockout from R32 to champion for all simulations.

    r32_teams_per_sim: shape (32, n_sims) — team index per position per sim
    all_teams: list mapping index → team name

    Returns:
        stage_counts: {team: {stage: count}}
        champions: np.ndarray shape (n_sims,) — champion index per sim
    """
    stage_map  = {"R32": 0, "R16": 1, "QF": 2, "SF": 3, "Final": 4}
    n_teams    = len(all_teams)
    # stage_counts[team_idx, stage_idx] = count
    stage_counts = np.zeros((n_teams, 5), dtype=np.int32)

    current = r32_teams_per_sim.copy()  # (32, n_sims) initially

    for round_idx, round_name in enumerate(ROUNDS):
        n_matches = current.shape[0] // 2
        next_round = np.zeros((n_matches, n_sims), dtype=np.int32)

        for m in range(n_matches):
            idx_a = current[2 * m]       # (n_sims,) team indices
            idx_b = current[2 * m + 1]

            # Sample scores for each unique pair
            # Vectorized per simulation
            ga = np.zeros(n_sims, dtype=np.int32)
            gb = np.zeros(n_sims, dtype=np.int32)

            # Group sims by matchup for cache lookup
            unique_pairs = set(zip(idx_a.tolist(), idx_b.tolist()))
            for (ai, bi) in unique_pairs:
                ta = all_teams[ai]
                tb = all_teams[bi]
                mask = (idx_a == ai) & (idx_b == bi)
                n_this = mask.sum()
                if n_this == 0:
                    continue

                key = (ta, tb) if (ta, tb) in score_cache else (tb, ta)
                swap = key != (ta, tb)
                flat = score_cache[key]
                g1, g2 = batch_sample(flat, n_this)
                if swap:
                    g1, g2 = g2, g1
                ga[mask] = g1
                gb[mask] = g2

            draw_mask = ga == gb
            p_a = np.array([
                penalty_win_prob(
                    elo_ratings.get(all_teams[ai], 1500),
                    elo_ratings.get(all_teams[bi], 1500)
                )
                for ai, bi in zip(idx_a, idx_b)
            ])
            penalty_a = np.random.random(n_sims) < p_a
            win_a = (ga > gb) | (draw_mask & penalty_a)

            winners = np.where(win_a, idx_a, idx_b)
            next_round[m] = winners

            # Track stage reached
            np.add.at(stage_counts[:, round_idx], winners, 1)

        current = next_round

    champions = current[0]  # (n_sims,) — winner of the Final
    return stage_counts, champions