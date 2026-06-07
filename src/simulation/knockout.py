# ============================================================
#  Knockout bracket simulator (R32 → Final)
# ============================================================
from src.simulation.match_engine import simulate_knockout


ROUNDS = ["R32", "R16", "QF", "SF", "Final"]


def simulate_bracket(
    bracket: list[tuple[str, str]],
    lambdas: dict,
    elo_ratings: dict,
    rho: float,
    stage_tracker: dict,
    round_name: str,
) -> list[str]:
    """
    Simulate one full knockout round.
    bracket: list of (team_a, team_b) matchups
    Returns list of winners.
    """
    winners = []
    for team_a, team_b in bracket:
        lam_a = lambdas.get((team_a, team_b), (1.2, 1.0))[0]
        lam_b = lambdas.get((team_a, team_b), (1.2, 1.0))[1]
        winner = simulate_knockout(
            team_a, team_b,
            lam_a, lam_b, rho,
            elo_ratings.get(team_a, 1500),
            elo_ratings.get(team_b, 1500),
        )
        stage_tracker[winner].append(round_name)
        winners.append(winner)
    return winners


def simulate_full_knockout(
    r32_bracket: list[tuple[str, str]],
    lambdas: dict,
    elo_ratings: dict,
    rho: float,
) -> dict:
    """
    Run the complete knockout tree from R32 to champion.
    Returns {team: [rounds_reached]} dict.
    """
    tracker = {team: [] for pair in r32_bracket for team in pair}

    round_pairs = r32_bracket
    for round_name in ROUNDS:
        winners = simulate_bracket(round_pairs, lambdas, elo_ratings, tracker, round_name)
        round_pairs = [(winners[i], winners[i+1]) for i in range(0, len(winners), 2)]
        if len(round_pairs) == 0:
            break

    return tracker
