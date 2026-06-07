# ============================================================
#  Group stage simulator + FIFA tiebreakers
# ============================================================
import itertools
import numpy as np
from src.simulation.match_engine import simulate_match


def simulate_group(teams: list[str], lambdas: dict, rho: float) -> list[dict]:
    """
    Simulate a single group (round-robin).
    lambdas: {team: {"att": float, "def": float}} or precomputed lam dict
    Returns sorted standings list.
    """
    stats = {t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0, "fp": 0, "results": {}} for t in teams}
    h2h   = {t: {o: {"pts": 0, "gf": 0, "ga": 0} for o in teams if o != t} for t in teams}

    for team_a, team_b in itertools.combinations(teams, 2):
        lam_a = lambdas.get((team_a, team_b), (1.2, 1.0))[0]
        lam_b = lambdas.get((team_a, team_b), (1.2, 1.0))[1]
        ga, gb = simulate_match(team_a, team_b, lam_a, lam_b, rho)

        # Update overall stats
        stats[team_a]["gf"] += ga; stats[team_a]["ga"] += gb
        stats[team_b]["gf"] += gb; stats[team_b]["ga"] += ga
        stats[team_a]["gd"]  = stats[team_a]["gf"] - stats[team_a]["ga"]
        stats[team_b]["gd"]  = stats[team_b]["gf"] - stats[team_b]["ga"]

        if ga > gb:
            stats[team_a]["pts"] += 3
            h2h[team_a][team_b]["pts"] += 3
        elif ga == gb:
            stats[team_a]["pts"] += 1
            stats[team_b]["pts"] += 1
            h2h[team_a][team_b]["pts"] += 1
            h2h[team_b][team_a]["pts"] += 1
        else:
            stats[team_b]["pts"] += 3
            h2h[team_b][team_a]["pts"] += 3

        h2h[team_a][team_b]["gf"] += ga; h2h[team_a][team_b]["ga"] += gb
        h2h[team_b][team_a]["gf"] += gb; h2h[team_b][team_a]["ga"] += ga

    return _sort_standings(stats, h2h, teams)


def _sort_standings(stats: dict, h2h: dict, teams: list) -> list[dict]:
    """Apply FIFA tiebreaker order."""
    def tiebreaker_key(team):
        h2h_pts = sum(h2h[team][o]["pts"] for o in teams if o != team)
        h2h_gd  = sum(h2h[team][o]["gf"] - h2h[team][o]["ga"] for o in teams if o != team)
        h2h_gf  = sum(h2h[team][o]["gf"] for o in teams if o != team)
        return (
            stats[team]["pts"],
            h2h_pts,
            h2h_gd,
            h2h_gf,
            stats[team]["gd"],
            stats[team]["gf"],
            -stats[team]["fp"],    # fewer cards is better
        )

    sorted_teams = sorted(teams, key=tiebreaker_key, reverse=True)
    return [{"team": t, "pos": i + 1, **stats[t]} for i, t in enumerate(sorted_teams)]
