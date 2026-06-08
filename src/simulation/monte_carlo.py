# ============================================================
#  Monte Carlo tournament simulation engine
# ============================================================
import numpy as np
from tqdm import tqdm
from src.config import N_SIMULATIONS
from src.simulation.group_stage import simulate_group
from src.simulation.third_place import select_best_third
from src.simulation.knockout import simulate_full_knockout

ALL_STAGES = ["group", "r32", "r16", "qf", "sf", "final", "champion"]


def run(
    groups: dict,
    lambdas: dict,
    elo_ratings: dict,
    rho: float,
    n_simulations: int = N_SIMULATIONS,
    verbose: bool = True,
) -> dict:
    """
    Run full Monte Carlo simulation.

    groups:      {"A": [team1, team2, team3, team4], ...}
    lambdas:     {(team_a, team_b): (lam_a, lam_b)}
    elo_ratings: {team: float}
    rho:         Dixon-Coles rho parameter
    """
    all_teams = [t for group in groups.values() for t in group]
    counts = {team: {s: 0 for s in ALL_STAGES} for team in all_teams}

    iterator = tqdm(range(n_simulations), desc="Simulating") if verbose else range(n_simulations)

    for _ in iterator:
        third_place_teams = []
        qualified = []

        # ── Group stage ──────────────────────────────────────
        for group_name, teams in groups.items():
            standings = simulate_group(teams, lambdas, rho)
            for entry in standings:
                counts[entry["team"]]["group"] += 1
                if entry["pos"] <= 2:
                    qualified.append(entry["team"])
                elif entry["pos"] == 3:
                    third_place_teams.append({
                        "team":  entry["team"],
                        "group": group_name,
                        "pts":   entry["pts"],
                        "gd":    entry["gd"],
                        "gf":    entry["gf"],
                        "fp":    entry.get("fp", 0),
                    })

        # ── Best third-place ─────────────────────────────────
        best_third = [t["team"] for t in select_best_third(third_place_teams)]
        r32_teams  = qualified + best_third

        for team in r32_teams:
            counts[team]["r32"] += 1

        # ── R32 bracket ──────────────────────────────────────
        r32_bracket = [(r32_teams[i], r32_teams[i + 1]) for i in range(0, 32, 2)]

        # ── Knockout rounds ──────────────────────────────────
        ko_tracker, champion = simulate_full_knockout(r32_bracket, lambdas, elo_ratings, rho)

        stage_map = {"R32": "r32", "R16": "r16", "QF": "qf", "SF": "sf", "Final": "final"}
        for team, rounds in ko_tracker.items():
            for r in rounds:
                mapped = stage_map.get(r)
                if mapped:
                    counts[team][mapped] += 1

        if champion:
            counts[champion]["champion"] += 1

    return {
        team: {s: round(counts[team][s] / n_simulations, 6) for s in ALL_STAGES}
        for team in all_teams
    }