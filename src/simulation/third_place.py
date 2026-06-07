# ============================================================
#  2026 best third-place team selection logic
# ============================================================
from src.config import THIRD_PLACE_SLOTS


def select_best_third(third_place_teams: list[dict]) -> list[dict]:
    """
    Rank all 12 third-place finishers and return the best 8.
    Each dict must have: team, group, pts, gd, gf, fp (fair play points)
    Lower fp = fewer cards = better.
    """
    ranked = sorted(
        third_place_teams,
        key=lambda t: (t["pts"], t["gd"], t["gf"], -t["fp"]),
        reverse=True,
    )
    return ranked[:THIRD_PLACE_SLOTS]
