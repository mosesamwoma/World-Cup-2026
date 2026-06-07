# ============================================================
#  Ensemble forecast combiner
# ============================================================
from src.config import ENSEMBLE_WEIGHTS


def combine(
    elo_probs: dict,
    dc_probs: dict,
    xgb_probs: dict,
    market_probs: dict = None,
) -> dict:
    """
    Weighted ensemble of model outputs.
    Each input dict: {"win_a": float, "draw": float, "win_b": float}
    """
    w = dict(ENSEMBLE_WEIGHTS)

    if market_probs is None:
        extra = w.pop("market")
        w["elo"]         += extra / 2
        w["dixon_coles"] += extra / 2
        market_probs = {"win_a": 0.0, "draw": 0.0, "win_b": 0.0}

    result = {}
    for key in ("win_a", "draw", "win_b"):
        result[key] = (
            w["elo"]         * elo_probs[key] +
            w["dixon_coles"] * dc_probs[key] +
            w["xgboost"]     * xgb_probs[key] +
            w.get("market", 0) * market_probs[key]
        )

    # Normalise to sum to 1.0
    total = sum(result.values())
    return {k: round(v / total, 6) for k, v in result.items()}
