# ============================================================
#  Ensemble forecast combiner — with finetunable weights
#  Fixed: L2 regularization + minimum 10% per model
# ============================================================
import numpy as np
import joblib
from src.config import ENSEMBLE_WEIGHTS, MODELS_DIR


def combine(
    elo_probs: dict,
    dc_probs: dict,
    xgb_probs: dict,
    market_probs: dict = None,
    weights: dict = None,
) -> dict:
    """
    Weighted ensemble of model outputs.
    Each input dict: {"win_a": float, "draw": float, "win_b": float}
    """
    w = dict(weights) if weights else dict(ENSEMBLE_WEIGHTS)

    if market_probs is None:
        extra = w.pop("market", 0)
        w["elo"]         = w.get("elo", 0)         + extra / 2
        w["dixon_coles"] = w.get("dixon_coles", 0) + extra / 2
        market_probs = {"win_a": 0.0, "draw": 0.0, "win_b": 0.0}

    result = {}
    for key in ("win_a", "draw", "win_b"):
        result[key] = (
            w["elo"]           * elo_probs[key] +
            w["dixon_coles"]   * dc_probs[key]  +
            w["xgboost"]       * xgb_probs[key] +
            w.get("market", 0) * market_probs[key]
        )

    total = sum(result.values())
    return {k: round(v / total, 6) for k, v in result.items()}


def train_weights(backtest_df, verbose: bool = True) -> dict:
    """
    Learn optimal ensemble weights by minimizing log loss.
    Fixed: L2 regularization prevents any single model dominating.
    Minimum 10% weight enforced per model.
    """
    from scipy.optimize import minimize
    from scipy.special import softmax

    df = backtest_df.dropna().copy()
    n  = len(df)

    if verbose:
        print(f"  Training ensemble weights on {n} backtest matches...")

    elo_p = df[["elo_win_a", "elo_draw", "elo_win_b"]].values
    dc_p  = df[["dc_win_a",  "dc_draw",  "dc_win_b"]].values
    xgb_p = df[["xgb_win_a", "xgb_draw", "xgb_win_b"]].values

    result_map = {1: 0, 0: 1, -1: 2}
    y        = np.array([result_map[r] for r in df["result"].values])
    y_onehot = np.eye(3)[y]

    def neg_log_loss(raw_weights):
        w = softmax(raw_weights)
        probs = w[0] * elo_p + w[1] * dc_p + w[2] * xgb_p
        probs = np.clip(probs, 1e-10, 1)
        probs /= probs.sum(axis=1, keepdims=True)
        ll = np.sum(y_onehot * np.log(probs)) / n
        # L2 regularization — pulls weights toward equal (1/3 each)
        # lambda=0.5 prevents any model collapsing to 0
        l2_penalty = 0.5 * np.sum((w - 1/3) ** 2)
        return -ll + l2_penalty

    # Start from equal weights → equal softmax inputs = zeros
    x0 = np.zeros(3)

    result = minimize(neg_log_loss, x0, method="L-BFGS-B")
    learned_w = softmax(result.x)

    # Enforce minimum 10% per model so nothing is ignored
    MIN_WEIGHT = 0.10
    learned_w  = np.clip(learned_w, MIN_WEIGHT, None)
    learned_w /= learned_w.sum()

    weights = {
        "elo":         round(float(learned_w[0]), 4),
        "dixon_coles": round(float(learned_w[1]), 4),
        "xgboost":     round(float(learned_w[2]), 4),
        "market":      0.0,
    }

    if verbose:
        print(f"  Learned weights:")
        print(f"    Elo:          {weights['elo']:.4f}")
        print(f"    Dixon-Coles:  {weights['dixon_coles']:.4f}")
        print(f"    XGBoost:      {weights['xgboost']:.4f}")
        print(f"  Log loss: {result.fun:.4f} (lower = better)")

    joblib.dump(weights, MODELS_DIR / "ensemble_weights.pkl")
    print(f"  Saved → models/ensemble_weights.pkl")
    return weights


def load_weights() -> dict:
    """Load learned weights if available, else fall back to config defaults."""
    path = MODELS_DIR / "ensemble_weights.pkl"
    if path.exists():
        return joblib.load(path)
    return dict(ENSEMBLE_WEIGHTS)