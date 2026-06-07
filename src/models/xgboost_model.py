# ============================================================
#  XGBoost match outcome classifier
# ============================================================
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from src.config import MODELS_DIR

FEATURE_COLS = [
    "elo_diff",
    "form5_ppg_home", "form5_gf_home", "form5_ga_home",
    "form5_ppg_away", "form5_gf_away", "form5_ga_away",
    "form10_ppg_home", "form10_gd_home",
    "form10_ppg_away", "form10_gd_away",
    "neutral", "is_host_home", "comp_weight",
]

# result labels: 1=home win, 0=draw, -1=away win → encode to 0,1,2
LABEL_MAP = {1: 2, 0: 1, -1: 0}   # home win=2, draw=1, away win=0


def get_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )


def train(df: pd.DataFrame) -> XGBClassifier:
    """
    Train XGBoost on feature matrix.
    df must have FEATURE_COLS + 'result' column.
    Uses temporal cross-validation — never random split.
    """
    X = df[FEATURE_COLS].values
    y = df["result"].map(LABEL_MAP).values

    model = get_model()
    model.fit(X, y)
    return model


def temporal_cv(df: pd.DataFrame, n_splits: int = 4) -> list[float]:
    """Time-series cross-validation scores (log loss)."""
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import log_loss

    df = df.sort_values("date").reset_index(drop=True)
    X  = df[FEATURE_COLS].values
    y  = df["result"].map(LABEL_MAP).values

    tscv   = TimeSeriesSplit(n_splits=n_splits)
    model  = get_model()
    scores = []

    for train_idx, test_idx in tscv.split(X):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict_proba(X[test_idx])
        scores.append(log_loss(y[test_idx], y_pred))

    return scores


def predict_proba(model: XGBClassifier, team_a_features: dict) -> dict:
    """
    Predict outcome probabilities for a single match.
    Returns {"win_a": float, "draw": float, "win_b": float}
    """
    X = np.array([[team_a_features[c] for c in FEATURE_COLS]])
    proba = model.predict_proba(X)[0]
    # proba order: [away win, draw, home win]
    return {"win_b": proba[0], "draw": proba[1], "win_a": proba[2]}


def save(model: XGBClassifier, path=None):
    path = path or MODELS_DIR / "xgboost_model.pkl"
    joblib.dump(model, path)
    print(f"Model saved → {path}")


def load(path=None) -> XGBClassifier:
    path = path or MODELS_DIR / "xgboost_model.pkl"
    return joblib.load(path)
