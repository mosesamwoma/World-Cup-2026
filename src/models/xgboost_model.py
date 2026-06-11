# ============================================================
#  XGBoost match outcome classifier
#  Improved: more features, better hyperparameters
# ============================================================
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
from src.config import MODELS_DIR

FEATURE_COLS = [
    "elo_diff",
    "form5_ppg_home",  "form5_gf_home",  "form5_ga_home",
    "form5_gsr_home",  "form5_cs_home",
    "form5_ppg_away",  "form5_gf_away",  "form5_ga_away",
    "form5_gsr_away",  "form5_cs_away",
    "form10_ppg_home", "form10_gd_home", "form10_gsr_home",
    "form10_ppg_away", "form10_gd_away", "form10_gsr_away",
    "days_rest_home",  "days_rest_away",
    "neutral", "is_host_home", "comp_weight",
]

LABEL_MAP = {1: 2, 0: 1, -1: 0}


def get_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )


def train(df: pd.DataFrame) -> XGBClassifier:
    """Train on full feature matrix using temporal order."""
    df = df.sort_values("date").reset_index(drop=True)

    # IMPROVED: use only matches from 1990 onwards
    # Pre-1990 data is noisy and has fewer teams with reliable stats
    df = df[df["date"] >= "1990-01-01"].copy()

    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols].fillna(0).values
    y = df["result"].map(LABEL_MAP).values

    model = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def temporal_cv(df: pd.DataFrame, n_splits: int = 4) -> list:
    """Time-series cross-validation scores (log loss)."""
    df = df.sort_values("date").reset_index(drop=True)
    df = df[df["date"] >= "1990-01-01"].copy()
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X  = df[available_cols].fillna(0).values
    y  = df["result"].map(LABEL_MAP).values

    tscv   = TimeSeriesSplit(n_splits=n_splits)
    model  = get_model()
    scores = []

    for train_idx, test_idx in tscv.split(X):
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict_proba(X[test_idx])
        scores.append(log_loss(y[test_idx], y_pred))

    return scores


def predict_proba(model: XGBClassifier, features: dict) -> dict:
    """Predict outcome probabilities for a single match."""
    available_cols = [c for c in FEATURE_COLS if c in features]
    X = np.array([[features.get(c, 0) for c in FEATURE_COLS]])
    proba = model.predict_proba(X)[0]
    return {"win_b": float(proba[0]), "draw": float(proba[1]), "win_a": float(proba[2])}


def save(model: XGBClassifier, path=None):
    path = path or MODELS_DIR / "xgboost_model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved → {path}")


def load(path=None) -> XGBClassifier:
    path = path or MODELS_DIR / "xgboost_model.pkl"
    return joblib.load(path)