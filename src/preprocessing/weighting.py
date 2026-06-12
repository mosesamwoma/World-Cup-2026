# ============================================================
#  Temporal decay + competition importance weighting
#  Fixed: uses LAMBDA_DECAY from config (now 0.15)
# ============================================================
import numpy as np
from datetime import date
from src.config import LAMBDA_DECAY, COMPETITION_WEIGHTS


def time_weight(match_date: date, reference_date: date = None) -> float:
    if reference_date is None:
        reference_date = date.today()
    years_ago = (reference_date - match_date).days / 365.25
    return float(np.exp(-LAMBDA_DECAY * years_ago))


def competition_weight(competition: str) -> float:
    return COMPETITION_WEIGHTS.get(competition, 1.0)


def final_weight(match_date: date, competition: str,
                 reference_date: date = None) -> float:
    return time_weight(match_date, reference_date) * competition_weight(competition)