import numpy as np
from src.models.dixon_coles import score_matrix, match_probs, rho_correction

def test_matrix_sums_to_one():
    m = score_matrix(1.5, 1.0, -0.1)
    assert abs(m.sum() - 1.0) < 0.01

def test_rho_correction_00():
    val = rho_correction(0, 0, 1.5, 1.0, -0.1)
    assert val != 1.0   # correction applied

def test_match_probs_sum():
    m = score_matrix(1.8, 0.9, -0.1)
    p = match_probs(m)
    assert abs(p["win_a"] + p["draw"] + p["win_b"] - 1.0) < 1e-6
