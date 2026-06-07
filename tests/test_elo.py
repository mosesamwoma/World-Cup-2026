import pytest
from src.ratings.elo import update_elo, expected_score

def test_expected_score_equal_teams():
    assert abs(expected_score(1500, 1500) - 0.5) < 1e-6

def test_elo_winner_gains():
    elo_a, elo_b = update_elo(1500, 1500, 2, 0, 1.0, True)
    assert elo_a > 1500
    assert elo_b < 1500

def test_elo_draw_even():
    elo_a, elo_b = update_elo(1500, 1500, 1, 1, 1.0, True)
    assert abs(elo_a - 1500) < 1e-6
    assert abs(elo_b - 1500) < 1e-6
