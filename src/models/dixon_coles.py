# ============================================================
#  Dixon-Coles Poisson model with rho low-score correction
#  Optimizer: coordinate descent — guaranteed convergence
#  Each team optimizes only 2 params (attack + defence)
#  Global params (home_adv, rho) optimized separately
#  Early stop when delta plateaus
# ============================================================
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.special import gammaln
from src.config import MAX_GOALS


def rho_correction(x: int, y: int, lam_a: float, lam_b: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1 - lam_a * lam_b * rho
    elif x == 1 and y == 0:
        return 1 + lam_b * rho
    elif x == 0 and y == 1:
        return 1 + lam_a * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def score_prob(ga: int, gb: int, lam_a: float, lam_b: float, rho: float) -> float:
    p = (poisson.pmf(ga, lam_a)
         * poisson.pmf(gb, lam_b)
         * rho_correction(ga, gb, lam_a, lam_b, rho))
    return max(float(p), 0.0)


def score_matrix(lam_a: float, lam_b: float, rho: float,
                 max_goals: int = MAX_GOALS) -> np.ndarray:
    m = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            m[i, j] = score_prob(i, j, lam_a, lam_b, rho)
    return m


def match_probs(matrix: np.ndarray) -> dict:
    win_a = float(np.sum(np.tril(matrix, -1)))
    draw  = float(np.sum(np.diag(matrix)))
    win_b = float(np.sum(np.triu(matrix, 1)))
    total = win_a + draw + win_b
    return {
        "win_a": win_a / total,
        "draw":  draw  / total,
        "win_b": win_b / total,
    }


def _pmf(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Vectorized Poisson PMF."""
    return np.exp(
        k * np.log(np.clip(lam, 1e-10, None)) - lam - gammaln(k + 1)
    )


def _rho_corr(hg: np.ndarray, ag: np.ndarray,
              lam_h: np.ndarray, lam_a: np.ndarray,
              rho: float) -> np.ndarray:
    """Vectorized rho correction for low scores."""
    rc  = np.ones(len(hg))
    m00 = (hg == 0) & (ag == 0)
    m10 = (hg == 1) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m11 = (hg == 1) & (ag == 1)
    rc[m00] = 1 - lam_h[m00] * lam_a[m00] * rho
    rc[m10] = 1 + lam_a[m10] * rho
    rc[m01] = 1 + lam_h[m01] * rho
    rc[m11] = 1 - rho
    return rc


def _global_nll(p2: np.ndarray, attack: np.ndarray, defence: np.ndarray,
                hi: np.ndarray, ai: np.ndarray,
                hg: np.ndarray, ag: np.ndarray,
                w: np.ndarray) -> float:
    """NLL for global params: home_adv and rho only."""
    home_adv, rho = float(p2[0]), float(p2[1])
    lam_h = np.exp(attack[hi] + defence[ai] + home_adv)
    lam_a = np.exp(attack[ai] + defence[hi])
    p     = _pmf(hg, lam_h) * _pmf(ag, lam_a)
    rc    = _rho_corr(hg, ag, lam_h, lam_a, rho)
    p     = p * np.clip(rc, 1e-10, None)
    return -float(np.sum(w * np.log(np.clip(p, 1e-10, None))))


def fit(df, weight_col: str = "final_weight"):
    """
    Fit Dixon-Coles using coordinate descent.

    Why this works when L-BFGS-B fails:
      - 270 teams = 542 parameters = too many for L-BFGS-B in one shot
      - Coordinate descent breaks it into sub-problems of 2 params each
      - Each team's attack+defence optimized using only that team's matches
      - home_adv and rho optimized globally (only 2 params)
      - Cycles until convergence — no evaluation limit possible

    Convergence: typically 20-25 cycles (~150-200s)
    Early stop: if delta improvement < 2% for 3 consecutive cycles
    """
    import time

    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Soft filter: exclude teams with near-zero weighted influence
    home_w  = df.groupby("home_team")[weight_col].sum()
    away_w  = df.groupby("away_team")[weight_col].sum()
    total_w = home_w.add(away_w, fill_value=0)
    active  = set(total_w[total_w >= 1.0].index)
    df      = df[
        df["home_team"].isin(active) &
        df["away_team"].isin(active)
    ].copy()

    teams    = sorted(set(df["home_team"]) | set(df["away_team"]))
    n        = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    print(f"  Fitting Dixon-Coles on {len(df):,} matches, {n} teams...")
    print(f"  Method: coordinate descent (guaranteed convergence)")

    # Precompute index arrays once
    hi = np.array([team_idx[t] for t in df["home_team"]])
    ai = np.array([team_idx[t] for t in df["away_team"]])
    hg = df["home_score"].values.astype(int)
    ag = df["away_score"].values.astype(int)
    w  = df[weight_col].values.astype(float)

    # Precompute per-team match masks — avoids repeated filtering in loop
    team_masks = [((hi == i) | (ai == i)) for i in range(n)]

    # Initialize parameters
    attack   = np.zeros(n)
    defence  = np.zeros(n)
    home_adv = 0.1
    rho      = -0.1

    MAX_CYCLES   = 30
    TOLERANCE    = 1e-4
    prev_delta   = float("inf")
    no_improve   = 0
    t0           = time.time()

    for cycle in range(MAX_CYCLES):
        prev_attack = attack.copy()

        # ── Step A: optimize each team's attack + defence ────
        # Each sub-problem has only 2 parameters → always converges fast
        for i in range(n):
            mask  = team_masks[i]
            hi_m  = hi[mask];  ai_m = ai[mask]
            hg_m  = hg[mask];  ag_m = ag[mask]
            w_m   = w[mask]

            # Capture loop variables explicitly to avoid closure issues
            def team_obj(p2, _hi=hi_m, _ai=ai_m, _hg=hg_m,
                         _ag=ag_m, _w=w_m, _i=i):
                a    = attack.copy();  a[_i]  = p2[0]
                d    = defence.copy(); d[_i]  = p2[1]
                lh   = np.exp(a[_hi] + d[_ai] + home_adv)
                la   = np.exp(a[_ai] + d[_hi])
                p    = _pmf(_hg, lh) * _pmf(_ag, la)
                rc   = _rho_corr(_hg, _ag, lh, la, rho)
                p    = p * np.clip(rc, 1e-10, None)
                return -float(np.sum(_w * np.log(np.clip(p, 1e-10, None))))

            result   = minimize(
                team_obj,
                [attack[i], defence[i]],
                method="L-BFGS-B",
                bounds=[(-3, 3), (-3, 3)],
                options={"maxiter": 50, "ftol": 1e-6},
            )
            attack[i]  = result.x[0]
            defence[i] = result.x[1]

        # ── Step B: optimize global params (home_adv, rho) ───
        r_global = minimize(
            _global_nll,
            [home_adv, rho],
            args=(attack, defence, hi, ai, hg, ag, w),
            method="L-BFGS-B",
            bounds=[(0, 1), (-1, 0)],
            options={"maxiter": 100},
        )
        home_adv = float(r_global.x[0])
        rho      = float(r_global.x[1])

        delta   = float(np.max(np.abs(attack - prev_attack)))
        elapsed = time.time() - t0

        print(f"  Cycle {cycle + 1:2d}: "
              f"delta={delta:.6f}  "
              f"home_adv={home_adv:.4f}  "
              f"rho={rho:.4f}  "
              f"t={elapsed:.0f}s")

        # Check convergence
        if delta < TOLERANCE:
            print(f"  ✅ Converged at cycle {cycle + 1} ({elapsed:.0f}s)")
            break

        # Early stop if delta is not improving meaningfully
        improvement = (prev_delta - delta) / (prev_delta + 1e-10)
        if cycle >= 5 and improvement < 0.02:
            no_improve += 1
        else:
            no_improve = 0

        if no_improve >= 3:
            print(f"  ✅ Early stop — delta plateaued at {delta:.6f} "
                  f"({elapsed:.0f}s)")
            break

        prev_delta = delta

    else:
        elapsed = time.time() - t0
        print(f"  ⚠️  Max cycles reached — results valid, "
              f"delta={delta:.6f} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Teams: {n}  |  Matches: {len(df):,}  |  "
          f"Total time: {elapsed:.0f}s")

    return {
        "teams":    teams,
        "attack":   dict(zip(teams, attack.tolist())),
        "defence":  dict(zip(teams, defence.tolist())),
        "home_adv": home_adv,
        "rho":      rho,
    }


def predict(team_a: str, team_b: str, params: dict,
            neutral: bool = True) -> dict:
    """Predict match outcome using fitted Dixon-Coles params."""
    home_adv = 0.0 if neutral else params["home_adv"]
    lam_a    = np.exp(
        params["attack"].get(team_a, 0) +
        params["defence"].get(team_b, 0) +
        home_adv
    )
    lam_b    = np.exp(
        params["attack"].get(team_b, 0) +
        params["defence"].get(team_a, 0)
    )
    matrix = score_matrix(lam_a, lam_b, params["rho"])
    probs  = match_probs(matrix)
    probs.update({
        "lambda_a": round(lam_a, 4),
        "lambda_b": round(lam_b, 4),
    })
    return probs