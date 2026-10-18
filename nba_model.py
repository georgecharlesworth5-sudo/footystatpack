"""
nba_model.py - turns team ratings and player form into probabilities.

Everything is a Normal distribution around an expected value, with
spreads measured from a walk-forward backtest of a full past season:

  * margin (home minus away):  expected = MARGIN_SCALE x rating gap,
    SD 14.5. The ratings are deliberately shrunk by the ridge penalty,
    so the raw gap is stretched by MARGIN_SCALE (fitted, ~1.14).
  * game total:                expected = rating total, SD 20.9.
  * team points:               from the expected margin + total, SD 13.
  * player props:              decayed average of recent games, adjusted for
    the opposition and game context (PROP_COEFS: what the opponent concedes,
    shot volume, pace, home/away, blowout risk, longer-term form, minutes
    trend). The SD is the player's own spread widened by a fitted per-stat
    scale. Threes use a negative-binomial (counts that skew right near
    zero) instead.

All lines are half-points, so there are no pushes.
"""
from __future__ import annotations

from math import erf, exp, lgamma, log, sqrt

MARGIN_SCALE = 1.14
MARGIN_SD = 14.5
TOTAL_SD = 20.9
TEAM_POINTS_SD = 13.0
PROP_SD_SCALE = 1.25
PROP_STALE_DAYS = 60       # last game further back than this: widen a little more
PROP_STALE_SCALE = 1.10
PROP_SD_FLOOR = {"points": 3.0, "rebounds": 1.5, "assists": 1.2, "threes": 0.8}


# Opposition / context adjustment for player props. Fitted by ridge regression on
# two full seasons of player games (2024-25 and 2025-26, ~32,000 player-games;
# tested walk-forward: fit on one season, scored on the next - average error fell
# 1.4-2.5 percent and the probabilities got better calibrated). Each stat's expected
# value is   form15 x (1 + bias + sum(coef x feature)) , where the features are:
#   home        +1 home / -1 away
#   b2b         1 if his team played yesterday
#   absmargin   |expected margin| / 10  (blowouts cut starters' minutes)
#   pace        average of both teams' possessions per game vs league (0.05 = 5 percent above)
#   odef_x      how much the OPPONENT concedes of x vs league (pts, fga=shots,
#               tpa=threes attempted, tpm, reb, ast, miss=missed shots)
#   aoff_x/ooff_x  his team's / the opponent's own output of x
#   form40      his 40-game average vs his 15-game average (a longer memory
#               stops one hot or cold fortnight dominating)
#   mintrend    minutes in his last 5 games vs his 40-game norm
# sd_scale = how much wider than his own game-to-game spread the real misses are.
PROP_COEFS = {
    "points": {
        "coef": {
            "home": 0.0015,
            "b2b": 0.0167,
            "absmargin": -0.0035,
            "pace": 0.0424,
            "odef_pts": 0.7299,
            "odef_fga": 0.2318,
            "odef_tpa": -0.1522,
            "form40": 0.653,
            "mintrend": 0.3101,
            "bias": -0.0129
        },
        "sd_scale": 1.191
    },
    "rebounds": {
        "coef": {
            "home": 0.0005,
            "b2b": 0.0053,
            "absmargin": 0.0075,
            "pace": 0.3486,
            "odef_reb": 0.5167,
            "odef_miss": -0.2373,
            "aoff_miss": -0.1365,
            "ooff_oreb": -0.0462,
            "odef_fga": 0.3089,
            "form40": 0.8414,
            "mintrend": 0.4065,
            "bias": -0.027
        },
        "sd_scale": 1.175
    },
    "assists": {
        "coef": {
            "home": 0.0096,
            "b2b": -0.009,
            "absmargin": -0.0027,
            "pace": 0.255,
            "odef_ast": 0.512,
            "odef_pts": 0.3352,
            "aoff_ast": -0.0484,
            "form40": 0.7148,
            "mintrend": 0.5004,
            "bias": -0.0143
        },
        "sd_scale": 1.196
    },
    "threes": {
        "coef": {
            "home": 0.009,
            "b2b": -0.0005,
            "absmargin": 0.0037,
            "pace": 0.0703,
            "odef_tpm": 0.0467,
            "odef_tpa": 0.4732,
            "ooff_tpa": -0.0106,
            "form40": 0.8917,
            "mintrend": 0.3767,
            "bias": -0.0252
        },
        "sd_scale": 1.159
    }
}


def prop_adjust(stat: str, form15: float, features: dict) -> tuple[float, dict]:
    """(context-adjusted expected value, each feature's contribution as a fraction)."""
    c = PROP_COEFS[stat]["coef"]
    parts = {k: v * features.get(k, 0.0) for k, v in c.items() if k != "bias"}
    factor = c["bias"] + sum(parts.values())
    return max(form15 * (1.0 + factor), 0.0), parts


def phi(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def p_over(mean: float, sd: float, line: float) -> float:
    return 1.0 - phi((line - mean) / sd)


def nb_cdf(k: int, mean: float, var: float) -> float:
    """P(X <= k) for a negative binomial with the given mean/variance
    (falls back to Poisson when variance <= mean)."""
    if k < 0:
        return 0.0
    if var <= mean * 1.0001:
        p0 = exp(-mean); total = p0; term = p0
        for i in range(1, k + 1):
            term *= mean / i; total += term
        return min(total, 1.0)
    r = mean * mean / (var - mean)
    q = mean / var          # success prob in the (r, p) parameterisation: p = mean/var
    term = exp(r * log(q)); total = term
    for i in range(1, k + 1):
        term *= (i - 1 + r) / i * (1 - q)
        total += term
    return min(total, 1.0)


def predict_game(ratings, home_id: int, away_id: int, home_b2b: int, away_b2b: int) -> dict:
    ph = ratings.expected_points(home_id, away_id, 1, home_b2b, away_b2b)
    pa = ratings.expected_points(away_id, home_id, 0, away_b2b, home_b2b)
    mid = (ph + pa) / 2
    margin = (ph - pa) * MARGIN_SCALE
    total = ph + pa
    home_pts = total / 2 + margin / 2
    away_pts = total / 2 - margin / 2
    return {"margin": margin, "total": total, "home_pts": home_pts, "away_pts": away_pts}


def moneyline(margin: float) -> dict:
    hw = phi(margin / MARGIN_SD)
    return {"home_win": round(hw, 4), "away_win": round(1 - hw, 4)}


def _half_lines(center: float, half_width: float) -> list[float]:
    lo = int(center - half_width)
    hi = int(center + half_width) + 1
    return [k + 0.5 for k in range(lo, hi)]


def spread_table(margin: float) -> list[dict]:
    """Handicap lines applied to the HOME team (e.g. -4.5 = home must win
    by 5+). cover = probability the home side beats that handicap; the
    away side covering the mirrored line is 1 - cover."""
    rows = []
    for L in _half_lines(-margin, 14):          # centred on the fair handicap
        rows.append({"line": L, "home_cover": round(phi((margin + L) / MARGIN_SD), 4)})
    return rows


def total_table(mean: float, sd: float, half_width: float) -> list[dict]:
    rows = []
    for L in _half_lines(mean, half_width):
        o = p_over(mean, sd, L)
        rows.append({"line": L, "over": round(o, 4), "under": round(1 - o, 4)})
    return rows


def prop_dist(stat: str, mean: float, sd: float, stale_days: int) -> tuple[float, float]:
    """(mean, sd used) for a prop."""
    scale = PROP_COEFS[stat]["sd_scale"] * (PROP_STALE_SCALE if stale_days > PROP_STALE_DAYS else 1.0)
    return mean, max(sd * scale, PROP_SD_FLOOR[stat])


def prop_table(stat: str, mean: float, sd: float) -> list[dict]:
    rows = []
    if stat == "threes":
        mean = max(mean, 0.05)
        var = max(sd * sd, mean * 1.05)
        for L in [k + 0.5 for k in range(0, int(mean + 4))]:
            under = nb_cdf(int(L - 0.5), mean, var)
            rows.append({"line": L, "over": round(1 - under, 4), "under": round(under, 4)})
        return rows
    for L in _half_lines(mean, 2.2 * sd):
        if L < 0.5:
            continue
        o = p_over(mean, sd, L)
        rows.append({"line": L, "over": round(o, 4), "under": round(1 - o, 4)})
    return rows
