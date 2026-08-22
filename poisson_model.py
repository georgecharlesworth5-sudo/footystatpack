"""
poisson_model.py

Classic attack/defense-strength Poisson model (the same family used by
most public football prediction models), applied to three metrics:
goals, corners, cards.

For each metric we compute:
  - team attack strength  = team's own average (for) / league average
  - team defense strength = team's own average (against) / league average
  - expected value (lambda) for a fixture = league_avg * attack * defense

Then we treat the total (home + away) as Poisson(lambda_home + lambda_away)
to get over/under probabilities, and treat home/away as independent
Poissons to get BTTS-style ("both teams card/corner") probabilities.

Note on corners/cards: this model assumes independence and stationarity
that's less clean than for goals (e.g. cards are influenced heavily by
game state and referee - Referee is captured as a separate diagnostic
field, not yet folded into the lambda itself). Treat the corners/cards
outputs as a solid baseline signal, not a finished edge - see NOTES in
build_statpack.py for suggested refinements.
"""

import math

from stats_engine import METRICS


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k)"""
    return sum(poisson_pmf(i, lam) for i in range(0, k + 1))


def over_under(lam_total: float, line: float) -> dict:
    """
    P(over) / P(under) a given line (e.g. 2.5) for a Poisson(lam_total)
    total. Lines are almost always X.5 in these markets so there's no
    push case to handle.
    """
    floor_line = math.floor(line)
    p_under_or_equal = poisson_cdf(floor_line, lam_total)
    return {
        "line": line,
        "over": round(1 - p_under_or_equal, 3),
        "under": round(p_under_or_equal, 3),
        "expected": round(lam_total, 2),
    }


def strengths(team_for_avg: float, team_against_avg: float, league_for_avg: float, league_against_avg: float) -> dict:
    """Attack/defense strength ratios, guarding against zero-division on
    sparse early-season data."""
    attack = team_for_avg / league_for_avg if league_for_avg else 1.0
    defense = team_against_avg / league_against_avg if league_against_avg else 1.0
    return {"attack": round(attack, 3), "defense": round(defense, 3)}


def expected_values(home_form: dict, away_form: dict, league_avg: dict, metric: str) -> tuple[float, float]:
    """
    Compute (lambda_home, lambda_away) for one metric ("goals", "corners",
    or "cards") given home team's home-form, away team's away-form, and
    league averages.
    """
    for_key = f"{metric}_for"
    against_key = f"{metric}_against"

    league_home_avg = league_avg.get(f"home_{metric}", 0) or 1.0
    league_away_avg = league_avg.get(f"away_{metric}", 0) or 1.0

    home_attack = (home_form.get(for_key, league_home_avg)) / league_home_avg
    away_defense = (away_form.get(against_key, league_home_avg)) / league_home_avg
    lam_home = league_home_avg * home_attack * away_defense

    away_attack = (away_form.get(for_key, league_away_avg)) / league_away_avg
    home_defense = (home_form.get(against_key, league_away_avg)) / league_away_avg
    lam_away = league_away_avg * away_attack * home_defense

    return round(lam_home, 3), round(lam_away, 3)


def match_result(lam_home: float, lam_away: float, max_goals: int = 10) -> dict:
    """
    Home Win / Draw / Away Win probabilities, from the same expected-goals
    values (lambda_home, lambda_away) already used for the goals O/U
    market. Builds the full grid of realistic scorelines (0-0 up to
    max_goals-max_goals), treating home and away goals as independent
    Poisson variables, then buckets each scoreline by which side has more
    goals. max_goals=10 each way is already far past any realistic
    scoreline's probability, so the bucketed totals sum to ~1.0.
    """
    home_win = draw = away_win = 0.0
    for h in range(max_goals + 1):
        p_h = poisson_pmf(h, lam_home)
        for a in range(max_goals + 1):
            p = p_h * poisson_pmf(a, lam_away)
            if h > a:
                home_win += p
            elif h == a:
                draw += p
            else:
                away_win += p

    # The three buckets won't sum to exactly 1.0 (scorelines above
    # max_goals each are vanishingly unlikely but not zero) - rescale so
    # they read as clean percentages that actually add up.
    total = home_win + draw + away_win
    if total > 0:
        home_win, draw, away_win = home_win / total, draw / total, away_win / total

    return {
        "home_win": round(home_win, 3),
        "draw": round(draw, 3),
        "away_win": round(away_win, 3),
    }


def predict_fixture(home_form: dict, away_form: dict, league_avg: dict, lines: dict) -> dict:
    """
    home_form / away_form: output of stats_engine.rolling_form() for the
    HOME venue slice (home team) and AWAY venue slice (away team)
    respectively - i.e. home team's home form, away team's away form.

    lines: dict of metric -> list of O/U lines to evaluate, e.g.
        {"goals": [1.5, 2.5, 3.5], "corners": [8.5, 9.5, 10.5], "cards": [3.5, 4.5]}
        Metrics not present in `lines` are still computed (expected value)
        but skip the over/under breakdown.

    Returns predictions for every tracked metric (see stats_engine.METRICS)
    plus BTTS and match-result (1X2) markets for full-match goals.
    """
    result = {}

    for metric in METRICS:
        lam_home, lam_away = expected_values(home_form, away_form, league_avg, metric)
        lam_total = lam_home + lam_away

        market = {
            "expected_home": lam_home,
            "expected_away": lam_away,
            "expected_total": round(lam_total, 2),
            "over_under": [over_under(lam_total, line) for line in lines.get(metric, [])],
        }

        if metric == "goals":
            # BTTS: both teams score >= 1
            p_home_scores = 1 - poisson_pmf(0, lam_home)
            p_away_scores = 1 - poisson_pmf(0, lam_away)
            market["btts_yes"] = round(p_home_scores * p_away_scores, 3)
            market["btts_no"] = round(1 - p_home_scores * p_away_scores, 3)

            # Match result (1X2), from the same lam_home/lam_away
            market["match_result"] = match_result(lam_home, lam_away)

        result[metric] = market

    return result
