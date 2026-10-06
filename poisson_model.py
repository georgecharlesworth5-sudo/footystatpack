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

## xG blending (goals only)

Actual goals scored/conceded over a rolling window are noisier than
they need to be - a team can meaningfully over- or under-perform the
quality of chances they actually created, purely down to finishing
variance. Where football-data.co.uk provides xG (HxG/AxG - not
available for every league/season, see fetch_data.py), the "goals"
metric blends actual goals with xG rather than using actual goals
alone, giving a steadier read on a team's true underlying level.
Corners/cards/half-splits are untouched - we have no xG equivalent
for those.
"""

import math

from stats_engine import METRICS

XG_BLEND_WEIGHT = 0.4  # how much weight xG gets vs actual goals, when both are available
MIN_XG_SAMPLE = 3      # below this many xG-having matches, trust is too thin - use actual goals alone

# ---------------------------------------------------------------------
# Model v2 (walk-forward backtest, ~3,100 matches, Nov 2025 onward)
#
# v1 trusted each team's recent form at face value and assumed pure Poisson
# counts. Replayed on past matches it stated ~93% on its 87%+ picks and
# landed ~83%. Two fixes closed the gap:
#
#  1. FORM_SHRINKAGE - each side's expected value is pulled this far toward
#     the league average (0 = trust form fully, 1 = ignore form). Team form
#     over 10-20 matches is mostly noise; 0.7 was near the optimum across
#     every form window tested (0.6-0.9 all scored within noise).
#  2. NB_ALPHA - negative binomial instead of Poisson: variance =
#     mu + alpha*mu^2, with alpha fitted per market from the same replay.
#     Corners and cards are visibly over-dispersed; goals barely are.
#
# Form window is also longer (20 matches, gentler decay - see stats_engine).
# ---------------------------------------------------------------------
MODEL_VERSION = "v2"
FORM_SHRINKAGE = 0.7

# (metric, level) -> alpha. level: "total" | "home" | "away".
NB_ALPHA = {
    ("goals", "total"): 0.0, ("goals", "home"): 0.0, ("goals", "away"): 0.012,
    ("corners", "total"): 0.016, ("corners", "home"): 0.095, ("corners", "away"): 0.101,
    ("cards", "total"): 0.037, ("cards", "home"): 0.029, ("cards", "away"): 0.0,
    ("first_half_goals", "total"): 0.0, ("second_half_goals", "total"): 0.0,
}

# Per-side O/U lines (a single team's own corner count, not the combined
# match total) - roughly half of the usual total-match lines, since one
# team's corners typically run a bit lower than the full-match total.
PER_SIDE_LINES = {
    "corners": [2.5, 3.5, 4.5, 5.5, 6.5],
    "goals": [0.5, 1.5],
    "cards": [0.5, 1.5],
}


def _blended_goal_value(actual: float, xg: float | None, xg_matches: int) -> float:
    """actual goals, blended toward xG when we have enough xG data to
    trust it; otherwise just the actual figure, unchanged."""
    if xg is None or xg_matches < MIN_XG_SAMPLE:
        return actual
    return round((1 - XG_BLEND_WEIGHT) * actual + XG_BLEND_WEIGHT * xg, 3)


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_cdf(k: int, lam: float) -> float:
    """P(X <= k)"""
    return sum(poisson_pmf(i, lam) for i in range(0, k + 1))


def nb_cdf(k: int, mu: float, alpha: float) -> float:
    """P(X <= k) for a negative binomial with mean mu and variance
    mu + alpha*mu^2. Falls back to Poisson when alpha is ~0."""
    if mu <= 0:
        return 1.0
    if alpha <= 1e-6:
        return poisson_cdf(k, mu)
    r = 1.0 / alpha
    lp, lq = math.log(r / (r + mu)), math.log(mu / (r + mu))
    total = 0.0
    for i in range(k + 1):
        total += math.exp(math.lgamma(i + r) - math.lgamma(r) - math.lgamma(i + 1) + r * lp + i * lq)
    return min(1.0, total)


def over_under(lam_total: float, line: float, alpha: float = 0.0) -> dict:
    """
    P(over) / P(under) a given line (e.g. 2.5) for a count with mean
    lam_total - Poisson when alpha=0, negative binomial otherwise. Lines
    are almost always X.5 in these markets so there's no push case.
    """
    floor_line = math.floor(line)
    p_under_or_equal = nb_cdf(floor_line, lam_total, alpha)
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
    league averages. For "goals" specifically, both the team-form figures
    and the league baseline are blended with xG where available (see
    XG_BLEND_WEIGHT above) - everything else is untouched.
    """
    for_key = f"{metric}_for"
    against_key = f"{metric}_against"

    league_home_avg = league_avg.get(f"home_{metric}", 0) or 1.0
    league_away_avg = league_avg.get(f"away_{metric}", 0) or 1.0

    home_form_for = home_form.get(for_key, league_home_avg)
    away_form_against = away_form.get(against_key, league_home_avg)
    away_form_for = away_form.get(for_key, league_away_avg)
    home_form_against = home_form.get(against_key, league_away_avg)

    if metric == "goals":
        league_home_avg = _blended_goal_value(
            league_home_avg, league_avg.get("home_xg"), league_avg.get("xg_matches", 0))
        league_away_avg = _blended_goal_value(
            league_away_avg, league_avg.get("away_xg"), league_avg.get("xg_matches", 0))
        home_form_for = _blended_goal_value(
            home_form_for, home_form.get("xg_for"), home_form.get("xg_matches", 0))
        away_form_against = _blended_goal_value(
            away_form_against, away_form.get("xg_against"), away_form.get("xg_matches", 0))
        away_form_for = _blended_goal_value(
            away_form_for, away_form.get("xg_for"), away_form.get("xg_matches", 0))
        home_form_against = _blended_goal_value(
            home_form_against, home_form.get("xg_against"), home_form.get("xg_matches", 0))

    home_attack = home_form_for / league_home_avg
    away_defense = away_form_against / league_home_avg
    lam_home = league_home_avg * home_attack * away_defense

    away_attack = away_form_for / league_away_avg
    home_defense = home_form_against / league_away_avg
    lam_away = league_away_avg * away_attack * home_defense

    # Pull toward the league average (see FORM_SHRINKAGE).
    lam_home = (1 - FORM_SHRINKAGE) * lam_home + FORM_SHRINKAGE * league_home_avg
    lam_away = (1 - FORM_SHRINKAGE) * lam_away + FORM_SHRINKAGE * league_away_avg

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

    total = home_win + draw + away_win
    if total > 0:
        home_win, draw, away_win = home_win / total, draw / total, away_win / total

    return {
        "home_win": round(home_win, 3),
        "draw": round(draw, 3),
        "away_win": round(away_win, 3),
    }


def predict_fixture(home_form: dict, away_form: dict, league_avg: dict, lines: dict,
                     metrics: list[str] | None = None) -> dict:
    """
    home_form / away_form: output of stats_engine.rolling_form() for the
    HOME venue slice (home team) and AWAY venue slice (away team)
    respectively - i.e. home team's home form, away team's away form.

    lines: dict of metric -> list of O/U lines to evaluate, e.g.
        {"goals": [1.5, 2.5, 3.5], "corners": [8.5, 9.5, 10.5], "cards": [3.5, 4.5]}
        Metrics not present in `lines` are still computed (expected value)
        but skip the over/under breakdown.

    metrics: which of stats_engine.METRICS to actually compute, defaulting
        to all of them. Pass a restricted list (e.g. ["goals"]) for a data
        source that doesn't carry every stat - e.g. MLS's results come from
        football-data.co.uk's "extra leagues" file, which has goals only,
        no corners/cards/half-time score. Computing corners/cards from
        columns that are always blank wouldn't just be "no data" - every
        row's HC/AC/HY/AY/HR/AR reads as 0 (see stats_engine.py's
        `int(row.get("HC") or 0)`), so the model would see zero variance
        and output a false, maximally-confident "under" on every
        corners/cards line. Leaving the metric out of the result dict
        entirely is what makes best_bets.py's `preds.get(metric_key)` /
        `if not m: continue` skip it cleanly, rather than quietly act on
        fabricated figures.

    Returns predictions for every requested metric plus BTTS and
    match-result (1X2) markets for full-match goals.
    """
    result = {}

    for metric in (metrics if metrics is not None else METRICS):
        lam_home, lam_away = expected_values(home_form, away_form, league_avg, metric)
        lam_total = lam_home + lam_away

        market = {
            "expected_home": lam_home,
            "expected_away": lam_away,
            "expected_total": round(lam_total, 2),
            "over_under": [over_under(lam_total, line, NB_ALPHA.get((metric, "total"), 0.0))
                           for line in lines.get(metric, [])],
        }

        if metric in PER_SIDE_LINES:
            market["home_over_under"] = [over_under(lam_home, line, NB_ALPHA.get((metric, "home"), 0.0))
                                         for line in PER_SIDE_LINES[metric]]
            market["away_over_under"] = [over_under(lam_away, line, NB_ALPHA.get((metric, "away"), 0.0))
                                         for line in PER_SIDE_LINES[metric]]

        if metric == "goals":
            p_home_scores = 1 - poisson_pmf(0, lam_home)
            p_away_scores = 1 - poisson_pmf(0, lam_away)
            market["btts_yes"] = round(p_home_scores * p_away_scores, 3)
            market["btts_no"] = round(1 - p_home_scores * p_away_scores, 3)

            market["match_result"] = match_result(lam_home, lam_away)

        result[metric] = market

    return result
