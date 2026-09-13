"""
nfl_model.py

NFL prediction model: Moneyline, Total Points, Team Points, and
Passing/Rushing TDs (team-level, not player props).

Two different statistical approaches are used deliberately, not
arbitrarily:

  - POINTS (moneyline, total points, team points): treated as roughly
    Normal-distributed. NFL scoring - points in 3s/6s/7s, driven by
    drives/possessions - isn't well modelled as a low-count Poisson
    process the way football goals are; the standard approach in real
    NFL power-rating/spread models is a Normal approximation on the
    score/margin instead.

  - PASSING/RUSHING TDs (team-level): treated as Poisson. Unlike total
    points, a team's TD count in a single category is a genuinely
    low, discrete count (usually 0-4) - much closer to the football
    goals case Poisson was built for than to the higher, more
    continuous point totals above. Same over/under approach as the
    football model's goals market.

Both share the same lesson learned building the football model: the
combined attack*defense factor gets clamped before use, since two
individually-plausible ratios can compound into an implausible result
when either side's sample is thin - confirmed directly, not assumed
(see FORM_CLAMP below).
"""

import math

STD_MARGIN = 13.5        # stdev of (home_score - away_score) around its predicted mean
STD_TEAM_POINTS = 10.0   # stdev of one team's own score around its predicted mean
STD_TOTAL = math.sqrt(2) * STD_TEAM_POINTS

FALLBACK_LEAGUE_POINTS = 22.0
FALLBACK_LEAGUE_TDS = 1.2  # a fallback for passing/rushing TDs specifically if league_avg is ever empty

# The combined attack*defense product for one side gets clamped to this
# range - confirmed necessary by testing: a thin-sample case (1-3 games)
# produced a "38 vs 66, 104-point total" prediction before this was
# added. Same reasoning applies to TDs, using the same bounds - no
# separate tuning done for TDs specifically yet, revisit if real
# results suggest they need their own range.
FORM_CLAMP = (0.6, 1.6)


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


def _normal_cdf(x: float, mean: float, std: float) -> float:
    if std <= 0:
        return 1.0 if x >= mean else 0.0
    z = (x - mean) / (std * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_cdf(k: int, lam: float) -> float:
    return sum(poisson_pmf(i, lam) for i in range(0, k + 1))


def over_under_poisson(lam: float, line: float) -> dict:
    floor_line = math.floor(line)
    p_under_or_equal = poisson_cdf(floor_line, lam)
    return {"line": line, "over": round(1 - p_under_or_equal, 3),
            "under": round(p_under_or_equal, 3), "expected": round(lam, 2)}


def _expected_stat(home_form: dict, away_form: dict, league_home_avg: float, league_away_avg: float,
                    for_key: str, against_key: str) -> tuple[float, float]:
    """Shared attack/defense-vs-league-average logic, used for BOTH
    points and TDs - same structure as the football goals model,
    parameterised by which stat's fields to read."""
    home_for = home_form.get(for_key)
    home_against = home_form.get(against_key)
    away_for = away_form.get(for_key)
    away_against = away_form.get(against_key)

    home_attack = (home_for if home_for is not None else league_home_avg) / league_home_avg
    away_defense = (away_against if away_against is not None else league_home_avg) / league_home_avg
    home_factor = _clamp(home_attack * away_defense, FORM_CLAMP)
    lam_home = league_home_avg * home_factor

    away_attack = (away_for if away_for is not None else league_away_avg) / league_away_avg
    home_defense = (home_against if home_against is not None else league_away_avg) / league_away_avg
    away_factor = _clamp(away_attack * home_defense, FORM_CLAMP)
    lam_away = league_away_avg * away_factor

    return round(lam_home, 3), round(lam_away, 3)


def expected_points(home_form: dict, away_form: dict, league_avg: dict) -> tuple[float, float]:
    league_home_avg = league_avg.get("home_points") or FALLBACK_LEAGUE_POINTS
    league_away_avg = league_avg.get("away_points") or FALLBACK_LEAGUE_POINTS
    return _expected_stat(home_form, away_form, league_home_avg, league_away_avg, "points_for", "points_against")


def expected_tds(home_form: dict, away_form: dict, league_avg: dict, td_type: str) -> tuple[float, float]:
    """td_type: 'passing_tds' or 'rushing_tds'."""
    league_home_avg = league_avg.get(f"home_{td_type}") or FALLBACK_LEAGUE_TDS
    league_away_avg = league_avg.get(f"away_{td_type}") or FALLBACK_LEAGUE_TDS
    return _expected_stat(home_form, away_form, league_home_avg, league_away_avg,
                           f"{td_type}_for", f"{td_type}_against")


def moneyline(exp_home: float, exp_away: float) -> dict:
    """No explicit draw probability - NFL ties are genuinely rare
    (~0.1% of games, only possible after a full overtime period)."""
    predicted_margin = exp_home - exp_away
    home_win = 1 - _normal_cdf(0, predicted_margin, STD_MARGIN)
    return {"home_win": round(home_win, 3), "away_win": round(1 - home_win, 3)}


def total_points_over_under(exp_home: float, exp_away: float, lines: list[float]) -> list[dict]:
    exp_total = exp_home + exp_away
    return [{"line": line, "over": round(1 - _normal_cdf(line, exp_total, STD_TOTAL), 3),
             "under": round(_normal_cdf(line, exp_total, STD_TOTAL), 3), "expected": round(exp_total, 1)}
            for line in lines]


def team_points_over_under(expected: float, lines: list[float]) -> list[dict]:
    return [{"line": line, "over": round(1 - _normal_cdf(line, expected, STD_TEAM_POINTS), 3),
             "under": round(_normal_cdf(line, expected, STD_TEAM_POINTS), 3), "expected": round(expected, 1)}
            for line in lines]


def predict_game(home_form: dict, away_form: dict, league_avg: dict,
                  total_lines: list[float], team_lines: list[float],
                  td_lines: list[float]) -> dict:
    """
    home_form / away_form: output of nfl_stats.rolling_form() for the
    HOME venue slice (home team) and AWAY venue slice (away team).
    td_lines: O/U lines applied to BOTH passing and rushing TDs (e.g.
    [0.5, 1.5, 2.5]) - team-level TD counts, not player props.
    """
    exp_home_pts, exp_away_pts = expected_points(home_form, away_form, league_avg)
    exp_home_pass_td, exp_away_pass_td = expected_tds(home_form, away_form, league_avg, "passing_tds")
    exp_home_rush_td, exp_away_rush_td = expected_tds(home_form, away_form, league_avg, "rushing_tds")

    return {
        "points": {
            "expected_home": exp_home_pts, "expected_away": exp_away_pts,
            "expected_total": round(exp_home_pts + exp_away_pts, 1),
            "moneyline": moneyline(exp_home_pts, exp_away_pts),
            "total_over_under": total_points_over_under(exp_home_pts, exp_away_pts, total_lines),
            "home_over_under": team_points_over_under(exp_home_pts, team_lines),
            "away_over_under": team_points_over_under(exp_away_pts, team_lines),
        },
        "passing_tds": {
            "expected_home": exp_home_pass_td, "expected_away": exp_away_pass_td,
            "home_over_under": [over_under_poisson(exp_home_pass_td, line) for line in td_lines],
            "away_over_under": [over_under_poisson(exp_away_pass_td, line) for line in td_lines],
        },
        "rushing_tds": {
            "expected_home": exp_home_rush_td, "expected_away": exp_away_rush_td,
            "home_over_under": [over_under_poisson(exp_home_rush_td, line) for line in td_lines],
            "away_over_under": [over_under_poisson(exp_away_rush_td, line) for line in td_lines],
        },
    }
