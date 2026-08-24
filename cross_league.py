"""
cross_league.py

Predicts fixtures between teams from DIFFERENT divisions (cup ties,
mainly the EFL Cup) - the same-league Poisson model in poisson_model.py
assumes both teams' attack/defense strengths are relative to the SAME
league average, which breaks down the moment the two teams come from
different divisions with different scoring/corners/cards baselines.

## The honest limitation

Within a division, "20% above average" is directly comparable between
any two teams in that division - they're measured against the same
baseline. But a "20% above average" League Two team and a "20% above
average" Championship team aren't obviously the same absolute quality -
divisions differ in overall standard, not just in what their averages
happen to be. We have NO cross-league match history to calibrate a true
quality gap from (our data source doesn't cover cup competitions at
all - see build_statpack.py notes), so this can't be derived
empirically. This uses three things instead:

  1. Each team's own division baseline (their league's average goals/
     corners/cards per game, already computed by league_averages()) -
     reconciled between the two divisions using the GEOMETRIC MEAN of
     both baselines as a shared reference point, rather than assuming
     one league's numbers are "correct" and the other should defer to
     them.
  2. Actual league position (see league_table.py) as a modest
     additional signal - a team's whole-season standing captures
     something a 10-game rolling form window doesn't always fully
     reflect.
  3. TIER_STRENGTH below - a deliberately hand-set, NOT data-derived
     estimate of the general quality gap between English divisions.
     Without it, two "exactly average for their own league" teams from
     different divisions come out as a dead-even coin flip, which is
     wrong - a Premier League team carries a real edge over a League
     One team independent of either team's specific current form, and
     the model had no way to express that until this was added. These
     numbers are a reasonable estimate, not a measured one - adjust
     them if they feel off in practice.

Cup predictions built on this should still be treated with real
caution, which is why they're visibly flagged as cross-division in the
output.
"""

from poisson_model import poisson_pmf, over_under

# Rough, hand-set relative quality between English divisions - see
# "TIER_STRENGTH" note above. Premier League = 1.0 baseline; each tier
# down reflects a genuine but not enormous quality gap (many
# Championship sides aren't far off Premier League standard, especially
# recently relegated ones; the gap widens further down). Scottish
# Premiership is placed roughly alongside the Championship/League One
# range - genuinely uncertain, adjust if it feels wrong.
TIER_STRENGTH = {
    "E0": 1.00,   # Premier League
    "E1": 0.88,   # Championship
    "E2": 0.78,   # League One
    "E3": 0.68,   # League Two
    "SC0": 0.85,  # Scottish Premiership
}


def tier_strength(league_code: str) -> float:
    return TIER_STRENGTH.get(league_code, 1.0)  # unknown league: neutral, no adjustment

# How much table position can swing a team's effective strength, as a
# multiplier: 1st place gets POSITION_SWING above 1.0, last place gets
# POSITION_SWING below 1.0, scaled linearly in between. Deliberately
# modest (+/-15%) - this supplements the form-based signal, it doesn't
# override it.
POSITION_SWING = 0.15


def _geomean(a: float, b: float) -> float:
    return (a * b) ** 0.5 if a > 0 and b > 0 else (a or b or 0)


def position_multiplier(position: int, total_teams: int) -> float:
    """1.0 + POSITION_SWING for 1st place, 1.0 - POSITION_SWING for
    last place, linear in between. Returns 1.0 (neutral) if we don't
    know the team's position or the league is degenerate."""
    if not position or not total_teams or total_teams <= 1:
        return 1.0
    # normalized 0..1, 1.0 = top of table
    normalized = (total_teams - position) / (total_teams - 1)
    return 1.0 + POSITION_SWING * (2 * normalized - 1)


# The combined attack*defense product for one side gets clamped to this
# range, rather than clamping each input ratio separately - clamping
# inputs alone doesn't work, since two only-moderately-elevated ratios
# (neither individually extreme) can still multiply together into an
# extreme joint result. Confirmed by testing: with per-input clamping
# of [0.5, 1.8], a Chelsea (weak home defense, ratio 1.6) vs Luton
# (strong away attack, ratio 1.7) case still produced a 74% away-win
# probability for a League One side at a Premier League ground -
# 1.6 x 1.7 = 2.72, still way beyond what either factor alone would
# suggest. Clamping the PRODUCT directly is what actually bounds the
# final result.
COMBINED_CLAMP = (0.6, 1.6)


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return max(bounds[0], min(bounds[1], value))


def cross_league_expected_values(
    home_form: dict, away_form: dict,
    home_league_avg: dict, away_league_avg: dict,
    home_league_code: str, away_league_code: str,
    home_position_mult: float, away_position_mult: float,
    metric: str,
) -> tuple[float, float]:
    """
    Like poisson_model.expected_values(), but for two teams from
    different divisions. Each team's attack/defense strength is still
    computed relative to their OWN division's average (so "above
    average for League Two" stays meaningful), but the final lambda
    uses the geometric mean of both divisions' baselines as a shared
    reference point, each side gets nudged by their league-position
    multiplier, and by their division's relative TIER_STRENGTH. The
    combined attack*defense product for each side is clamped (see
    COMBINED_CLAMP) to prevent two individually-reasonable ratios
    compounding into an implausible result - see the comment above
    COMBINED_CLAMP for a real example this was built to catch.
    """
    for_key, against_key = f"{metric}_for", f"{metric}_against"

    home_league_home_avg = home_league_avg.get(f"home_{metric}", 0) or 1.0
    home_league_away_avg = home_league_avg.get(f"away_{metric}", 0) or 1.0
    away_league_home_avg = away_league_avg.get(f"home_{metric}", 0) or 1.0
    away_league_away_avg = away_league_avg.get(f"away_{metric}", 0) or 1.0

    home_attack = (home_form.get(for_key, home_league_home_avg)) / home_league_home_avg
    away_defense = (away_form.get(against_key, away_league_home_avg)) / away_league_home_avg
    away_attack = (away_form.get(for_key, away_league_away_avg)) / away_league_away_avg
    home_defense = (home_form.get(against_key, home_league_away_avg)) / home_league_away_avg

    # Tier gap only actually matters for goals-adjacent metrics - a
    # weaker division's teams still concede/win corners and cards at
    # broadly similar rates (those aren't primarily a quality signal
    # the way goalscoring is), so only apply the tier adjustment to
    # goals and the two half-goal splits, not corners/cards.
    if metric in ("goals", "first_half_goals", "second_half_goals"):
        home_tier, away_tier = tier_strength(home_league_code), tier_strength(away_league_code)
        tier_ratio_home = home_tier / away_tier if away_tier else 1.0
        tier_ratio_away = away_tier / home_tier if home_tier else 1.0
    else:
        tier_ratio_home = tier_ratio_away = 1.0

    home_factor = _clamp(home_attack * away_defense * tier_ratio_home, COMBINED_CLAMP)
    away_factor = _clamp(away_attack * home_defense * tier_ratio_away, COMBINED_CLAMP)

    shared_home_baseline = _geomean(home_league_home_avg, away_league_home_avg)
    shared_away_baseline = _geomean(home_league_away_avg, away_league_away_avg)

    lam_home = shared_home_baseline * home_factor * home_position_mult
    lam_away = shared_away_baseline * away_factor * away_position_mult

    return round(lam_home, 3), round(lam_away, 3)


def predict_cross_league_fixture(
    home_form: dict, away_form: dict,
    home_league_avg: dict, away_league_avg: dict,
    home_league_code: str, away_league_code: str,
    home_position_mult: float, away_position_mult: float,
    lines: dict,
) -> dict:
    """Same output shape as poisson_model.predict_fixture(), for a
    cross-league (cup) fixture."""
    from stats_engine import METRICS
    result = {}

    for metric in METRICS:
        lam_home, lam_away = cross_league_expected_values(
            home_form, away_form, home_league_avg, away_league_avg,
            home_league_code, away_league_code,
            home_position_mult, away_position_mult, metric,
        )
        lam_total = lam_home + lam_away

        market = {
            "expected_home": lam_home,
            "expected_away": lam_away,
            "expected_total": round(lam_total, 2),
            "over_under": [over_under(lam_total, line) for line in lines.get(metric, [])],
        }

        if metric == "goals":
            p_home_scores = 1 - poisson_pmf(0, lam_home)
            p_away_scores = 1 - poisson_pmf(0, lam_away)
            market["btts_yes"] = round(p_home_scores * p_away_scores, 3)
            market["btts_no"] = round(1 - p_home_scores * p_away_scores, 3)

            from poisson_model import match_result
            market["match_result"] = match_result(lam_home, lam_away)

        result[metric] = market

    return result
