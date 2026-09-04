"""
stats_engine.py

Turns raw match rows (from fetch_data.py) into per-team rolling form:
goals, corners, cards, first-half goals, second-half goals - for/against,
with home/away splits.

This is the layer the Poisson model builds on top of.
"""

from collections import defaultdict
from datetime import datetime

# Central list of tracked metrics. Add a new stat by adding it here plus
# the extraction logic in build_team_match_log/league_averages - everything
# else (rolling_form, the Poisson model loop) reads this list rather than
# hardcoding metric names.
METRICS = ["goals", "corners", "cards", "first_half_goals", "second_half_goals"]

# Metrics where cross-division quality actually matters enough to adjust
# for (see TIER_STRENGTH below) - corners/cards aren't primarily a
# quality signal the way goalscoring is, so they're left unscaled even
# when a match came from a different division.
TIER_ADJUSTED_METRICS = {"goals", "first_half_goals", "second_half_goals"}

# Rough, hand-set relative quality between English divisions (plus Serie
# A, neutral since it doesn't currently interact with English tiers).
# Originally built for the EFL Cup's cross-division predictions
# (cross_league.py imports this from here) - now also used to correct a
# related but different problem: a newly promoted/relegated team's
# rolling form window blends in matches from their OLD division, and
# without this, those matches count at full face value even though
# they're not the same standard of opposition. E.g. a newly-promoted
# team's Championship-winning form doesn't automatically translate to
# Premier League form - this discounts it appropriately. See
# _tier_adjusted_value below for how it's actually applied.
#
# These numbers are a reasonable estimate, not a measured one - adjust
# them if they feel off in practice (same guidance as when they were
# first added for the cup predictions).
TIER_STRENGTH = {
    "E0": 1.00,   # Premier League
    "E1": 0.70,   # Championship
    "E2": 0.55,   # League One
    "E3": 0.45,   # League Two
    "SC0": 0.63,  # Scottish Premiership
}


def tier_strength(league_code: str) -> float:
    return TIER_STRENGTH.get(league_code, 1.0)  # unknown league: neutral, no adjustment


def _tier_adjusted_value(value: float, is_for: bool, match_division: str, current_division: str | None) -> float:
    """
    Scales a single match's stat value to translate it into "as if played
    in current_division" terms, when the match was actually played in a
    different division. Left unchanged if we don't know the current
    division, or the match was already in that division.

    For "for" stats (e.g. goals scored): a team's scoring rate against a
    WEAKER division should be discounted when judging them against a
    STRONGER one (they won't score as freely against tougher defenses) -
    scale by strength(match_division) / strength(current_division).

    For "against" stats (e.g. goals conceded): the opposite - conceding
    few goals against a weaker division's attacks doesn't mean they'll
    concede as few against a stronger one - scale by
    strength(current_division) / strength(match_division).
    """
    if not current_division or match_division == current_division:
        return value
    match_strength = tier_strength(match_division)
    current_strength = tier_strength(current_division)
    if match_strength <= 0 or current_strength <= 0:
        return value
    ratio = (match_strength / current_strength) if is_for else (current_strength / match_strength)
    return value * ratio


def _parse_date(d: str) -> datetime:
    # football-data.co.uk uses dd/mm/yy or dd/mm/yyyy depending on era
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            continue
    return datetime.min


def build_team_match_log(rows: list[dict]) -> dict[str, list[dict]]:
    """
    Reshape match rows (one row per fixture) into one row per team-appearance,
    tagged with venue and the metrics relevant to that team. Sorted oldest
    to newest per team.

    Each entry: {date, venue, opponent, div, goals_for, goals_against,
                 corners_for, corners_against, cards_for, cards_against,
                 first_half_goals_for, first_half_goals_against,
                 second_half_goals_for, second_half_goals_against,
                 xg_for, xg_against}

    "div" is which division this specific match was played in (e.g.
    "E1") - needed so rolling_form can tell when a match in a team's
    window came from a different division than the one we're currently
    judging their form against (see TIER_STRENGTH above).

    xg_for/xg_against are None when the source row had no HxG/AxG value
    (not every league/season has it) - downstream code must handle that,
    never assume it's always present.
    """
    log: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        try:
            dt = _parse_date(row["Date"])
            home, away = row["HomeTeam"], row["AwayTeam"]
            div = row.get("Div", "")

            hg, ag = int(row["FTHG"]), int(row["FTAG"])
            hc, ac = int(row.get("HC") or 0), int(row.get("AC") or 0)
            hy, ay = int(row.get("HY") or 0), int(row.get("AY") or 0)
            hr, ar = int(row.get("HR") or 0), int(row.get("AR") or 0)
            h_cards, a_cards = hy + 2 * hr, ay + 2 * ar  # weighted card count (red=2)

            # HTHG/HTAG = goals at half-time = first-half goals directly.
            # Second-half goals = full-time total minus that.
            hth, ath = int(row.get("HTHG") or 0), int(row.get("HTAG") or 0)
            h_2nd, a_2nd = hg - hth, ag - ath
        except (ValueError, KeyError):
            continue  # skip malformed / incomplete rows (e.g. postponed games)

        try:
            h_xg = float(row.get("HxG") or "")
            a_xg = float(row.get("AxG") or "")
        except ValueError:
            h_xg = a_xg = None  # not every match has xG - leave as None, not 0

        log[home].append({
            "date": dt, "venue": "H", "opponent": away, "div": div,
            "goals_for": hg, "goals_against": ag,
            "corners_for": hc, "corners_against": ac,
            "cards_for": h_cards, "cards_against": a_cards,
            "first_half_goals_for": hth, "first_half_goals_against": ath,
            "second_half_goals_for": h_2nd, "second_half_goals_against": a_2nd,
            "xg_for": h_xg, "xg_against": a_xg,
            "referee": row.get("Referee", ""),
        })
        log[away].append({
            "date": dt, "venue": "A", "opponent": home, "div": div,
            "goals_for": ag, "goals_against": hg,
            "corners_for": ac, "corners_against": hc,
            "cards_for": a_cards, "cards_against": h_cards,
            "first_half_goals_for": ath, "first_half_goals_against": hth,
            "second_half_goals_for": a_2nd, "second_half_goals_against": h_2nd,
            "xg_for": a_xg, "xg_against": h_xg,
            "referee": row.get("Referee", ""),
        })

    for team in log:
        log[team].sort(key=lambda r: r["date"])

    return log


# How much each match's weight decays per position back from the most
# recent one in the window - e.g. with WEIGHT_DECAY=0.85, the most recent
# match counts 1.0, the one before it 0.85, the one before that 0.7225,
# and so on. This is decay by RECENCY RANK (how many games back), not by
# calendar date - deliberately, since a team's rolling window blends in
# last season's matches early in a new campaign (see build_team_match_log
# notes), and pure calendar-day decay would treat "10 months ago, last
# game of last season" the same as "10 months ago mid-season", which
# isn't the distinction that actually matters. Rank-based decay means
# this season's matches - being the most recent by definition - dominate
# the average as soon as there are any, without needing to explicitly
# know or check which season a match came from.
WEIGHT_DECAY = 0.85

# On top of the value rescaling above, a cross-division match's WEIGHT in
# the average is ALSO reduced by this factor (only for TIER_ADJUSTED_METRICS
# - corners/cards keep their normal recency weight). Added after a real
# case exposed the gap: value-only rescaling wasn't enough when a team
# has very few current-division matches so far - a relegated team who
# started brilliantly in their new (weaker) division still had their
# excellent recent form diluted by several rescaled-but-still-numerous
# old-division matches carrying full weight. Cross-division matches
# should count for less, not just be worth less.
CROSS_DIVISION_WEIGHT_PENALTY = 0.4


def rolling_form(team_log: list[dict], venue: str | None = None, window: int = 10,
                  current_division: str | None = None) -> dict:
    """
    Recency-weighted average of the last `window` matches for a team
    (optionally filtered to venue='H' or 'A' only) - the most recent
    match counts most, decaying by WEIGHT_DECAY per position further
    back. Returns per-90 weighted averages for goals/corners/cards, both
    for and against.

    current_division: if given, any match in the window played in a
    DIFFERENT division gets its goals-related values tier-adjusted (see
    TIER_STRENGTH/_tier_adjusted_value above) AND its weight further
    reduced (see CROSS_DIVISION_WEIGHT_PENALTY above) before being
    folded into the average - this is what stops a promoted/relegated
    team's old-division form counting at anywhere near full strength.
    Corners/cards are never adjusted this way (neither value nor
    weight). Pass None to skip this entirely (raw values, plain
    recency weights, used as-is).

    If venue is set, we look back further in the full log to find
    `window` matches at that venue (not just the last N overall).
    """
    if venue:
        matches = [m for m in team_log if m["venue"] == venue][-window:]
    else:
        matches = team_log[-window:]

    n = len(matches)
    if n == 0:
        return {"matches": 0}

    # matches is oldest-to-newest; weight the LAST (most recent) one
    # highest. Reversed so base_weights[0] pairs with the most recent
    # match. This is the PLAIN recency weight, used as-is for
    # corners/cards and for goals when a match is in the current
    # division; goals-adjacent metrics additionally penalise
    # cross-division matches on top of this (see weighted_avg below).
    base_weights = [WEIGHT_DECAY ** i for i in range(n)]
    base_weights.reverse()

    def weighted_avg(key, metric):
        if metric in TIER_ADJUSTED_METRICS and current_division:
            is_for = key.endswith("_for")
            values, weights = [], []
            for m, w in zip(matches, base_weights):
                match_div = m.get("div", "")
                values.append(_tier_adjusted_value(m[key], is_for, match_div, current_division))
                weights.append(w * CROSS_DIVISION_WEIGHT_PENALTY if match_div != current_division else w)
        else:
            values, weights = [m[key] for m in matches], base_weights
        total_w = sum(weights)
        return round(sum(v * w for v, w in zip(values, weights)) / total_w, 2)

    result = {"matches": n}
    for metric in METRICS:
        result[f"{metric}_for"] = weighted_avg(f"{metric}_for", metric)
        result[f"{metric}_against"] = weighted_avg(f"{metric}_against", metric)

    # xG: only average over matches that actually HAD an xG value - not
    # every league/season has it, so mixing in absent values would
    # silently corrupt the average. Same recency weighting (and, when
    # applicable, cross-division weight penalty) applied, using only the
    # xG-having matches. Tier-adjusted the same way as goals, since xG is
    # itself a goals-adjacent quality measure. xg_matches tells the
    # caller how much to trust the figure (e.g. 2 games' worth of xG
    # isn't a lot to blend on).
    xg_pairs = [(m, w) for m, w in zip(matches, base_weights) if m["xg_for"] is not None and m["xg_against"] is not None]
    if xg_pairs:
        if current_division:
            xg_for_vals, xg_for_weights = [], []
            xg_against_vals, xg_against_weights = [], []
            for m, w in xg_pairs:
                match_div = m.get("div", "")
                penalty = CROSS_DIVISION_WEIGHT_PENALTY if match_div != current_division else 1.0
                xg_for_vals.append(_tier_adjusted_value(m["xg_for"], True, match_div, current_division))
                xg_for_weights.append(w * penalty)
                xg_against_vals.append(_tier_adjusted_value(m["xg_against"], False, match_div, current_division))
                xg_against_weights.append(w * penalty)
            result["xg_for"] = round(sum(v * w for v, w in zip(xg_for_vals, xg_for_weights)) / sum(xg_for_weights), 2)
            result["xg_against"] = round(sum(v * w for v, w in zip(xg_against_vals, xg_against_weights)) / sum(xg_against_weights), 2)
        else:
            xg_weight_total = sum(w for _, w in xg_pairs)
            result["xg_for"] = round(sum(m["xg_for"] * w for m, w in xg_pairs) / xg_weight_total, 2)
            result["xg_against"] = round(sum(m["xg_against"] * w for m, w in xg_pairs) / xg_weight_total, 2)
        result["xg_matches"] = len(xg_pairs)
    else:
        result["xg_for"] = result["xg_against"] = None
        result["xg_matches"] = 0

    return result


def league_averages(rows: list[dict]) -> dict:
    """League-wide per-match averages, used to normalise team strength in
    the Poisson model (an attack strength of 1.0 = league average)."""
    n = 0
    xg_n = 0
    totals = defaultdict(float)
    for row in rows:
        try:
            hg, ag = int(row["FTHG"]), int(row["FTAG"])
            hc, ac = int(row.get("HC") or 0), int(row.get("AC") or 0)
            hy, ay = int(row.get("HY") or 0), int(row.get("AY") or 0)
            hr, ar = int(row.get("HR") or 0), int(row.get("AR") or 0)
            hth, ath = int(row.get("HTHG") or 0), int(row.get("HTAG") or 0)
            h_2nd, a_2nd = hg - hth, ag - ath
        except (ValueError, KeyError):
            continue
        n += 1
        totals["home_goals"] += hg
        totals["away_goals"] += ag
        totals["home_corners"] += hc
        totals["away_corners"] += ac
        totals["home_cards"] += hy + 2 * hr
        totals["away_cards"] += ay + 2 * ar
        totals["home_first_half_goals"] += hth
        totals["away_first_half_goals"] += ath
        totals["home_second_half_goals"] += h_2nd
        totals["away_second_half_goals"] += a_2nd

        try:
            h_xg, a_xg = float(row.get("HxG") or ""), float(row.get("AxG") or "")
            totals["home_xg"] += h_xg
            totals["away_xg"] += a_xg
            xg_n += 1
        except ValueError:
            pass  # this match had no xG - just doesn't contribute to the xG average

    if n == 0:
        return {}

    result = {k: round(v / n, 3) for k, v in totals.items() if not k.endswith("_xg")}
    result |= {"matches": n}
    if xg_n > 0:
        result["home_xg"] = round(totals["home_xg"] / xg_n, 3)
        result["away_xg"] = round(totals["away_xg"] / xg_n, 3)
    result["xg_matches"] = xg_n
    return result


def team_form_summary(team_log: list[dict], window: int = 10, current_division: str | None = None) -> dict:
    """Convenience wrapper: overall / home / away rolling form for one
    team. current_division is passed through to rolling_form - see its
    docstring for what it does."""
    return {
        "overall": rolling_form(team_log, None, window, current_division),
        "home": rolling_form(team_log, "H", window, current_division),
        "away": rolling_form(team_log, "A", window, current_division),
    }
