"""
best_bets.py

Computes the "Best Bets" list - every pick, across every league AND
sport, within a near-term window, that clears a high confidence bar.
A single FLAT list, not grouped by market/category - deliberately
restructured away from an earlier {"goals": {...}, "corners": {...},
...} shape, since the point of this list is "everything that qualifies
today", not "here's how goals picks are doing vs corners picks".
Covers match-level markets (e.g. "Over 2.5 Goals") AND team-level
splits (e.g. "Arsenal Over 4.5 Corners"), and both straight team-win
and BTTS as their own pick types within the same flat list.

Each pick is a dict with a uniform shape - see _make_pick() below -
regardless of which market or scope it came from, so nothing
downstream needs to special-case (match vs team-level, over vs
team-win, etc.) beyond reading a few common fields.

This used to live only in the dashboard's JavaScript, recalculated fresh
every time the page loaded. It's been ported here so there's ONE
authoritative version: the pipeline can now log exactly what was picked
(track_bets.py) and reconcile it against real results later, and the
dashboard just displays whatever Python computed rather than
recalculating it itself. If you change the ranking logic (window size,
exclusion rules, thresholds), this is the only place to change it -
dashboard.html's JS best-bets code has been removed.
"""

from datetime import date, timedelta

NEAR_TERM_WINDOW_DAYS = 0  # only fixtures happening TODAY

# A pick needs to clear this bar to count as a "best bet" at all.
MIN_CONFIDENCE = 0.9

# Separate, lower bar specifically for straight team-win picks - a
# lower threshold than MIN_CONFIDENCE is still a meaningful edge for a
# match-result bet, which is inherently a 3-way market (home/draw/away)
# rather than a coin-flip over/under line, so 75% here is comparably
# strong to 90% on a two-way market.
TEAM_WIN_MIN_CONFIDENCE = 0.75

METRIC_LABELS = {
    "goals": "Goals",
    "corners": "Corners",
    "cards": "Cards",
    "first_half_goals": "1st-Half Goals",
    "second_half_goals": "2nd-Half Goals",
}


def _parse_date(d: str):
    from datetime import datetime
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(d, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _eligible_pool(statpack: dict, window_days: int = NEAR_TERM_WINDOW_DAYS, today: date | None = None) -> list[dict]:
    """Every fixture, across all leagues, within the near-term window and
    NOT carrying a caution flag - cross-league or thin-sample fixtures
    are still excluded (those reflect genuine uncertainty in the
    prediction itself), but a fuzzy name match ("Name Check") is a
    DIFFERENT kind of flag - it's uncertain about which raw fixture
    name maps to which known team, not about the prediction's own
    reliability once that mapping is made. Deliberately included here
    since a correct name match still produces a perfectly good pick;
    the flag stays visible on the fixture card itself so it's still
    something worth a quick human glance before relying on it, just
    not a reason to hide it from Best Bets outright.
    Each entry is tagged with its league code/name."""
    today = today or date.today()
    window_end = today + timedelta(days=window_days)

    pool = []
    for code, league in statpack.get("generated_leagues", {}).items():
        for fx in league.get("upcoming_fixtures", []):
            fx_date = _parse_date(fx.get("date", ""))
            if fx_date and fx_date > window_end:
                continue
            if fx.get("cross_league_data") or fx.get("low_sample"):
                continue
            pool.append({**fx, "league_code": code, "league_name": league.get("league_name", code)})
    return pool


def _make_pick(fx: dict, metric: str, scope: str, team: str | None,
               direction: str, line, confidence: float, label: str) -> dict:
    """Uniform shape for every pick, regardless of sport/market/scope.

    sport: "football" (this module) or "nfl" (see nfl_best_bets.py) -
       lets the dashboard merge both lists and reconcile picks against
       the right results source later.
    scope: "match" (the combined/total market) or "home"/"away" (a
       team-level split, e.g. one side's own corner count).
    team: the specific team this pick concerns, for scope="home"/"away"
       or a team_win pick - None for a match-level pick.
    direction: "Over"/"Yes" for O/U-style picks, or the picked team's
       name for team_win (matches track_bets.py's existing pick_id
       scheme, where "direction" already doubles as a team name there).
    label: a ready-to-display string, computed once here rather than
       reconstructed differently in every place that shows a pick.
    """
    return {
        "sport": "football",
        "home_team": fx["home_team"],
        "away_team": fx["away_team"],
        "league_name": fx["league_name"],
        "league_code": fx["league_code"],
        "date": fx.get("date", ""),
        "time": fx.get("time", ""),
        "metric": metric,
        "scope": scope,
        "team": team,
        "direction": direction,
        "line": line,
        "confidence": round(confidence, 3),
        "label": label,
    }


def compute_best_bets(statpack: dict, window_days: int = NEAR_TERM_WINDOW_DAYS, today: date | None = None) -> list[dict]:
    """
    Returns a single flat list of every football pick clearing its
    threshold today - match-level over markets, team-level over splits
    (corners/goals/cards, wherever the model computes a home/away
    split), BTTS Yes, and straight team-win. Sorted highest-confidence
    first. No fixed length - as long (or short, or empty) as the
    genuinely strong picks for today happen to be.
    """
    pool = _eligible_pool(statpack, window_days, today)
    picks = []

    for fx in pool:
        preds = fx.get("predictions", {})

        for metric_key, metric_label in METRIC_LABELS.items():
            m = preds.get(metric_key)
            if not m:
                continue

            if m.get("over_under"):
                top_over = max(m["over_under"], key=lambda ou: ou["over"])
                if top_over["over"] >= MIN_CONFIDENCE:
                    label = f"{fx['home_team']} v {fx['away_team']} - Over {top_over['line']} {metric_label}"
                    picks.append(_make_pick(fx, metric_key, "match", None, "Over",
                                             top_over["line"], top_over["over"], label))

            # Team-level splits - only present for metrics the model
            # actually computes a home/away split for (currently
            # corners/goals/cards - see poisson_model.PER_SIDE_LINES).
            # Missing entirely for first_half_goals/second_half_goals,
            # which .get() below handles by simply finding nothing.
            for scope, side_key, team in (("home", "home_over_under", fx["home_team"]),
                                           ("away", "away_over_under", fx["away_team"])):
                side_ou = m.get(side_key)
                if not side_ou:
                    continue
                top_side = max(side_ou, key=lambda ou: ou["over"])
                if top_side["over"] >= MIN_CONFIDENCE:
                    label = f"{team} - Over {top_side['line']} {metric_label}"
                    picks.append(_make_pick(fx, metric_key, scope, team, "Over",
                                             top_side["line"], top_side["over"], label))

        g = preds.get("goals")
        if g and "btts_yes" in g and g["btts_yes"] >= MIN_CONFIDENCE:
            label = f"{fx['home_team']} v {fx['away_team']} - BTTS Yes"
            picks.append(_make_pick(fx, "btts", "match", None, "Yes", None, g["btts_yes"], label))

        mr = preds.get("goals", {}).get("match_result")
        if mr:
            if mr.get("home_win", 0) >= TEAM_WIN_MIN_CONFIDENCE:
                label = f"{fx['home_team']} to win"
                picks.append(_make_pick(fx, "team_win", "home", fx["home_team"],
                                         fx["home_team"], None, mr["home_win"], label))
            if mr.get("away_win", 0) >= TEAM_WIN_MIN_CONFIDENCE:
                label = f"{fx['away_team']} to win"
                picks.append(_make_pick(fx, "team_win", "away", fx["away_team"],
                                         fx["away_team"], None, mr["away_win"], label))

    picks.sort(key=lambda p: p["confidence"], reverse=True)
    return picks
