"""
best_bets.py

Computes the "Best Bets" list - every Over/Under (and BTTS Yes/No) pick
across all leagues, within a near-term window, that clears a high
confidence bar. Unlike a "top N" ranking, the list here can be any
length - some gameweeks might have none, others several.

This used to live only in the dashboard's JavaScript, recalculated fresh
every time the page loaded. It's been ported here so there's ONE
authoritative version: the pipeline can now log exactly what was picked
(track_bets.py) and reconcile it against real results later, and the
dashboard just displays whatever Python computed rather than
recalculating it itself. If you change the ranking logic (window size,
exclusion rules, category list), this is the only place to change it -
dashboard.html's JS best-bets code has been removed.
"""

from datetime import date, timedelta

NEAR_TERM_WINDOW_DAYS = 0  # only fixtures happening TODAY - narrowed from an earlier
                           # 2-day window on request

# A pick needs to clear this bar to count as a "best bet" at all. Raised
# from an earlier 0.6 to 0.9 - the list is no longer a "top 5" ranking,
# it's every pick that clears the bar, so the bar itself needs to be
# high enough that everything shown is genuinely a strong signal, not
# just "the best of a mediocre bunch" the way a top-5 cap could tolerate.
MIN_CONFIDENCE = 0.9

# Separate, lower bar specifically for straight team-win picks (below) -
# a lower threshold than MIN_CONFIDENCE is still a meaningful edge for a
# match-result bet, which is inherently a 3-way market (home/draw/away)
# rather than a coin-flip over/under line, so 75% here is comparably
# strong to 90% on a two-way market.
TEAM_WIN_MIN_CONFIDENCE = 0.75

METRIC_LABELS = {
    "goals": "Full-Time Goals",
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
    NOT carrying a caution flag (thin-sample/cross-league/fuzzy-matched).
    Each entry is tagged with its league code/name."""
    today = today or date.today()
    window_end = today + timedelta(days=window_days)

    pool = []
    for code, league in statpack.get("generated_leagues", {}).items():
        for fx in league.get("upcoming_fixtures", []):
            fx_date = _parse_date(fx.get("date", ""))
            if fx_date and fx_date > window_end:
                continue
            if fx.get("cross_league_data") or fx.get("fuzzy_name_match") or fx.get("low_sample"):
                continue
            pool.append({**fx, "league_code": code, "league_name": league.get("league_name", code)})
    return pool


def compute_best_bets(statpack: dict, window_days: int = NEAR_TERM_WINDOW_DAYS, today: date | None = None) -> dict:
    """
    Returns {metric_key: {"over": [...], "under": [...]}} for every metric
    in METRIC_LABELS, plus "btts": {"over": [...yes...], "under": [...no...]}
    (named over/under for a consistent shape - "over" = Yes, "under" = No),
    plus "team_win": [...] - a flat list (not over/under shaped) of
    straight team-win picks clearing TEAM_WIN_MIN_CONFIDENCE.

    Every entry shown clears MIN_CONFIDENCE (or TEAM_WIN_MIN_CONFIDENCE
    for team_win) - there's no fixed count, the list is as long (or
    short, or empty) as the genuinely strong picks for that gameweek
    happen to be. Sorted highest-confidence first.

    Each over/under entry: {home_team, away_team, league_name, league_code,
                 date, time, direction, line (None for BTTS), confidence}
    Each team_win entry: {home_team, away_team, team, league_name,
                 league_code, date, time, confidence}
    """
    pool = _eligible_pool(statpack, window_days, today)
    categories = {}

    for metric_key in METRIC_LABELS:
        # Under markets deliberately not computed - not used, and no
        # point doing the work just to throw it away. "under" stays in
        # the output shape as an empty list so anything downstream that
        # expects the key to exist (e.g. dashboard code iterating over
        # it) doesn't break, it just always finds nothing there.
        best_over = []
        for fx in pool:
            m = fx.get("predictions", {}).get(metric_key)
            if not m or not m.get("over_under"):
                continue
            top_over = None
            for ou in m["over_under"]:
                if top_over is None or ou["over"] > top_over["confidence"]:
                    top_over = {"line": ou["line"], "direction": "Over", "confidence": ou["over"]}
            if top_over:
                best_over.append(_entry(fx, top_over))
        best_over.sort(key=lambda e: e["confidence"], reverse=True)
        best_over = [e for e in best_over if e["confidence"] >= MIN_CONFIDENCE]
        categories[metric_key] = {"over": best_over, "under": []}

    # BTTS "No" is the structural equivalent of an under market here -
    # same reasoning as above, not computed, kept as an empty list.
    btts_yes = []
    for fx in pool:
        g = fx.get("predictions", {}).get("goals")
        if not g or "btts_yes" not in g:
            continue
        btts_yes.append(_entry(fx, {"line": None, "direction": "Yes", "confidence": g["btts_yes"]}))
    btts_yes.sort(key=lambda e: e["confidence"], reverse=True)
    btts_yes = [e for e in btts_yes if e["confidence"] >= MIN_CONFIDENCE]
    categories["btts"] = {"over": btts_yes, "under": []}

    # Straight team-win picks - a different shape to everything else
    # above (a flat list, not {"over":[...], "under":[...]}), since this
    # isn't an over/under market at all - each entry says WHICH team
    # (home or away) is favoured, at what confidence. No opposite-
    # direction equivalent is computed (same reasoning as the removed
    # under markets - draws/the other team winning aren't wanted here).
    team_win_picks = []
    for fx in pool:
        mr = fx.get("predictions", {}).get("goals", {}).get("match_result")
        if not mr:
            continue
        if mr.get("home_win", 0) >= TEAM_WIN_MIN_CONFIDENCE:
            team_win_picks.append(_team_win_entry(fx, fx["home_team"], mr["home_win"]))
        if mr.get("away_win", 0) >= TEAM_WIN_MIN_CONFIDENCE:
            team_win_picks.append(_team_win_entry(fx, fx["away_team"], mr["away_win"]))
    team_win_picks.sort(key=lambda e: e["confidence"], reverse=True)
    categories["team_win"] = team_win_picks

    return categories


def _entry(fx: dict, pick: dict) -> dict:
    return {
        "home_team": fx["home_team"],
        "away_team": fx["away_team"],
        "league_name": fx["league_name"],
        "league_code": fx["league_code"],
        "date": fx.get("date", ""),
        "time": fx.get("time", ""),
        "direction": pick["direction"],
        "line": pick["line"],
        "confidence": round(pick["confidence"], 3),
    }


def _team_win_entry(fx: dict, team: str, confidence: float) -> dict:
    return {
        "home_team": fx["home_team"],
        "away_team": fx["away_team"],
        "team": team,
        "league_name": fx["league_name"],
        "league_code": fx["league_code"],
        "date": fx.get("date", ""),
        "time": fx.get("time", ""),
        "confidence": round(confidence, 3),
    }
