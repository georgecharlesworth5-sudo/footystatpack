"""
best_bets.py

Computes the "Best Bets" list - the highest-confidence Over and Under
(and BTTS Yes/No) picks across all leagues, within a near-term window.

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

NEAR_TERM_WINDOW_DAYS = 4  # one weekend round (Fri-Mon) - matches the dashboard's league-view default

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
    Returns {metric_key: {"over": [...top 5...], "under": [...top 5...]}}
    for every metric in METRIC_LABELS, plus "btts": {"over": [...yes...], "under": [...no...]}
    (named over/under for a consistent shape - "over" = Yes, "under" = No).

    Each entry: {home_team, away_team, league_name, league_code, date, time,
                 direction, line (None for BTTS), confidence}
    """
    pool = _eligible_pool(statpack, window_days, today)
    categories = {}

    for metric_key in METRIC_LABELS:
        best_over, best_under = [], []
        for fx in pool:
            m = fx.get("predictions", {}).get(metric_key)
            if not m or not m.get("over_under"):
                continue
            top_over, top_under = None, None
            for ou in m["over_under"]:
                if top_over is None or ou["over"] > top_over["confidence"]:
                    top_over = {"line": ou["line"], "direction": "Over", "confidence": ou["over"]}
                under_conf = 1 - ou["over"]
                if top_under is None or under_conf > top_under["confidence"]:
                    top_under = {"line": ou["line"], "direction": "Under", "confidence": under_conf}
            if top_over:
                best_over.append(_entry(fx, top_over))
            if top_under:
                best_under.append(_entry(fx, top_under))
        best_over.sort(key=lambda e: e["confidence"], reverse=True)
        best_under.sort(key=lambda e: e["confidence"], reverse=True)
        categories[metric_key] = {"over": best_over[:5], "under": best_under[:5]}

    btts_yes, btts_no = [], []
    for fx in pool:
        g = fx.get("predictions", {}).get("goals")
        if not g or "btts_yes" not in g:
            continue
        btts_yes.append(_entry(fx, {"line": None, "direction": "Yes", "confidence": g["btts_yes"]}))
        btts_no.append(_entry(fx, {"line": None, "direction": "No", "confidence": g["btts_no"]}))
    btts_yes.sort(key=lambda e: e["confidence"], reverse=True)
    btts_no.sort(key=lambda e: e["confidence"], reverse=True)
    categories["btts"] = {"over": btts_yes[:5], "under": btts_no[:5]}

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
