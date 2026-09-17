"""
nfl_best_bets.py

NFL equivalent of the football side's best_bets.py - same flat pick
shape (see best_bets.py's _make_pick docstring for the field meanings),
just "sport": "nfl" instead of "football" and scanning NFL's own
prediction shape (points/passing_tds/rushing_tds/moneyline) instead of
football's (goals/corners/cards/etc). Kept as a separate module rather
than merged into best_bets.py, since the two pipelines run on separate
schedules against separate data sources - the dashboard concatenates
both lists together for display, each computed independently.

Covers: moneyline (team to win, the NFL equivalent of team_win),
total points (match-level), and team-level splits for points/passing
TDs/rushing TDs (home and away separately - "team corners, goals,
passing TD etc." from the original ask, NFL side).
"""

from datetime import date, timedelta

NEAR_TERM_WINDOW_DAYS = 0
MIN_CONFIDENCE = 0.9
MONEYLINE_MIN_CONFIDENCE = 0.75  # same reasoning as football's TEAM_WIN_MIN_CONFIDENCE -
                                  # moneyline is a 2-way market here (NFL ties are ~0.1%
                                  # of games), so 75% is a comparably strong edge to 90%
                                  # on a true coin-flip O/U line.

METRIC_LABELS = {
    "points": "Points",
    "passing_tds": "Passing TDs",
    "rushing_tds": "Rushing TDs",
}


def _parse_date(d: str):
    from datetime import datetime
    try:
        return datetime.strptime(d, "%d/%m/%Y").date()
    except (ValueError, TypeError):
        return None


def _eligible_pool(pack: dict, window_days: int = NEAR_TERM_WINDOW_DAYS, today: date | None = None) -> list[dict]:
    today = today or date.today()
    window_end = today + timedelta(days=window_days)
    pool = []
    for fx in pack.get("upcoming_fixtures", []):
        fx_date = _parse_date(fx.get("date", ""))
        if fx_date and fx_date > window_end:
            continue
        if fx.get("low_sample"):
            continue
        pool.append(fx)
    return pool


def _make_pick(fx: dict, metric: str, scope: str, team: str | None,
               direction: str, line, confidence: float, label: str) -> dict:
    return {
        "sport": "nfl",
        "home_team": fx["home_team"],
        "away_team": fx["away_team"],
        "league_name": "NFL",
        "league_code": "NFL",
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


def compute_nfl_best_bets(pack: dict, window_days: int = NEAR_TERM_WINDOW_DAYS, today: date | None = None) -> list[dict]:
    pool = _eligible_pool(pack, window_days, today)
    picks = []

    for fx in pool:
        preds = fx.get("predictions", {})

        ml = preds.get("points", {}).get("moneyline")
        if ml:
            if ml.get("home_win", 0) >= MONEYLINE_MIN_CONFIDENCE:
                label = f"{fx['home_team']} to win"
                picks.append(_make_pick(fx, "moneyline", "home", fx["home_team"],
                                         fx["home_team"], None, ml["home_win"], label))
            if ml.get("away_win", 0) >= MONEYLINE_MIN_CONFIDENCE:
                label = f"{fx['away_team']} to win"
                picks.append(_make_pick(fx, "moneyline", "away", fx["away_team"],
                                         fx["away_team"], None, ml["away_win"], label))

        for metric_key, metric_label in METRIC_LABELS.items():
            m = preds.get(metric_key)
            if not m:
                continue
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

        total_ou = preds.get("points", {}).get("total_over_under")
        if total_ou:
            top_total = max(total_ou, key=lambda ou: ou["over"])
            if top_total["over"] >= MIN_CONFIDENCE:
                label = f"{fx['home_team']} v {fx['away_team']} - Over {top_total['line']} Total Points"
                picks.append(_make_pick(fx, "points", "match", None, "Over",
                                         top_total["line"], top_total["over"], label))

    picks.sort(key=lambda p: p["confidence"], reverse=True)
    return picks
