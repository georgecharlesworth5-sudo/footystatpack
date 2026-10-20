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
TDs/rushing TDs/passing yards/rushing yards (home and away separately -
"team corners, goals, passing TD etc." from the original ask, NFL
side).
"""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from calibration import split_by_calibration

NEAR_TERM_WINDOW_DAYS = 0
MIN_CONFIDENCE = 0.87  # lowered from an earlier 0.9, kept in sync with best_bets.py's
                        # same change - the two lists get merged together on the dashboard,
                        # so this threshold should stay consistent across both sports.
MONEYLINE_MIN_CONFIDENCE = 0.75  # same reasoning as football's TEAM_WIN_MIN_CONFIDENCE -
                                  # moneyline is a 2-way market here (NFL ties are ~0.1%
                                  # of games), so 75% is a comparably strong edge to a
                                  # coin-flip O/U line - unaffected by this change.

METRIC_LABELS = {
    "points": "Points",
    "passing_tds": "Passing TDs",
    "rushing_tds": "Rushing TDs",
    "passing_yards": "Passing Yards",
    "rushing_yards": "Rushing Yards",
}


def _parse_date(d: str):
    from datetime import datetime
    try:
        return datetime.strptime(d, "%d/%m/%Y").date()
    except (ValueError, TypeError):
        return None

# Same rule as the Today page: a game kicking off after midnight and before
# 6am UK time (US evening games, mostly) counts as part of the previous day.
LONDON = ZoneInfo("Europe/London")

# Picks stay on Best Bets for this many hours after kick-off, so they don't
# vanish the moment a game starts. They're flagged "started" and are never
# logged to the Track Record (see track_bets.py) - only picks seen BEFORE
# kick-off count.
POST_KICKOFF_GRACE_HOURS = 5
LATE_NIGHT_CUTOFF_HOUR = 6


def _late_night_next_day(fx_date, fx_time: str, window_end) -> bool:
    """True for a fixture dated the day after the window that kicks off
    before the cutoff hour (e.g. 00:30 UK on the 15th still belongs to the
    14th's Best Bets, matching the Today tab)."""
    if not fx_date or fx_date != window_end + timedelta(days=1):
        return False
    try:
        return int(str(fx_time).split(":")[0]) < LATE_NIGHT_CUTOFF_HOUR
    except (ValueError, IndexError):
        return False


def _kickoff_time(fx: dict):
    """The fixture's kick-off as an aware UK datetime, or None if unusable."""
    fx_date = _parse_date(fx.get("date", ""))
    if not fx_date:
        return None
    try:
        hh, mm = str(fx.get("time", "")).split(":")[:2]
        return datetime(fx_date.year, fx_date.month, fx_date.day, int(hh), int(mm), tzinfo=LONDON)
    except (ValueError, TypeError):
        return None         # no usable kick-off time: keep it


def _already_kicked_off(fx: dict, now) -> bool:
    """True once kick-off has passed."""
    if now is None:
        return False
    kickoff = _kickoff_time(fx)
    return kickoff is not None and kickoff <= now


def _past_grace(fx: dict, now) -> bool:
    """True once kick-off was more than POST_KICKOFF_GRACE_HOURS ago - the
    point a pick finally drops off Best Bets."""
    if now is None:
        return False
    kickoff = _kickoff_time(fx)
    return kickoff is not None and kickoff + timedelta(hours=POST_KICKOFF_GRACE_HOURS) <= now


def _eligible_pool(pack: dict, window_days: int = NEAR_TERM_WINDOW_DAYS, today: date | None = None, now=None) -> list[dict]:
    today = today or date.today()
    window_end = today + timedelta(days=window_days)
    pool = []
    for fx in pack.get("upcoming_fixtures", []):
        fx_date = _parse_date(fx.get("date", ""))
        if fx_date and fx_date > window_end and not _late_night_next_day(fx_date, fx.get("time", ""), window_end):
            continue
        if fx.get("low_sample"):
            continue
        if _past_grace(fx, now):
            continue
        pool.append({**fx, "started": _already_kicked_off(fx, now)})
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
        **({"started": True} if fx.get("started") else {}),
    }


def _bar_for(pick: dict) -> float:
    return MONEYLINE_MIN_CONFIDENCE if pick["metric"] == "moneyline" else MIN_CONFIDENCE


def compute_nfl_best_bets(pack: dict, window_days: int = NEAR_TERM_WINDOW_DAYS, today: date | None = None,
                          calibration: dict | None = None) -> list[dict]:
    """The picks to SHOW (see compute_nfl_best_bets_split)."""
    return compute_nfl_best_bets_split(pack, window_days, today, calibration)[0]


def compute_nfl_best_bets_split(pack: dict, window_days: int = NEAR_TERM_WINDOW_DAYS, today: date | None = None,
                                calibration: dict | None = None, now=None) -> tuple[list[dict], list[dict]]:
    """(shown, held_back) - same calibration split as best_bets.py: raw
    qualification first, then each pick's confidence is shifted by how
    its market has performed, and picks that no longer clear their bar
    are held back (still logged and measured, just not shown)."""
    # Real runs (no explicit date) drop games that have already kicked off.
    if now is None and today is None:
        now = datetime.now(LONDON)
    pool = _eligible_pool(pack, window_days, today, now)
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

    return split_by_calibration(picks, calibration, _bar_for)
