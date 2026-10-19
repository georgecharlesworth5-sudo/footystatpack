"""
nba_best_bets.py

NBA equivalent of nfl_best_bets.py - same flat pick shape, "sport": "nba".

SHADOW MODE: NBA is new, so for its first few weeks every pick it
qualifies is logged and settled (so the Track Record measures it) but NONE
is shown in Best Bets. Flip SHADOW_MODE to False once the hit rates have
earned it (see the NBA block on the Track Record tab).

Markets: moneyline, spread (handicap), game total, team totals, and
player props (points / rebounds / assists / threes, Over only).

There are no bookmaker prices, so the line choice is the model's own:
for each market the pick is the line whose probability JUST clears the
bar - i.e. the best price that still meets the confidence standard.
(Choosing the safest line instead would give 99% picks at 1.01 odds.)
Compare the line shown with what your bookmaker offers: a bookmaker line
equal to or easier than ours is a bet the model likes.

Player props are "if plays" - the model has no injury feed, so check the
player is in the team news before betting.

To keep the daily volume sensible there's at most one pick per market per
fixture (two for team totals - one per team) and PROPS_PER_FIXTURE props,
chosen from the highest-minutes players (the steadiest numbers).
"""
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

SHADOW_MODE = True

NEAR_TERM_WINDOW_DAYS = 0
MONEYLINE_MIN = 0.75
SPREAD_MIN = 0.80
TOTAL_MIN = 0.80
PROP_MIN = 0.85
PROPS_PER_FIXTURE = 3

PROP_LABELS = {"points": "Points", "rebounds": "Rebounds", "assists": "Assists", "threes": "Threes"}

LONDON = ZoneInfo("Europe/London")
POST_KICKOFF_GRACE_HOURS = 5
LATE_NIGHT_CUTOFF_HOUR = 6     # tip-offs before 06:00 UK belong to the previous day


def _parse_date(d: str):
    try:
        return datetime.strptime(d, "%d/%m/%Y").date()
    except (ValueError, TypeError):
        return None


def _late_night_next_day(fx_date, fx_time: str, window_end) -> bool:
    if not fx_date or fx_date != window_end + timedelta(days=1):
        return False
    try:
        return int(str(fx_time).split(":")[0]) < LATE_NIGHT_CUTOFF_HOUR
    except (ValueError, IndexError):
        return False


def _kickoff_time(fx: dict):
    d = _parse_date(fx.get("date", ""))
    if not d:
        return None
    try:
        hh, mm = str(fx.get("time", "")).split(":")[:2]
        return datetime(d.year, d.month, d.day, int(hh), int(mm), tzinfo=LONDON)
    except (ValueError, TypeError):
        return None


def _eligible_pool(pack: dict, window_days: int, today, now) -> list[dict]:
    today = today or date.today()
    window_end = today + timedelta(days=window_days)
    pool = []
    for fx in pack.get("upcoming_fixtures", []):
        d = _parse_date(fx.get("date", ""))
        if d and d > window_end and not _late_night_next_day(d, fx.get("time", ""), window_end):
            continue
        k = _kickoff_time(fx)
        if now is not None and k is not None and k + timedelta(hours=POST_KICKOFF_GRACE_HOURS) <= now:
            continue
        pool.append({**fx, "started": bool(now is not None and k is not None and k <= now)})
    return pool


def _make_pick(fx: dict, metric: str, scope: str, team, direction: str, line, confidence: float, label: str) -> dict:
    return {
        "sport": "nba",
        "home_team": fx["home_team"],
        "away_team": fx["away_team"],
        "league_name": "NBA",
        "league_code": "NBA",
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


def _fmt(line: float) -> str:
    return f"{line:+g}"


def _best_price(rows: list[dict], key: str, bar: float):
    """The row whose probability (rows[key]) is lowest while still >= bar."""
    ok = [r for r in rows if r[key] >= bar]
    return min(ok, key=lambda r: r[key]) if ok else None


def picks_for_fixture(fx: dict) -> list[dict]:
    preds = fx.get("predictions", {})
    home, away = fx["home_team"], fx["away_team"]
    out = []

    ml = preds.get("moneyline")
    if ml:
        for scope, team, key in (("home", home, "home_win"), ("away", away, "away_win")):
            if ml.get(key, 0) >= MONEYLINE_MIN:
                out.append(_make_pick(fx, "moneyline", scope, team, team, None, ml[key], f"{team} to win"))

    spread = preds.get("spread")
    if spread:
        # home handicap L covers with p = home_cover; the away side takes -L with 1 - p.
        rows_home = [{"line": r["line"], "p": r["home_cover"]} for r in spread]
        rows_away = [{"line": -r["line"], "p": 1 - r["home_cover"]} for r in spread]
        cands = []
        for scope, team, rows in (("home", home, rows_home), ("away", away, rows_away)):
            r = _best_price(rows, "p", SPREAD_MIN)
            if r:
                cands.append(_make_pick(fx, "spread", scope, team, "Cover", r["line"], r["p"],
                                        f"{team} {_fmt(r['line'])}"))
        if cands:                       # one spread pick per fixture: the more confident side
            out.append(max(cands, key=lambda p: p["confidence"]))

    total = preds.get("total")
    if total:
        cands = []
        for direction, key in (("Over", "over"),):          # Overs only - no Under picks
            r = _best_price(total, key, TOTAL_MIN)
            if r:
                cands.append(_make_pick(fx, "points", "match", None, direction, r["line"], r[key],
                                        f"{home} v {away} - {direction} {r['line']} Total Points"))
        if cands:
            out.append(max(cands, key=lambda p: p["confidence"]))

    for scope, team in (("home", home), ("away", away)):
        tt = preds.get("team_totals", {}).get(scope)
        if not tt:
            continue
        cands = []
        for direction, key in (("Over", "over"),):          # Overs only - no Under picks
            r = _best_price(tt, key, TOTAL_MIN)
            if r:
                cands.append(_make_pick(fx, "points", scope, team, direction, r["line"], r[key],
                                        f"{team} - {direction} {r['line']} Team Points"))
        if cands:
            out.append(max(cands, key=lambda p: p["confidence"]))

    prop_picks = []
    for pl in sorted(fx.get("props", []), key=lambda p: -p.get("mins", 0)):
        for stat, label in PROP_LABELS.items():
            rows = pl.get("lines", {}).get(stat)
            if not rows:
                continue
            r = _best_price(rows, "over", PROP_MIN)
            if r:
                prop_picks.append(_make_pick(
                    fx, stat, "p:" + pl["player"], pl["team"], "Over", r["line"], r["over"],
                    f"{pl['player']} - Over {r['line']} {label} (if plays)"))
        if len(prop_picks) >= PROPS_PER_FIXTURE * 3:
            break
    # steadiest first: the picks list is already in minutes order; keep the first few
    seen_players, kept = {}, []
    for p in prop_picks:
        if len(kept) >= PROPS_PER_FIXTURE:
            break
        if seen_players.get(p["scope"], 0) >= 1:     # one prop per player
            continue
        seen_players[p["scope"]] = 1
        kept.append(p)
    out.extend(kept)
    return out


def compute_nba_best_bets_split(pack: dict, window_days: int = NEAR_TERM_WINDOW_DAYS, today: date | None = None,
                                now=None) -> tuple[list[dict], list[dict]]:
    """(shown, held_back). In shadow mode everything is held back - still
    logged and settled by track_bets.py, never shown."""
    if now is None and today is None:
        now = datetime.now(LONDON)
    picks = []
    for fx in _eligible_pool(pack, window_days, today, now):
        picks.extend(picks_for_fixture(fx))
    if SHADOW_MODE:
        return [], picks
    return picks, []
