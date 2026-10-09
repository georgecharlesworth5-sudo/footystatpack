"""
track_bets.py

Keeps a running log of Best Bets picks and checks them against real
results once the matches have been played, so you can see an actual
hit rate rather than just trusting the model.

## Flat pick shape

best_bets.py and nfl_best_bets.py both return a single flat list of
picks (match-level AND team-level markets), each tagged with a "sport"
("football" or "nfl"). log_new_picks() iterates that flat list directly.

Each pick's scope ("match", "home", or "away") is part of both its
pick_id and how its actual result gets computed - a team-level pick
(e.g. "Arsenal Over 4.5 Corners") settles against ARSENAL's own corner
count, not the match total, whereas the exact same metric/direction at
scope="match" settles against the combined total.

## Settlement, per sport

  * Football settles against data/<LEAGUE>.csv (football-data.co.uk
    results). A pick matches the result row with the same home/away pair
    whose date is close to the pick's frozen match date - see
    _find_result_row for the window. That window is asymmetric on
    purpose: a postponed fixture is PLAYED LATER than the date it was
    logged under, so the window reaches much further forward than back.
    (It used to be a symmetric 14 days, which left postponed games that
    had since been played - e.g. Motherwell v Aberdeen, 22/08 -> 15/09 -
    sitting pending forever.)

  * NFL settles against data/nfl_team_games.csv (nflverse, one row per
    team per game). Pick team names are converted back to nflverse codes
    via TEAM_NAMES, the game is found by (home, away) + a small date
    window (picks carry UK dates, nflverse carries US dates, so a
    Thursday/Monday night game can be a day apart), and each market reads
    its own column: points / passing_tds / rushing_tds / passing_yards /
    rushing_yards, at match/home/away scope, plus moneyline. An NFL tie
    voids a moneyline pick (no result either way) rather than counting as
    a miss.

## Pick lifecycle
  1. The first time a pick's exact (sport, home, away, date, metric,
     scope, direction) combination appears in Best Bets, it's logged as
     "pending" with whatever line/confidence was showing that day - a
     FREEZE, not a moving target.
  2. Once the result is available, each run settles it as hit or miss.
     If the result isn't published yet, it stays pending and gets
     checked again next run.
  3. "void" = the pick never produced a real outcome (postponed and
     rescheduled under a new date, an NFL moneyline tie, or no result
     more than a week after the match date - see void_overdue_picks).
     Kept in the log as a record, excluded from every count. Overdue
     voids are still re-checked each run and settle if a result
     turns up late.

## What the dashboard gets (bets_log.js)

Beyond the headline hit rate, bets_log.js carries per-sport blocks
(overall / by market / by league / by confidence band / recent form), a
list of recent settled picks, and the pending picks grouped by match and
bucketed as:
    upcoming  - match date is today or later, nothing wrong
    awaiting  - played within the last week, result not in the feed yet
                (normal - football-data.co.uk lags, see the dashboard's
                freshness strip)
    overdue   - more than a week old and still unsettled: postponed, or
                a team-name mismatch. These are the ones worth a look.

## Held-back picks (calibration)

calibration.py shifts each pick's confidence by how its market has
actually performed; picks that no longer clear their bar are "held
back": logged here with shown="0" and settled like any other, but kept
out of the headline record, recent list and pending list. Their own hit
rate is reported as block["held_back"] - the out-of-sample check on the
filter. Per-market evidence (the break-even table) uses ALL picks.

Run this after build_statpack.py and after nfl_build_statpack.py - it
reads statpack.json's and nfl_statpack.json's "best_bets" lists as its
source of new picks. Both workflows run it.
"""

import csv
import json
from datetime import date, datetime
from pathlib import Path

from nfl_build_statpack import TEAM_NAMES as NFL_TEAM_NAMES

NFL_CODE_BY_NAME = {name: code for code, name in NFL_TEAM_NAMES.items()}
NFL_TEAM_METRICS = ("points", "passing_tds", "rushing_tds", "passing_yards", "rushing_yards")

from poisson_model import MODEL_VERSION

LOG_COLUMNS = [
    "pick_id", "logged_date", "match_date", "league_code", "league_name",
    "home_team", "away_team", "sport", "metric", "scope", "team",
    "direction", "line", "confidence", "label",
    "status", "actual_value", "result", "settled_date",
    "shown", "model",
]

# Football result matching window, in days relative to the pick's frozen
# match date: a result may be dated up to BACK days before it (small
# fixture-file date slips) or up to FORWARD days after it (postponed and
# since played). Used by BOTH reconcile_pending and reaudit_settled -
# they must agree, or settled picks would flip back to pending.
FOOTBALL_BACK_DRIFT_DAYS = 7
FOOTBALL_FORWARD_DRIFT_DAYS = 45

# NFL: picks carry UK dates, nflverse carries US dates. A late US game
# (Thursday/Monday night) lands on the NEXT UK day, so the pick date is
# 0-1 days after the game date. A little slack either side.
NFL_BACK_DRIFT_DAYS = 2   # pick date up to 2 days BEFORE the game date
NFL_FORWARD_DRIFT_DAYS = 3  # pick date up to 3 days AFTER the game date

# Pending picks played within this many days are "awaiting results"
# (feed lag, normal); older than that they're "overdue" (needs a look).
AWAITING_RESULT_DAYS = 7

RECENT_PICKS_LIMIT = 60

CONFIDENCE_BANDS = [
    (0.0, 0.75, "under 75%"),
    (0.75, 0.80, "75-80%"),
    (0.80, 0.85, "80-85%"),
    (0.85, 0.87, "85-87%"),
    (0.87, 0.90, "87-90%"),
    (0.90, 0.93, "90-93%"),
    (0.93, 0.96, "93-96%"),
    (0.96, 1.01, "96%+"),
]


def _parse_date(d: str):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(d, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _sport(pick: dict) -> str:
    return pick.get("sport") or "football"


def _shown(pick: dict) -> bool:
    """False for picks the calibration filter held back (logged and
    settled so their markets keep being measured, but never shown in
    Best Bets). Rows from before this column existed have no value -
    they were all shown."""
    return pick.get("shown") != "0"


def _counts_toward_record(pick: dict) -> bool:
    """best_bets.py doesn't generate "Under"/"No" picks any more, but
    older logged rows may still carry them - they're excluded from every
    hit rate and pending count, as before."""
    return pick["direction"] not in ("Under", "No")


def make_pick_id(sport: str, home: str, away: str, match_date: str, metric: str, scope: str, direction: str) -> str:
    return f"{sport}|{home}|{away}|{match_date}|{metric}|{scope}|{direction}"


def load_log(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {row["pick_id"]: row for row in csv.DictReader(f)}


def save_log(path: Path, log: dict[str, dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        for row in log.values():
            writer.writerow({col: row.get(col, "") for col in LOG_COLUMNS})


def log_new_picks(log: dict[str, dict], all_picks: list[dict], today: date, shown: bool = True) -> int:
    """Add any not-yet-seen pick from a flat picks list (the output of
    best_bets / nfl_best_bets - concatenate both sports before calling
    if tracking more than one). shown=False logs calibration-held-back
    picks, which are tracked and settled like any other but kept out of
    the record. The logged "confidence" is always the model's RAW
    probability (pick["raw_confidence"] when calibration has replaced
    "confidence" with its adjusted value), so calibration is always
    measured against the model's own output. Returns how many new rows
    were added."""
    added = 0
    for pick in all_picks:
        sport = pick.get("sport") or "football"
        pick_id = make_pick_id(sport, pick["home_team"], pick["away_team"],
                                pick["date"], pick["metric"], pick["scope"], pick["direction"])
        if pick_id in log:
            continue  # already logged - frozen, don't touch it again

        log[pick_id] = {
            "pick_id": pick_id,
            "logged_date": today.isoformat(),
            "match_date": pick["date"],
            "league_code": pick["league_code"],
            "league_name": pick["league_name"],
            "home_team": pick["home_team"],
            "away_team": pick["away_team"],
            "sport": sport,
            "metric": pick["metric"],
            "scope": pick["scope"],
            "team": pick.get("team") or "",
            "direction": pick["direction"],
            "line": pick["line"] if pick["line"] is not None else "",
            "confidence": pick.get("raw_confidence", pick["confidence"]),
            "label": pick.get("label", ""),
            "status": "pending",
            "actual_value": "",
            "result": "",
            "settled_date": "",
            "shown": "1" if shown else "0",
            # Football picks come from the versioned Poisson/NB model; blank
            # = logged before versioning (v1). NFL has its own model.
            "model": MODEL_VERSION if sport == "football" else "",
        }
        added += 1
    return added


# ---------------------------------------------------------------------
# Football settlement
# ---------------------------------------------------------------------

def _actual_metric_value(row: dict, metric: str, scope: str = "match"):
    """Returns the actual value for one football metric, at the given scope.

    scope="match": the combined/total value.
    scope="home"/"away": that ONE side's own value only - e.g. a pick
    on "Arsenal Over 4.5 Corners" settles against Arsenal's own corner
    count, not the match total, even though the metric ("corners") and
    direction ("Over") are identical to a match-level pick.

    team_win ignores scope entirely (it's already team-specific by
    construction) - returns the actual winning team's name, or None
    for a draw, which correctly falls out as a miss against any picked
    team's name.
    """
    try:
        fthg, ftag = int(row["FTHG"]), int(row["FTAG"])
        hthg, athg = int(row.get("HTHG") or 0), int(row.get("HTAG") or 0)
        hc, ac = int(row.get("HC") or 0), int(row.get("AC") or 0)
        hy, ay = int(row.get("HY") or 0), int(row.get("AY") or 0)
        hr, ar = int(row.get("HR") or 0), int(row.get("AR") or 0)
    except (ValueError, KeyError, TypeError):
        return None

    if metric == "team_win":
        if fthg > ftag:
            return row.get("HomeTeam")
        if ftag > fthg:
            return row.get("AwayTeam")
        return None

    home_cards, away_cards = hy + 2 * hr, ay + 2 * ar
    home_2h, away_2h = fthg - hthg, ftag - athg

    if scope == "home":
        return {"goals": fthg, "corners": hc, "cards": home_cards,
                "first_half_goals": hthg, "second_half_goals": home_2h}.get(metric)
    if scope == "away":
        return {"goals": ftag, "corners": ac, "cards": away_cards,
                "first_half_goals": athg, "second_half_goals": away_2h}.get(metric)
    return {"goals": fthg + ftag, "corners": hc + ac, "cards": home_cards + away_cards,
            "first_half_goals": hthg + athg, "second_half_goals": home_2h + away_2h,
            "btts": 1 if (fthg > 0 and ftag > 0) else 0}.get(metric)


def _check_hit(actual, direction: str, line) -> bool:
    if direction == "Over":
        return actual > line
    if direction == "Under":
        return actual < line
    if direction == "Yes":
        return actual == 1
    if direction == "No":
        return actual == 0
    # Anything else falling through here is a team_win/moneyline pick -
    # "direction" holds the picked team's name, "actual" holds the actual
    # winning team's name (or None for a draw). Hit only if they match.
    return actual is not None and actual == direction


def _load_football_results(data_dir: Path) -> dict[tuple[str, str], list[dict]]:
    results_by_pair: dict[tuple[str, str], list[dict]] = {}
    for csv_path in data_dir.glob("*.csv"):
        if csv_path.name.startswith("nfl_") or csv_path.name == "fixture_odds.csv":
            continue  # not football results (NFL has its own loader; odds have no FTHG)
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if not row.get("FTHG") or not row.get("HomeTeam"):
                    continue
                key = (row["HomeTeam"], row["AwayTeam"])
                results_by_pair.setdefault(key, []).append(row)
    return results_by_pair


def _find_result_row(candidates: list[dict], match_date: date) -> dict | None:
    """The result row for this pairing closest to the pick's frozen match
    date, within [match_date - BACK, match_date + FORWARD]. The window is
    wider forward than back because the failure it exists for is a
    postponement (played later than scheduled), while still excluding the
    previous season's meeting of the same two teams."""
    row, best_drift = None, None
    for candidate in candidates:
        candidate_date = _parse_date(candidate.get("Date", ""))
        if candidate_date is None:
            continue
        delta = (candidate_date - match_date).days
        if -FOOTBALL_BACK_DRIFT_DAYS <= delta <= FOOTBALL_FORWARD_DRIFT_DAYS:
            drift = abs(delta)
            if best_drift is None or drift < best_drift:
                row, best_drift = candidate, drift
    return row


def _football_outcome(pick: dict, results_by_pair: dict, match_date: date):
    """Returns (found_row, actual, display_actual, result) - result is
    "hit"/"miss", or None if the result row is missing or unusable."""
    candidates = results_by_pair.get((pick["home_team"], pick["away_team"]), [])
    row = _find_result_row(candidates, match_date)
    if row is None:
        return False, None, None, None

    actual = _actual_metric_value(row, pick["metric"], pick.get("scope", "match"))
    if actual is None and pick["metric"] != "team_win":
        return True, None, None, None

    line = float(pick["line"]) if pick["line"] not in ("", None) else None
    result = "hit" if _check_hit(actual, pick["direction"], line) else "miss"
    return True, actual, (actual if actual is not None else "Draw"), result


# ---------------------------------------------------------------------
# NFL settlement
# ---------------------------------------------------------------------

def _load_nfl_games(data_dir: Path) -> dict[tuple[str, str], list[dict]]:
    """nfl_team_games.csv has one row per team per game. Pairs the two
    rows of each game up and indexes completed games by
    (home_code, away_code) -> [{"date", "home": row, "away": row}]."""
    path = data_dir / "nfl_team_games.csv"
    if not path.exists():
        return {}

    by_game: dict[str, dict[str, dict]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_game.setdefault(row["game_id"], {})[row["venue"]] = row

    index: dict[tuple[str, str], list[dict]] = {}
    for sides in by_game.values():
        home, away = sides.get("H"), sides.get("A")
        if not home or not away:
            continue
        try:
            float(home["points_for"]), float(away["points_for"])
            game_date = datetime.strptime(home["gameday"], "%Y-%m-%d").date()
        except (ValueError, KeyError, TypeError):
            continue  # unplayed / malformed row - no result to settle against
        index.setdefault((home["team"], away["team"]), []).append(
            {"date": game_date, "home": home, "away": away})
    return index


def _find_nfl_game(pick: dict, nfl_games: dict, match_date: date) -> dict | None:
    home_code = NFL_CODE_BY_NAME.get(pick["home_team"])
    away_code = NFL_CODE_BY_NAME.get(pick["away_team"])
    if not home_code or not away_code:
        return None
    best, best_drift = None, None
    for game in nfl_games.get((home_code, away_code), []):
        delta = (match_date - game["date"]).days  # UK pick date minus US game date
        if -NFL_BACK_DRIFT_DAYS <= delta <= NFL_FORWARD_DRIFT_DAYS:
            drift = abs(delta)
            if best_drift is None or drift < best_drift:
                best, best_drift = game, drift
    return best


def _nfl_outcome(pick: dict, nfl_games: dict, match_date: date):
    """Returns (found, actual, display_actual, result) where result is
    "hit"/"miss"/"void" (tie on a moneyline), or None if no completed
    game was found."""
    game = _find_nfl_game(pick, nfl_games, match_date)
    if game is None:
        return False, None, None, None

    home, away = game["home"], game["away"]
    metric, scope = pick["metric"], pick.get("scope", "match")

    if metric == "moneyline":
        home_pts, away_pts = float(home["points_for"]), float(away["points_for"])
        if home_pts == away_pts:
            return True, None, "Tie", "void"
        winner = pick["home_team"] if home_pts > away_pts else pick["away_team"]
        return True, winner, winner, "hit" if _check_hit(winner, pick["direction"], None) else "miss"

    if metric not in NFL_TEAM_METRICS:
        return True, None, None, None

    column = f"{metric}_for"
    try:
        home_val, away_val = float(home[column]), float(away[column])
    except (ValueError, KeyError, TypeError):
        return True, None, None, None
    actual = home_val if scope == "home" else away_val if scope == "away" else home_val + away_val
    actual = int(actual) if float(actual).is_integer() else actual

    line = float(pick["line"]) if pick["line"] not in ("", None) else None
    return True, actual, actual, "hit" if _check_hit(actual, pick["direction"], line) else "miss"


# ---------------------------------------------------------------------
# Reconcile / audit / void
# ---------------------------------------------------------------------

def reconcile_pending(log: dict[str, dict], data_dir: Path, today: date) -> tuple[dict, dict]:
    """Try to settle every pending pick whose result is available.
    Returns ({"football": n, "nfl": n} settled this run,
             {"football": n, "nfl": n} past-due picks still waiting)."""
    football_results = _load_football_results(data_dir)
    nfl_games = _load_nfl_games(data_dir)

    settled = {"football": 0, "nfl": 0}
    waiting = {"football": 0, "nfl": 0}

    for pick in log.values():
        # Picks auto-voided for going overdue (void_overdue_picks) are
        # still re-checked: if the result feed catches up later, they
        # settle after all rather than being lost from the record.
        if pick["status"] != "pending" and not _voided_for_overdue(pick):
            continue
        sport = _sport(pick)
        match_date = _parse_date(pick["match_date"])
        if match_date is None:
            continue

        if sport == "nfl":
            # NFL data only ever contains COMPLETED games, so a found game
            # means it's been played - no need to wait for the UK date to
            # roll past (a Monday-night game is "today" until the small hours).
            if match_date > today:
                continue
            found, actual, display_actual, result = _nfl_outcome(pick, nfl_games, match_date)
        elif sport == "football":
            if match_date >= today:
                continue
            found, actual, display_actual, result = _football_outcome(pick, football_results, match_date)
        else:
            continue

        if result is None:
            if pick["status"] == "pending":
                waiting[sport] += 1
            continue

        pick["status"] = "void" if result == "void" else "settled"
        pick["actual_value"] = display_actual
        pick["result"] = "" if result == "void" else result
        pick["settled_date"] = today.isoformat()
        settled[sport] += 1

    return settled, waiting


def reaudit_settled(log: dict[str, dict], data_dir: Path) -> tuple[list[dict], list[dict]]:
    """Re-check every already-settled FOOTBALL pick against the current
    matching logic and data: fix any whose recorded actual/result no
    longer matches, and revert to pending any whose result row can no
    longer be found at all (rather than leave an unverifiable settlement
    standing). Uses the same _find_result_row window as reconcile_pending.

    NFL picks get the corrections half only - nflverse occasionally
    revises a stat after the fact, so a settled NFL pick is re-read, but
    one that can't be found is left as settled rather than reverted
    (the NFL file only ever grows, so a vanished game means a data hiccup,
    not a wrong settlement)."""
    football_results = _load_football_results(data_dir)
    nfl_games = _load_nfl_games(data_dir)

    corrections, reverted = [], []
    for pick in log.values():
        if pick["status"] != "settled":
            continue
        sport = _sport(pick)
        match_date = _parse_date(pick["match_date"])
        if match_date is None:
            continue

        if sport == "football":
            found, actual, display_actual, correct_result = _football_outcome(pick, football_results, match_date)
            if not found:
                reverted.append({
                    "pick_id": pick["pick_id"], "home_team": pick["home_team"], "away_team": pick["away_team"],
                    "metric": pick["metric"], "direction": pick["direction"], "line": pick["line"],
                    "old_actual": pick["actual_value"], "old_result": pick["result"],
                })
                pick["status"] = "pending"
                pick["actual_value"] = ""
                pick["result"] = ""
                pick["settled_date"] = ""
                continue
        elif sport == "nfl":
            found, actual, display_actual, correct_result = _nfl_outcome(pick, nfl_games, match_date)
            if not found:
                continue
        else:
            continue

        if correct_result is None or correct_result == "void":
            continue  # row found but unusable - leave the existing settlement alone

        old_actual, old_result = pick["actual_value"], pick["result"]
        if str(old_actual) != str(display_actual) or old_result != correct_result:
            corrections.append({
                "pick_id": pick["pick_id"], "home_team": pick["home_team"], "away_team": pick["away_team"],
                "metric": pick["metric"], "direction": pick["direction"], "line": pick["line"],
                "old_actual": old_actual, "new_actual": display_actual,
                "old_result": old_result, "new_result": correct_result,
            })
            pick["actual_value"] = display_actual
            pick["result"] = correct_result

    return corrections, reverted


def void_postponed_picks(log: dict[str, dict], statpack: dict) -> list[dict]:
    """
    A pick's match_date is frozen at the moment it's first logged. If the
    fixture is later postponed, that frozen date becomes permanently wrong
    - reconcile_pending will never find a result for it, and once the
    rescheduled fixture becomes eligible for Best Bets again,
    log_new_picks logs it as a brand new pick under the new date (match_date
    is part of the pick_id, so it's a different id). Left alone, the
    ORIGINAL entry sits pending forever.

    Detects this by checking every pending FOOTBALL pick's (home, away)
    against the CURRENT football fixture list: if that pairing still
    has an upcoming fixture but at a DIFFERENT date than what's frozen
    on the pick, the original has been superseded - void it (status="void",
    not deleted).

    (A postponed fixture that has ALREADY been played is handled by
    reconcile_pending's forward-looking result window instead - this
    function only deals with ones that haven't happened yet.)

    NFL picks are skipped - NFL has no equivalent fixture-list source
    wired in here.

    Returns the list of voided picks, for reporting. Doesn't save
    anything itself.
    """
    current_dates: dict[tuple[str, str], str] = {}
    for league in statpack.get("generated_leagues", {}).values():
        for fx in league.get("upcoming_fixtures", []):
            current_dates[(fx["home_team"], fx["away_team"])] = fx.get("date", "")

    voided = []
    for pick in log.values():
        if pick["status"] != "pending" or _sport(pick) != "football":
            continue
        current_date = current_dates.get((pick["home_team"], pick["away_team"]))
        if current_date and current_date != pick["match_date"]:
            voided.append({
                "pick_id": pick["pick_id"], "home_team": pick["home_team"], "away_team": pick["away_team"],
                "metric": pick["metric"], "direction": pick["direction"], "line": pick["line"],
                "old_match_date": pick["match_date"], "new_match_date": current_date,
            })
            pick["status"] = "void"

    return voided


OVERDUE_MARK = "overdue"  # stored in actual_value to tell an auto-voided pick apart from other voids


def _voided_for_overdue(pick: dict) -> bool:
    return pick["status"] == "void" and pick.get("actual_value") == OVERDUE_MARK


def void_overdue_picks(log: dict[str, dict], today: date) -> list[dict]:
    """Void every pending pick whose match date is more than
    AWAITING_RESULT_DAYS old with no result - almost always a
    postponement (international breaks etc.) that hasn't got a new date
    yet. Runs AFTER void_postponed_picks, which handles the case where
    the new date is already known.

    Voided picks leave every count, same as other voids. They're marked
    with actual_value="overdue", and reconcile_pending keeps re-checking
    them - so if it turns out the game WAS played and the results feed
    was just slow, the pick settles normally instead of being lost.
    Returns the voided picks, for reporting."""
    voided = []
    for pick in log.values():
        if pick["status"] != "pending":
            continue
        if _pending_bucket(_parse_date(pick["match_date"]), today) != "overdue":
            continue
        pick["status"] = "void"
        pick["actual_value"] = OVERDUE_MARK
        pick["settled_date"] = today.isoformat()
        voided.append(pick)
    return voided


# ---------------------------------------------------------------------
# Summaries for the dashboard
# ---------------------------------------------------------------------

def _pct(hits: int, total: int) -> int:
    return round(100 * hits / total) if total else 0


def _pending_bucket(match_date: date | None, today: date) -> str:
    if match_date is None:
        return "overdue"
    if match_date >= today:
        return "upcoming"
    return "awaiting" if (today - match_date).days <= AWAITING_RESULT_DAYS else "overdue"


def _category_breakdown(settled: list[dict]) -> list[dict]:
    """Per-(sport, metric, direction) hit rates. team_win/moneyline
    collapse into one category per sport regardless of which team was
    picked - direction there is a team name, not Over/Under.

    Team-level picks (scope="home"/"away") are grouped with their
    match-level counterpart under the same metric+direction."""
    by_category: dict[tuple, dict] = {}
    for p in settled:
        is_win = p["metric"] in ("team_win", "moneyline")
        direction = "" if is_win else p["direction"]
        key = (_sport(p), p["metric"], direction)
        entry = by_category.setdefault(key, {
            "sport": _sport(p), "metric": p["metric"], "direction": direction, "hits": 0, "total": 0})
        entry["total"] += 1
        if p["result"] == "hit":
            entry["hits"] += 1
    for c in by_category.values():
        c["pct"] = _pct(c["hits"], c["total"])
    return sorted(by_category.values(), key=lambda c: (-c["total"], c["metric"]))


def _league_breakdown(settled: list[dict]) -> list[dict]:
    by_league: dict[str, dict] = {}
    for p in settled:
        name = p["league_name"] or p["league_code"]
        entry = by_league.setdefault(name, {"league": name, "sport": _sport(p), "hits": 0, "total": 0})
        entry["total"] += 1
        if p["result"] == "hit":
            entry["hits"] += 1
    for e in by_league.values():
        e["pct"] = _pct(e["hits"], e["total"])
    return sorted(by_league.values(), key=lambda e: (-e["total"], e["league"]))


def _wilson_lower(hits: int, total: int, z: float = 1.96) -> float:
    """Lower bound of the 95% Wilson interval for a hit rate - the
    pessimistic read of a small sample ("it's hit 71% so far, but with
    this few picks the true rate could plausibly be as low as ...")."""
    if not total:
        return 0.0
    p = hits / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5)
    return (centre - margin) / denom


def _market_breakdown(settled: list[dict]) -> list[dict]:
    """Per-MARKET hit rates with the odds needed to break even.

    A market here is one specific bet type: (sport, metric, match-vs-team,
    direction, line) - e.g. "Match Goals Over 1.5" or "Team Corners Over
    2.5" (home and away pooled; they share lines). Moneyline/team_win
    collapse to "team to win".

    breakeven_odds = 1 / hit rate: the decimal price at which flat
    staking on this market so far would have broken even. Anything
    above it would have been profitable, anything below a loss.
    safe_odds does the same with the pessimistic (Wilson 95% lower)
    hit rate, so a market with a handful of lucky picks doesn't look
    cheaper than it really is. Neither knows the actual price - the
    point is a bar to compare real odds against."""
    markets: dict[tuple, dict] = {}
    for p in settled:
        is_win = p["metric"] in ("team_win", "moneyline")
        level = "team" if (is_win or p["scope"] in ("home", "away")) else "match"
        line = "" if is_win else p["line"]
        direction = "" if is_win else p["direction"]
        key = (_sport(p), p["metric"], level, direction, line)
        entry = markets.setdefault(key, {
            "sport": _sport(p), "metric": p["metric"], "level": level,
            "direction": direction, "line": line, "hits": 0, "total": 0, "conf_sum": 0.0})
        entry["total"] += 1
        entry["conf_sum"] += float(p["confidence"] or 0)
        if p["result"] == "hit":
            entry["hits"] += 1

    out = []
    for m in markets.values():
        rate = m["hits"] / m["total"]
        lower = _wilson_lower(m["hits"], m["total"])
        out.append({
            "sport": m["sport"], "metric": m["metric"], "level": m["level"],
            "direction": m["direction"], "line": m["line"],
            "hits": m["hits"], "total": m["total"], "pct": _pct(m["hits"], m["total"]),
            "avg_confidence": round(100 * m["conf_sum"] / m["total"]),
            "breakeven_odds": round(1 / rate, 2) if rate else None,
            "safe_odds": round(1 / lower, 2) if lower else None,
        })
    return sorted(out, key=lambda m: (-m["total"], m["metric"]))


def _calibration(settled: list[dict]) -> list[dict]:
    """Does "93% confident" actually win ~93% of the time? For each
    confidence band: how many picks, the model's average stated
    confidence, and the actual hit rate. A big gap between the last two
    is the model being over- or under-confident in that band."""
    out = []
    for low, high, label in CONFIDENCE_BANDS:
        in_band = []
        for p in settled:
            try:
                conf = float(p["confidence"])
            except (ValueError, TypeError):
                continue
            if low <= conf < high:
                in_band.append((conf, p["result"] == "hit"))
        if not in_band:
            continue
        hits = sum(1 for _, h in in_band if h)
        out.append({
            "band": label, "total": len(in_band), "hits": hits,
            "pct": _pct(hits, len(in_band)),
            "avg_confidence": round(100 * sum(c for c, _ in in_band) / len(in_band)),
        })
    return out


def _conf_slice(picks: list[dict], low: float, high: float) -> dict:
    sub = []
    for p in picks:
        try:
            if low <= float(p["confidence"]) < high:
                sub.append(p)
        except (ValueError, TypeError):
            continue
    hits = sum(1 for p in sub if p["result"] == "hit")
    confs = [float(p["confidence"]) for p in sub]
    rate = hits / len(sub) if sub else 0
    return {
        "total": len(sub), "hits": hits, "pct": _pct(hits, len(sub)),
        "avg_confidence": round(100 * sum(confs) / len(confs)) if confs else 0,
        "breakeven_odds": round(1 / rate, 2) if rate else None,
        "markets": _market_breakdown(sub),
    }


def _at_80(picks: list[dict]) -> dict:
    """The 80% check, for judging the 80% bar on its own. "all" is every
    settled pick rated 80%+; "low" is just the 80-87% slice - the picks
    that only exist because some markets have a bar below the default
    87% (it's where an overconfident model would show first). Shown and
    held-back picks both count. Each carries its break-even odds, and
    "low" a per-market split."""
    return {"all": _conf_slice(picks, 0.80, 1.01), "low": _conf_slice(picks, 0.80, 0.87)}


def _form(settled: list[dict], today: date) -> dict:
    """Hit rate over the last 7 / 30 days, by match date."""
    out = {}
    for days in (7, 30):
        recent = [p for p in settled
                  if (d := _parse_date(p["match_date"])) is not None and 0 <= (today - d).days < days]
        hits = sum(1 for p in recent if p["result"] == "hit")
        out[f"last_{days}"] = {"hits": hits, "total": len(recent), "pct": _pct(hits, len(recent))}
    return out


def _block(picks: list[dict], today: date) -> dict:
    """Everything the dashboard shows for one slice (all / football / nfl)."""
    counted = [p for p in picks if _counts_toward_record(p) and _shown(p)]
    settled = [p for p in counted if p["status"] == "settled"]
    hits = sum(1 for p in settled if p["result"] == "hit")

    # The record above is only picks that were SHOWN. The held-back ones
    # (calibration filter) are tracked separately - how they do is the
    # out-of-sample test of whether the filter is removing the right picks.
    held = [p for p in picks if _counts_toward_record(p) and not _shown(p) and p["status"] == "settled"]
    held_hits = sum(1 for p in held if p["result"] == "hit")

    pending = {"upcoming": 0, "awaiting": 0, "overdue": 0}
    for p in counted:
        if p["status"] == "pending":
            pending[_pending_bucket(_parse_date(p["match_date"]), today)] += 1

    return {
        "overall": {"hits": hits, "total": len(settled), "pct": _pct(hits, len(settled))},
        "form": _form(settled, today),
        "by_category": _category_breakdown(settled),
        # Markets use every settled pick, shown or held back - this is
        # evidence about the market itself, not about the filter.
        "markets": _market_breakdown([p for p in picks if _counts_toward_record(p) and p["status"] == "settled"]),
        "held_back": {"hits": held_hits, "total": len(held), "pct": _pct(held_hits, len(held))},
        "leagues": _league_breakdown(settled),
        "calibration": _calibration(settled),
        "at_80": _at_80([p for p in picks if _counts_toward_record(p) and p["status"] == "settled"]),
        "pending": {**pending, "total": sum(pending.values())},
    }


def _fallback_label(p: dict) -> str:
    """Legacy rows logged before labels existed."""
    line = f" {p['line']}" if p["line"] not in ("", None) else ""
    return f"{p['home_team']} v {p['away_team']} - {p['direction']}{line} {p['metric'].replace('_', ' ')}"


def _recent_picks(log: dict[str, dict]) -> list[dict]:
    settled = [p for p in log.values()
               if p["status"] == "settled" and _counts_toward_record(p) and _shown(p) and _parse_date(p["match_date"])]
    settled.sort(key=lambda p: (_parse_date(p["match_date"]), float(p["confidence"] or 0)), reverse=True)
    return [{
        "sport": _sport(p),
        "date": _parse_date(p["match_date"]).isoformat(),
        "match": f"{p['home_team']} v {p['away_team']}",
        "label": p["label"] or _fallback_label(p),
        "league": p["league_name"],
        "metric": p["metric"],
        "direction": p["direction"],
        "line": p["line"],
        "actual": p["actual_value"],
        "result": p["result"],
        "confidence": float(p["confidence"] or 0),
    } for p in settled[:RECENT_PICKS_LIMIT]]


def _pending_matches(log: dict[str, dict], today: date) -> list[dict]:
    """Pending picks grouped by match (a single game can carry a dozen
    picks - listing 116 individual rows would bury the point). Overdue
    first, since those are the ones that need attention."""
    groups: dict[tuple, dict] = {}
    for p in log.values():
        if p["status"] != "pending" or not _counts_toward_record(p) or not _shown(p):
            continue
        match_date = _parse_date(p["match_date"])
        key = (_sport(p), p["home_team"], p["away_team"], p["match_date"])
        group = groups.setdefault(key, {
            "sport": _sport(p),
            "date": match_date.isoformat() if match_date else "",
            "match": f"{p['home_team']} v {p['away_team']}",
            "league": p["league_name"],
            "picks": 0,
            "bucket": _pending_bucket(match_date, today),
            "days_ago": (today - match_date).days if match_date else None,
        })
        group["picks"] += 1
    order = {"overdue": 0, "awaiting": 1, "upcoming": 2}
    return sorted(groups.values(), key=lambda g: (order[g["bucket"]], g["date"], g["match"]))


def compute_summary(log: dict[str, dict], today: date | None = None) -> dict:
    """Everything bets_log.js carries.

    "summary" keeps its original shape (overall / by_category /
    pending_count, now across BOTH sports) so anything still reading
    BETS_LOG.summary keeps working - the tab label does."""
    today = today or date.today()
    picks = list(log.values())

    blocks = {
        "all": _block(picks, today),
        "football": _block([p for p in picks if _sport(p) == "football"], today),
        "nfl": _block([p for p in picks if _sport(p) == "nfl"], today),
    }
    logged_dates = [p["logged_date"] for p in picks if p.get("logged_date")]

    return {
        "generated": today.isoformat(),
        "logging_since": min(logged_dates) if logged_dates else "",
        "summary": {
            "overall": blocks["all"]["overall"],
            "by_category": blocks["all"]["by_category"],
            "pending_count": blocks["all"]["pending"]["total"],
        },
        "blocks": blocks,
        "recent": _recent_picks(log),
        "pending_matches": _pending_matches(log, today),
    }


if __name__ == "__main__":
    base = Path(__file__).parent
    data_dir = base / "data"
    log_path = base / "bets_log.csv"
    statpack_path = base / "statpack.json"
    nfl_statpack_path = base / "nfl_statpack.json"

    if not statpack_path.exists():
        raise SystemExit("statpack.json not found - run build_statpack.py first.")

    with open(statpack_path) as f:
        statpack = json.load(f)

    today = date.today()
    log = load_log(log_path)

    # Both sports' best_bets lists are the SAME flat pick shape (see
    # best_bets.py / nfl_best_bets.py), tagged "sport" precisely so they
    # can be logged together here. nfl_statpack.json is written by the
    # separate NFL workflow to the same repo, so it's read here if
    # present; its absence isn't an error (e.g. the NFL workflow hasn't
    # run yet on a fresh repo) - football logging still proceeds.
    #
    # "shadow_picks" are picks that cleared the model's raw bar but not
    # the calibrated one (see calibration.py) - logged as shown="0" so
    # their markets keep being measured, never shown in Best Bets.
    picks = list(statpack.get("best_bets", []))
    shadow = list(statpack.get("shadow_picks", []))
    if nfl_statpack_path.exists():
        with open(nfl_statpack_path) as f:
            nfl_statpack = json.load(f)
        picks.extend(nfl_statpack.get("best_bets", []))
        shadow.extend(nfl_statpack.get("shadow_picks", []))
        print(f"Including NFL picks: {len(picks)} shown / {len(shadow)} held back in total.")
    else:
        print("No nfl_statpack.json found - logging football picks only.")

    added = log_new_picks(log, picks, today)
    added_shadow = log_new_picks(log, shadow, today, shown=False)
    print(f"Logged {added} new pick(s) + {added_shadow} held-back pick(s).")

    settled, waiting = reconcile_pending(log, data_dir, today)
    print(f"Settled {settled['football']} football + {settled['nfl']} NFL pick(s) this run; "
          f"{waiting['football']} football + {waiting['nfl']} NFL past-due pick(s) still waiting on results.")

    corrections, reverted = reaudit_settled(log, data_dir)
    if corrections:
        print(f"Corrected {len(corrections)} previously-settled pick(s):")
        for c in corrections:
            print(f"  {c['home_team']} v {c['away_team']} - {c['direction']} {c['line']} {c['metric']}: "
                  f"actual {c['old_actual']} -> {c['new_actual']}, result {c['old_result']} -> {c['new_result']}")
    if reverted:
        print(f"Reverted {len(reverted)} unverifiable settled pick(s) back to pending:")
        for r in reverted:
            print(f"  {r['home_team']} v {r['away_team']} - {r['direction']} {r['line']} {r['metric']} "
                  f"(was: actual {r['old_actual']}, result {r['old_result']})")

    voided = void_postponed_picks(log, statpack)
    if voided:
        print(f"Voided {len(voided)} pick(s) whose fixture was postponed/rescheduled:")
        for v in voided:
            print(f"  {v['home_team']} v {v['away_team']} - {v['direction']} {v['line']} {v['metric']} "
                  f"(was frozen at {v['old_match_date']}, now scheduled {v['new_match_date']})")

    overdue_voided = void_overdue_picks(log, today)
    if overdue_voided:
        matches = {(p["home_team"], p["away_team"], p["match_date"]) for p in overdue_voided}
        print(f"Voided {len(overdue_voided)} pick(s) across {len(matches)} match(es) for going overdue "
              f"(no result {AWAITING_RESULT_DAYS}+ days after the match date - will still settle if a result turns up):")
        for h, a, d in sorted(matches):
            print(f"  {h} v {a} ({d})")

    save_log(log_path, log)
    print(f"Log saved to {log_path} ({len(log)} total picks).")

    summary = compute_summary(log, today)
    for name, block in summary["blocks"].items():
        o, p = block["overall"], block["pending"]
        print(f"  {name:<9} {o['hits']}/{o['total']} ({o['pct']}%) settled | pending: "
              f"{p['upcoming']} upcoming, {p['awaiting']} awaiting results, {p['overdue']} overdue")

    js_path = base / "bets_log.js"
    with open(js_path, "w") as f:
        f.write("const BETS_LOG = ")
        json.dump(summary, f, default=str)
        f.write(";\n")
    print(f"Dashboard log data written to {js_path}")
