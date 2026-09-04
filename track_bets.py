"""
track_bets.py

Keeps a running log of Best Bets picks and checks them against real
results once the matches have been played, so you can see an actual
hit rate rather than just trusting the model.

How a pick's lifecycle works:
  1. The first time a (fixture, metric, direction) combination appears
     in Best Bets, it's logged as "pending" with whatever line/
     confidence was showing that day. This is a FREEZE, not a moving
     target - if the same combination is still in Best Bets days later
     with a slightly different number (form can shift a little before
     kickoff), we don't touch the logged row again. The log reflects
     what was actually shown at the time, not a constantly-updated
     target.
  2. Once the fixture's date has passed, each run tries to reconcile
     it: look up the actual match result in the results CSVs
     (data/<LEAGUE>.csv) and check whether the frozen pick's line/
     direction actually hit. If the result isn't in the data yet
     (football-data.co.uk hasn't published it), it stays pending and
     gets checked again on the next run.

Run this after build_statpack.py, since it reads statpack.json's
"best_bets" section (see best_bets.py) as its source of new picks.

## team_win picks

Every other category in best_bets.py has the same {"over": [...],
"under": [...]} shape. "team_win" doesn't - it's a flat list of straight
team-to-win picks, since it isn't an over/under market at all. Handled
as its own case throughout this file:
  - logged with metric="team_win" and direction=<the picked team's name>
    (not "Over"/"Under"/etc) - this still fits the existing
    (home, away, match_date, metric, direction) pick_id scheme cleanly,
    since a team name is just as valid a "direction" value for
    disambiguating picks.
  - settled by comparing the picked team's name against whichever team
    (if either) actually won - see _actual_metric_value and _check_hit.

Output: bets_log.csv (the durable log, committed to the repo) and
bets_log.js (a summary + recent entries, in the same embeddable-script
pattern as statpack_data.js, for the dashboard's Track Record tab).
"""

import csv
import json
from datetime import date, datetime
from pathlib import Path

from best_bets import METRIC_LABELS

LOG_COLUMNS = [
    "pick_id", "logged_date", "match_date", "league_code", "league_name",
    "home_team", "away_team", "metric", "direction", "line", "confidence",
    "status", "actual_value", "result", "settled_date",
]


def _parse_date(d: str):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(d, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def make_pick_id(home: str, away: str, match_date: str, metric: str, direction: str) -> str:
    return f"{home}|{away}|{match_date}|{metric}|{direction}"


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


def _make_pick_row(entry: dict, today: date, metric: str, direction: str, line) -> dict:
    return {
        "pick_id": make_pick_id(entry["home_team"], entry["away_team"], entry["date"], metric, direction),
        "logged_date": today.isoformat(),
        "match_date": entry["date"],
        "league_code": entry["league_code"],
        "league_name": entry["league_name"],
        "home_team": entry["home_team"],
        "away_team": entry["away_team"],
        "metric": metric,
        "direction": direction,
        "line": line if line is not None else "",
        "confidence": entry["confidence"],
        "status": "pending",
        "actual_value": "",
        "result": "",
        "settled_date": "",
    }


def log_new_picks(log: dict[str, dict], best_bets: dict, today: date) -> int:
    """Add any not-yet-seen (fixture, metric, direction) picks. Returns
    how many new rows were added."""
    added = 0
    for metric, sides in best_bets.items():
        if metric == "team_win":
            # Flat list, not {"over": [...], "under": [...]} like every
            # other category - each entry is a straight team-to-win pick,
            # so "direction" here is the picked team's name rather than
            # Over/Under/Yes/No.
            for entry in sides:
                row = _make_pick_row(entry, today, metric, entry["team"], None)
                if row["pick_id"] in log:
                    continue
                log[row["pick_id"]] = row
                added += 1
            continue

        for direction_list in (sides.get("over", []), sides.get("under", [])):
            for entry in direction_list:
                row = _make_pick_row(entry, today, metric, entry["direction"], entry["line"])
                if row["pick_id"] in log:
                    continue  # already logged - frozen, don't touch it again
                log[row["pick_id"]] = row
                added += 1
    return added


def _actual_metric_value(row: dict, metric: str):
    try:
        fthg, ftag = int(row["FTHG"]), int(row["FTAG"])
        hthg, athg = int(row.get("HTHG") or 0), int(row.get("HTAG") or 0)
        hc, ac = int(row.get("HC") or 0), int(row.get("AC") or 0)
        hy, ay = int(row.get("HY") or 0), int(row.get("AY") or 0)
        hr, ar = int(row.get("HR") or 0), int(row.get("AR") or 0)
    except (ValueError, KeyError, TypeError):
        return None

    if metric == "goals":
        return fthg + ftag
    if metric == "corners":
        return hc + ac
    if metric == "cards":
        return (hy + 2 * hr) + (ay + 2 * ar)
    if metric == "first_half_goals":
        return hthg + athg
    if metric == "second_half_goals":
        return (fthg - hthg) + (ftag - athg)
    if metric == "btts":
        return 1 if (fthg > 0 and ftag > 0) else 0
    if metric == "team_win":
        # Returns the actual WINNING team's name (a string, not a
        # number) - or None for a draw, which can never match a picked
        # team's name so it correctly falls out as a miss below.
        if fthg > ftag:
            return row.get("HomeTeam")
        if ftag > fthg:
            return row.get("AwayTeam")
        return None
    return None


def _check_hit(actual, direction: str, line) -> bool:
    if direction == "Over":
        return actual > line
    if direction == "Under":
        return actual < line
    if direction == "Yes":
        return actual == 1
    if direction == "No":
        return actual == 0
    # Anything else falling through here is a team_win pick - "direction"
    # holds the picked team's name, "actual" holds the actual winning
    # team's name (or None for a draw). Hit only if they match exactly.
    return actual is not None and actual == direction


def reconcile_pending(log: dict[str, dict], data_dir: Path, today: date) -> tuple[int, int]:
    """Try to settle any pending picks whose match date has passed.
    Returns (settled_count, still_pending_past_date_count)."""
    # Load every league's results, keyed by (home, away) -> LIST of rows.
    # Team names in the log are already the canonical resolved names
    # (same ones used in data/<LEAGUE>.csv), so exact match is enough -
    # no fuzzy resolution needed at this stage.
    #
    # A plain dict overwrite here was the bug behind a real reported
    # issue: fetch_data.py pulls TWO seasons of history, and a fixture
    # pairing (e.g. Man City v Bournemouth) recurs every season, so
    # there can genuinely be two rows with the same (home, away) key -
    # last season's meeting and this season's. Overwriting silently
    # picked whichever happened to be read last, which could reconcile
    # a pick against the wrong season's score entirely. Keeping every
    # row and selecting the one closest to the pick's own match_date
    # fixes that.
    results_by_pair: dict[tuple[str, str], list[dict]] = {}
    for csv_path in data_dir.glob("*.csv"):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if not row.get("FTHG") or not row.get("HomeTeam"):
                    continue  # not yet played, or malformed row
                key = (row["HomeTeam"], row["AwayTeam"])
                results_by_pair.setdefault(key, []).append(row)

    # Only accept a match within this many days of the pick's logged
    # match_date - close enough to absorb a postponed/rearranged fixture,
    # far enough to never accidentally grab a different season's meeting.
    MAX_DATE_DRIFT_DAYS = 14

    settled, still_waiting = 0, 0
    for pick in log.values():
        if pick["status"] != "pending":
            continue
        match_date = _parse_date(pick["match_date"])
        if match_date is None or match_date >= today:
            continue  # not due yet

        candidates = results_by_pair.get((pick["home_team"], pick["away_team"]), [])
        row = _find_result_row(candidates, match_date, MAX_DATE_DRIFT_DAYS)

        if row is None:
            still_waiting += 1  # result not published yet - retry next run
            continue

        actual = _actual_metric_value(row, pick["metric"])
        if actual is None and pick["metric"] != "team_win":
            still_waiting += 1
            continue
        # For team_win, actual=None legitimately means "it was a draw" -
        # a real, settleable outcome (the pick just misses) - so it does
        # NOT fall into the "still waiting" bucket the way a missing/
        # malformed row does for every other metric.

        line = float(pick["line"]) if pick["line"] not in ("", None) else None
        hit = _check_hit(actual, pick["direction"], line)

        pick["status"] = "settled"
        pick["actual_value"] = actual if actual is not None else "Draw"
        pick["result"] = "hit" if hit else "miss"
        pick["settled_date"] = today.isoformat()
        settled += 1

    return settled, still_waiting


def _find_result_row(candidates: list[dict], match_date: date, max_drift_days: int = 14) -> dict | None:
    """Pick the candidate row whose date is closest to match_date, within
    max_drift_days. Shared by reconcile_pending and reaudit_settled so
    both use identical matching logic."""
    row, best_drift = None, None
    for candidate in candidates:
        candidate_date = _parse_date(candidate.get("Date", ""))
        if candidate_date is None:
            continue
        drift = abs((candidate_date - match_date).days)
        if drift <= max_drift_days and (best_drift is None or drift < best_drift):
            row = candidate
            best_drift = drift
    return row


def reaudit_settled(log: dict[str, dict], data_dir: Path) -> tuple[list[dict], list[dict]]:
    """
    Re-check every ALREADY-SETTLED pick against the current matching
    logic and data, and correct any that were reconciled incorrectly.

    This exists because a real bug was found and fixed here: results
    were looked up by (home, away) team names alone, with no season/date
    awareness. Since a fixture pairing recurs every season, that could
    silently match a pick against the WRONG season's score. Settled
    picks are never touched by the normal pending->settled flow, so
    fixing the matching logic alone doesn't correct rows that already
    settled incorrectly under the old logic - this does that, once, and
    is safe to leave running on every future run too: it's a no-op for
    anything that was already reconciled correctly.

    A second, related problem this also handles: if NO candidate exists
    within tolerance for a settled pick, that pick's original settlement
    can no longer be verified against current data - meaning it was very
    likely one of the picks wrongly settled by the old buggy logic
    against a different season's meeting (exactly the Man City v
    Bournemouth and Doncaster v Barnsley cases this was built to catch).
    Rather than silently leaving a value we can no longer verify sitting
    there looking like ground truth, those picks are REVERTED to
    "pending" - they'll get properly re-settled once the real match
    result is published and a genuine within-tolerance candidate exists.

    Returns (corrections, reverted) - two lists of records for
    reporting. Doesn't save anything itself.
    """
    results_by_pair: dict[tuple[str, str], list[dict]] = {}
    for csv_path in data_dir.glob("*.csv"):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if not row.get("FTHG") or not row.get("HomeTeam"):
                    continue
                key = (row["HomeTeam"], row["AwayTeam"])
                results_by_pair.setdefault(key, []).append(row)

    corrections, reverted = [], []
    for pick in log.values():
        if pick["status"] != "settled":
            continue
        match_date = _parse_date(pick["match_date"])
        if match_date is None:
            continue

        candidates = results_by_pair.get((pick["home_team"], pick["away_team"]), [])
        row = _find_result_row(candidates, match_date)
        if row is None:
            # No verifiable candidate exists right now - can't confirm
            # this settlement is correct, so don't let it keep counting
            # toward the hit rate. Revert to pending; it'll be properly
            # re-settled once the real result is published.
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

        actual = _actual_metric_value(row, pick["metric"])
        if actual is None and pick["metric"] != "team_win":
            continue
        display_actual = actual if actual is not None else "Draw"

        line = float(pick["line"]) if pick["line"] not in ("", None) else None
        correct_result = "hit" if _check_hit(actual, pick["direction"], line) else "miss"

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


def compute_summary(log: dict[str, dict]) -> dict:
    """Overall and per-(metric, direction) hit rates, from settled picks only.

    Note: for team_win picks, "direction" is a team name, so each
    distinct team ever picked gets its own by_category row (e.g.
    "team_win_Arsenal") rather than one aggregated "team_win" row - a
    minor cosmetic quirk, not a bug, since the overall hit-rate figure
    still correctly includes every settled team_win pick regardless."""
    settled = [p for p in log.values() if p["status"] == "settled"]
    overall_hits = sum(1 for p in settled if p["result"] == "hit")

    by_category: dict[str, dict] = {}
    for p in settled:
        key = f"{p['metric']}_{p['direction']}"
        by_category.setdefault(key, {"metric": p["metric"], "direction": p["direction"], "hits": 0, "total": 0})
        by_category[key]["total"] += 1
        if p["result"] == "hit":
            by_category[key]["hits"] += 1

    for c in by_category.values():
        c["pct"] = round(100 * c["hits"] / c["total"]) if c["total"] else 0

    return {
        "overall": {
            "hits": overall_hits,
            "total": len(settled),
            "pct": round(100 * overall_hits / len(settled)) if settled else 0,
        },
        "by_category": sorted(by_category.values(), key=lambda c: (-c["total"], c["metric"])),
        "pending_count": sum(1 for p in log.values() if p["status"] == "pending"),
    }


if __name__ == "__main__":
    base = Path(__file__).parent
    data_dir = base / "data"
    log_path = base / "bets_log.csv"
    statpack_path = base / "statpack.json"

    if not statpack_path.exists():
        raise SystemExit("statpack.json not found - run build_statpack.py first.")

    with open(statpack_path) as f:
        statpack = json.load(f)

    today = date.today()
    log = load_log(log_path)

    added = log_new_picks(log, statpack.get("best_bets", {}), today)
    print(f"Logged {added} new pick(s).")

    settled, still_waiting = reconcile_pending(log, data_dir, today)
    print(f"Settled {settled} pick(s) this run; {still_waiting} past-due pick(s) still waiting on results.")

    corrections, reverted = reaudit_settled(log, data_dir)
    if corrections:
        print(f"Corrected {len(corrections)} previously-settled pick(s):")
        for c in corrections:
            print(f"  {c['home_team']} v {c['away_team']} - {c['direction']} {c['line']} {c['metric']}: "
                  f"actual {c['old_actual']} -> {c['new_actual']}, result {c['old_result']} -> {c['new_result']}")
    if reverted:
        print(f"Reverted {len(reverted)} unverifiable settled pick(s) back to pending "
              f"(no current-season result available yet to confirm them):")
        for r in reverted:
            print(f"  {r['home_team']} v {r['away_team']} - {r['direction']} {r['line']} {r['metric']} "
                  f"(was: actual {r['old_actual']}, result {r['old_result']})")

    save_log(log_path, log)
    print(f"Log saved to {log_path} ({len(log)} total picks).")

    summary = compute_summary(log)
    print(f"Overall: {summary['overall']['hits']}/{summary['overall']['total']} "
          f"({summary['overall']['pct']}%) settled, {summary['pending_count']} pending.")

    recent = sorted(log.values(), key=lambda p: p.get("match_date", ""), reverse=True)[:100]

    out = {"summary": summary, "recent": recent}
    js_path = base / "bets_log.js"
    with open(js_path, "w") as f:
        f.write("const BETS_LOG = ")
        json.dump(out, f, default=str)
        f.write(";\n")
    print(f"Dashboard log data written to {js_path}")
