"""
track_bets.py

Keeps a running log of Best Bets picks and checks them against real
results once the matches have been played, so you can see an actual
hit rate rather than just trusting the model.

## Flat pick shape (updated)

best_bets.py was restructured to return a single flat list of picks
(covering match-level AND team-level markets, across football only for
now) rather than a shape grouped by market category. This file was
updated to match: log_new_picks() now iterates that flat list directly
rather than the old {"goals": {"over":[...],...}, "team_win":[...]}
nesting.

Each pick's scope ("match", "home", or "away") is now part of both its
pick_id and how its actual result gets computed - a team-level pick
(e.g. "Arsenal Over 4.5 Corners") settles against ARSENAL's own corner
count, not the match total, whereas the exact same metric/direction at
scope="match" settles against the combined total. See
_actual_metric_value below.

## NFL picks - logged, but NOT YET reconciled

nfl_best_bets.py produces picks in the same flat shape, tagged
"sport": "nfl". This file logs them fine (they show up as "pending"),
but reconcile_pending() deliberately skips settling them for now -
this reads football's data/<LEAGUE>.csv results files, which have
nothing to do with NFL's own results data (different source, different
column names entirely). Wiring up NFL settlement is a real follow-up,
not done here - explicitly scoped out for now rather than half-built,
per the same reasoning that motivated this restructure in the first
place (get Best Bets right first, Track Record can catch up after).

## Pick lifecycle (unchanged)
  1. The first time a pick's exact (sport, home, away, date, metric,
     scope, direction) combination appears in Best Bets, it's logged as
     "pending" with whatever line/confidence was showing that day - a
     FREEZE, not a moving target.
  2. Once the fixture's date has passed, each run tries to reconcile
     it against the real result. If the result isn't published yet,
     it stays pending and gets checked again next run.

Run this after build_statpack.py (and, once NFL settlement exists,
after nfl_build_statpack.py too), since it reads statpack.json's
"best_bets" list as its source of new football picks.
"""

import csv
import json
from datetime import date, datetime
from pathlib import Path

LOG_COLUMNS = [
    "pick_id", "logged_date", "match_date", "league_code", "league_name",
    "home_team", "away_team", "sport", "metric", "scope", "team",
    "direction", "line", "confidence", "label",
    "status", "actual_value", "result", "settled_date",
]


def _parse_date(d: str):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(d, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


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


def log_new_picks(log: dict[str, dict], all_picks: list[dict], today: date) -> int:
    """Add any not-yet-seen pick from the flat picks list (the output of
    best_bets.compute_best_bets() and/or nfl_best_bets.compute_nfl_best_bets() -
    concatenate both before calling this if tracking more than one
    sport). Returns how many new rows were added."""
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
            "confidence": pick["confidence"],
            "label": pick.get("label", ""),
            "status": "pending",
            "actual_value": "",
            "result": "",
            "settled_date": "",
        }
        added += 1
    return added


def _actual_metric_value(row: dict, metric: str, scope: str = "match"):
    """Returns the actual value for one metric, at the given scope.

    scope="match": the combined/total value (unchanged behaviour from
    before this file supported team-level picks at all).
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
    # Anything else falling through here is a team_win pick - "direction"
    # holds the picked team's name, "actual" holds the actual winning
    # team's name (or None for a draw). Hit only if they match exactly.
    return actual is not None and actual == direction


def reconcile_pending(log: dict[str, dict], data_dir: Path, today: date) -> tuple[int, int]:
    """Try to settle any pending FOOTBALL picks whose match date has
    passed. NFL picks are deliberately skipped here - see this file's
    module docstring - and stay pending indefinitely until that's
    built. Returns (settled_count, still_pending_past_date_count)."""
    results_by_pair: dict[tuple[str, str], list[dict]] = {}
    for csv_path in data_dir.glob("*.csv"):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if not row.get("FTHG") or not row.get("HomeTeam"):
                    continue
                key = (row["HomeTeam"], row["AwayTeam"])
                results_by_pair.setdefault(key, []).append(row)

    MAX_DATE_DRIFT_DAYS = 14

    settled, still_waiting = 0, 0
    for pick in log.values():
        if pick["status"] != "pending":
            continue
        if (pick.get("sport") or "football") != "football":
            continue  # NFL settlement not built yet - stays pending, not an error

        match_date = _parse_date(pick["match_date"])
        if match_date is None or match_date >= today:
            continue

        candidates = results_by_pair.get((pick["home_team"], pick["away_team"]), [])
        row = _find_result_row(candidates, match_date, MAX_DATE_DRIFT_DAYS)

        if row is None:
            still_waiting += 1
            continue

        scope = pick.get("scope", "match")
        actual = _actual_metric_value(row, pick["metric"], scope)
        if actual is None and pick["metric"] != "team_win":
            still_waiting += 1
            continue

        line = float(pick["line"]) if pick["line"] not in ("", None) else None
        hit = _check_hit(actual, pick["direction"], line)

        pick["status"] = "settled"
        pick["actual_value"] = actual if actual is not None else "Draw"
        pick["result"] = "hit" if hit else "miss"
        pick["settled_date"] = today.isoformat()
        settled += 1

    return settled, still_waiting


def _find_result_row(candidates: list[dict], match_date: date, max_drift_days: int = 14) -> dict | None:
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
    """Re-check every already-settled FOOTBALL pick against the current
    matching logic and data (see this function's original docstring
    reasoning - unchanged). NFL picks are never settled yet, so there's
    nothing here for them to re-audit."""
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
        if pick["status"] != "settled" or (pick.get("sport") or "football") != "football":
            continue
        match_date = _parse_date(pick["match_date"])
        if match_date is None:
            continue

        candidates = results_by_pair.get((pick["home_team"], pick["away_team"]), [])
        row = _find_result_row(candidates, match_date)
        if row is None:
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

        scope = pick.get("scope", "match")
        actual = _actual_metric_value(row, pick["metric"], scope)
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
    """Overall and per-(metric, direction) hit rates, from settled picks
    only - EXCLUDING "Under"/"No" (best_bets.py doesn't generate these
    any more, but older logged rows may still carry them).

    team_win/moneyline picks all collapse into a single category each
    regardless of which specific team was picked - direction there is
    a team name, not Over/Under.

    Team-level picks (scope="home"/"away") are grouped together with
    their match-level counterpart under the same metric+direction
    category for now (e.g. "Arsenal Over 4.5 Corners" counts alongside
    "Over 9.5 Corners" under "Corners Over") - a coarser breakdown than
    might eventually be wanted, but a reasonable starting point rather
    than over-building this before it's clear it's needed."""
    settled = [p for p in log.values() if p["status"] == "settled" and p["direction"] not in ("Under", "No")]
    overall_hits = sum(1 for p in settled if p["result"] == "hit")

    by_category: dict[str, dict] = {}
    for p in settled:
        if p["metric"] in ("team_win", "moneyline"):
            key = p["metric"]
        else:
            key = f"{p['metric']}_{p['direction']}"
        by_category.setdefault(key, {
            "metric": p["metric"],
            "direction": "" if p["metric"] in ("team_win", "moneyline") else p["direction"],
            "hits": 0, "total": 0,
        })
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
    nfl_statpack_path = base / "nfl_statpack.json"

    if not statpack_path.exists():
        raise SystemExit("statpack.json not found - run build_statpack.py first.")

    with open(statpack_path) as f:
        statpack = json.load(f)

    today = date.today()
    log = load_log(log_path)

    # Both sports' best_bets lists are the SAME flat pick shape (see
    # best_bets.py / nfl_best_bets.py), tagged "sport" precisely so they
    # can be logged together here - this was the whole point of that
    # shape, but the actual merge was missing: this only ever read
    # statpack.json, so NFL picks were never logged at all (not stuck
    # pending - simply never entered into bets_log.csv in the first
    # place). nfl_statpack.json is written by the separate NFL workflow
    # to the same repo, so it's read here if present; its absence isn't
    # an error (e.g. the NFL workflow hasn't run yet on a fresh repo) -
    # football logging still proceeds either way.
    picks = list(statpack.get("best_bets", []))
    if nfl_statpack_path.exists():
        with open(nfl_statpack_path) as f:
            nfl_statpack = json.load(f)
        nfl_picks = nfl_statpack.get("best_bets", [])
        picks.extend(nfl_picks)
        print(f"Including {len(nfl_picks)} NFL pick(s) alongside {len(statpack.get('best_bets', []))} football pick(s).")
    else:
        print("No nfl_statpack.json found - logging football picks only.")

    added = log_new_picks(log, picks, today)
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
        print(f"Reverted {len(reverted)} unverifiable settled pick(s) back to pending:")
        for r in reverted:
            print(f"  {r['home_team']} v {r['away_team']} - {r['direction']} {r['line']} {r['metric']} "
                  f"(was: actual {r['old_actual']}, result {r['old_result']})")

    save_log(log_path, log)
    print(f"Log saved to {log_path} ({len(log)} total picks).")

    summary = compute_summary(log)
    print(f"Overall: {summary['overall']['hits']}/{summary['overall']['total']} "
          f"({summary['overall']['pct']}%) settled, {summary['pending_count']} pending.")

    out = {"summary": summary}
    js_path = base / "bets_log.js"
    with open(js_path, "w") as f:
        f.write("const BETS_LOG = ")
        json.dump(out, f, default=str)
        f.write(";\n")
    print(f"Dashboard log data written to {js_path}")
