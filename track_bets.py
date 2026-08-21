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


def log_new_picks(log: dict[str, dict], best_bets: dict, today: date) -> int:
    """Add any not-yet-seen (fixture, metric, direction) picks. Returns
    how many new rows were added."""
    added = 0
    for metric, sides in best_bets.items():
        for direction_list in (sides.get("over", []), sides.get("under", [])):
            for entry in direction_list:
                pick_id = make_pick_id(entry["home_team"], entry["away_team"],
                                        entry["date"], metric, entry["direction"])
                if pick_id in log:
                    continue  # already logged - frozen, don't touch it again
                log[pick_id] = {
                    "pick_id": pick_id,
                    "logged_date": today.isoformat(),
                    "match_date": entry["date"],
                    "league_code": entry["league_code"],
                    "league_name": entry["league_name"],
                    "home_team": entry["home_team"],
                    "away_team": entry["away_team"],
                    "metric": metric,
                    "direction": entry["direction"],
                    "line": entry["line"] if entry["line"] is not None else "",
                    "confidence": entry["confidence"],
                    "status": "pending",
                    "actual_value": "",
                    "result": "",
                    "settled_date": "",
                }
                added += 1
    return added


def _actual_metric_value(row: dict, metric: str) -> float | None:
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
    return None


def _check_hit(actual: float, direction: str, line) -> bool:
    if direction == "Over":
        return actual > line
    if direction == "Under":
        return actual < line
    if direction == "Yes":
        return actual == 1
    if direction == "No":
        return actual == 0
    return False


def reconcile_pending(log: dict[str, dict], data_dir: Path, today: date) -> tuple[int, int]:
    """Try to settle any pending picks whose match date has passed.
    Returns (settled_count, still_pending_past_date_count)."""
    # Load every league's results once, keyed by (home, away) -> row.
    # Team names in the log are already the canonical resolved names
    # (same ones used in data/<LEAGUE>.csv), so exact match is enough -
    # no fuzzy resolution needed at this stage.
    results_by_pair: dict[tuple[str, str], dict] = {}
    for csv_path in data_dir.glob("*.csv"):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if not row.get("FTHG") or not row.get("HomeTeam"):
                    continue  # not yet played, or malformed row
                key = (row["HomeTeam"], row["AwayTeam"])
                results_by_pair[key] = row

    settled, still_waiting = 0, 0
    for pick in log.values():
        if pick["status"] != "pending":
            continue
        match_date = _parse_date(pick["match_date"])
        if match_date is None or match_date >= today:
            continue  # not due yet

        row = results_by_pair.get((pick["home_team"], pick["away_team"]))
        if row is None:
            still_waiting += 1  # result not published yet - retry next run
            continue

        actual = _actual_metric_value(row, pick["metric"])
        if actual is None:
            still_waiting += 1
            continue

        line = float(pick["line"]) if pick["line"] not in ("", None) else None
        hit = _check_hit(actual, pick["direction"], line)

        pick["status"] = "settled"
        pick["actual_value"] = actual
        pick["result"] = "hit" if hit else "miss"
        pick["settled_date"] = today.isoformat()
        settled += 1

    return settled, still_waiting


def compute_summary(log: dict[str, dict]) -> dict:
    """Overall and per-(metric, direction) hit rates, from settled picks only."""
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

    save_log(log_path, log)
    print(f"Log saved to {log_path} ({len(log)} total picks).")

    summary = compute_summary(log)
    print(f"Overall: {summary['overall']['hits']}/{summary['overall']['total']} "
          f"({summary['overall']['pct']}%) settled, {summary['pending_count']} pending.")

    # Recent settled + all pending, for the dashboard - not the full log
    # (which only grows), to keep the embedded JS file small.
    recent = sorted(log.values(), key=lambda p: p.get("match_date", ""), reverse=True)[:100]

    out = {"summary": summary, "recent": recent}
    js_path = base / "bets_log.js"
    with open(js_path, "w") as f:
        f.write("const BETS_LOG = ")
        json.dump(out, f, default=str)
        f.write(";\n")
    print(f"Dashboard log data written to {js_path}")
