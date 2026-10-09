"""UK horse racing revision guide - data builder.

Pulls Betfair's free daily UK win-market price files, boils each race down
to one row (course, code, race type, distance, field size, favourite,
winner, how the market moved), keeps that history in data/racing_races.csv
and writes racing_data.js for the dashboard's Racing tab.

Source: https://promo.betfair.com/betfairsp/prices/dwbfpricesukwinDDMMYYYY.csv
One file per day, and the file dated D holds the races run on D-1 (it is
published the morning after). The race date always comes from the rows
themselves, never from the file name.

Each run only downloads files it hasn't processed yet (plus the last few
days again, in case a file was incomplete when first fetched), so the
first run does the whole backfill and every later run is a handful of
requests. Usage:

    python racing.py                       # normal / daily run
    python racing.py --start 2023-05-01    # first run: how far back to go
    python racing.py --from-dir some/dir   # read files from a folder instead
                                           # of downloading (used for testing)

Standard library only.

Definitions worth knowing:
  * "Favourite" = shortest Betfair Starting Price (BSP) in the race.
  * Profit/ROI is for a 1-unit stake at BSP with Betfair's 5% commission
    taken off winnings. It is a guide to value, not a bookmaker SP return.
  * Season: jumps run 1 May - 30 Apr (labelled 2025/26); the Flat is the
    calendar year (labelled 2025) and includes all-weather racing.
  * Bumpers (NHF) count as jumps for the season, as they are National Hunt.
  * Dead heats count as a win for whoever dead-heated, at full odds.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

BASE_URL = "https://promo.betfair.com/betfairsp/prices/dwbfpricesukwin{d}.csv"
DATA_DIR = Path(__file__).parent / "data"
RACES_CSV = DATA_DIR / "racing_races.csv"
DONE_FILE = DATA_DIR / "racing_files_done.txt"
OUT_JS = Path(__file__).parent / "racing_data.js"

DEFAULT_START = date(2023, 5, 1)
REFETCH_RECENT_DAYS = 3        # re-download the newest files each run
SEASONS_SHOWN = 3              # per code: the latest N seasons (incl. current)
COMMISSION = 0.05
MIN_RUNNERS = 2

COLUMNS = [
    "event_id", "date", "time", "course", "race_name", "code", "rtype",
    "furlongs", "runners", "winners", "fav_bsp", "fav_morning", "fav_won",
    "win_bsp", "win_morning", "win_rank", "steam_bsp", "steam_ratio",
    "steam_won", "season",
]


# ------------------------------------------------------------------ parsing

_COURSE_RE = re.compile(r"^(.*?)\s+\d{1,2}(?:st|nd|rd|th)\s+[A-Za-z]{3}$")
_DIST_RE = re.compile(r"^(?:(\d+)m)?(?:(\d+)f)?(?:(\d+)y)?\s*(.*)$")


def parse_course(menu_hint: str) -> str:
    m = _COURSE_RE.match(menu_hint.strip())
    return (m.group(1) if m else menu_hint).strip()


def parse_event_name(name: str):
    """'2m7f Hcap Hrd' -> (furlongs 23.0, 'Hcap Hrd'). furlongs is None if
    the distance can't be read."""
    m = _DIST_RE.match(name.strip())
    if not m:
        return None, name.strip()
    miles, furlongs, yards, rest = m.groups()
    if miles is None and furlongs is None and yards is None:
        return None, rest
    f = (int(miles or 0) * 8) + int(furlongs or 0) + (int(yards or 0) / 220)
    return round(f, 2), rest


def classify_code(rest: str) -> str:
    """flat / hurdle / chase / bumper."""
    tokens = set(rest.replace("/", " ").split())
    if "Chs" in tokens or "XC" in tokens:
        return "chase"
    if "Hrd" in tokens:
        return "hurdle"
    if "NHF" in tokens:
        return "bumper"
    return "flat"


def classify_type(rest: str, code: str) -> str:
    r = rest.lower()
    if re.search(r"\bgr[pd]\s*\d", r):
        return "Group/Graded"
    if re.search(r"\blist", r):
        return "Listed"
    if "nursery" in r:
        return "Nursery"
    if "hcap" in r:
        return "Handicap"
    if "mdn" in r:
        return "Maiden"
    if "nov" in r:
        return "Novice"
    if "juv" in r:
        return "Juvenile"
    if re.search(r"\b(sell|clm|claim)", r):
        return "Selling/Claiming"
    if "hunt" in r:
        return "Hunters"
    if code == "bumper":
        return "Bumper"
    if "class" in r or "cond" in r:
        return "Conditions/Class"
    return "Other"


def season_of(day: date, group: str) -> str:
    if group == "flat":
        return str(day.year)
    start = day.year if day.month >= 5 else day.year - 1
    return f"{start}/{(start + 1) % 100:02d}"


def group_of(code: str) -> str:
    return "flat" if code == "flat" else "jumps"


def _f(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def race_row(rows: list[dict]):
    """One race's runner rows -> a single summary dict, or None when the
    race can't be used (void, abandoned, no prices)."""
    first = rows[0]
    try:
        when = datetime.strptime(first["event_dt"].strip(), "%d-%m-%Y %H:%M")
    except (KeyError, ValueError):
        return None

    runners = [r for r in rows if _f(r.get("bsp")) > 1.0]
    if len(runners) < MIN_RUNNERS:
        return None
    winners = [r for r in runners if str(r.get("win_lose", "")).strip() == "1"]
    if not winners:
        return None

    for r in runners:
        r["_bsp"] = _f(r["bsp"])
        r["_morning"] = _f(r.get("morningwap"))
    runners.sort(key=lambda r: r["_bsp"])
    fav = runners[0]
    winner = min(winners, key=lambda r: r["_bsp"])
    fav_won = any(w["_bsp"] <= fav["_bsp"] + 1e-9 for w in winners)
    win_rank = 1 + sum(1 for r in runners if r["_bsp"] < winner["_bsp"] - 1e-9)

    steam = None
    for r in runners:
        if r["_morning"] > 1.0:
            ratio = r["_morning"] / r["_bsp"]
            if steam is None or ratio > steam[0]:
                steam = (ratio, r)
    steam_won = ""
    steam_bsp = steam_ratio = ""
    if steam:
        steam_ratio = round(steam[0], 2)
        steam_bsp = round(steam[1]["_bsp"], 2)
        steam_won = int(any(steam[1] is w for w in winners))

    furlongs, rest = parse_event_name(first.get("event_name", ""))
    code = classify_code(rest)
    return {
        "event_id": first["event_id"].strip(),
        "date": when.date().isoformat(),
        "time": when.strftime("%H:%M"),
        "course": parse_course(first.get("menu_hint", "")),
        "race_name": first.get("event_name", "").strip(),
        "code": code,
        "rtype": classify_type(rest, code),
        "furlongs": "" if furlongs is None else furlongs,
        "runners": len(runners),
        "winners": len(winners),
        "fav_bsp": round(fav["_bsp"], 2),
        "fav_morning": round(fav["_morning"], 2) if fav["_morning"] > 1 else "",
        "fav_won": int(fav_won),
        "win_bsp": round(winner["_bsp"], 2),
        "win_morning": round(winner["_morning"], 2) if winner["_morning"] > 1 else "",
        "win_rank": win_rank,
        "steam_bsp": steam_bsp,
        "steam_ratio": steam_ratio,
        "steam_won": steam_won,
        "season": season_of(when.date(), group_of(code)),
    }


def parse_file(text: str) -> list[dict]:
    reader = csv.DictReader(text.splitlines())
    by_event: dict[str, list[dict]] = defaultdict(list)
    for row in reader:
        row = {(k or "").strip().lower(): (v or "") for k, v in row.items()}
        if row.get("event_id"):
            by_event[row["event_id"].strip()].append(row)
    out = []
    for rows in by_event.values():
        r = race_row(rows)
        if r:
            out.append(r)
    return out


# ------------------------------------------------------------- fetch & store

def load_races() -> dict[str, dict]:
    if not RACES_CSV.exists():
        return {}
    with open(RACES_CSV, newline="") as f:
        return {r["event_id"]: r for r in csv.DictReader(f)}


def save_races(races: dict[str, dict]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    rows = sorted(races.values(), key=lambda r: (r["date"], r["time"], r["course"]))
    with open(RACES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def load_done() -> set[str]:
    if not DONE_FILE.exists():
        return set()
    return {ln.strip() for ln in DONE_FILE.read_text().splitlines() if ln.strip()}


def save_done(done: set[str]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    DONE_FILE.write_text("\n".join(sorted(done)) + "\n")


def download(file_date: date) -> str | None:
    """Returns the file text, "" when there's no file for that day (404),
    or None when the download failed and should be retried next run."""
    url = BASE_URL.format(d=file_date.strftime("%d%m%Y"))
    req = urllib.request.Request(url, headers={"User-Agent": "footystatpack-racing/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return ""
            err = f"HTTP {e.code}"
        except Exception as e:  # network blips
            err = str(e)
        time.sleep(2 * (attempt + 1))
    print(f"  {file_date}: gave up ({err})")
    return None


def read_local(directory: Path, file_date: date) -> str | None:
    p = directory / f"dwbfpricesukwin{file_date.strftime('%d%m%Y')}.csv"
    return p.read_text(errors="replace") if p.exists() else ""


def update(start: date, end: date, from_dir: Path | None) -> None:
    races = load_races()
    done = load_done()
    recent_cutoff = end - timedelta(days=REFETCH_RECENT_DAYS)

    # File D holds races from D-1, so the first file needed is start+1.
    todo = []
    d = start + timedelta(days=1)
    while d <= end + timedelta(days=1):
        if d > date.today():
            break
        if d.isoformat() not in done or d >= recent_cutoff:
            todo.append(d)
        d += timedelta(days=1)

    print(f"{len(todo)} file(s) to fetch ({len(races)} races already stored).")
    added = fetched_ok = 0
    for i, fd in enumerate(todo, 1):
        text = read_local(from_dir, fd) if from_dir else download(fd)
        if text is None:
            continue
        fetched_ok += 1
        # A missing file for a recent day may just not be published yet.
        if text or fd < recent_cutoff:
            done.add(fd.isoformat())
        for r in parse_file(text) if text else []:
            if r["event_id"] not in races:
                added += 1
            races[r["event_id"]] = r
        if i % 50 == 0:
            print(f"  ...{i}/{len(todo)} files, {len(races)} races")
            save_races(races)
            save_done(done)
        if not from_dir:
            time.sleep(0.2)

    save_races(races)
    save_done(done)
    print(f"Done: {fetched_ok}/{len(todo)} files read, {added} new race(s), {len(races)} total.")


# -------------------------------------------------------------- aggregation

def dist_band(code: str, furlongs) -> str:
    if furlongs in ("", None):
        return "Unknown"
    f = float(furlongs)
    if code == "flat":
        if f <= 6.5:
            return "Sprint (5-6f)"
        if f < 10:
            return "7f-1m1f"
        if f < 14:
            return "1m2f-1m5f"
        return "1m6f+"
    if f <= 17.5:
        return "Up to 2m1f"
    if f < 22:
        return "2m2f-2m5f"
    if f < 26:
        return "2m6f-3m1f"
    return "3m2f+"


def field_band(n: int) -> str:
    if n <= 6:
        return "6 or fewer"
    if n <= 9:
        return "7-9"
    if n <= 12:
        return "10-12"
    if n <= 16:
        return "13-16"
    return "17+"


def price_band(bsp: float) -> str:
    if bsp <= 2.0:
        return "Evens-shot or shorter (to 2.0)"
    if bsp <= 4.0:
        return "2.0-4.0"
    if bsp <= 8.0:
        return "4.0-8.0"
    if bsp <= 16.0:
        return "8.0-16"
    return "16+"


DIST_ORDER = {
    "flat": ["Sprint (5-6f)", "7f-1m1f", "1m2f-1m5f", "1m6f+", "Unknown"],
    "jumps": ["Up to 2m1f", "2m2f-2m5f", "2m6f-3m1f", "3m2f+", "Unknown"],
}
FIELD_ORDER = ["6 or fewer", "7-9", "10-12", "13-16", "17+"]
PRICE_ORDER = ["Evens-shot or shorter (to 2.0)", "2.0-4.0", "4.0-8.0", "8.0-16", "16+"]


def stats(races: list[dict]) -> dict:
    n = len(races)
    if not n:
        return {"n": 0}
    fav_wins = sum(int(r["fav_won"]) for r in races)
    profit = 0.0
    for r in races:
        if int(r["fav_won"]):
            profit += (float(r["fav_bsp"]) - 1) * (1 - COMMISSION)
        else:
            profit -= 1
    top3 = sum(1 for r in races if int(r["win_rank"]) <= 3)
    return {
        "n": n,
        "fav": round(100 * fav_wins / n, 1),
        "roi": round(100 * profit / n, 1),
        "top3": round(100 * top3 / n, 1),
        "favp": round(statistics.mean(float(r["fav_bsp"]) for r in races), 2),
        "medw": round(statistics.median(float(r["win_bsp"]) for r in races), 1),
    }


def breakdown(races: list[dict], key, order=None) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in races:
        groups[key(r)].append(r)
    names = order if order else sorted(groups, key=lambda k: -len(groups[k]))
    out = []
    for name in names:
        if name in groups:
            out.append({"name": name, **stats(groups[name])})
    return out


def winners_by_price(races: list[dict]) -> list[dict]:
    n = len(races)
    groups: dict[str, int] = defaultdict(int)
    for r in races:
        groups[price_band(float(r["win_bsp"]))] += 1
    return [{"name": b, "n": groups[b], "share": round(100 * groups[b] / n, 1)}
            for b in PRICE_ORDER if n and groups.get(b)]


def window_summary(races: list[dict], group: str, with_courses: bool) -> dict:
    out = {
        "overall": stats(races),
        "by_code": breakdown(races, lambda r: r["code"], None),
        "by_type": breakdown(races, lambda r: r["rtype"]),
        "by_dist": breakdown(races, lambda r: dist_band(r["code"], r["furlongs"]), DIST_ORDER[group]),
        "by_field": breakdown(races, lambda r: field_band(int(r["runners"])), FIELD_ORDER),
        "winner_price": winners_by_price(races),
    }
    by_course: dict[str, list[dict]] = defaultdict(list)
    for r in races:
        by_course[r["course"]].append(r)
    courses = {}
    for name, rs in by_course.items():
        entry = {"overall": stats(rs)}
        if with_courses:
            entry["by_type"] = breakdown(rs, lambda r: r["rtype"])
            entry["by_dist"] = breakdown(rs, lambda r: dist_band(r["code"], r["furlongs"]), DIST_ORDER[group])
            entry["by_field"] = breakdown(rs, lambda r: field_band(int(r["runners"])), FIELD_ORDER)
            entry["by_code"] = breakdown(rs, lambda r: r["code"], None)
        courses[name] = entry
    out["courses"] = courses
    return out


def build_summary(races: dict[str, dict]) -> dict:
    rows = list(races.values())
    result = {}
    for group, label in (("flat", "Flat"), ("jumps", "Jumps")):
        mine = [r for r in rows if group_of(r["code"]) == group]
        seasons = sorted({r["season"] for r in mine})[-SEASONS_SHOWN:]
        mine = [r for r in mine if r["season"] in seasons]
        windows = {"all": window_summary(mine, group, True)}
        for s in seasons:
            windows[s] = window_summary([r for r in mine if r["season"] == s], group, False)
        dates = [r["date"] for r in mine]
        result[group] = {
            "label": label,
            "seasons": seasons,
            "current": seasons[-1] if seasons else "",
            "first_date": min(dates) if dates else "",
            "last_date": max(dates) if dates else "",
            "windows": windows,
        }
    return {
        "generated": date.today().isoformat(),
        "source": "Betfair Starting Price files",
        "commission": COMMISSION,
        "races_stored": len(rows),
        "codes": result,
    }


def write_js(summary: dict) -> None:
    with open(OUT_JS, "w") as f:
        f.write("const RACING_DATA = ")
        json.dump(summary, f, separators=(",", ":"))
        f.write(";\n")
    print(f"Wrote {OUT_JS.name} ({OUT_JS.stat().st_size // 1024} KB).")


def label_report(races: dict[str, dict]) -> None:
    """Prints how many races landed in 'Other' and the most common labels
    there, so any race description the classifier doesn't know shows up in
    the Actions log rather than silently skewing the numbers."""
    other = defaultdict(int)
    unknown_dist = 0
    for r in races.values():
        if r["rtype"] == "Other":
            other[re.sub(r"^[\dmfy]+\s*", "", r["race_name"])] += 1
        if r["furlongs"] == "":
            unknown_dist += 1
    if other:
        top = sorted(other.items(), key=lambda kv: -kv[1])[:8]
        print(f"Race type 'Other': {sum(other.values())} race(s). Most common labels: {top}")
    if unknown_dist:
        print(f"Races with unreadable distance: {unknown_dist}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", help="first race date to cover (YYYY-MM-DD)")
    ap.add_argument("--end", help="last race date (default yesterday)")
    ap.add_argument("--from-dir", help="read dwbfpricesukwin*.csv from this folder")
    args = ap.parse_args()

    start = date.fromisoformat(args.start) if args.start else DEFAULT_START
    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    update(start, end, Path(args.from_dir) if args.from_dir else None)

    races = load_races()
    if not races:
        print("No races stored - nothing to summarise. (Check the download log above.)")
        return 1
    label_report(races)
    write_js(build_summary(races))
    return 0


if __name__ == "__main__":
    sys.exit(main())
