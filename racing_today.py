"""Today's UK racecard -> racing_today.js (read by the dashboard's Racing tab).

Source: Sporting Life's public racecard feed
    https://www.sportinglife.com/api/horse-racing/racing/racecards/YYYY-MM-DD
It lists every meeting with each race's time, name, distance, handicap flag
and number of runners. This script turns each race into the same labels the
history uses (code, race type, field size) so the dashboard can look up how
that kind of race has behaved. It does NOT fetch prices or tips.

    python racing_today.py                  # today
    python racing_today.py --date 2026-10-12  # tomorrow, say
    python racing_today.py --debug          # print what it parsed

If the feed can't be read the existing racing_today.js is left alone and the
script exits with an error, so a blocked or changed feed never wipes the page.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

URL = "https://www.sportinglife.com/api/horse-racing/racing/racecards/{day}"
OUT_JS = Path(__file__).parent / "racing_today.js"
UK_COUNTRIES = {"ENG", "SCO", "WAL", "NIR", "GB", "UK"}
LONDON = ZoneInfo("Europe/London")
UTC = ZoneInfo("UTC")


def fetch(day: str):
    req = urllib.request.Request(URL.format(day=day), headers={
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.sportinglife.com/racing/racecards",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def furlongs(text: str):
    """'1m 1f 11y' -> 9.05 ; '2m4' f -> 20.5 ; '6f' -> 6.0"""
    if not text:
        return None
    t = text.lower().replace("½", ".5").replace("¼", ".25").replace("¾", ".75")
    m = re.fullmatch(r"\s*(?:(\d+)\s*m)?\s*(?:(\d+(?:\.\d+)?)\s*f)?\s*(?:(\d+)\s*y)?\s*", t)
    if not m or not any(m.groups()):
        return None
    miles, f, y = m.groups()
    return round(int(miles or 0) * 8 + float(f or 0) + int(y or 0) / 220, 2)


def classify_code(name: str, surface: str) -> str:
    n = name.lower()
    if re.search(r"\b(national hunt flat|nh flat|bumper|n\.?h\.? flat)\b", n):
        return "bumper"
    if re.search(r"\b(chase|steeplechase|steeple-chase|chs)\b", n):
        return "chase"
    if re.search(r"\b(hurdle|hrd|hurdles)\b", n):
        return "hurdle"
    return "flat"


def classify_type(name: str, code: str, handicap: bool) -> str:
    n = name.lower()
    if re.search(r"\b(group|grade)\s*(1|2|3|one|two|three|i{1,3})\b", n):
        return "Group/Graded"
    if re.search(r"\blisted\b", n):
        return "Listed"
    if "nursery" in n:
        return "Nursery"
    if handicap or "handicap" in n:
        return "Handicap"
    if "maiden" in n or "mdn" in n:
        return "Maiden"
    if re.search(r"\b(novice|novices|beginners|introductory|graduation)\b", n):
        return "Novice"
    if "juvenile" in n:
        return "Juvenile"
    if re.search(r"\b(selling|claiming|seller)\b", n):
        return "Selling/Claiming"
    if re.search(r"\bhunters?\b", n):
        return "Hunters"
    if code == "bumper":
        return "Bumper"
    return "Conditions/Class"


def meetings_from(payload):
    if isinstance(payload, dict):
        for k in ("meetings", "data", "racecards"):
            if isinstance(payload.get(k), list):
                return payload[k]
        return [payload]
    return payload or []


def parse(payload, day: str):
    races = []
    for m in meetings_from(payload):
        if not isinstance(m, dict):
            continue
        ms = m.get("meeting_summary") or {}
        course = ms.get("course") or {}
        country = ((course.get("country") or {}).get("short_name") or "").upper()
        if country and country not in UK_COUNTRIES:
            continue
        surface = (ms.get("surface_summary") or "").strip()
        for r in m.get("races") or []:
            if r.get("hidden"):
                continue
            name = (r.get("name") or "").strip()
            name = re.sub(r"\s*\((GBB|GBBPlus)[^)]*\)\s*$", "", name).strip()
            hhmm = r.get("time") or ""
            when = ""
            try:  # the feed's times are UTC; the dashboard shows UK time
                dt = datetime.fromisoformat(f"{r.get('date') or day}T{hhmm}").replace(tzinfo=UTC)
                when = dt.astimezone(LONDON).strftime("%H:%M")
            except ValueError:
                when = hhmm
            code = classify_code(name, surface)
            handicap = bool(r.get("has_handicap"))
            races.append({
                "time": when,
                "course": r.get("course_name") or course.get("name") or "",
                "name": name,
                "code": code,
                "rtype": classify_type(name, code, handicap),
                "furlongs": furlongs(r.get("distance") or ""),
                "distance": r.get("distance") or "",
                "runners": int(r.get("ride_count") or 0),
                "going": r.get("going") or ms.get("going") or "",
                "surface": (r.get("course_surface") or {}).get("surface") or surface,
                "class": r.get("race_class") or "",
            })
    races = [r for r in races if r["runners"] > 0]
    races.sort(key=lambda r: (r["time"], r["course"]))
    return races


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (default today, UK time)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    day = args.date or datetime.now(LONDON).date().isoformat()
    try:
        payload = fetch(day)
    except Exception as exc:  # noqa: BLE001 - report any failure, keep old file
        print(f"Could not read the racecard feed for {day}: {exc}")
        return 1
    races = parse(payload, day)
    if args.debug:
        for r in races:
            print(r["time"], r["course"], "|", r["name"], "|", r["code"], r["rtype"], r["distance"], r["runners"], "runners")
    if not races:
        print(f"Feed answered but no UK races were found for {day}. Leaving racing_today.js as it is.")
        return 1
    data = {"date": day, "generated": datetime.now(LONDON).strftime("%Y-%m-%d %H:%M"), "races": races}
    OUT_JS.write_text("const RACING_TODAY = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"Wrote {OUT_JS.name}: {len(races)} UK races on {day}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
