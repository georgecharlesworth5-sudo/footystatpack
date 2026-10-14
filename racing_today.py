"""Today's UK and Irish racecard -> racing_today.js (read by the dashboard's Racing tab).

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
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

URL = "https://www.sportinglife.com/api/horse-racing/racing/racecards/{day}"
OUT_JS = Path(__file__).parent / "racing_today.js"
# The feed spells countries inconsistently ("Wale" for Wales, "Eire" for
# Ireland), so match on the long name too.
UK_SHORT = {"ENG", "SCO", "SCOT", "WAL", "WALE", "WLS", "NIR", "NI", "GB", "UK"}
UK_LONG = {"england", "scotland", "wales", "northern ireland", "great britain", "united kingdom"}


IRE_SHORT = {"IRE", "EIRE", "IRL", "ROI"}
IRE_LONG = {"eire", "ireland", "republic of ireland"}


def is_ire(country: dict) -> bool:
    short = (country.get("short_name") or "").strip().upper()
    long_ = (country.get("long_name") or "").strip().lower()
    return short in IRE_SHORT or long_ in IRE_LONG


def is_uk(country: dict) -> bool:
    short = (country.get("short_name") or "").strip().upper()
    long_ = (country.get("long_name") or "").strip().lower()
    return short in UK_SHORT or long_ in UK_LONG
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


def parse(payload, day: str, skipped=None):
    races = []
    skipped = skipped if skipped is not None else []
    for m in meetings_from(payload):
        if not isinstance(m, dict):
            continue
        ms = m.get("meeting_summary") or {}
        course = ms.get("course") or {}
        country = course.get("country") or {}
        region = "ire" if (country and is_ire(country)) else "uk"
        if country and region == "uk" and not is_uk(country):
            skipped.append(f"{course.get('name', '?')} ({country.get('long_name') or country.get('short_name')})")
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
                "region": region,
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


DETAIL_URL = "https://www.sportinglife.com/api/horse-racing/race/{rid}"
PROBE_OUT = Path(__file__).parent / "data" / "racing_probe.txt"


def probe_results() -> int:
    """Looks at yesterday's finished races in the feed and writes what it
    finds (finishing positions, any starting-price fields) to
    data/racing_probe.txt, so we can see whether a results view is
    possible. Changes nothing else."""
    from datetime import timedelta
    day = (datetime.now(LONDON).date() - timedelta(days=1)).isoformat()
    lines = [f"Probe for {day}, run {datetime.now(LONDON):%Y-%m-%d %H:%M} UK"]
    try:
        payload = fetch(day)
    except Exception as exc:  # noqa: BLE001
        lines.append(f"Could not read the card list: {exc}")
        PROBE_OUT.parent.mkdir(exist_ok=True)
        PROBE_OUT.write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return 1
    ids = []
    for m in meetings_from(payload):
        country = ((m.get("meeting_summary") or {}).get("course") or {}).get("country") or {}
        if country and not is_uk(country):
            continue
        for r in (m.get("races") or [])[:2]:
            rid = ((r.get("race_summary_reference") or {}).get("id"))
            if rid:
                ids.append((rid, r.get("course_name"), r.get("time"), r.get("name")))
    lines.append(f"UK races found: {len(ids)}; probing up to 3")
    lines.append("List-level keys on a race: " + ", ".join(sorted(((meetings_from(payload)[0].get('races') or [{}])[0]).keys())) if meetings_from(payload) else "no meetings")
    for rid, course, when, name in ids[:3]:
        lines.append(f"\n=== race {rid}: {when} {course} {name}")
        try:
            req = urllib.request.Request(DETAIL_URL.format(rid=rid), headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                d = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            lines.append(f"detail failed: {exc}")
            continue
        lines.append("top-level keys: " + ", ".join(sorted(d.keys())))
        rs = d.get("race_summary") or {}
        lines.append("race_summary keys: " + ", ".join(sorted(rs.keys())))
        lines.append(f"race_stage={rs.get('race_stage')} winning_time={d.get('winning_time')} result={str(d.get('result'))[:200]}")
        for ride in (d.get("rides") or [])[:20]:
            b = ride.get("betting") or {}
            lines.append(f"  pos={ride.get('finish_position')} status={ride.get('ride_status')} {(ride.get('horse') or {}).get('name')} betting={json.dumps(b)[:300]}")
    PROBE_OUT.parent.mkdir(exist_ok=True)
    PROBE_OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (default today, UK time)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--tomorrow", action="store_true", help="fetch tomorrow's card instead of today's")
    ap.add_argument("--probe-results", action="store_true",
                    help="inspect yesterday's results in the feed and write data/racing_probe.txt")
    args = ap.parse_args()
    if args.probe_results:
        return probe_results()
    day = args.date or (datetime.now(LONDON).date() + timedelta(days=int(args.tomorrow))).isoformat()
    try:
        payload = fetch(day)
    except Exception as exc:  # noqa: BLE001 - report any failure, keep old file
        print(f"Could not read the racecard feed for {day}: {exc}")
        return 1
    skipped: list[str] = []
    races = parse(payload, day, skipped)
    print("Meetings skipped (not UK or Irish): " + (", ".join(skipped) or "none"))
    if args.debug:
        for r in races:
            print(r["time"], r["course"], "|", r["name"], "|", r["code"], r["rtype"], r["distance"], r["runners"], "runners")
    if not races:
        print(f"Feed answered but no UK or Irish races were found for {day}. Leaving racing_today.js as it is.")
        return 1
    data = {"date": day, "generated": datetime.now(LONDON).strftime("%Y-%m-%d %H:%M"), "races": races}
    OUT_JS.write_text("const RACING_TODAY = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"Wrote {OUT_JS.name}: {len(races)} races on {day} ({sum(1 for r in races if r['region'] == 'uk')} UK, {sum(1 for r in races if r['region'] == 'ire')} Irish).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
