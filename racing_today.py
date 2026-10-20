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
import csv
import json
import re
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

URL = "https://www.sportinglife.com/api/horse-racing/racing/racecards/{day}"
OUT_JS = Path(__file__).parent / "racing_today.js"
OUT_JS_TOMORROW = Path(__file__).parent / "racing_tomorrow.js"
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
        lines.append(f"race_stage={rs.get('race_stage')} going={rs.get('going')!r} class={rs.get('race_class')!r} winning_time={d.get('winning_time')} result={str(d.get('result'))[:200]}")
        r0 = (d.get("rides") or [{}])[0]
        lines.append("ride keys: " + ", ".join(sorted(r0.keys())))
        lines.append("horse keys: " + ", ".join(sorted((r0.get("horse") or {}).keys())) + f" | jockey={str(r0.get('jockey'))[:120]} | trainer={str(r0.get('trainer'))[:120]}")
        for ride in (d.get("rides") or [])[:20]:
            b = ride.get("betting") or {}
            lines.append(f"  pos={ride.get('finish_position')} status={ride.get('ride_status')} {(ride.get('horse') or {}).get('name')} betting={json.dumps(b)[:300]}")
    PROBE_OUT.parent.mkdir(exist_ok=True)
    PROBE_OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


RESULTS_JSON = Path(__file__).parent / "data" / "racing_results.json"
RESULTS_JS = Path(__file__).parent / "racing_results.js"
RUNNERS_CSV = Path(__file__).parent / "data" / "racing_runners.csv"
RUNNER_COLS = ["date", "time", "course", "region", "code", "rtype", "going", "class", "ran", "horse", "jockey", "trainer",
               "age", "draw", "weight", "odds", "pos"]


def _nm(x):
    """A name from either a plain string or a {'name': ...} object."""
    if isinstance(x, dict):
        return str(x.get("name") or x.get("full_name") or x.get("short_name") or "").strip()
    return str(x or "").strip()


def runner_rows(detail, meta):
    """One row per runner (non-runners left out): who rode, who trained, the final
    price and where it finished. Used for going / jockey / trainer analysis later."""
    out = []
    for ride in detail.get("rides") or []:
        status = str(ride.get("ride_status") or "").upper()
        if "NON" in status or "WITHDRAW" in status:
            continue
        horse = ride.get("horse") if isinstance(ride.get("horse"), dict) else {}
        dec = to_decimal((ride.get("betting") or {}).get("current_odds"))
        pos = ride.get("finish_position")
        pos = int(pos) if isinstance(pos, (int, float)) or (isinstance(pos, str) and pos.isdigit()) else 0
        out.append({**meta, "horse": _nm(horse) or _nm(ride.get("horse")), "jockey": _nm(ride.get("jockey")),
                    "trainer": _nm(ride.get("trainer")), "age": horse.get("age", ride.get("age", "")),
                    "draw": ride.get("draw_number", ""), "weight": ride.get("handicap", ""),
                    "odds": round(dec, 2) if dec else "", "pos": pos})
    return out


def load_runner_keys():
    try:
        with open(RUNNERS_CSV, newline="") as f:
            return {(r["date"], r["time"], r["course"]) for r in csv.DictReader(f)}
    except OSError:
        return set()


def to_decimal(odds):
    """'5/2' -> 3.5, 'Evs'/'EVENS' -> 2.0, '2.5' -> 2.5. None if unreadable."""
    if odds is None:
        return None
    t = str(odds).strip().lower()
    if t in ("evs", "evens", "even"):
        return 2.0
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", t)
    if m and float(m.group(2)) > 0:
        return 1 + float(m.group(1)) / float(m.group(2))
    try:
        v = float(t)
        return v if v > 1 else None
    except ValueError:
        return None


def score_race(detail):
    """Reads a finished race's detail and returns the outcome measured against
    the betting (final odds), or None if the race isn't finished/readable.
    Ties in the betting at the top make favourite or 2nd favourite ambiguous,
    and those fields are left as None rather than guessed."""
    rides = detail.get("rides") or []
    runners = []
    for ride in rides:
        status = str(ride.get("ride_status") or "").upper()
        if "NON" in status or "WITHDRAW" in status:
            continue
        b = ride.get("betting") or {}
        dec = to_decimal(b.get("current_odds"))
        pos = ride.get("finish_position")
        pos = int(pos) if isinstance(pos, (int, float)) or (isinstance(pos, str) and pos.isdigit()) else 0
        runners.append({"dec": dec, "pos": pos})
    if len(runners) < 2:
        return None
    if not any(r["pos"] == 1 for r in runners):
        return None                      # not finished (or abandoned)
    priced = [r for r in runners if r["dec"] is not None]
    if len(priced) < 2:
        return None
    priced.sort(key=lambda r: r["dec"])
    winners = [r for r in runners if r["pos"] == 1]
    out = {"ran": len(runners), "fav": None, "fav_pos": None, "sec_pos": None,
           "win_rank": None, "fav_odds": None}
    d0, d1 = priced[0]["dec"], priced[1]["dec"]
    out["fav_odds"] = round(d0, 2)
    if d0 != d1:                          # a clear favourite
        out["fav_pos"] = priced[0]["pos"]
        out["fav"] = 1 if priced[0]["pos"] == 1 else 0
        if len(priced) < 3 or d1 != priced[2]["dec"]:   # and a clear 2nd favourite
            out["sec_pos"] = priced[1]["pos"]
    wdec = winners[0]["dec"]
    if wdec is not None and len(winners) == 1:
        out["win_rank"] = 1 + sum(1 for r in priced if r["dec"] < wdec)
    return out


def load_results():
    try:
        return json.loads(RESULTS_JSON.read_text())
    except (OSError, ValueError):
        return {}


def collect_results(days, pause: float = 0.0) -> int:
    """For each day, reads the card, opens every race, and stores the outcome
    of those that have finished. Safe to re-run: a race is stored once."""
    store = load_results()
    runner_keys = load_runner_keys()
    new_runners = []
    added = 0
    for day in days:
        try:
            payload = fetch(day)
        except Exception as exc:  # noqa: BLE001
            print(f"{day}: could not read the card list ({exc})")
            continue
        ids = {}
        for m in meetings_from(payload):
            for r in m.get("races") or []:
                rid = (r.get("race_summary_reference") or {}).get("id")
                if rid:
                    ids[(r.get("course_name") or "", r.get("time") or "")] = rid
        races = parse(payload, day)
        # parse() gives UK-time strings; match back to ids through the raw list
        raw = {}
        for m in meetings_from(payload):
            for r in m.get("races") or []:
                rid = (r.get("race_summary_reference") or {}).get("id")
                if rid:
                    raw[(r.get("course_name") or (m.get("meeting_summary") or {}).get("course", {}).get("name") or "", r.get("name"))] = rid
        done_here = 0
        for r in races:
            key = f"{day}|{r['time']}|{r['course']}"
            have_runners = (day, r["time"], r["course"]) in runner_keys
            if key in store and store[key].get("ran") and "going" in store[key] and have_runners:
                continue
            rid = raw.get((r["course"], r["name"])) or raw.get((r["course"], re.sub(r"\s*\((GBB|GBBPlus)[^)]*\)\s*$", "", r["name"]).strip()))
            if not rid:
                print(f"  {key}: no race id matched ({r['name']})")
                continue
            try:
                req = urllib.request.Request(DETAIL_URL.format(rid=rid), headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    d = json.loads(resp.read().decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                print(f"  {key}: detail failed ({exc})")
                continue
            if pause:
                time.sleep(pause)
            res = score_race(d)
            if not res:
                rides = d.get("rides") or []
                print(f"  {key}: not scored - {len(rides)} rides, "
                      f"{sum(1 for x in rides if x.get('finish_position') in (1, '1'))} winner(s), "
                      f"{sum(1 for x in rides if (x.get('betting') or {}).get('current_odds'))} with odds")
                continue
            rsum = d.get("race_summary") or {}
            going = (rsum.get("going") or r.get("going") or "").strip()
            klass = (rsum.get("race_class") or r.get("class") or "").strip()
            is_new = not (key in store and store[key].get("ran"))
            store[key] = {"date": day, "time": r["time"], "course": r["course"], "region": r["region"],
                          "code": r["code"], "rtype": r["rtype"], "furlongs": r["furlongs"],
                          "name": r["name"], "going": going, "class": klass, **res}
            if not have_runners:
                meta = {"date": day, "time": r["time"], "course": r["course"], "region": r["region"], "code": r["code"],
                        "rtype": r["rtype"], "going": going, "class": klass, "ran": res["ran"]}
                new_runners += runner_rows(d, meta)
                runner_keys.add((day, r["time"], r["course"]))
            if is_new:
                added += 1
            done_here += 1
        print(f"{day}: {done_here} new results ({len(races)} races on the card)")
    # keep the file to the last ~400 days so it stays small
    keep = sorted(store, reverse=True)[:6000]
    store = {k: store[k] for k in keep}
    RESULTS_JSON.parent.mkdir(exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(store, separators=(",", ":"), sort_keys=True))
    rows = sorted(store.values(), key=lambda x: (x["date"], x["time"], x["course"]))
    RESULTS_JS.write_text("const RACING_RESULTS = " + json.dumps({"updated": datetime.now(LONDON).strftime("%Y-%m-%d %H:%M"), "races": rows}, separators=(",", ":")) + ";\n")
    if new_runners:
        RUNNERS_CSV.parent.mkdir(exist_ok=True)
        fresh = not RUNNERS_CSV.exists()
        with open(RUNNERS_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RUNNER_COLS)
            if fresh:
                w.writeheader()
            w.writerows(new_runners)
        print(f"Runner rows added: {len(new_runners)} ({sum(1 for x in new_runners if x['jockey'])} with a jockey, "
              f"{sum(1 for x in new_runners if x['trainer'])} with a trainer, {sum(1 for x in new_runners if x['going'])} with going).")
    print(f"Stored {len(store)} results in total ({added} new).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (default today, UK time)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--tomorrow", action="store_true", help="fetch tomorrow's card instead of today's")
    ap.add_argument("--probe-results", action="store_true",
                    help="inspect yesterday's results in the feed and write data/racing_probe.txt")
    ap.add_argument("--results", action="store_true",
                    help="record the outcomes of yesterday's and today's finished races")
    ap.add_argument("--backfill-days", type=int, default=0,
                    help="with --results: go back this many days instead of just yesterday and today")
    args = ap.parse_args()
    if args.results:
        today = datetime.now(LONDON).date()
        if args.backfill_days > 0:
            days = [(today - timedelta(days=i)).isoformat() for i in range(args.backfill_days, -1, -1)]
        else:
            days = [args.date] if args.date else [(today - timedelta(days=1)).isoformat(), today.isoformat()]
        return collect_results(days, pause=0.2 if args.backfill_days > 0 else 0.0)
    if args.probe_results:
        return probe_results()
    today = datetime.now(LONDON).date()
    if args.date:
        return write_card(args.date, OUT_JS, "RACING_TODAY", args.debug)
    if args.tomorrow:
        return write_card((today + timedelta(days=1)).isoformat(), OUT_JS_TOMORROW, "RACING_TOMORROW", args.debug)
    # Default: refresh BOTH cards every run, so the page always has today's
    # card even if the morning run is late or missed.
    rc = write_card(today.isoformat(), OUT_JS, "RACING_TODAY", args.debug)
    write_card((today + timedelta(days=1)).isoformat(), OUT_JS_TOMORROW, "RACING_TOMORROW", args.debug)
    return rc


def write_card(day: str, out: Path, var: str, debug: bool) -> int:
    try:
        payload = fetch(day)
    except Exception as exc:  # noqa: BLE001 - report any failure, keep old file
        print(f"Could not read the racecard feed for {day}: {exc}")
        return 1
    skipped: list[str] = []
    races = parse(payload, day, skipped)
    print("Meetings skipped (not UK or Irish): " + (", ".join(skipped) or "none"))
    if debug:
        for r in races:
            print(r["time"], r["course"], "|", r["name"], "|", r["code"], r["rtype"], r["distance"], r["runners"], "runners")
    if not races:
        print(f"Feed answered but no UK or Irish races were found for {day}. Leaving {out.name} as it is.")
        return 1
    data = {"date": day, "generated": datetime.now(LONDON).strftime("%Y-%m-%d %H:%M"), "races": races}
    out.write_text(f"const {var} = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"Wrote {out.name}: {len(races)} races on {day} ({sum(1 for r in races if r['region'] == 'uk')} UK, {sum(1 for r in races if r['region'] == 'ire')} Irish).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
