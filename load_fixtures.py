"""
load_fixtures.py

Loads upcoming fixtures from CSVs you download yourself from
fixturedownload.com - one file per league.

Why manual rather than automated: fixturedownload.com's robots.txt
blocks automated access to their download endpoints, and their terms &
conditions prohibit storing their content in another electronic
retrieval system. Their site publishes full-season fixture lists (all
38 rounds, months in advance) via a "Download as CSV" button - exactly
what we need, but it's meant to be used the way it's presented: a
person clicking download, not a script polling it. So: you download,
we just read what's on disk.

## One-time setup (and again whenever you want the latest fixtures -
##  e.g. once a week, or whenever a new round's dates/times firm up):

Visit each of these pages and click "Download as CSV", saving into a
`fixtures_manual/` folder next to this script, using EXACTLY these
filenames:

    League                  URL                                                          Save as
    Premier League          https://fixturedownload.com/results/epl-2026                  E0.csv
    Championship             https://fixturedownload.com/results/championship-2026         E1.csv
    League One               https://fixturedownload.com/results/efl-league-one-2026       E2.csv
    League Two               https://fixturedownload.com/results/efl-league-two-2026       E3.csv
    Scottish Premiership     https://fixturedownload.com/results/scottish-premiership-2026  SC0.csv

(Season slugs will roll over to e.g. "epl-2027" next season - check
fixturedownload.com/index if a URL above 404s.)

fixturedownload.com's CSV columns are: Round Number, Date, Location,
Home Team, Away Team, Result. "Date" combines date+time
(dd/mm/yyyy HH:MM) and "Result" is blank ("-") for matches not yet
played - this module splits Date into our Date/Time fields and treats
any row with no result as an upcoming fixture.

## Timezone note

fixturedownload.com shows the exact same generic "your time zone is
not set" disclaimer on every competition page, but that text turns out
to be meaningless as a signal - it appears regardless of what the page
actually does underneath. Verified directly against real fixtures:

  - Premier League (epl-2026): raw time is ALREADY correct UK local
    time. Confirmed against Arsenal v Coventry - the site shows
    "20:00", and that match's real kickoff (per Arsenal's own site and
    the Premier League) is 8pm UK time. No adjustment needed.
  - The other leagues: reported wrong (an hour off) once the BST
    correction below was removed for everyone - i.e. unlike the EPL
    page, they DO need it.

So this isn't a single site-wide behaviour - it's inconsistent
per-competition-page, for reasons we can't see into. Rather than
guess further, NEEDS_BST_CORRECTION below is a per-league flag you can
flip based on what you actually observe. If a league's times ever look
an hour out (either direction), that's the fix: toggle its flag here.
"""

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

LEAGUE_FILES = {
    "E0": "E0.csv",
    "E1": "E1.csv",
    "E2": "E2.csv",
    "E3": "E3.csv",
    "SC0": "SC0.csv",
}

# See "Timezone note" above. True = this league's raw fixturedownload.com
# times are fixed UTC+0 and need +1hr during BST; False = already correct
# local UK time, use as-is.
NEEDS_BST_CORRECTION = {
    "E0": False,  # Premier League - confirmed correct as-is (Arsenal v Coventry check)
    "E1": True,   # Championship
    "E2": True,   # League One
    "E3": True,   # League Two
    "SC0": True,  # Scottish Premiership
}


def _last_sunday(year: int, month: int) -> int:
    """Day-of-month of the last Sunday in a given month/year."""
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    d = datetime(year, month, last_day)
    return last_day - ((d.weekday() + 1) % 7)  # weekday(): Mon=0 .. Sun=6


def _is_bst(dt_utc: datetime) -> bool:
    """UK clocks go forward 1 hour at 01:00 UTC on the last Sunday of
    March, and back at 01:00 UTC on the last Sunday of October."""
    year = dt_utc.year
    start = datetime(year, 3, _last_sunday(year, 3), 1, 0, tzinfo=timezone.utc)
    end = datetime(year, 10, _last_sunday(year, 10), 1, 0, tzinfo=timezone.utc)
    return start <= dt_utc < end


def _to_uk_local(date_str: str, time_str: str) -> tuple[str, str]:
    """Given a fixturedownload.com date/time treated as fixed UTC+0,
    return (date, time) adjusted to actual UK local clock time."""
    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
    except ValueError:
        return date_str, time_str  # malformed/missing time - leave as-is

    dt_utc = dt.replace(tzinfo=timezone.utc)
    if _is_bst(dt_utc):
        dt_utc += timedelta(hours=1)
    return dt_utc.strftime("%d/%m/%Y"), dt_utc.strftime("%H:%M")


def load_league_fixtures(fixtures_dir: Path, div_code: str) -> list[dict]:
    """Load one league's manually-downloaded fixturedownload.com CSV,
    returning only matches that haven't been played yet, in our
    internal fixture format: Div, Date, Time, HomeTeam, AwayTeam."""
    path = fixtures_dir / LEAGUE_FILES[div_code]
    if not path.exists():
        return []

    needs_correction = NEEDS_BST_CORRECTION.get(div_code, True)

    fixtures = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = (row.get("Result") or "").strip()
            if result and result != "-":
                continue  # already played

            date_time = (row.get("Date") or "").strip()
            date_part, _, time_part = date_time.partition(" ")
            if time_part and needs_correction:
                date_part, time_part = _to_uk_local(date_part, time_part)

            fixtures.append({
                "Div": div_code,
                "Date": date_part,
                "Time": time_part,
                "HomeTeam": (row.get("Home Team") or "").strip(),
                "AwayTeam": (row.get("Away Team") or "").strip(),
            })
    return fixtures


def load_all_fixtures(fixtures_dir: Path) -> list[dict]:
    """Load every league's manually-downloaded fixture file, skipping
    any that haven't been downloaded yet (prints what's missing)."""
    all_fixtures = []
    for code in LEAGUE_FILES:
        league_fixtures = load_league_fixtures(fixtures_dir, code)
        if not league_fixtures:
            print(f"  [info] no fixtures file for {code} yet "
                  f"(expected {fixtures_dir / LEAGUE_FILES[code]}) - skipping")
        all_fixtures.extend(league_fixtures)
    return all_fixtures


if __name__ == "__main__":
    fixtures_dir = Path(__file__).parent / "fixtures_manual"
    fixtures_dir.mkdir(exist_ok=True)
    fixtures = load_all_fixtures(fixtures_dir)
    print(f"\n{len(fixtures)} upcoming fixtures loaded across all leagues.")
