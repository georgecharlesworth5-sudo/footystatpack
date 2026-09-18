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
    La Liga                  https://fixturedownload.com/results/la-liga-2026              SP1.csv
    Bundesliga                https://fixturedownload.com/results/bundesliga-2026           D1.csv

(Season slugs will roll over to e.g. "epl-2027" next season - check
fixturedownload.com/index if a URL above 404s. The La Liga/Bundesliga
slugs above are best guesses following the same pattern as the others -
check fixturedownload.com/index directly if either 404s.)

fixturedownload.com's CSV columns are: Round Number, Date, Location,
Home Team, Away Team, Result. "Date" combines date+time
(dd/mm/yyyy HH:MM) and "Result" is blank ("-") for matches not yet
played - this module splits Date into our Date/Time fields and treats
any row with no result as an upcoming fixture.

## Delimiter - confirmed inconsistent, so this is auto-detected

fixturedownload.com's own "Download as CSV" button doesn't reliably
produce a comma-separated file despite the name and extension - the
Bundesliga download came through genuinely TAB-separated, which a
plain comma-delimited csv.DictReader can't parse at all (it reads the
whole header as one giant column name, and every field then comes back
empty - confirmed directly: that's exactly what a real "no history
found for ['', '']" skip message turned out to be, not a team-name
mismatch the way Serie A/La Liga's issues were). Rather than assume
one delimiter and require every future download to happen to match it,
each file's actual delimiter is sniffed from its own header line
before parsing - see _detect_delimiter below.

## Timezone note

fixturedownload.com shows the exact same generic "your time zone is
not set" disclaimer on every competition page, but that text turns out
to be meaningless as a signal - it appears regardless of what the page
actually does underneath. Verified directly against real fixtures:

  - Premier League (epl-2026): raw time is ALREADY correct UK local
    time. Confirmed against Arsenal v Coventry - the site shows
    "20:00", and that match's real kickoff (per Arsenal's own site and
    the Premier League) is 8pm UK time. No adjustment needed.
  - The other English/Scottish leagues: reported wrong (an hour off)
    once the BST correction below was removed for everyone - i.e.
    unlike the EPL page, they DO need it.

So this isn't a single site-wide behaviour - it's inconsistent
per-competition-page, for reasons we can't see into. Rather than
guess further, LEAGUE_TIME_ADJUSTMENT below is a per-league setting
you can adjust based on what you actually observe. If a league's
times ever look off, that's the fix: adjust its entry here.

For a foreign league, the correction ISN'T the same shape as the BST
one above: European countries covered so far are a CONSTANT 1 hour
ahead of the UK year-round (all shift clocks on the same EU-wide
dates, so the gap never changes) - it's not a seasonal correction,
it's a flat offset. Confirmed directly for Serie A and La Liga that
the correction actually needed is +1hr (i.e. -1 passed to
_apply_flat_offset, which subtracts), NOT the -1hr a naive "always
UK+1" assumption would suggest - fixturedownload.com's raw time for
both of those turned out not to be genuine local time in the way
first assumed. Bundesliga defaults to the same -1 given that pattern
now holding for two leagues in a row, but - same as always - check a
real fixture once loaded rather than trusting the assumption blind.
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
    "I1": "I1.csv",
    "SP1": "SP1.csv",
    "D1": "D1.csv",
}

# Each entry is one of:
#   "bst"      - apply the seasonal UK BST correction (see _to_uk_local)
#   <int>      - apply a FLAT hour offset year-round (for foreign
#                leagues where the gap to UK time never changes) - a
#                positive number means fixturedownload.com's raw time
#                needs that many hours SUBTRACTED to reach UK time,
#                negative means ADDED
#   None / 0   - no adjustment, use the raw time as-is
LEAGUE_TIME_ADJUSTMENT = {
    "E0": None,   # Premier League - confirmed correct as-is (Arsenal v Coventry check)
    "E1": "bst",  # Championship
    "E2": "bst",  # League One
    "E3": "bst",  # League Two
    "SC0": "bst", # Scottish Premiership
    "I1": -1,     # Serie A - CONFIRMED needing +1hr, same reversed-sign finding as
                  # La Liga below (5:45 shown, needed 7:45, with the old hours=1
                  # code) - fixturedownload.com's raw Serie A time isn't genuine
                  # Italian local time either, same as La Liga.
    "SP1": -1,    # La Liga - CONFIRMED needing +1hr (not the -1hr originally assumed
                  # by the "always UK+1" reasoning) - real-world check showed the
                  # -1hr version was 2 hours out, meaning the true correction runs the
                  # opposite direction to what a flat "always UK+1" rule would suggest.
                  # fixturedownload.com's raw La Liga time evidently isn't genuine
                  # Spanish local time the way assumed.
    "D1": -1,     # Bundesliga - NOT YET independently confirmed, but set to -1 rather
                  # than an untested None default - Italy AND Spain have both now
                  # confirmed needing this same reversed correction, a real pattern
                  # worth acting on. Still worth checking a real fixture once loaded.
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


def _apply_flat_offset(date_str: str, time_str: str, hours: float) -> tuple[str, str]:
    """Subtract a constant number of hours (positive = subtract,
    negative = add) - no seasonal logic, just a fixed shift."""
    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
    except ValueError:
        return date_str, time_str
    dt -= timedelta(hours=hours)
    return dt.strftime("%d/%m/%Y"), dt.strftime("%H:%M")


def _detect_delimiter(header_line: str) -> str:
    """fixturedownload.com's own CSV export has been observed to come
    through as either genuinely comma-separated OR genuinely
    tab-separated, inconsistently between leagues/downloads, despite
    always being named/labelled as a CSV. A plain csv.DictReader
    assuming comma silently mis-parses a tab-separated file entirely -
    every field comes back empty rather than raising an error, which
    is exactly what happened with a real Bundesliga upload. Picking
    the delimiter that actually appears in the header line, per file,
    avoids depending on which format happens to come out of any given
    download."""
    return "\t" if header_line.count("\t") > header_line.count(",") else ","


def load_league_fixtures(fixtures_dir: Path, div_code: str) -> list[dict]:
    """Load one league's manually-downloaded fixturedownload.com CSV,
    returning only matches that haven't been played yet, in our
    internal fixture format: Div, Date, Time, HomeTeam, AwayTeam."""
    path = fixtures_dir / LEAGUE_FILES[div_code]
    if not path.exists():
        return []

    adjustment = LEAGUE_TIME_ADJUSTMENT.get(div_code)

    with open(path, encoding="utf-8-sig") as f:
        first_line = f.readline()
    delimiter = _detect_delimiter(first_line)

    fixtures = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            result = (row.get("Result") or "").strip()
            if result and result != "-":
                continue  # already played

            date_time = (row.get("Date") or "").strip()
            date_part, _, time_part = date_time.partition(" ")
            if time_part and adjustment == "bst":
                date_part, time_part = _to_uk_local(date_part, time_part)
            elif time_part and isinstance(adjustment, (int, float)) and adjustment:
                date_part, time_part = _apply_flat_offset(date_part, time_part, adjustment)

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
