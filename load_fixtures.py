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

Earlier versions of this file assumed fixturedownload.com's times were
fixed UTC+0 (no daylight-saving adjustment) and added an hour during
BST. That assumption turned out to be WRONG - verified directly
against a real fixture: fixturedownload.com's EPL page shows "20:00"
for the Arsenal v Coventry season-opener, and that match's actual
kickoff (per Arsenal's own site and the Premier League) is confirmed
as 8pm UK time - i.e. the site's raw value was already correct local
UK clock time, and our "correction" was pushing it an hour late. The
BST-adjustment code has been removed - times are now used as-is.
"""

import csv
from pathlib import Path

LEAGUE_FILES = {
    "E0": "E0.csv",
    "E1": "E1.csv",
    "E2": "E2.csv",
    "E3": "E3.csv",
    "SC0": "SC0.csv",
}


def load_league_fixtures(fixtures_dir: Path, div_code: str) -> list[dict]:
    """Load one league's manually-downloaded fixturedownload.com CSV,
    returning only matches that haven't been played yet, in our
    internal fixture format: Div, Date, Time, HomeTeam, AwayTeam."""
    path = fixtures_dir / LEAGUE_FILES[div_code]
    if not path.exists():
        return []

    fixtures = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result = (row.get("Result") or "").strip()
            if result and result != "-":
                continue  # already played

            date_time = (row.get("Date") or "").strip()
            date_part, _, time_part = date_time.partition(" ")

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
