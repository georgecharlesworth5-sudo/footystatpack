"""
fetch_mls.py

Pulls MLS (Major League Soccer) results and upcoming fixtures from
API-Football (api-sports.io), and writes them in the SAME shapes the
rest of the pipeline already expects:

  - data/MLS.csv            - one row per finished match, same columns
                               fetch_data.py writes for the other 9
                               leagues (Div, Date, HomeTeam, AwayTeam,
                               FTHG, FTAG, HTHG, HTAG, HC, AC, HY, AY,
                               HR, AR, ...) - so stats_engine.py and
                               build_statpack.py need no MLS-specific
                               code at all, just "MLS" added to
                               build_statpack.LEAGUE_NAMES.
  - fixtures_manual/MLS.csv - one row per not-yet-played fixture, in
                               fixturedownload.com's own column shape
                               (Round Number, Date, Location, Home
                               Team, Away Team, Result) - so
                               load_fixtures.py needs no MLS-specific
                               code either, just "MLS" added to its
                               LEAGUE_FILES dict.

Why MLS needed its own fetch script rather than just another league
code in fetch_data.py: football-data.co.uk DOES carry MLS, but only in
its "extra leagues" combined file (a totally different column layout -
goals and odds only, no corners or cards at all). There's no corners/
cards data for MLS available from the source the rest of this project
uses. API-Football's free tier (100 requests/day) does have it, at the
cost of needing a real API key and some request budgeting - see below.

## One-time setup

1. Create a free account at https://www.api-football.com (or via
   RapidAPI) and grab your API key from the dashboard.
2. Set it as an environment variable named API_FOOTBALL_KEY - as a
   GitHub Actions repository secret (Settings -> Secrets and
   variables -> Actions -> New repository secret) for the automated
   runs, AND as a local environment variable (e.g. in ~/.zshrc on Mac,
   or via `setx` on Windows) for running this locally. NEVER hardcode
   it into this file or any committed file.

## Why this doesn't just fetch everything in one go

The free plan is capped at 100 requests/day, and a full MLS season is
~500+ matches. The /fixtures endpoint itself is cheap (the WHOLE
season's fixture list - results and upcoming - comes back in a single
request), but each match's corners/cards figures need their own
separate /fixtures/statistics call. Backfilling a mid-season's worth of
history at MAX_STATS_REQUESTS_PER_RUN per run, run twice a day (this
project's existing cron), takes a week or two to catch up - after
that, steady state is just the handful of games actually played since
the last run, comfortably inside the daily cap.

This script is deliberately stateless beyond the CSV files themselves:
"what's left to backfill" is recomputed each run as "finished fixtures
API-Football knows about, minus ones already sitting in data/MLS.csv"
(matched by FixtureID, an extra column tacked onto the end of our usual
columns - harmless to the rest of the pipeline, which only reads the
columns it knows about).
"""

import csv
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

API_BASE = "https://v3.football.api-sports.io"

# Conservative on purpose: update-statpack.yml's cron currently fires
# twice a day (every 12 hours). 40 stats requests/run x 2 runs/day = 80,
# leaving real headroom under the 100/day cap for the (cheap, usually
# 1-request) fixtures-list call each run, plus the one-off league-ID
# lookup on the very first run ever. Lower this if the cron frequency
# ever goes up, or raise it if it's ever reduced to once a day.
MAX_STATS_REQUESTS_PER_RUN = 40

# Same column set fetch_data.py writes for the other leagues (see its
# KEEP_COLUMNS) minus the ones API-Football doesn't give us (shots
# breakdowns, FTR/HTR - both trivially derivable, so left for
# stats_engine.py to not need at all) plus FixtureID on the end, which
# the rest of the pipeline simply ignores (extra dict keys are harmless
# to csv.DictReader-based consumers) and we use to track what's already
# been backfilled.
RESULTS_COLUMNS = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "HTHG", "HTAG",
    "HC", "AC", "HY", "AY", "HR", "AR",
    "Referee", "FixtureID",
]

# Status codes API-Football uses that mean "this match has a final
# result" - AET/PEN are extra-time/penalties finishes, which still
# carry a normal final score (penalty-shootout goals aren't folded into
# FTHG/FTAG, same as how football-data.co.uk's own cup data works).
FINISHED_STATUSES = {"FT", "AET", "PEN"}
NOT_STARTED_STATUS = "NS"


def _api_get(path: str, params: dict) -> dict:
    api_key = os.environ.get("API_FOOTBALL_KEY")
    if not api_key:
        print("[error] API_FOOTBALL_KEY environment variable is not set - "
              "see the setup notes at the top of this file.")
        sys.exit(1)

    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{API_BASE}/{path}?{query}"
    request = urllib.request.Request(url, headers={"x-apisports-key": api_key})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"[error] API-Football request failed ({e.code}) for {path}: {e.read().decode('utf-8', 'ignore')}")
        raise

    errors = payload.get("errors")
    if errors:
        # API-Football returns HTTP 200 with an "errors" object for
        # things like a used-up daily quota, rather than an HTTP error
        # code - has to be checked explicitly, an HTTP-status-only check
        # would miss this entirely.
        print(f"[warning] API-Football returned errors for {path}: {errors}")
    return payload


def find_league_id(cache_path: Path) -> int:
    """Resolves MLS's numeric league ID once, then caches it locally so
    every future run skips this lookup entirely. Deliberately not
    hardcoded - rather than trust an unverified number, this asks the
    API directly and checks the result actually says "Major League
    Soccer" / USA before trusting it."""
    if cache_path.exists():
        return int(cache_path.read_text().strip())

    payload = _api_get("leagues", {"search": "MLS"})
    candidates = payload.get("response", [])
    for entry in candidates:
        league = entry.get("league", {})
        country = entry.get("country", {})
        if (league.get("name", "").strip().lower() == "major league soccer"
                and country.get("name", "").strip().lower() in ("usa", "united states")):
            league_id = league["id"]
            cache_path.write_text(str(league_id))
            print(f"[debug] Resolved MLS league ID to {league_id} (cached at {cache_path})")
            return league_id

    raise RuntimeError(
        f"Could not find 'Major League Soccer' / USA in API-Football's leagues search "
        f"response: {candidates}. Check your API key is valid and has quota left."
    )


def _stat_value(statistics: list[dict], stat_type: str) -> int:
    for entry in statistics:
        if entry.get("type") == stat_type:
            value = entry.get("value")
            return int(value) if value not in (None, "") else 0
    return 0


def fetch_fixture_statistics(fixture_id: int) -> dict | None:
    """One request. Returns {"home": {...}, "away": {...}} each with
    corners/yellow/red counts, or None if the API has no stats for this
    fixture yet (can happen briefly right after a match ends)."""
    payload = _api_get("fixtures/statistics", {"fixture": fixture_id})
    response = payload.get("response", [])
    if len(response) != 2:
        return None

    by_team_id = {}
    for side in response:
        team_id = side.get("team", {}).get("id")
        stats = side.get("statistics", [])
        by_team_id[team_id] = {
            "corners": _stat_value(stats, "Corner Kicks"),
            "yellow": _stat_value(stats, "Yellow Cards"),
            "red": _stat_value(stats, "Red Cards"),
        }
    return by_team_id


def _to_uk_local(iso_date: str) -> tuple[str, str]:
    """API-Football's fixture.date is a real ISO 8601 timestamp with its
    own UTC offset (unlike fixturedownload.com's ambiguous raw times,
    which needed per-league guesswork to convert - see
    load_fixtures.py). Converting via zoneinfo from an explicit,
    correctly-offset source timestamp needs no guessing at all.
    Returns (dd/mm/yyyy, HH:MM) in UK local time."""
    dt = datetime.fromisoformat(iso_date)
    dt_uk = dt.astimezone(ZoneInfo("Europe/London"))
    return dt_uk.strftime("%d/%m/%Y"), dt_uk.strftime("%H:%M")


def load_existing_results(path: Path) -> tuple[list[dict], set[str]]:
    if not path.exists():
        return [], set()
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    known_ids = {row["FixtureID"] for row in rows if row.get("FixtureID")}
    return rows, known_ids


def write_results(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in RESULTS_COLUMNS})


def write_upcoming_fixtures(path: Path, fixtures: list[dict]) -> None:
    """fixturedownload.com's own column shape, so load_fixtures.py's
    existing per-league loader works completely unchanged for MLS -
    see load_league_fixtures()'s Date/Result parsing. Dates are written
    already in correct UK local time (see _to_uk_local), so MLS gets
    LEAGUE_TIME_ADJUSTMENT = None, same as the one other league
    (Premier League) that needs no further correction."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Round Number", "Date", "Location", "Home Team", "Away Team", "Result"])
        writer.writeheader()
        for fx in fixtures:
            date_part, time_part = _to_uk_local(fx["fixture"]["date"])
            writer.writerow({
                "Round Number": fx.get("league", {}).get("round", ""),
                "Date": f"{date_part} {time_part}",
                "Location": "",
                "Home Team": fx["teams"]["home"]["name"],
                "Away Team": fx["teams"]["away"]["name"],
                "Result": "-",
            })


def fetch_mls(data_dir: Path, fixtures_dir: Path) -> None:
    data_dir.mkdir(exist_ok=True)
    fixtures_dir.mkdir(exist_ok=True)

    league_id = find_league_id(data_dir / "mls_league_id.txt")
    season = datetime.now(timezone.utc).year  # MLS runs within a single calendar year

    print(f"[debug] Fetching MLS (league {league_id}) season {season} fixture list...")
    payload = _api_get("fixtures", {"league": league_id, "season": season})
    all_fixtures = payload.get("response", [])
    print(f"[debug] {len(all_fixtures)} total fixture(s) returned for the season")

    existing_rows, known_ids = load_existing_results(data_dir / "MLS.csv")

    finished = [fx for fx in all_fixtures if fx["fixture"]["status"]["short"] in FINISHED_STATUSES]
    not_started = [fx for fx in all_fixtures if fx["fixture"]["status"]["short"] == NOT_STARTED_STATUS]

    to_backfill = [fx for fx in finished if str(fx["fixture"]["id"]) not in known_ids]
    print(f"[debug] {len(finished)} finished fixture(s) total, {len(to_backfill)} not yet in data/MLS.csv")

    new_rows = []
    requests_used = 0
    for fx in to_backfill:
        if requests_used >= MAX_STATS_REQUESTS_PER_RUN:
            print(f"[debug] Hit this run's budget ({MAX_STATS_REQUESTS_PER_RUN} stats requests) - "
                  f"{len(to_backfill) - requests_used} fixture(s) left for next run.")
            break

        fixture_id = fx["fixture"]["id"]
        stats = fetch_fixture_statistics(fixture_id)
        requests_used += 1
        if stats is None:
            print(f"[info] No statistics available yet for fixture {fixture_id} - will retry next run")
            continue

        home_id = fx["teams"]["home"]["id"]
        away_id = fx["teams"]["away"]["id"]
        home_stats = stats.get(home_id, {"corners": 0, "yellow": 0, "red": 0})
        away_stats = stats.get(away_id, {"corners": 0, "yellow": 0, "red": 0})
        date_part, time_part = _to_uk_local(fx["fixture"]["date"])
        halftime = fx.get("score", {}).get("halftime") or {}

        new_rows.append({
            "Div": "MLS",
            "Date": date_part, "Time": time_part,
            "HomeTeam": fx["teams"]["home"]["name"], "AwayTeam": fx["teams"]["away"]["name"],
            "FTHG": fx["goals"]["home"], "FTAG": fx["goals"]["away"],
            "HTHG": halftime.get("home") or 0, "HTAG": halftime.get("away") or 0,
            "HC": home_stats["corners"], "AC": away_stats["corners"],
            "HY": home_stats["yellow"], "AY": away_stats["yellow"],
            "HR": home_stats["red"], "AR": away_stats["red"],
            "Referee": fx["fixture"].get("referee") or "",
            "FixtureID": str(fixture_id),
        })

    if new_rows:
        write_results(data_dir / "MLS.csv", existing_rows + new_rows)
        print(f"[debug] Added {len(new_rows)} newly-finished fixture(s) to data/MLS.csv "
              f"({requests_used} statistics request(s) used this run)")
    else:
        print(f"[debug] No new finished fixtures with stats to add this run "
              f"({requests_used} statistics request(s) used)")

    write_upcoming_fixtures(fixtures_dir / "MLS.csv", not_started)
    print(f"[debug] Wrote {len(not_started)} upcoming fixture(s) to fixtures_manual/MLS.csv")


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"
    fixtures_dir = Path(__file__).parent / "fixtures_manual"
    fetch_mls(data_dir, fixtures_dir)
