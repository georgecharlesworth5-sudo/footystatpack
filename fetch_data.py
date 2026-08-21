"""
fetch_data.py

Pulls historical + in-progress season match data from football-data.co.uk
for the 4 English leagues + Scottish Premiership, and the upcoming
fixtures list.

football-data.co.uk publishes free CSVs per league per season at:
    https://www.football-data.co.uk/mmz4281/<season_code>/<league_code>.csv

Season code format: two-digit start year + two-digit end year, e.g. "2526"
for the 2025/26 season, "2627" for 2026/27.

League codes we care about:
    E0  = Premier League
    E1  = Championship
    E2  = League One
    E3  = League Two
    SC0 = Scottish Premiership

There's also a combined upcoming-fixtures file (all leagues, no results yet):
    https://www.football-data.co.uk/fixtures.csv

Run this on a schedule (GitHub Actions cron) - the site updates at least
twice a week.
"""

import csv
import io
import ssl
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Some Windows Python installs (and older Mac ones) don't have their
# certificate store wired up correctly, which makes every HTTPS request
# fail with CERTIFICATE_VERIFY_FAILED even though the site itself is
# fine. We first try normal verified HTTPS; if that specific error shows
# up, we fall back to an unverified context. This is safe here because
# we're only ever reading public, non-sensitive CSV data from one known
# domain - not sending credentials or handling anything private.
_UNVERIFIED_CONTEXT = ssl._create_unverified_context()

LEAGUES = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "SC0": "Scottish Premiership",
}

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# Columns we actually need downstream. football-data.co.uk CSVs carry a lot
# of bookmaker odds columns we don't care about for this project.
KEEP_COLUMNS = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR",
    "Referee",
    "HS", "AS", "HST", "AST",
    "HC", "AC",          # corners
    "HY", "AY", "HR", "AR",  # cards
]

USER_AGENT = "Mozilla/5.0 (compatible; StatPackBot/1.0; personal use)"


def _fetch_url(url: str, retries: int = 3, backoff: float = 2.0) -> str:
    """Fetch a URL as text, with basic retry on transient failures.
    Falls back to an unverified SSL context if the local machine's
    certificate store is misconfigured (common on Windows)."""
    last_err = None
    use_unverified = False

    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            context = _UNVERIFIED_CONTEXT if use_unverified else None
            with urlopen(req, timeout=30, context=context) as resp:
                raw = resp.read()
                # Some football-data.co.uk files (fixtures.csv in
                # particular) are saved with a UTF-8 byte-order-mark
                # (BOM) at the start, which - if decoded as plain
                # latin-1 - turns into stray "ï»¿" characters glued onto
                # the first column name (e.g. "ï»¿Div" instead of "Div"),
                # silently breaking any column lookup on that field.
                # utf-8-sig strips the BOM automatically if present; if
                # the file isn't UTF-8 at all (older files with accented
                # names in plain latin-1), fall back to latin-1.
                try:
                    return raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    return raw.decode("latin-1")
        except URLError as e:
            last_err = e
            if isinstance(e.reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(e):
                if not use_unverified:
                    print("  [info] Certificate verification failed - retrying without SSL verification "
                          "(safe here: public read-only data, no credentials involved).")
                    use_unverified = True
                    continue
            time.sleep(backoff * (attempt + 1))
        except HTTPError as e:
            last_err = e
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts: {last_err}")


def fetch_league_season(league_code: str, season_code: str) -> list[dict]:
    """Download and parse one league/season CSV into a list of row dicts,
    trimmed to KEEP_COLUMNS. Skips gracefully if the file doesn't exist
    yet (e.g. season hasn't started, or league has no Div column match)."""
    url = BASE_URL.format(season=season_code, code=league_code)
    try:
        text = _fetch_url(url)
    except RuntimeError as e:
        print(f"  [skip] {league_code} {season_code}: {e}")
        return []

    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        if not row.get("HomeTeam"):
            continue
        trimmed = {col: row.get(col, "") for col in KEEP_COLUMNS}
        trimmed["League"] = league_code
        trimmed["Season"] = season_code
        rows.append(trimmed)
    return rows


def fetch_all(season_codes: list[str], out_dir: Path) -> dict[str, list[dict]]:
    """Fetch every league for every given season code. Returns a dict keyed
    by league_code -> list of match rows (across all requested seasons),
    and writes a combined CSV per league to out_dir for caching/inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_data: dict[str, list[dict]] = {code: [] for code in LEAGUES}

    for code in LEAGUES:
        for season in season_codes:
            print(f"Fetching {LEAGUES[code]} ({code}) season {season}...")
            rows = fetch_league_season(code, season)
            all_data[code].extend(rows)
            time.sleep(1)  # be polite to a free public data source

        # cache to disk
        out_path = out_dir / f"{code}.csv"
        if all_data[code]:
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=KEEP_COLUMNS + ["League", "Season"])
                writer.writeheader()
                writer.writerows(all_data[code])
            print(f"  -> {len(all_data[code])} matches cached to {out_path}")

    return all_data


def fetch_fixtures() -> list[dict]:
    """Download the combined upcoming-fixtures file and filter to our
    5 target leagues."""
    text = _fetch_url(FIXTURES_URL)

    if len(text) < 2000:
        # A real fixtures.csv is normally tens of thousands of characters
        # (hundreds of matches across many countries). If it's tiny, we
        # likely didn't get the real file - print what we did get so we
        # can see what's going on (e.g. an error/redirect page).
        print(f"  [warning] fixtures.csv response looks too small ({len(text)} chars) "
              f"to be the real file. Raw content received:")
        print("  " + "-" * 60)
        print(text[:1500])
        print("  " + "-" * 60)

    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)

    all_divs = sorted(set(r.get("Div", "") for r in rows if r.get("Div")))
    print(f"  [debug] fixtures.csv has {len(rows)} total rows across {len(all_divs)} "
          f"divisions: {all_divs}")

    fixtures = []
    for row in rows:
        div = row.get("Div", "")
        if div in LEAGUES:
            fixtures.append({
                "Div": div,
                "League": LEAGUES[div],
                "Date": row.get("Date", ""),
                "Time": row.get("Time", ""),
                "HomeTeam": row.get("HomeTeam", ""),
                "AwayTeam": row.get("AwayTeam", ""),
            })
    return fixtures


if __name__ == "__main__":
    # Example: current + previous season, so form models have a full
    # season of history even early in a new campaign.
    seasons = ["2526", "2627"]
    data_dir = Path(__file__).parent / "data"
    fetch_all(seasons, data_dir)

    fixtures = fetch_fixtures()
    print(f"\n{len(fixtures)} upcoming fixtures found across target leagues.")
