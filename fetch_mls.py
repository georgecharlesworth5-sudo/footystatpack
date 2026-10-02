"""
fetch_mls.py

Pulls MLS (Major League Soccer) results from football-data.co.uk's
"extra leagues" file and writes them in the same shape fetch_data.py
writes for the other 9 leagues - data/MLS.csv, with Div/Date/Time/
HomeTeam/AwayTeam/FTHG/FTAG columns, so stats_engine.py and
build_statpack.py need no MLS-specific parsing code at all, just "MLS"
added to build_statpack.LEAGUE_NAMES (already done).

## Why this is a separate script, and why it's goals-only

football-data.co.uk's MAIN per-league files (the ones fetch_data.py
reads for the 4 English leagues + Scottish Premiership + the other 4
in LEAGUES there) don't cover MLS at all. MLS only shows up in a
DIFFERENT file the site publishes - a combined, multi-country "extra
leagues" CSV - at a different URL, in a different column layout:
goals, half-time score and match odds only, NO corners or cards.

That ruled out treating MLS as just another entry in fetch_data.py's
own LEAGUES dict (different URL shape, different columns entirely).
A genuine corners/cards source for MLS would mean paying for an API;
this project runs on football-data.co.uk's free data elsewhere, so
MLS instead gets one real trade-off: goals markets only, same as
every other league, but no corners/cards predictions for this one
competition. See build_statpack.GOALS_ONLY_LEAGUES for how the rest
of the pipeline is told to leave those markets out for MLS rather
than silently computing them as a false, maximally-confident zero
(football-data.co.uk's blank HC/AC/HY/AY/HR/AR columns would otherwise
read as "0 corners, 0 cards, every match" to stats_engine.py's
`int(row.get("HC") or 0)`-style parsing - a fabricated signal, not an
absent one).

## Upcoming fixtures

football-data.co.uk's "extra leagues" file is RESULTS only (played
matches), and its own separate all-leagues fixtures.csv (which
fetch_data.py reads for upcoming matches) only covers the 9 leagues in
fetch_data.py's own LEAGUES dict - not MLS. So unlike those 9,
fixtures_manual/MLS.csv is NOT written by this script - it's one more
manual weekly fixturedownload.com download, exactly like every league
in load_fixtures.py other than the Premier League. See that file's
docstring for the download URL/filename convention; MLS is now in
load_fixtures.LEAGUE_FILES alongside the others.

## No API key, no setup

Nothing to configure here - this reads the same free, public, no-login
football-data.co.uk site the rest of the project already relies on
for the other leagues.
"""

import csv
import io
import ssl
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Same reasoning as fetch_data.py's identical constant - some Windows/
# older Mac Python installs have a broken local certificate store, which
# makes every HTTPS request fail with CERTIFICATE_VERIFY_FAILED even
# though the site itself is fine. Falls back to this only if that
# specific error shows up. Safe here: public, non-sensitive, read-only
# data from one known domain, no credentials involved.
_UNVERIFIED_CONTEXT = ssl._create_unverified_context()

EXTRA_LEAGUES_URL = "https://www.football-data.co.uk/new/USA.csv"
USER_AGENT = "Mozilla/5.0 (compatible; StatPackBot/1.0; personal use)"

# How many of the most recent distinct seasons in the file to keep, for
# the same reason fetch_data.py keeps "current + previous season" for
# the other leagues: enough history for rolling form to mean something,
# without the pipeline doing form-weighting math over several years of
# matches it'll never use. Sorting the season labels found in the file
# and keeping the last N of them avoids having to guess the exact label
# format in advance (could be "2025", "2025/2026", etc. - unverified,
# since the real file couldn't be fetched directly during development,
# only described in earlier research - see the debug print in fetch_mls
# below, which makes whatever the real format turns out to be visible
# in the very first Action run rather than silently wrong).
SEASONS_TO_KEEP = 2

RESULTS_COLUMNS = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "HTHG", "HTAG",
    "HC", "AC", "HY", "AY", "HR", "AR",
    "Referee",
]


def _fetch_url(url: str, retries: int = 3, backoff: float = 2.0) -> str:
    """Identical approach to fetch_data.py's own helper of the same
    name - see that file's comments for the certificate-fallback and
    BOM-handling reasoning. Duplicated rather than imported so this
    script stays fully standalone, same as every other fetch_*.py here."""
    last_err = None
    use_unverified = False

    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            context = _UNVERIFIED_CONTEXT if use_unverified else None
            with urlopen(req, timeout=30, context=context) as resp:
                raw = resp.read()
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


def _is_mls_row(row: dict) -> bool:
    """Identify MLS specifically within the combined multi-country file.

    The exact spelling of the Country/League columns for MLS couldn't be
    confirmed against a live fetch during development (football-data.co.uk
    blocks this environment's own outbound requests - GitHub Actions'
    runner has no such restriction, so this only gets properly tested on
    a real run). Matching loosely (country mentions USA, league mentions
    MLS OR is USA's flagged top division) rather than on one exact
    hardcoded string, so a near-miss in capitalisation/wording doesn't
    silently match zero rows. select_mls_rows() below prints every
    distinct (Country, League) pairing the file actually contains,
    specifically so a wrong guess here is immediately visible and fixable
    from the Action log rather than failing silently or crashing deep in
    the pipeline - same reasoning as fetch_data.fetch_fixtures()'s own
    debug print of the divisions it found.
    """
    country = (row.get("Country") or "").strip().lower()
    league = (row.get("League") or "").strip().lower()
    if "usa" not in country and "united states" not in country:
        return False
    return "mls" in league or league in ("1", "usa1", "usa 1")


def select_mls_rows(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)

    mls_rows = [r for r in rows if _is_mls_row(r)]

    if not mls_rows:
        distinct = sorted(set((r.get("Country", ""), r.get("League", "")) for r in rows))
        raise RuntimeError(
            "No MLS rows found in football-data.co.uk's extra-leagues file. "
            "This almost certainly means the Country/League spelling assumed in "
            "_is_mls_row() doesn't match the real file - here are every distinct "
            f"(Country, League) pairing actually present, so the right one can be "
            f"picked out: {distinct}"
        )

    seasons_present = sorted(set(r.get("Season", "") for r in mls_rows))
    print(f"[debug] Found {len(mls_rows)} MLS row(s) across {len(seasons_present)} "
          f"season(s) in the file: {seasons_present}")

    seasons_to_keep = set(seasons_present[-SEASONS_TO_KEEP:])
    kept = [r for r in mls_rows if r.get("Season", "") in seasons_to_keep]
    print(f"[debug] Keeping the {len(seasons_to_keep)} most recent season(s) "
          f"({sorted(seasons_to_keep)}): {len(kept)} match(es)")
    return kept


def _parse_date(raw: str) -> str:
    """football-data.co.uk's main files use dd/mm/yyyy throughout - the
    extra-leagues file is assumed to match (unverified directly, same
    caveat as _is_mls_row above). If it turns out to use yyyy-mm-dd or
    mm/dd/yyyy instead, every row will fail this and the resulting
    "0 matches found" from the pipeline afterwards is the signal to come
    back and adjust this function against a real downloaded sample."""
    raw = (raw or "").strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue
    return raw  # leave as-is rather than crash - downstream date parsing will just skip it


def build_results_rows(mls_rows: list[dict]) -> list[dict]:
    out = []
    for r in mls_rows:
        out.append({
            "Div": "MLS",
            "Date": _parse_date(r.get("Date", "")),
            "Time": (r.get("Time") or "").strip(),
            "HomeTeam": (r.get("Home") or "").strip(),
            "AwayTeam": (r.get("Away") or "").strip(),
            "FTHG": (r.get("HG") or "").strip(),
            "FTAG": (r.get("AG") or "").strip(),
            # Not available in this file - left blank rather than 0, and
            # never computed on by the model for MLS at all (see
            # build_statpack.GOALS_ONLY_LEAGUES) - blank here is just for
            # a human glancing at the CSV, not something the pipeline
            # reads for this league.
            "HTHG": "", "HTAG": "",
            "HC": "", "AC": "", "HY": "", "AY": "", "HR": "", "AR": "",
            "Referee": "",
        })
    return out


def write_results(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def fetch_mls(data_dir: Path) -> None:
    data_dir.mkdir(exist_ok=True)

    print(f"[debug] Fetching MLS results from {EXTRA_LEAGUES_URL} ...")
    text = _fetch_url(EXTRA_LEAGUES_URL)

    mls_rows = select_mls_rows(text)
    results_rows = build_results_rows(mls_rows)

    out_path = data_dir / "MLS.csv"
    write_results(out_path, results_rows)
    print(f"[debug] Wrote {len(results_rows)} MLS match(es) to {out_path}")


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"
    fetch_mls(data_dir)
