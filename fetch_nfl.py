"""
fetch_nfl.py

Pulls NFL data from nflverse (nflverse-data GitHub releases + nfldata),
an openly-licensed (CC-BY 4.0) project maintained specifically for this
kind of use - same spirit as football-data.co.uk for the football side
of this project, verified as a genuine, legitimate open-data source
before building against it.

Two source files, joined together:

  1. stats_team_week_<season>.csv - team-game-level offensive/defensive
     stats (passing_tds, rushing_tds, etc.), one row per team per game
     they've PLAYED. Does not include points/final score at all.
     URL pattern confirmed by direct testing:
       https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_<season>.csv

  2. games.csv - the season schedule AND results (home_score/away_score,
     blank for games not yet played) - this is what supplies points, and
     also the list of upcoming fixtures to predict.
       https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv

Home/away for each stats_team_week row is derived from its own game_id
(format: <season>_<week>_<away_team>_<home_team>) rather than needing a
separate lookup - confirmed directly against real data before relying
on it.

Combines last season + current season, same reasoning as the football
side: full history for a stable read, current season weighted more
heavily by the recency-decay already built into nfl_stats.py's rolling
form - not a special case here, just naturally true once combined.
"""

import csv
import io
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

USER_AGENT = "Mozilla/5.0 (compatible; StatPackBot/1.0; personal use)"

STATS_TEAM_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{season}.csv"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def _fetch_url(url: str, retries: int = 3, backoff: float = 2.0) -> str:
    """Same resilient fetch pattern as the football side's fetch_data.py -
    retries transient failures, never raises uncaught (returns None
    instead) so one bad fetch doesn't take down the whole pipeline."""
    last_err = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8-sig")
        except (URLError, HTTPError) as e:
            last_err = e
            time.sleep(backoff * (attempt + 1))
    print(f"  [skip] {url}: failed after {retries} attempts: {last_err}")
    return None


def _parse_venue_from_game_id(game_id: str, team: str) -> str | None:
    """nflverse game_id format: <season>_<week>_<away_team>_<home_team>.
    Confirmed directly against real data (e.g. '2025_01_TB_ATL' - ATL's
    own row for this game_id has ATL as the LAST component, meaning
    ATL was home). Returns 'H', 'A', or None if the format doesn't
    match what's expected (better to skip a row than misclassify it)."""
    parts = game_id.split("_")
    if len(parts) != 4:
        return None
    _, _, away, home = parts
    if team == home:
        return "H"
    if team == away:
        return "A"
    return None


def fetch_team_stats(season: int) -> list[dict]:
    """Fetches one season's team-game stats. Returns [] gracefully on
    failure (e.g. a not-yet-existing future season, or a transient
    fetch error) - same reasoning as the football side: a missing
    season shouldn't crash the whole pipeline."""
    url = STATS_TEAM_URL.format(season=season)
    text = _fetch_url(url)
    if text is None:
        return []
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        if row.get("season_type") != "REG":
            continue  # regular season only - same reasoning as the football side (playoffs: tiny sample, different stakes)
        venue = _parse_venue_from_game_id(row.get("game_id", ""), row.get("team", ""))
        if venue is None:
            continue
        try:
            passing_tds = int(row.get("passing_tds") or 0)
            rushing_tds = int(row.get("rushing_tds") or 0)
        except ValueError:
            continue
        rows.append({
            "season": row.get("season"), "week": row.get("week"), "game_id": row.get("game_id"),
            "team": row.get("team"), "opponent_team": row.get("opponent_team"), "venue": venue,
            "passing_tds": passing_tds, "rushing_tds": rushing_tds,
        })
    return rows


def fetch_games() -> list[dict]:
    """Fetches the full schedule (all seasons, past results AND future
    fixtures with blank scores in the same file)."""
    text = _fetch_url(GAMES_URL)
    if text is None:
        return []
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def build_team_game_rows(team_stats_rows: list[dict], games_rows: list[dict]) -> list[dict]:
    """
    Joins team_stats (passing/rushing TDs, no points) with games (points,
    no TD breakdown) on game_id, and folds in each row's OPPONENT's
    passing/rushing TDs from the same game_id (a self-join within
    team_stats) to get the "against" side of those two stats - the
    source file only gives each team's own offensive production, not
    what they conceded, so that has to be derived by looking up the
    other team's row for the same game.

    Returns one row per team-game actually played (blank/future games
    in games.csv are excluded here - they have no team_stats entry to
    join against in the first place, since that file only covers played
    games).
    """
    games_by_id = {g["game_id"]: g for g in games_rows}
    stats_by_game = {}
    for row in team_stats_rows:
        stats_by_game.setdefault(row["game_id"], {})[row["team"]] = row

    combined = []
    for game_id, teams_in_game in stats_by_game.items():
        game = games_by_id.get(game_id)
        if game is None:
            continue  # shouldn't happen given both sources cover the same nflverse game_ids, but don't assume
        try:
            home_score = float(game["home_score"])
            away_score = float(game["away_score"])
        except (ValueError, KeyError, TypeError):
            continue  # game not actually finished despite having a team_stats row (shouldn't happen, but don't guess)

        for team, row in teams_in_game.items():
            opponent = row["opponent_team"]
            opponent_row = teams_in_game.get(opponent)
            if opponent_row is None:
                continue  # can't get the "against" side without the opponent's own row

            points_for = home_score if row["venue"] == "H" else away_score
            points_against = away_score if row["venue"] == "H" else home_score

            combined.append({
                "season": row["season"], "week": row["week"], "game_id": game_id,
                "team": team, "opponent": opponent, "venue": row["venue"],
                "points_for": points_for, "points_against": points_against,
                "passing_tds_for": row["passing_tds"], "passing_tds_against": opponent_row["passing_tds"],
                "rushing_tds_for": row["rushing_tds"], "rushing_tds_against": opponent_row["rushing_tds"],
            })
    return combined


def fetch_upcoming_fixtures(games_rows: list[dict]) -> list[dict]:
    """Fixtures with no score yet - both the source of predictions AND
    confirms directly whether the field is populated ahead of kickoff,
    since that was left genuinely unresolved earlier and needs a live
    check against real data rather than more guessing."""
    upcoming = []
    for g in games_rows:
        if g.get("home_score") not in (None, "", "NA"):
            continue
        upcoming.append(g)
    return upcoming


def cache_team_game_data(rows: list[dict], out_path: Path) -> None:
    fieldnames = ["season", "week", "game_id", "team", "opponent", "venue",
                  "points_for", "points_against",
                  "passing_tds_for", "passing_tds_against",
                  "rushing_tds_for", "rushing_tds_against"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {len(rows)} team-game rows cached to {out_path}")


def cache_upcoming_fixtures(fixtures: list[dict], out_path: Path) -> None:
    fieldnames = ["game_id", "season", "week", "gameday", "gametime",
                  "away_team", "home_team", "spread_line", "total_line",
                  "home_moneyline", "away_moneyline", "over_odds", "under_odds"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for g in fixtures:
            writer.writerow({col: g.get(col, "") for col in fieldnames})
    print(f"  -> {len(fixtures)} upcoming fixture(s) cached to {out_path}")

    # Directly settles a real open question from earlier: does
    # spread_line/total_line actually populate ahead of kickoff, or
    # only get filled in after the fact? Checked here, once, against
    # real data rather than left as a guess.
    with_lines = sum(1 for g in fixtures if g.get("spread_line") not in (None, "", "NA"))
    print(f"  [debug] {with_lines}/{len(fixtures)} upcoming fixtures have a spread_line populated ahead of kickoff")


if __name__ == "__main__":
    from datetime import date

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    current_year = date.today().year
    # Last season + current season, same combining logic as the football side
    seasons = [current_year - 1, current_year]

    print("Fetching team-game stats...")
    all_team_stats = []
    for season in seasons:
        print(f"  Season {season}...")
        rows = fetch_team_stats(season)
        print(f"    {len(rows)} team-game rows")
        all_team_stats.extend(rows)

    print("Fetching schedule/results...")
    games = fetch_games()
    print(f"  {len(games)} total games across all NFL history")

    combined = build_team_game_rows(all_team_stats, games)
    cache_team_game_data(combined, data_dir / "nfl_team_games.csv")

    upcoming = fetch_upcoming_fixtures([g for g in games if g.get("season") in (str(s) for s in seasons)])
    cache_upcoming_fixtures(upcoming, data_dir / "nfl_upcoming.csv")
