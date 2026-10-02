"""
nfl_build_statpack.py

Orchestrates the NFL pipeline - the equivalent of the football side's
build_statpack.py:
  1. Read fetch_nfl.py's cached data (nfl_team_games.csv, nfl_upcoming.csv)
  2. Build rolling form + league averages (nfl_stats.py)
  3. For each upcoming fixture, run the prediction model (nfl_model.py)
  4. Blend in real market odds where available (nfl_market_odds.py)
  5. Write nfl_statpack.json / nfl_statpack_data.js for the dashboard

Run this after fetch_nfl.py, same relationship as the football side.
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from nfl_stats import build_team_game_log, team_form_summary, league_averages
from nfl_model import predict_game
from nfl_market_odds import parse_market_row, blend_moneyline, blend_total_points
from nfl_best_bets import compute_nfl_best_bets

# Full names for display - nflverse uses 2-3 letter codes throughout.
TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

DEFAULT_TOTAL_LINES = [40.5, 44.5, 48.5]

DEFAULT_TEAM_LINES = [20.5, 24.5]
DEFAULT_TD_LINES = [0.5, 1.5, 2.5]

LOW_SAMPLE_THRESHOLD = 5  # same reasoning as the football side - a prediction built on fewer
                          # games than this gets flagged, not hidden, but treated with caution


def load_team_games(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_upcoming(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_player_stats(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


STAT_LEADER_CATEGORIES = [
    ("passing", "passing_yards"),
    ("rushing", "rushing_yards"),
    ("receiving", "receiving_yards"),
]


def stat_leaders_for_team(players: list[dict]) -> dict:
    """Top player by yards in each category, for one team in one game.
    A category is left out entirely (not shown as "0 yds") if nobody on
    the team recorded any - e.g. a team with no completed passes still
    technically has a "leading passer" at 0 yards, which isn't a
    meaningful stat to surface."""
    leaders = {}
    for category, field in STAT_LEADER_CATEGORIES:
        best = None
        for p in players:
            try:
                yards = float(p.get(field) or 0)
            except ValueError:
                continue
            if yards <= 0:
                continue
            if best is None or yards > best["yards"]:
                best = {"player": p.get("player", ""), "yards": yards}
        if best:
            best["yards"] = int(best["yards"]) if best["yards"] == int(best["yards"]) else best["yards"]
            leaders[category] = best
    return leaders


def build_recent_results(team_games_rows: list[dict], player_stats_rows: list[dict]) -> list[dict]:
    """
    Completed games from the single most recent (season, week) found in
    team_games_rows, each with the final score and each team's stat
    leaders (top passer/rusher/receiver by yards, from stats_player_week -
    see fetch_nfl.py). Deliberately just the latest week rather than
    every game ever played - this is "what just happened", the
    completed-game equivalent of upcoming_fixtures, not a full season
    archive (nflverse's own files remain the source of truth for
    historical lookups beyond that).

    team_games_rows has two rows per game (one per team, see
    fetch_nfl.build_team_game_rows) - paired back into one record per
    game here using each row's own "venue" (H/A) field.
    """
    if not team_games_rows:
        return []

    def season_week_key(row):
        try:
            return (int(row["season"]), int(row["week"]))
        except (ValueError, KeyError, TypeError):
            return (0, 0)

    latest = max(season_week_key(r) for r in team_games_rows)
    latest_rows = [r for r in team_games_rows if season_week_key(r) == latest]

    games_by_id: dict[str, dict] = {}
    for row in latest_rows:
        game = games_by_id.setdefault(row["game_id"], {})
        game[row["venue"]] = row

    # Index player stats by (game_id, team) once, rather than scanning
    # the full player-stats file per game - that file covers every game
    # all season, this is a handful of games.
    players_by_game_team: dict[tuple[str, str], list[dict]] = {}
    for p in player_stats_rows:
        if p.get("game_id") not in games_by_id:
            continue  # cheap pre-filter - only bother indexing this week's games
        players_by_game_team.setdefault((p["game_id"], p.get("team")), []).append(p)

    results = []
    for game_id, sides in games_by_id.items():
        home_row, away_row = sides.get("H"), sides.get("A")
        if home_row is None or away_row is None:
            continue  # shouldn't happen - a game_id with only one side's row is incomplete data

        home_code, away_code = home_row["team"], away_row["team"]
        date, time = convert_et_to_uk(home_row.get("gameday", ""), home_row.get("gametime", ""))

        try:
            home_score = int(float(home_row["points_for"]))
            away_score = int(float(away_row["points_for"]))
        except (ValueError, KeyError):
            continue  # malformed score - skip rather than show a broken result

        results.append({
            "game_id": game_id,
            "home_team": TEAM_NAMES.get(home_code, home_code),
            "away_team": TEAM_NAMES.get(away_code, away_code),
            "home_code": home_code, "away_code": away_code,
            "home_score": home_score, "away_score": away_score,
            "date": date, "time": time,
            "week": home_row.get("week", ""),
            "stat_leaders": {
                "home": stat_leaders_for_team(players_by_game_team.get((game_id, home_code), [])),
                "away": stat_leaders_for_team(players_by_game_team.get((game_id, away_code), [])),
            },
        })
    return results


def convert_et_to_uk(gameday: str, gametime: str) -> tuple[str, str]:
    """
    nflverse's gametime is US Eastern (confirmed directly: the 2025
    season opener, DAL@PHI, shows 20:20 - the real, well-known 8:20pm ET
    kickoff for that game). Converts to UK date/time using proper
    IANA timezone data rather than a flat offset - the NFL season spans
    September to February, crossing both the US and UK's DST transition
    dates, and those dates don't align (UK's BST ends the last Sunday
    of October, US's EDT ends the first Sunday of November) - there's a
    ~1 week window each year where the gap between them is 4 hours
    instead of the usual 5. zoneinfo handles this correctly by
    construction; a hand-rolled offset would need to reimplement both
    countries' DST rules to get that week right.

    Returns (date, time) in our usual dd/mm/yyyy, HH:MM format. Falls
    back to the raw values unchanged if either input is missing/
    malformed, rather than crashing the whole build over one fixture.
    """
    try:
        dt_et = datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return gameday, gametime
    dt_et = dt_et.replace(tzinfo=ZoneInfo("America/New_York"))
    dt_uk = dt_et.astimezone(ZoneInfo("Europe/London"))
    return dt_uk.strftime("%d/%m/%Y"), dt_uk.strftime("%H:%M")


def build_fixture_card(home_code: str, away_code: str, home_form: dict, away_form: dict,
                        league_avg: dict, market_row: dict | None = None) -> dict:
    # Uses each team's OVERALL rolling form (home + away games pooled),
    # not the home-only/away-only split the football side uses. Unlike
    # football, splitting by venue this early in a 17-game NFL season
    # means a team with e.g. 2 home games and 0 away games so far would
    # get an "away form" that's 100% last season's games - discarding
    # all of this season's data - to make up a venue-specific sample.
    # NFL home-field advantage is also real but comparatively small, so
    # that trade (lose same-season signal to preserve a venue split) is
    # a bad one here. League-wide home/away scoring averages (league_avg,
    # passed through unchanged) still give the model a home-field
    # baseline - it's just no longer conditioned on each TEAM's own
    # venue-split history on top of that.
    predictions = predict_game(
        home_form["overall"], away_form["overall"], league_avg,
        DEFAULT_TOTAL_LINES, DEFAULT_TEAM_LINES, DEFAULT_TD_LINES,
    )

    market_blended = False
    if market_row:
        market_probs = parse_market_row(market_row)
        if market_probs:
            new_ml = blend_moneyline(predictions["points"]["moneyline"], market_probs)
            if new_ml != predictions["points"]["moneyline"]:
                market_blended = True
            predictions["points"]["moneyline"] = new_ml
            new_totals = blend_total_points(predictions["points"]["total_over_under"], market_probs)
            if new_totals != predictions["points"]["total_over_under"]:
                market_blended = True
            predictions["points"]["total_over_under"] = new_totals

    card = {
        "home_team": TEAM_NAMES.get(home_code, home_code),
        "away_team": TEAM_NAMES.get(away_code, away_code),
        "home_code": home_code, "away_code": away_code,
        "home_form_sample": home_form["overall"].get("matches", 0),
        "away_form_sample": away_form["overall"].get("matches", 0),
        "predictions": predictions,
    }
    if market_blended:
        card["market_blended"] = True

    home_n = home_form["overall"].get("matches", 0)
    away_n = away_form["overall"].get("matches", 0)
    if home_n < LOW_SAMPLE_THRESHOLD or away_n < LOW_SAMPLE_THRESHOLD:
        card["low_sample"] = True
        card["note"] = (
            f"Thin form sample ({home_n} home team / {away_n} away team games) - probabilities here "
            f"can swing hard on a single result. Treat as a rough signal, not a settled read."
        )
    return card


def build_nfl_statpack(data_dir: Path) -> dict:
    team_games = load_team_games(data_dir / "nfl_team_games.csv")
    upcoming = load_upcoming(data_dir / "nfl_upcoming.csv")
    player_stats = load_player_stats(data_dir / "nfl_player_stats.csv")

    print(f"[debug] {len(team_games)} team-game rows loaded")
    print(f"[debug] {len(upcoming)} upcoming fixture(s) loaded")
    print(f"[debug] {len(player_stats)} player-game rows loaded")

    team_logs = build_team_game_log(team_games)
    league_avg = league_averages(team_games)
    team_forms = {team: team_form_summary(log) for team, log in team_logs.items()}

    # Index market rows by (away_code, home_code) for lookup per fixture -
    # games.csv's own game_id already encodes exactly this pairing.
    market_by_matchup = {(row["away_team"], row["home_team"]): row for row in upcoming}

    fixture_cards = []
    market_blended_count = 0
    for fx in upcoming:
        home_code, away_code = fx.get("home_team"), fx.get("away_team")
        if home_code not in team_forms or away_code not in team_forms:
            print(f"[debug] skipping {away_code} @ {home_code}: no form data for one or both teams yet")
            continue

        market_row = market_by_matchup.get((away_code, home_code))
        card = build_fixture_card(home_code, away_code, team_forms[home_code], team_forms[away_code],
                                   league_avg, market_row=market_row)
        card["date"], card["time"] = convert_et_to_uk(fx.get("gameday", ""), fx.get("gametime", ""))
        card["week"] = fx.get("week", "")
        if card.get("market_blended"):
            market_blended_count += 1
        fixture_cards.append(card)

    print(f"[debug] {market_blended_count}/{len(fixture_cards)} fixtures have market odds blended in")

    recent_results = build_recent_results(team_games, player_stats)
    print(f"[debug] {len(recent_results)} completed game(s) in the most recent week, with stat leaders")

    pack = {
        "league_averages": league_avg,
        "team_form": team_forms,
        "upcoming_fixtures": fixture_cards,
        "recent_results": recent_results,
    }
    pack["best_bets"] = compute_nfl_best_bets(pack)
    print(f"[debug] {len(pack['best_bets'])} NFL best bet(s) qualified today")
    return pack


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"
    pack = build_nfl_statpack(data_dir)

    out_path = Path(__file__).parent / "nfl_statpack.json"
    with open(out_path, "w") as f:
        json.dump(pack, f, indent=2, default=str)

    js_path = Path(__file__).parent / "nfl_statpack_data.js"
    with open(js_path, "w") as f:
        f.write("const NFL_STATPACK_DATA = ")
        json.dump(pack, f, default=str)
        f.write(";\n")

    print(f"Stat pack written to {out_path}")
    print(f"Dashboard data written to {js_path}")
