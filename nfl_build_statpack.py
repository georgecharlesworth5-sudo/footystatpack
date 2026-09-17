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
    predictions = predict_game(
        home_form["home"], away_form["away"], league_avg,
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
        "home_form_sample": home_form["home"].get("matches", 0),
        "away_form_sample": away_form["away"].get("matches", 0),
        "predictions": predictions,
    }
    if market_blended:
        card["market_blended"] = True

    home_n = home_form["home"].get("matches", 0)
    away_n = away_form["away"].get("matches", 0)
    if home_n < LOW_SAMPLE_THRESHOLD or away_n < LOW_SAMPLE_THRESHOLD:
        card["low_sample"] = True
        card["note"] = (
            f"Thin form sample ({home_n} home / {away_n} away games) - probabilities here "
            f"can swing hard on a single result. Treat as a rough signal, not a settled read."
        )
    return card


def build_nfl_statpack(data_dir: Path) -> dict:
    team_games = load_team_games(data_dir / "nfl_team_games.csv")
    upcoming = load_upcoming(data_dir / "nfl_upcoming.csv")

    print(f"[debug] {len(team_games)} team-game rows loaded")
    print(f"[debug] {len(upcoming)} upcoming fixture(s) loaded")

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

    pack = {
        "league_averages": league_avg,
        "team_form": team_forms,
        "upcoming_fixtures": fixture_cards,
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
