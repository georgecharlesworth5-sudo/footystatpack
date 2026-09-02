"""
build_statpack.py

Orchestrates the whole pipeline:
  1. Load match data (from fetch_data.py output, or cached CSVs)
  2. Build team form logs + league averages (stats_engine.py)
  3. For each upcoming fixture, run the Poisson model (poisson_model.py)
  4. Write out a single statpack.json - the file the dashboard reads

Run this weekly (e.g. Thursday morning) once fetch_data.py has refreshed
the cached CSVs and fixtures list.

NOTES / suggested refinements once v1 is working:
  - Corners/cards models would benefit from a referee-specific card
    average layered on top of team form (some refs card far more than
    others) - Referee is already captured per match in stats_engine, just
    not used in the lambda yet.
  - Consider decaying older matches (weight recent form more) rather than
    a flat rolling window.
  - Home/away split sample sizes get thin for teams early in a season -
    the model currently blends in the prior season automatically via
    fetch_data's multi-season pull, but a explicit shrinkage-to-league-
    average would be more robust than a hard rolling window.
  - Promoted/relegated teams: their most recent matches may be in a
    *different* division's CSV to the one they're now playing in (e.g.
    a team promoted from League Two into League One has no League One
    history yet). We now build one combined match log across all 5
    leagues so their actual recent form still gets used - but the O/U
    lines are still normalised against the league they're playing in
    NOW, which is the right target but means a team's cross-league form
    is an approximation, not a perfect like-for-like read. Fixtures
    using cross-league form are flagged with "cross_league_data": true
    so it's visible in the output which predictions to treat with more
    caution.
"""

import csv
import difflib
import json
from datetime import datetime, date
from pathlib import Path

from stats_engine import build_team_match_log, team_form_summary, league_averages
from poisson_model import predict_fixture
from best_bets import compute_best_bets
from cup_predictions import build_cup_predictions
from league_table import compute_league_table, team_position


def _parse_fixture_date(d: str) -> date | None:
    """fixtures.csv dates come as dd/mm/yyyy (sometimes dd/mm/yy)."""
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(d, fmt).date()
        except ValueError:
            continue
    return None

DEFAULT_LINES = {
    "goals": [1.5, 2.5, 3.5],
    "corners": [8.5, 9.5, 10.5, 11.5],
    "cards": [2.5, 3.5, 4.5],
    "first_half_goals": [0.5, 1.5],
    "second_half_goals": [0.5, 1.5, 2.5],
}

LEAGUE_NAMES = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "SC0": "Scottish Premiership",
    "I1": "Serie A",
}


def load_cached_league(data_dir: Path, code: str) -> list[dict]:
    path = data_dir / f"{code}.csv"
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# Generic club-name words that appear across many unrelated clubs
# ("Derby County" vs "Newport County", "Bristol City" vs "Leicester
# City"). Left in, these dominate character-level fuzzy-match scores
# and can make an entirely different club look like the closest match
# (confirmed: "Derby County" scored HIGHER similarity to "Newport
# County" than to the correct "Derby"). Stripping them out before
# comparing gives a much safer first pass.
GENERIC_TEAM_WORDS = {
    "city", "county", "town", "united", "rovers", "albion", "wanderers",
    "athletic", "academy", "alexandra", "forest", "argyle", "orient", "town",
}


def _core_name(name: str) -> str:
    words = [w for w in name.replace("'", "").split() if w.lower() not in GENERIC_TEAM_WORDS]
    return " ".join(words).strip().lower() or name.strip().lower()


def resolve_team_name(name: str, known_names: list[str]) -> tuple[str | None, str]:
    """
    Team names aren't always consistent across data sources (e.g.
    football-data.co.uk's own results/fixtures files disagreeing:
    "Sheffield Wed" vs "Sheffield Weds"; fixturedownload.com using
    fuller or shorter names: "Derby County" vs "Derby", "Spurs" vs
    "Tottenham"). Resolution order, cheapest/most-certain first:
      1. Exact match.
      2. Known nickname/abbreviation table - hand-verified, treated as
         fully reliable.
      3. Match on the "core" name with generic suffix words (City,
         County, Town, United, etc.) stripped - if exactly ONE known
         name shares that same stripped core, it's almost certainly
         the same club; deterministic, treated as fully reliable.
      4. Fall back to closest-match fuzzy string matching on the full
         name - this is the only tier that's actually produced a wrong
         match in practice (Derby County briefly resolved to Newport
         County, since "County" inflated the character-similarity
         score). Flagged as uncertain in the caller.

    Returns (resolved_name_or_None, tier) where tier is one of
    "exact", "alias", "core", "fuzzy", "none".
    """
    if name in known_names:
        return name, "exact"

    if name in KNOWN_ALIASES and KNOWN_ALIASES[name] in known_names:
        return KNOWN_ALIASES[name], "alias"

    target_core = _core_name(name)
    core_matches = [k for k in known_names if _core_name(k) == target_core]
    if len(core_matches) == 1:
        return core_matches[0], "core"

    matches = difflib.get_close_matches(name, known_names, n=1, cutoff=0.6)
    if matches:
        return matches[0], "fuzzy"
    return None, "none"


# Known nickname/abbreviation pairs that don't share enough characters
# for fuzzy matching to catch reliably (e.g. "Spurs" vs "Tottenham"
# share almost no substring). Add to this as new mismatches turn up -
# check the [debug] skip messages for names that got missed.
KNOWN_ALIASES = {
    "Spurs": "Tottenham",
    "Man Utd": "Man United",
    "Nottingham Forest": "Nott'm Forest",
    "Notts Forest": "Nott'm Forest",
    "Wolverhampton": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
    "Sheffield United": "Sheffield United",
    "West Bromwich Albion": "West Brom",
    "Preston North End": "Preston",
    "Queens Park Rangers": "QPR",
    "MK Dons": "Milton Keynes Dons",
    "Bristol Rovers": "Bristol Rvs",
}


def build_fixture_card(home_team: str, away_team: str, home_form: dict, away_form: dict,
                        league_avg: dict, cross_league: bool = False, name_matches: dict | None = None) -> dict:
    predictions = predict_fixture(
        home_form["home"], away_form["away"], league_avg, DEFAULT_LINES
    )
    card = {
        "home_team": home_team,
        "away_team": away_team,
        "home_form_sample": home_form["home"].get("matches", 0),
        "away_form_sample": away_form["away"].get("matches", 0),
        "predictions": predictions,
    }
    if cross_league:
        card["cross_league_data"] = True
        card["note"] = ("One or both teams' recent form comes from a different division "
                         "(promotion/relegation) - treat this prediction with more caution "
                         "until they've built up matches in the current league.")
    if name_matches:
        card["fuzzy_name_match"] = name_matches
        card.setdefault("note", "")
        card["note"] = (card["note"] + " " if card["note"] else "") + (
            f"Team name(s) matched via fuzzy matching, not exact: {name_matches}. "
            f"Double-check these are actually the same club."
        )
    # A prediction built on only a handful of matches can show a stark
    # 0%/100% split that LOOKS like a confident signal but is really just
    # small-sample noise (e.g. one match with zero cards makes "cards"
    # look like a guaranteed under). Flag it rather than let it read as
    # equally trustworthy as a fixture backed by a full rolling window.
    LOW_SAMPLE_THRESHOLD = 5
    home_n = home_form["home"].get("matches", 0)
    away_n = away_form["away"].get("matches", 0)
    if home_n < LOW_SAMPLE_THRESHOLD or away_n < LOW_SAMPLE_THRESHOLD:
        card["low_sample"] = True
        card.setdefault("note", "")
        card["note"] = (card["note"] + " " if card["note"] else "") + (
            f"Thin form sample ({home_n} home / {away_n} away matches) - probabilities here "
            f"can swing hard on a single result. Treat as a rough signal, not a settled read."
        )
    return card


def compute_data_freshness(all_rows_by_league: dict[str, list[dict]]) -> dict:
    """For each league, find the most recent played-match date in its
    data, plus how many days old that is. Surfaces the publishing-lag
    issue directly on the dashboard rather than requiring anyone to dig
    through CSVs on GitHub to notice a league's results have gone stale."""
    today = date.today()
    freshness = {}
    for code, name in LEAGUE_NAMES.items():
        rows = all_rows_by_league.get(code, [])
        latest = None
        for row in rows:
            if not row.get("FTHG"):
                continue  # not yet played
            d = _parse_fixture_date(row.get("Date", ""))
            if d and (latest is None or d > latest):
                latest = d
        if latest is None:
            freshness[code] = {"league_name": name, "latest_date": None, "days_old": None}
        else:
            freshness[code] = {
                "league_name": name,
                "latest_date": latest.strftime("%d/%m/%Y"),
                "days_old": (today - latest).days,
            }
    return freshness


def build_statpack(data_dir: Path, fixtures: list[dict], cup_fixtures_path: Path | None = None) -> dict:
    statpack = {"generated_leagues": {}}

    # Load every league's rows once and build ONE combined match log across
    # all 5 leagues. This means a promoted/relegated team's most recent
    # matches (played in a different division) still feed their rolling
    # form, rather than the team appearing to have zero history the moment
    # they change league.
    all_rows_by_league = {code: load_cached_league(data_dir, code) for code in LEAGUE_NAMES}
    combined_rows = [row for rows in all_rows_by_league.values() for row in rows]
    combined_team_logs = build_team_match_log(combined_rows)

    print("[debug] rows loaded per league:")
    for code, rows in all_rows_by_league.items():
        print(f"  {code} ({LEAGUE_NAMES[code]}): {len(rows)} rows")
    print(f"[debug] {len(combined_team_logs)} distinct teams found across all leagues: "
          f"{sorted(combined_team_logs.keys())}")

    # Track, per team, which league their TRUE most recent match was in
    # (by actual date, not by which league's CSV happened to be processed
    # last) - used to flag cross-league fixtures. Getting this wrong in
    # one direction matters a lot: a promoted team's old, lower-division
    # form would otherwise look "current" indefinitely, since their new
    # league's rows and old league's rows don't arrive in date order just
    # from concatenating each league's file.
    most_recent_league: dict[str, str] = {}
    most_recent_date: dict[str, date] = {}
    for row in combined_rows:
        home, away, div = row.get("HomeTeam"), row.get("AwayTeam"), row.get("Div") or row.get("League")
        row_date = _parse_fixture_date(row.get("Date", ""))
        if row_date is None:
            continue  # can't compare undated rows - skip rather than risk a wrong overwrite
        for team in (home, away):
            if not team:
                continue
            if team not in most_recent_date or row_date > most_recent_date[team]:
                most_recent_date[team] = row_date
                most_recent_league[team] = div

    league_averages_by_code: dict[str, dict] = {}

    for code, name in LEAGUE_NAMES.items():
        rows = all_rows_by_league[code]
        if not rows:
            continue

        league_avg = league_averages(rows)
        league_averages_by_code[code] = league_avg
        # Only compute form summaries for teams that appear somewhere in
        # the combined log AND have played in (or been promoted/relegated
        # into) this league - i.e. every team, using their full combined
        # history for form, but keyed under whichever leagues they're
        # relevant to.
        team_forms = {team: team_form_summary(log) for team, log in combined_team_logs.items()}

        # This league's own current-season table, for showing each team's
        # position alongside their fixture (same league_table.py already
        # used for the EFL Cup's cross-division adjustment - just applied
        # here to every regular in-league fixture too).
        league_table = compute_league_table(rows)
        league_team_count = len(league_table)

        league_fixtures = [f for f in fixtures if f.get("Div") == code]

        # Drop fixtures whose date has already passed. football-data.co.uk's
        # fixtures.csv is a live snapshot that isn't refreshed continuously,
        # so a match can briefly still appear as "upcoming" for a while
        # after kickoff - there's no value in predicting an already-played
        # game, so filter those out here rather than showing stale cards.
        today = date.today()
        future_fixtures = []
        for f in league_fixtures:
            fx_date = _parse_fixture_date(f.get("Date", ""))
            if fx_date is not None and fx_date < today:
                print(f"[debug] dropping past fixture {f['HomeTeam']} vs {f['AwayTeam']} "
                      f"({f.get('Date')}) - already played")
                continue
            future_fixtures.append(f)
        league_fixtures = future_fixtures

        fixture_cards = []
        known_names = list(team_forms.keys())
        for fx in league_fixtures:
            home_raw, away_raw = fx["HomeTeam"], fx["AwayTeam"]
            home, home_tier = resolve_team_name(home_raw, known_names)
            away, away_tier = resolve_team_name(away_raw, known_names)

            if home is None or away is None:
                missing = [n for n, resolved in [(home_raw, home), (away_raw, away)] if resolved is None]
                print(f"[debug] skipping {home_raw} vs {away_raw} ({code}): no history found for {missing} "
                      f"(even after fuzzy matching)")
                continue

            # Only the "fuzzy" tier is genuinely uncertain (alias-table
            # and core-stripped matches are deterministic and reliable,
            # so they're resolved silently without a caution flag - see
            # resolve_team_name's docstring).
            name_matches = {}
            if home_tier == "fuzzy":
                name_matches[home_raw] = home
            if away_tier == "fuzzy":
                name_matches[away_raw] = away
            if name_matches:
                print(f"[debug] fuzzy-matched fixture name(s) (flagged, uncertain): {name_matches}")
            elif home != home_raw or away != away_raw:
                print(f"[debug] resolved fixture name(s) via alias/core match (trusted, not flagged): "
                      f"{ {k: v for k, v in [(home_raw, home), (away_raw, away)] if k != v} }")

            cross_league = most_recent_league.get(home) != code or most_recent_league.get(away) != code
            card = build_fixture_card(home, away, team_forms[home], team_forms[away],
                                       league_avg, cross_league=cross_league, name_matches=name_matches)
            card["date"] = fx.get("Date", "")
            card["time"] = fx.get("Time", "")

            home_pos = team_position(league_table, home)
            away_pos = team_position(league_table, away)
            card["home_position"] = home_pos["position"] if home_pos else None
            card["away_position"] = away_pos["position"] if away_pos else None
            card["league_team_count"] = league_team_count

            fixture_cards.append(card)

        # Only surface team_form entries relevant to this league's own
        # squad list (i.e. teams that appear in this league's own CSV),
        # to avoid every league's JSON listing every team in England.
        league_team_names = set(r["HomeTeam"] for r in rows) | set(r["AwayTeam"] for r in rows)
        relevant_team_forms = {t: team_forms[t] for t in league_team_names if t in team_forms}

        statpack["generated_leagues"][code] = {
            "league_name": name,
            "league_averages": league_avg,
            "team_form": relevant_team_forms,
            "upcoming_fixtures": fixture_cards,
        }

    # Computed here (not in the dashboard's JS) so there's one
    # authoritative version of "what the best bets are right now" -
    # track_bets.py logs exactly this, and the dashboard just displays it.
    statpack["best_bets"] = compute_best_bets(statpack)
    statpack["data_freshness"] = compute_data_freshness(all_rows_by_league)

    if cup_fixtures_path is not None:
        cup_cards = build_cup_predictions(
            cup_fixtures_path, all_rows_by_league, team_forms, most_recent_league,
            league_averages_by_code, LEAGUE_NAMES, resolve_team_name,
        )
        statpack["efl_cup"] = {"fixtures": cup_cards}
        print(f"[debug] EFL Cup: {len(cup_cards)} fixture(s) predicted")

    return statpack


if __name__ == "__main__":
    from load_fixtures import load_all_fixtures

    data_dir = Path(__file__).parent / "data"
    fixtures_dir = Path(__file__).parent / "fixtures_manual"
    fixtures_dir.mkdir(exist_ok=True)
    cup_fixtures_dir = Path(__file__).parent / "cup_fixtures_manual"
    cup_fixtures_dir.mkdir(exist_ok=True)
    cup_fixtures_path = cup_fixtures_dir / "efl_cup.txt"

    fixtures = load_all_fixtures(fixtures_dir)

    pack = build_statpack(data_dir, fixtures, cup_fixtures_path=cup_fixtures_path)

    out_path = Path(__file__).parent / "statpack.json"
    with open(out_path, "w") as f:
        json.dump(pack, f, indent=2, default=str)

    # Also write a JS version (a plain global variable assignment) so the
    # dashboard can load it via a <script src="..."> tag. This means the
    # dashboard works by just double-clicking dashboard.html - no local
    # server needed - since browsers block fetch() of local JSON files
    # opened directly from disk (file:// CORS restrictions), but a
    # <script> tag has no such restriction.
    js_path = Path(__file__).parent / "statpack_data.js"
    with open(js_path, "w") as f:
        f.write("const STATPACK_DATA = ")
        json.dump(pack, f, default=str)
        f.write(";\n")

    print(f"Stat pack written to {out_path}")
    print(f"Dashboard data written to {js_path}")
