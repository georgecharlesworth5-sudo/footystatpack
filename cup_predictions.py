"""
cup_predictions.py

Builds predictions for EFL Cup (or any other cross-league cup) fixtures
that a normal league round doesn't cover - see cross_league.py for why
this needs a different approach than the regular per-league predictions.

## Fixture input format

Since no data source we're allowed to use covers cup fixtures (see the
notes in cross_league.py and build_statpack.py), fixtures are entered
by hand into cup_fixtures_manual/efl_cup.txt - a simple text format
you can type directly on mobile, no CSV needed:

    Date: 25/08/2026
    Cardiff v Norwich
    Doncaster v Middlesbrough
    Barnsley v Crewe

    Date: 26/08/2026
    Bradford City v Burnley
    Newcastle v West Brom

One "Date: dd/mm/yyyy" line starts a new round; every following line
until the next Date line (or end of file) is one fixture as
"Home v Away" (the " v " separator, with spaces, matters). Blank lines
are ignored. Team names should match how they appear elsewhere in the
stat pack (e.g. "Sheffield Weds" not "Sheff Wed") where possible - the
same fuzzy/alias name resolution used for league fixtures also runs
here, so close variants usually still resolve correctly.
"""

from pathlib import Path

from stats_engine import METRICS
from league_table import compute_league_table, team_position
from cross_league import predict_cross_league_fixture, position_multiplier
from poisson_model import predict_fixture

DEFAULT_LINES = {
    "goals": [1.5, 2.5, 3.5],
    "corners": [8.5, 9.5, 10.5, 11.5],
    "cards": [2.5, 3.5, 4.5],
    "first_half_goals": [0.5, 1.5],
    "second_half_goals": [0.5, 1.5, 2.5],
}


def parse_cup_fixtures(path: Path) -> list[dict]:
    """Parse the plain-text manual fixture file. Returns a list of
    {date, home_team, away_team} - team names exactly as typed, not
    yet resolved against known names."""
    if not path.exists():
        return []

    fixtures = []
    current_date = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("date:"):
                current_date = line.split(":", 1)[1].strip()
                continue
            if " v " not in line:
                print(f"[debug] cup fixtures: skipping unparseable line: {line!r}")
                continue
            home, away = line.split(" v ", 1)
            fixtures.append({
                "date": current_date,
                "home_team": home.strip(),
                "away_team": away.strip(),
            })
    return fixtures


def build_cup_predictions(
    cup_fixtures_path: Path,
    all_rows_by_league: dict[str, list[dict]],
    team_forms: dict[str, dict],
    most_recent_league: dict[str, str],
    league_averages_by_code: dict[str, dict],
    league_names: dict[str, str],
    resolve_team_name,
) -> list[dict]:
    """
    Builds a prediction card for each manually-entered cup fixture.
    Reuses everything build_statpack.py already computed for the
    league predictions (team forms, most-recent-league lookup, league
    averages) rather than recalculating any of it.
    """
    raw_fixtures = parse_cup_fixtures(cup_fixtures_path)
    if not raw_fixtures:
        return []

    known_names = list(team_forms.keys())

    # Cache league tables per league code - only compute each once even
    # if many fixtures reference the same league.
    tables_cache: dict[str, list[dict]] = {}

    def get_table(league_code: str) -> list[dict]:
        if league_code not in tables_cache:
            tables_cache[league_code] = compute_league_table(all_rows_by_league.get(league_code, []))
        return tables_cache[league_code]

    cards = []
    for fx in raw_fixtures:
        home_raw, away_raw = fx["home_team"], fx["away_team"]
        home, home_tier = resolve_team_name(home_raw, known_names)
        away, away_tier = resolve_team_name(away_raw, known_names)

        if home is None or away is None:
            missing = [n for n, resolved in [(home_raw, home), (away_raw, away)] if resolved is None]
            print(f"[debug] skipping cup fixture {home_raw} v {away_raw}: no history found for {missing}")
            continue

        home_league = most_recent_league.get(home)
        away_league = most_recent_league.get(away)
        if home_league is None or away_league is None:
            print(f"[debug] skipping cup fixture {home_raw} v {away_raw}: "
                  f"couldn't determine current league for {'home' if home_league is None else 'away'} team")
            continue

        name_matches = {}
        if home_tier == "fuzzy":
            name_matches[home_raw] = home
        if away_tier == "fuzzy":
            name_matches[away_raw] = away

        same_league = home_league == away_league

        if same_league:
            # Both teams in the same division - this is really just a
            # normal same-league prediction, no cross-league adjustment
            # needed. Reuse the regular model directly.
            league_avg = league_averages_by_code[home_league]
            predictions = predict_fixture(
                team_forms[home]["home"], team_forms[away]["away"], league_avg, DEFAULT_LINES
            )
        else:
            home_table = get_table(home_league)
            away_table = get_table(away_league)
            home_pos_entry = team_position(home_table, home)
            away_pos_entry = team_position(away_table, away)
            home_mult = position_multiplier(
                home_pos_entry["position"] if home_pos_entry else None, len(home_table)
            )
            away_mult = position_multiplier(
                away_pos_entry["position"] if away_pos_entry else None, len(away_table)
            )
            predictions = predict_cross_league_fixture(
                team_forms[home]["home"], team_forms[away]["away"],
                league_averages_by_code[home_league], league_averages_by_code[away_league],
                home_league, away_league,
                home_mult, away_mult, DEFAULT_LINES,
            )

        card = {
            "home_team": home, "away_team": away,
            "home_league": league_names.get(home_league, home_league),
            "away_league": league_names.get(away_league, away_league),
            "cross_division": not same_league,
            "date": fx["date"] or "", "time": "",
            "home_form_sample": team_forms[home]["home"].get("matches", 0),
            "away_form_sample": team_forms[away]["away"].get("matches", 0),
            "predictions": predictions,
        }
        if not same_league:
            home_pos_entry = team_position(get_table(home_league), home)
            away_pos_entry = team_position(get_table(away_league), away)
            card["home_position"] = home_pos_entry["position"] if home_pos_entry else None
            card["away_position"] = away_pos_entry["position"] if away_pos_entry else None
            card["note"] = (
                f"Cross-division fixture ({card['home_league']} v {card['away_league']}) - "
                f"predictions use each team's own league form with a rough cross-league adjustment. "
                f"Treat with more caution than same-league predictions."
            )
        if name_matches:
            card["fuzzy_name_match"] = name_matches

        cards.append(card)

    return cards
