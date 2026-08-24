"""
league_table.py

Computes real league table standings (position, points, goal difference)
from the match results we already have. Built for the EFL Cup cross-
league prediction feature: a team's actual league position is one of
the only meaningful signals we have for comparing teams across
different divisions, since our within-division attack/defense strength
numbers (see poisson_model.py) aren't directly comparable between
divisions - a "20% above average" League Two team and a "20% above
average" Premier League team are not the same absolute quality.

Standard English football tie-break order: points, then goal
difference, then goals for.
"""

from collections import defaultdict
from stats_engine import _parse_date


def detect_current_season(rows: list[dict]) -> str | None:
    """Auto-detect the most recent season code present (e.g. "2627"),
    so the table reflects only the current campaign, not last season's
    results mixed in - our combined data intentionally holds two
    seasons for form purposes, but a table needs just the one."""
    seasons = {row.get("Season") for row in rows if row.get("Season")}
    if not seasons:
        return None
    return max(seasons)  # season codes sort correctly as strings, e.g. "2526" < "2627"


def compute_league_table(rows: list[dict], season: str | None = None) -> list[dict]:
    """
    Returns a list of team standings, sorted by rank (1st place first).
    Each entry: {team, played, won, drawn, lost, goals_for, goals_against,
                 goal_difference, points, position}

    Only counts matches from the given season (auto-detected if not
    specified) - a team's position should reflect their current
    campaign, not a blend of this season and last.
    """
    if season is None:
        season = detect_current_season(rows)

    stats: dict[str, dict] = defaultdict(lambda: {
        "played": 0, "won": 0, "drawn": 0, "lost": 0,
        "goals_for": 0, "goals_against": 0,
    })

    for row in rows:
        if season is not None and row.get("Season") != season:
            continue
        try:
            home, away = row["HomeTeam"], row["AwayTeam"]
            hg, ag = int(row["FTHG"]), int(row["FTAG"])
        except (ValueError, KeyError):
            continue  # not yet played, or malformed row

        h, a = stats[home], stats[away]
        h["played"] += 1
        a["played"] += 1
        h["goals_for"] += hg
        h["goals_against"] += ag
        a["goals_for"] += ag
        a["goals_against"] += hg

        if hg > ag:
            h["won"] += 1
            a["lost"] += 1
        elif hg < ag:
            a["won"] += 1
            h["lost"] += 1
        else:
            h["drawn"] += 1
            a["drawn"] += 1

    table = []
    for team, s in stats.items():
        points = s["won"] * 3 + s["drawn"]
        gd = s["goals_for"] - s["goals_against"]
        table.append({
            "team": team, "played": s["played"], "won": s["won"],
            "drawn": s["drawn"], "lost": s["lost"],
            "goals_for": s["goals_for"], "goals_against": s["goals_against"],
            "goal_difference": gd, "points": points,
        })

    # Standard tie-break order: points desc, then GD desc, then GF desc
    table.sort(key=lambda t: (-t["points"], -t["goal_difference"], -t["goals_for"]))
    for i, t in enumerate(table):
        t["position"] = i + 1

    return table


def team_position(table: list[dict], team: str) -> dict | None:
    """Look up one team's standing from a computed table, or None if
    they're not in it (e.g. haven't played any current-season matches
    yet, or the name doesn't match - no fuzzy resolution here, pass in
    the already-resolved canonical name)."""
    for entry in table:
        if entry["team"] == team:
            return entry
    return None
