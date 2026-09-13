"""
nfl_stats.py

Turns the joined team-game dataset (fetch_nfl.py's nfl_team_games.csv)
into per-team rolling form: points, passing TDs, and rushing TDs - for
and against, recency-weighted, home/away splits. Same architecture as
the football project's stats_engine.py, adapted for what NFL data
actually looks like.
"""

from collections import defaultdict

# Same reasoning as the football model (stats_engine.WEIGHT_DECAY): most
# recent game counts most, decaying by this factor per game further
# back. Combining last season + current season means this is what lets
# early-season predictions lean on real current-season games as soon as
# there are any, rather than treating a 10-months-ago game the same as
# last week's.
WEIGHT_DECAY = 0.85

STAT_KEYS = ["points", "passing_tds", "rushing_tds"]


def build_team_game_log(rows: list[dict]) -> dict[str, list[dict]]:
    """One row per team-appearance, sorted oldest to newest per team.
    Each entry: {season, week, venue, opponent, points_for,
    points_against, passing_tds_for, passing_tds_against,
    rushing_tds_for, rushing_tds_against}."""
    log: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        try:
            entry = {
                "season": int(row["season"]), "week": int(row["week"]),
                "venue": row["venue"], "opponent": row["opponent"],
                "points_for": float(row["points_for"]), "points_against": float(row["points_against"]),
                "passing_tds_for": int(row["passing_tds_for"]), "passing_tds_against": int(row["passing_tds_against"]),
                "rushing_tds_for": int(row["rushing_tds_for"]), "rushing_tds_against": int(row["rushing_tds_against"]),
            }
        except (ValueError, KeyError, TypeError):
            continue
        log[row["team"]].append(entry)

    for team in log:
        log[team].sort(key=lambda r: (r["season"], r["week"]))
    return log


def rolling_form(team_log: list[dict], venue: str | None = None, window: int = 10) -> dict:
    """Recency-weighted average points/passing TDs/rushing TDs for and
    against - same weighting scheme as the football model's
    rolling_form. Returns None for each stat if there's no data at all
    (brand new team/venue combo), rather than a misleading zero."""
    if venue:
        matches = [m for m in team_log if m["venue"] == venue][-window:]
    else:
        matches = team_log[-window:]

    n = len(matches)
    if n == 0:
        result = {"matches": 0}
        for key in STAT_KEYS:
            result[f"{key}_for"] = None
            result[f"{key}_against"] = None
        return result

    weights = [WEIGHT_DECAY ** i for i in range(n)]
    weights.reverse()
    total_weight = sum(weights)

    def w_avg(field):
        return round(sum(m[field] * w for m, w in zip(matches, weights)) / total_weight, 3)

    result = {"matches": n}
    for key in STAT_KEYS:
        result[f"{key}_for"] = w_avg(f"{key}_for")
        result[f"{key}_against"] = w_avg(f"{key}_against")
    return result


def team_form_summary(team_log: list[dict], window: int = 10) -> dict:
    """Convenience wrapper: overall / home / away rolling form for one team."""
    return {
        "overall": rolling_form(team_log, None, window),
        "home": rolling_form(team_log, "H", window),
        "away": rolling_form(team_log, "A", window),
    }


def league_averages(rows: list[dict]) -> dict:
    """League-wide average home/away values per game, for normalising
    team strength (same role as the football model's league_averages)."""
    totals = defaultdict(float)
    n = 0
    for row in rows:
        try:
            venue = row["venue"]
            points = float(row["points_for"])
            passing_tds = int(row["passing_tds_for"])
            rushing_tds = int(row["rushing_tds_for"])
        except (ValueError, KeyError, TypeError):
            continue
        side = "home" if venue == "H" else "away"
        totals[f"{side}_points"] += points
        totals[f"{side}_passing_tds"] += passing_tds
        totals[f"{side}_rushing_tds"] += rushing_tds
        totals[f"{side}_matches"] += 1
        n += 1

    if not totals:
        return {}

    result = {}
    for side in ("home", "away"):
        matches = totals.get(f"{side}_matches", 0)
        if matches == 0:
            continue
        result[f"{side}_points"] = round(totals[f"{side}_points"] / matches, 3)
        result[f"{side}_passing_tds"] = round(totals[f"{side}_passing_tds"] / matches, 3)
        result[f"{side}_rushing_tds"] = round(totals[f"{side}_rushing_tds"] / matches, 3)
    result["matches"] = n
    return result
