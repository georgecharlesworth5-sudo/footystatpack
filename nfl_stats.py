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
#
# Lowered from an earlier 0.85 to make that lean much stronger - at 0.85,
# the most recent 2 games only carried ~28% of total weight in a 10-game
# window, nowhere near enough to reflect "this season's form" once a team
# has only played 1-2 games so far this year. At 0.50, the most recent 2
# games carry ~75% of the total weight, with the remaining (mostly
# last-season-tail, early in a new campaign) games still contributing a
# real but clearly secondary ~25%.
WEIGHT_DECAY = 0.50

# Decay for the per-play measures below (plays per game, EPA per play).
# Backtested separately from WEIGHT_DECAY on 2018-2026 NFL games: these
# per-play measures do best around 0.85-0.92 (flat between those), and
# noticeably worse at 0.50 - per-play efficiency is a steadier signal than
# raw points, so it wants a longer memory rather than a short one.
EPA_WEIGHT_DECAY = 0.85
MIN_PLAY_GAMES = 3   # need at least this many games with play data, else no per-play form

STAT_KEYS = ["points", "passing_tds", "rushing_tds", "passing_yards", "rushing_yards"]

# Per-play measures (optional - only present if fetch_nfl.py supplies
# plays_for/plays_against/epa_for/epa_against columns). "plays" = pass
# attempts + sacks + carries; EPA = passing EPA + rushing EPA. The "for"
# value is the team's own offence; the "against" value is the opponent's
# offence in the same games (i.e. what this team's defence allowed).
PLAY_KEYS = ["plays", "epa_pp"]

# All-games form (used for the points estimate and the EPA estimate).
# Instead of splitting a team's form into home games and away games - which
# five games into a season leaves only 2-3 current-season games per team -
# every game counts, each one expressed relative to the league average for
# the venue it was played at (so home advantage is handled league-wide, not
# re-learned per team from a tiny sample). Backtested on 2018-2026 games,
# including early-season weeks (week <= 6): all games together, decay 0.85,
# window 17 beat the venue-split form on total points, margin and moneyline,
# and extra weight on current-season games did NOT help (a 5-game sample is
# too noisy to lean on harder). TDs and yardage keep their old venue-split
# form (WEIGHT_DECAY) - those weren't re-tested.
ALL_GAMES_DECAY = 0.85
ALL_GAMES_WINDOW = 17
MIN_ALL_GAMES = 3


def _add_ratio_fields(entry: dict, league: dict) -> None:
    """Express one team-game relative to the league average for its venue:
    pf_r / pa_r (points for / against as a ratio), plays_f_r / plays_a_r
    (plays as a ratio), epa_f_d / epa_a_d (EPA per play as a deviation).
    'for' is compared with the league's average for the team's own venue,
    'against' with the opponent's venue (the opponent's offence)."""
    side = "home" if entry["venue"] == "H" else "away"
    opp = "away" if side == "home" else "home"
    lg_pf, lg_pa = league.get(f"{side}_points"), league.get(f"{opp}_points")
    if lg_pf and lg_pa:
        entry["pf_r"] = entry["points_for"] / lg_pf
        entry["pa_r"] = entry["points_against"] / lg_pa
    if "plays_for" in entry:
        lg_plays_f, lg_plays_a = league.get(f"{side}_plays"), league.get(f"{opp}_plays")
        lg_epa_f, lg_epa_a = league.get(f"{side}_epa_pp"), league.get(f"{opp}_epa_pp")
        if lg_plays_f and lg_plays_a and lg_epa_f is not None and lg_epa_a is not None:
            entry["plays_f_r"] = entry["plays_for"] / lg_plays_f
            entry["plays_a_r"] = entry["plays_against"] / lg_plays_a
            entry["epa_f_d"] = entry["epa_pp_for"] - lg_epa_f
            entry["epa_a_d"] = entry["epa_pp_against"] - lg_epa_a


def all_games_form(team_log: list[dict], window: int = ALL_GAMES_WINDOW, decay: float = ALL_GAMES_DECAY) -> dict:
    """Recency-weighted form over a team's last `window` games of ANY venue,
    from the venue-normalised fields above. Keys are prefixed 'all_'; each
    group is None if fewer than MIN_ALL_GAMES games have the data."""
    recent = team_log[-window:]
    out: dict = {"all_matches": len(recent)}
    groups = {"pts": ["pf_r", "pa_r"], "play": ["plays_f_r", "plays_a_r", "epa_f_d", "epa_a_d"]}
    for keys in groups.values():
        have = [m for m in recent if all(k in m for k in keys)]
        if len(have) >= MIN_ALL_GAMES:
            w = [decay ** i for i in range(len(have))]
            w.reverse()
            tw = sum(w)
            for k in keys:
                out[f"all_{k}"] = round(sum(m[k] * wi for m, wi in zip(have, w)) / tw, 4)
        else:
            for k in keys:
                out[f"all_{k}"] = None
    return out


def build_team_game_log(rows: list[dict]) -> dict[str, list[dict]]:
    """One row per team-appearance, sorted oldest to newest per team.
    Each entry: {season, week, venue, opponent, points_for,
    points_against, passing_tds_for, passing_tds_against,
    rushing_tds_for, rushing_tds_against}, plus (when available) per-play
    fields and the league-relative fields used by all_games_form."""
    log: dict[str, list[dict]] = defaultdict(list)
    league = league_averages(rows)
    for row in rows:
        try:
            entry = {
                "season": int(row["season"]), "week": int(row["week"]),
                "venue": row["venue"], "opponent": row["opponent"],
                "points_for": float(row["points_for"]), "points_against": float(row["points_against"]),
                "passing_tds_for": int(row["passing_tds_for"]), "passing_tds_against": int(row["passing_tds_against"]),
                "rushing_tds_for": int(row["rushing_tds_for"]), "rushing_tds_against": int(row["rushing_tds_against"]),
                "passing_yards_for": float(row["passing_yards_for"]), "passing_yards_against": float(row["passing_yards_against"]),
                "rushing_yards_for": float(row["rushing_yards_for"]), "rushing_yards_against": float(row["rushing_yards_against"]),
            }
        except (ValueError, KeyError, TypeError):
            continue
        # Optional per-play fields - skipped silently for rows that don't
        # have them, so older cached data still loads and the model just
        # falls back to points-only.
        try:
            plays_for = float(row["plays_for"])
            plays_against = float(row["plays_against"])
            if plays_for >= 30 and plays_against >= 30:
                entry["plays_for"] = plays_for
                entry["plays_against"] = plays_against
                entry["epa_pp_for"] = float(row["epa_for"]) / plays_for
                entry["epa_pp_against"] = float(row["epa_against"]) / plays_against
                entry["ppp_for"] = entry["points_for"] / plays_for
        except (ValueError, KeyError, TypeError):
            pass
        if league:
            _add_ratio_fields(entry, league)
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
        result.update(all_games_form(team_log))  # venue-independent, so still available
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

    # Per-play form (plays per game, EPA per play): same window, but its
    # own (longer) decay, and only over games that actually have play
    # data. None if there aren't enough such games.
    play_matches = [m for m in matches if "plays_for" in m]
    if len(play_matches) >= MIN_PLAY_GAMES:
        pw = [EPA_WEIGHT_DECAY ** i for i in range(len(play_matches))]
        pw.reverse()
        ptotal = sum(pw)
        for key in PLAY_KEYS:
            for side in ("for", "against"):
                field = f"{key}_{side}"
                result[field] = round(sum(m[field] * w for m, w in zip(play_matches, pw)) / ptotal, 4)
        result["play_matches"] = len(play_matches)
    else:
        for key in PLAY_KEYS:
            result[f"{key}_for"] = None
            result[f"{key}_against"] = None
        result["play_matches"] = len(play_matches)
    # All-games form (same values whichever venue slice this is called for).
    result.update(all_games_form(team_log))
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
            passing_yards = float(row["passing_yards_for"])
            rushing_yards = float(row["rushing_yards_for"])
        except (ValueError, KeyError, TypeError):
            continue
        side = "home" if venue == "H" else "away"
        totals[f"{side}_points"] += points
        totals[f"{side}_passing_tds"] += passing_tds
        totals[f"{side}_rushing_tds"] += rushing_tds
        totals[f"{side}_passing_yards"] += passing_yards
        totals[f"{side}_rushing_yards"] += rushing_yards
        totals[f"{side}_matches"] += 1
        n += 1
        # Optional per-play league baselines (per-game averages of plays,
        # points per play and EPA per play, same way the backtest did it).
        try:
            plays_for = float(row["plays_for"])
            if plays_for >= 30:
                epa_pp = float(row["epa_for"]) / plays_for
                totals[f"{side}_plays"] += plays_for
                totals[f"{side}_ppp"] += points / plays_for
                totals[f"{side}_epa_pp"] += epa_pp
                totals[f"{side}_play_matches"] += 1
        except (ValueError, KeyError, TypeError):
            pass

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
        result[f"{side}_passing_yards"] = round(totals[f"{side}_passing_yards"] / matches, 3)
        result[f"{side}_rushing_yards"] = round(totals[f"{side}_rushing_yards"] / matches, 3)
        play_matches = totals.get(f"{side}_play_matches", 0)
        if play_matches >= 50:
            result[f"{side}_plays"] = round(totals[f"{side}_plays"] / play_matches, 3)
            result[f"{side}_ppp"] = round(totals[f"{side}_ppp"] / play_matches, 4)
            result[f"{side}_epa_pp"] = round(totals[f"{side}_epa_pp"] / play_matches, 4)
    result["matches"] = n
    return result
