"""
stats_engine.py

Turns raw match rows (from fetch_data.py) into per-team rolling form:
goals, corners, cards, first-half goals, second-half goals - for/against,
with home/away splits.

This is the layer the Poisson model builds on top of.
"""

from collections import defaultdict
from datetime import datetime

# Central list of tracked metrics. Add a new stat by adding it here plus
# the extraction logic in build_team_match_log/league_averages - everything
# else (rolling_form, the Poisson model loop) reads this list rather than
# hardcoding metric names.
METRICS = ["goals", "corners", "cards", "first_half_goals", "second_half_goals"]


def _parse_date(d: str) -> datetime:
    # football-data.co.uk uses dd/mm/yy or dd/mm/yyyy depending on era
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(d, fmt)
        except ValueError:
            continue
    return datetime.min


def build_team_match_log(rows: list[dict]) -> dict[str, list[dict]]:
    """
    Reshape match rows (one row per fixture) into one row per team-appearance,
    tagged with venue and the metrics relevant to that team. Sorted oldest
    to newest per team.

    Each entry: {date, venue, opponent, goals_for, goals_against,
                 corners_for, corners_against, cards_for, cards_against,
                 first_half_goals_for, first_half_goals_against,
                 second_half_goals_for, second_half_goals_against,
                 xg_for, xg_against}

    xg_for/xg_against are None when the source row had no HxG/AxG value
    (not every league/season has it) - downstream code must handle that,
    never assume it's always present.
    """
    log: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        try:
            dt = _parse_date(row["Date"])
            home, away = row["HomeTeam"], row["AwayTeam"]

            hg, ag = int(row["FTHG"]), int(row["FTAG"])
            hc, ac = int(row.get("HC") or 0), int(row.get("AC") or 0)
            hy, ay = int(row.get("HY") or 0), int(row.get("AY") or 0)
            hr, ar = int(row.get("HR") or 0), int(row.get("AR") or 0)
            h_cards, a_cards = hy + 2 * hr, ay + 2 * ar  # weighted card count (red=2)

            # HTHG/HTAG = goals at half-time = first-half goals directly.
            # Second-half goals = full-time total minus that.
            hth, ath = int(row.get("HTHG") or 0), int(row.get("HTAG") or 0)
            h_2nd, a_2nd = hg - hth, ag - ath
        except (ValueError, KeyError):
            continue  # skip malformed / incomplete rows (e.g. postponed games)

        try:
            h_xg = float(row.get("HxG") or "")
            a_xg = float(row.get("AxG") or "")
        except ValueError:
            h_xg = a_xg = None  # not every match has xG - leave as None, not 0

        log[home].append({
            "date": dt, "venue": "H", "opponent": away,
            "goals_for": hg, "goals_against": ag,
            "corners_for": hc, "corners_against": ac,
            "cards_for": h_cards, "cards_against": a_cards,
            "first_half_goals_for": hth, "first_half_goals_against": ath,
            "second_half_goals_for": h_2nd, "second_half_goals_against": a_2nd,
            "xg_for": h_xg, "xg_against": a_xg,
            "referee": row.get("Referee", ""),
        })
        log[away].append({
            "date": dt, "venue": "A", "opponent": home,
            "goals_for": ag, "goals_against": hg,
            "corners_for": ac, "corners_against": hc,
            "cards_for": a_cards, "cards_against": h_cards,
            "first_half_goals_for": ath, "first_half_goals_against": hth,
            "second_half_goals_for": a_2nd, "second_half_goals_against": h_2nd,
            "xg_for": a_xg, "xg_against": h_xg,
            "referee": row.get("Referee", ""),
        })

    for team in log:
        log[team].sort(key=lambda r: r["date"])

    return log


def rolling_form(team_log: list[dict], venue: str | None = None, window: int = 10) -> dict:
    """
    Average the last `window` matches for a team (optionally filtered to
    venue='H' or 'A' only). Returns per-90 averages for goals/corners/cards,
    both for and against.

    If venue is set, we look back further in the full log to find `window`
    matches at that venue (not just the last N overall).
    """
    if venue:
        matches = [m for m in team_log if m["venue"] == venue][-window:]
    else:
        matches = team_log[-window:]

    n = len(matches)
    if n == 0:
        return {"matches": 0}

    def avg(key):
        return round(sum(m[key] for m in matches) / n, 2)

    result = {"matches": n}
    for metric in METRICS:
        result[f"{metric}_for"] = avg(f"{metric}_for")
        result[f"{metric}_against"] = avg(f"{metric}_against")

    # xG: only average over matches that actually HAD an xG value - not
    # every league/season has it, so mixing in absent values would
    # silently corrupt the average. xg_matches tells the caller how much
    # to trust the figure (e.g. 2 games' worth of xG isn't a lot to
    # blend on).
    xg_matches = [m for m in matches if m["xg_for"] is not None and m["xg_against"] is not None]
    if xg_matches:
        result["xg_for"] = round(sum(m["xg_for"] for m in xg_matches) / len(xg_matches), 2)
        result["xg_against"] = round(sum(m["xg_against"] for m in xg_matches) / len(xg_matches), 2)
        result["xg_matches"] = len(xg_matches)
    else:
        result["xg_for"] = result["xg_against"] = None
        result["xg_matches"] = 0

    return result


def league_averages(rows: list[dict]) -> dict:
    """League-wide per-match averages, used to normalise team strength in
    the Poisson model (an attack strength of 1.0 = league average)."""
    n = 0
    xg_n = 0
    totals = defaultdict(float)
    for row in rows:
        try:
            hg, ag = int(row["FTHG"]), int(row["FTAG"])
            hc, ac = int(row.get("HC") or 0), int(row.get("AC") or 0)
            hy, ay = int(row.get("HY") or 0), int(row.get("AY") or 0)
            hr, ar = int(row.get("HR") or 0), int(row.get("AR") or 0)
            hth, ath = int(row.get("HTHG") or 0), int(row.get("HTAG") or 0)
            h_2nd, a_2nd = hg - hth, ag - ath
        except (ValueError, KeyError):
            continue
        n += 1
        totals["home_goals"] += hg
        totals["away_goals"] += ag
        totals["home_corners"] += hc
        totals["away_corners"] += ac
        totals["home_cards"] += hy + 2 * hr
        totals["away_cards"] += ay + 2 * ar
        totals["home_first_half_goals"] += hth
        totals["away_first_half_goals"] += ath
        totals["home_second_half_goals"] += h_2nd
        totals["away_second_half_goals"] += a_2nd

        try:
            h_xg, a_xg = float(row.get("HxG") or ""), float(row.get("AxG") or "")
            totals["home_xg"] += h_xg
            totals["away_xg"] += a_xg
            xg_n += 1
        except ValueError:
            pass  # this match had no xG - just doesn't contribute to the xG average

    if n == 0:
        return {}

    result = {k: round(v / n, 3) for k, v in totals.items() if not k.endswith("_xg")}
    result |= {"matches": n}
    if xg_n > 0:
        result["home_xg"] = round(totals["home_xg"] / xg_n, 3)
        result["away_xg"] = round(totals["away_xg"] / xg_n, 3)
    result["xg_matches"] = xg_n
    return result


def team_form_summary(team_log: list[dict], window: int = 10) -> dict:
    """Convenience wrapper: overall / home / away rolling form for one team."""
    return {
        "overall": rolling_form(team_log, None, window),
        "home": rolling_form(team_log, "H", window),
        "away": rolling_form(team_log, "A", window),
    }
