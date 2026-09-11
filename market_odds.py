"""
market_odds.py

Blends real bookmaker market odds (from football-data.co.uk's
fixtures.csv - see fetch_data.py's fetch_fixtures/cache_fixture_odds)
into our own model's predictions for match result and total goals.

## Why blend rather than just show the market's own numbers

The market is, honestly, better-informed than our simple rolling-form
model - it prices in injuries, team news, rest, motivation, and
everything else a bookmaker's trading team knows that we can't see
from historical scorelines alone. But it's not infallible, and our own
model has real signal too (it's what the rest of this whole project is
built on). Blending the two - rather than fully replacing our model
with the market, or ignoring the market entirely - is the same
philosophy as the xG blend in poisson_model.py: lean on a more
reliable external signal without discarding our own.

## De-vigging

Bookmaker odds always imply MORE than 100% combined probability (their
margin/vig) - e.g. AvgH + AvgD + AvgA implied probabilities might sum
to 106%. Blending against that raw, inflated total would systematically
bias every prediction toward "more likely than the market really
thinks", for every outcome, in every match - so odds are normalised
("de-vigged") to sum to exactly 100% first, giving the market's actual
implied view rather than its raw pricing.

## Scope

Only match result (1X2) and total goals over/under 2.5 get blended -
those are the only markets football-data.co.uk's fixtures.csv actually
carries odds for. Corners, cards, and every other goals line (1.5, 3.5)
stay pure-model, same as before.
"""

MARKET_WEIGHT = 0.5  # how much trust the market gets vs our own model, 0-1.
                      # 0.5 = equal weight - a deliberately balanced starting
                      # point, not a claim that either side is more right.
                      # Adjust once real results give a read on which way
                      # to lean.


def implied_probs_from_odds(odds: list[float]) -> list[float] | None:
    """
    Converts a set of decimal odds (e.g. [AvgH, AvgD, AvgA]) into
    de-vigged ("fair") probabilities that sum to exactly 1.0. Returns
    None if any odds value is missing/invalid - callers should treat
    that as "no market data available", not as a probability of zero.
    """
    try:
        raw = [1.0 / float(o) for o in odds]
    except (ValueError, ZeroDivisionError, TypeError):
        return None
    overround = sum(raw)
    if overround <= 0:
        return None
    return [p / overround for p in raw]


def parse_fixture_odds_row(row: dict) -> dict | None:
    """
    Given one row from the cached fixture_odds.csv (see
    fetch_data.cache_fixture_odds), returns the de-vigged match-result
    and over/under-2.5 probabilities, or None if the row's odds are
    incomplete/unusable.
    """
    match_result_probs = implied_probs_from_odds([row.get("AvgH"), row.get("AvgD"), row.get("AvgA")])
    ou25_probs = implied_probs_from_odds([row.get("AvgOver25"), row.get("AvgUnder25")])

    if match_result_probs is None and ou25_probs is None:
        return None

    result = {}
    if match_result_probs is not None:
        result["home_win"], result["draw"], result["away_win"] = match_result_probs
    if ou25_probs is not None:
        result["over_2_5"], result["under_2_5"] = ou25_probs
    return result


def load_fixture_odds(path) -> list[dict]:
    """Loads the cached fixture_odds.csv (if present) into a plain
    list of row dicts - kept as raw dicts rather than pre-parsed, since
    build_statpack.py needs the team names/date for matching before
    the odds themselves are parsed."""
    import csv
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def resolve_odds_rows(odds_rows: list[dict], known_names: list[str], resolve_team_name) -> list[dict]:
    """
    Resolves each odds row's HomeTeam/AwayTeam through the SAME name-
    resolution used for the main fixture list (build_statpack.py's
    resolve_team_name, passed in here rather than imported, to avoid a
    circular import between the two modules).

    This has to run before find_odds_for_fixture's exact-match lookup
    ever has a chance of working - fixture_odds.csv comes from
    football-data.co.uk's own fixtures.csv, a genuinely different
    source to fixturedownload.com (used for the main fixtures_manual/
    lists), and the two don't always spell team names the same way.
    Confirmed as a real issue, not a hypothetical: without this step,
    cached odds existed for every fixture in one real case, yet zero
    fixtures ever showed as market-blended, because the exact string
    match silently failed on every single one.

    Rows where either team can't be resolved at all are dropped - an
    unmatchable row has no use here regardless of what odds it carries.
    """
    resolved = []
    for row in odds_rows:
        home, _ = resolve_team_name(row.get("HomeTeam", ""), known_names)
        away, _ = resolve_team_name(row.get("AwayTeam", ""), known_names)
        if home is None or away is None:
            continue
        new_row = dict(row)
        new_row["HomeTeam"] = home
        new_row["AwayTeam"] = away
        resolved.append(new_row)
    return resolved


# Same reasoning as track_bets.py's MAX_DATE_DRIFT_DAYS: the fixture list
# and the odds cache are two independently-fetched snapshots of
# football-data.co.uk, so a fixture that's been postponed/rescheduled
# between the two fetches could show a slightly different date in each -
# an exact date match would silently drop that fixture's odds entirely.
MAX_DATE_DRIFT_DAYS = 3


def find_odds_for_fixture(odds_rows: list[dict], home_team: str, away_team: str, date: str) -> dict | None:
    """
    Matches one fixture (by resolved home/away team name, and date within
    MAX_DATE_DRIFT_DAYS) against the cached odds rows. Team-name matching
    is exact, but odds_rows must already have been resolved through the
    same name-resolution used for the main fixture list before calling
    this (see resolve_odds_rows below) - without that step, this
    silently matches nothing whenever football-data.co.uk's own naming
    differs from fixturedownload.com's, which happens often enough to be
    a real, confirmed issue.
    """
    from datetime import datetime

    def _parse(d):
        try:
            return datetime.strptime(d, "%d/%m/%Y")
        except (ValueError, TypeError):
            return None

    target_date = _parse(date)
    best_row, best_drift = None, None
    for row in odds_rows:
        if row.get("HomeTeam") != home_team or row.get("AwayTeam") != away_team:
            continue
        row_date = _parse(row.get("Date", ""))
        if target_date is None or row_date is None:
            if row.get("Date") == date:  # fall back to exact string match if either date is unparseable
                return parse_fixture_odds_row(row)
            continue
        drift = abs((row_date - target_date).days)
        if drift <= MAX_DATE_DRIFT_DAYS and (best_drift is None or drift < best_drift):
            best_row, best_drift = row, drift
    if best_row is not None:
        return parse_fixture_odds_row(best_row)
    return None


def blend_match_result(model_result: dict, market_probs: dict, weight: float = MARKET_WEIGHT) -> dict:
    """
    Blends our own model's home_win/draw/away_win with the market's
    de-vigged equivalents. If market_probs doesn't have match-result
    data (e.g. only O/U odds were available), returns model_result
    unchanged.
    """
    if "home_win" not in market_probs:
        return model_result
    blended = {
        "home_win": (1 - weight) * model_result["home_win"] + weight * market_probs["home_win"],
        "draw": (1 - weight) * model_result["draw"] + weight * market_probs["draw"],
        "away_win": (1 - weight) * model_result["away_win"] + weight * market_probs["away_win"],
    }
    # Blending two already-normalised distributions can leave a tiny
    # rounding gap from 1.0 - renormalise so the three options still
    # sum to exactly 100% for display.
    total = sum(blended.values())
    return {k: round(v / total, 3) for k, v in blended.items()}


def blend_over_under_25(model_over: float, market_probs: dict, weight: float = MARKET_WEIGHT) -> float | None:
    """
    Blends our own model's P(over 2.5) with the market's de-vigged
    equivalent. Returns None if market_probs doesn't have O/U-2.5 data
    (e.g. only match-result odds were available) - caller should treat
    that as "leave the model's own figure alone", not as zero.
    """
    if "over_2_5" not in market_probs:
        return None
    return round((1 - weight) * model_over + weight * market_probs["over_2_5"], 3)
