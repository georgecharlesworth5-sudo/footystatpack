"""
nfl_market_odds.py

Blends real Vegas lines (moneyline, total points) into our own model's
NFL predictions - same philosophy as the football project's
market_odds.py: lean on a more informed external signal without fully
replacing our own model, blend rather than override.

## American odds, not decimal

nflverse's games.csv carries moneylines and over/under odds in AMERICAN
format (e.g. "145", "-160"), confirmed directly from real data - not
the decimal format football-data.co.uk uses. Converted to implied
probability:
    positive odds (e.g. +145): 100 / (odds + 100)
    negative odds (e.g. -160): -odds / (-odds + 100)

## De-vigging

Same reasoning as football: raw implied probabilities from odds always
sum to slightly more than 100% (the sportsbook's margin). Normalising
back to exactly 100% before blending avoids systematically biasing
every prediction toward "more likely than the market really thinks".

## Scope

Only moneyline and total points get blended - team-level points,
passing TDs, and rushing TDs have no market equivalent in this data
source, so they stay pure-model, same as corners/cards on the football
side.
"""

MARKET_WEIGHT = 0.5  # same deliberately-balanced starting point as the football side


def american_to_implied_prob(odds) -> float | None:
    try:
        odds = float(odds)
    except (ValueError, TypeError):
        return None
    if odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return -odds / (-odds + 100.0)


def de_vig_pair(prob_a: float | None, prob_b: float | None) -> tuple[float, float] | None:
    """De-vigs a two-outcome market (moneyline, or an over/under pair)."""
    if prob_a is None or prob_b is None:
        return None
    total = prob_a + prob_b
    if total <= 0:
        return None
    return prob_a / total, prob_b / total


def parse_market_row(row: dict) -> dict:
    """
    Extracts whatever usable de-vigged probabilities are present in one
    fixture's row from the cached upcoming-fixtures file. Returns a
    dict with only the keys that were actually derivable - callers
    check for key presence rather than assuming every fixture has both
    markets available (confirmed directly: only a partial subset of
    upcoming fixtures have lines populated this far ahead of kickoff).
    """
    result = {}

    home_ml = american_to_implied_prob(row.get("home_moneyline"))
    away_ml = american_to_implied_prob(row.get("away_moneyline"))
    devigged = de_vig_pair(home_ml, away_ml)
    if devigged:
        result["home_win"], result["away_win"] = devigged

    over_p = american_to_implied_prob(row.get("over_odds"))
    under_p = american_to_implied_prob(row.get("under_odds"))
    devigged_total = de_vig_pair(over_p, under_p)
    if devigged_total and row.get("total_line") not in (None, "", "NA"):
        try:
            result["total_line"] = float(row["total_line"])
            result["total_over"], result["total_under"] = devigged_total
        except ValueError:
            pass

    return result


def blend_moneyline(model_ml: dict, market_probs: dict, weight: float = MARKET_WEIGHT) -> dict:
    """Blends our own home_win/away_win with the market's de-vigged
    equivalent. Returns model_ml unchanged if no moneyline market data
    is available for this fixture."""
    if "home_win" not in market_probs:
        return model_ml
    blended_home = (1 - weight) * model_ml["home_win"] + weight * market_probs["home_win"]
    return {"home_win": round(blended_home, 3), "away_win": round(1 - blended_home, 3)}


def blend_total_points(model_over_under: list[dict], market_probs: dict, weight: float = MARKET_WEIGHT) -> list[dict]:
    """
    Blends our own total-points O/U list with the market's line, ONLY
    for whichever of our own lines matches the market's total_line
    exactly - a market line of 47.5 shouldn't get blended into our
    separate 44.5 evaluation, those are different bets. If the market's
    line isn't one we already evaluate, it's added as an extra entry so
    the market signal isn't silently dropped.
    """
    if "total_line" not in market_probs:
        return model_over_under

    market_line = market_probs["total_line"]
    result = []
    matched = False
    for entry in model_over_under:
        if entry["line"] == market_line:
            blended_over = (1 - weight) * entry["over"] + weight * market_probs["total_over"]
            result.append({**entry, "over": round(blended_over, 3), "under": round(1 - blended_over, 3)})
            matched = True
        else:
            result.append(entry)
    if not matched:
        result.append({
            "line": market_line,
            "over": round(market_probs["total_over"], 3),
            "under": round(market_probs["total_under"], 3),
            "expected": model_over_under[0]["expected"] if model_over_under else None,
        })
    # The market's own line (if it didn't match one of ours) got
    # appended at the end regardless of its numeric value - re-sort so
    # a line like 43.5 doesn't show up out of order after 44.5/48.5.
    result.sort(key=lambda e: e["line"])
    return result
