"""
calibration.py

Corrects the model's stated confidence using how its picks have actually
performed.

## The problem

Across ~600 settled picks the model said ~90% on average and landed ~81%.
That overconfidence isn't uniform: cards and team-level picks land close
to what the model says, while match goals Over 1.5, match corners Over 8.5
and the half-goals markets land 9-18 points below it (a plain Poisson
underestimates the spread on those counts). A single global confidence bar
therefore lets the overconfident markets flood Best Bets.

## The fix

For each MARKET - one specific bet type, (sport, metric, match-vs-team,
direction, line), e.g. "football Match Goals Over 1.5" - compare the
model's average stated probability with the hit rate it actually achieved,
and shift future picks in that market by the gap:

    calibrated = stated + (actual - stated_average) * n / (n + K)

The n/(n+K) factor shrinks the correction when a market has few settled
picks, so two unlucky games can't sink a market on their own; with K=30 a
market needs roughly 30 settled picks before half the observed gap is
trusted, and about 100 before most of it is. Markets with no history are
left alone.

## What this deliberately does NOT do

 * It never touches the model. The raw probability stays the thing
   that's logged and measured (log "confidence" column = raw), so
   calibration is always computed against the model's own output - no
   feedback loop of correcting an already-corrected number.
 * It never stops measuring a market. Picks that clear the raw bar but
   fall short once calibrated are "held back" - not shown in Best Bets,
   but still logged (shown="0") and settled, so every market keeps
   accumulating evidence and can earn its way back in. They also give a
   genuine out-of-sample check: how do the picks the filter removed
   actually do?
 * It's derived from bets_log.csv every time it's needed - there's no
   calibration file to commit or go stale.

It is a correction for overconfidence, not a profitability test - hit
rate says nothing about the price (see the break-even odds table on the
Track Record tab).
"""

import csv
from pathlib import Path

SHRINKAGE_K = 30       # pseudo-count: bigger = slower to trust a market's own gap
MAX_ADJUSTMENT = 0.25  # never move a probability by more than this, whatever the data says


def _norm_line(line) -> str:
    if line in ("", None):
        return ""
    try:
        return f"{float(line):g}"
    except (ValueError, TypeError):
        return str(line)


def market_key(sport: str, metric: str, scope: str, direction: str, line) -> str:
    """One key per bet type. Home and away team-level picks pool together
    (they share lines); team_win/moneyline collapse to "the team to win"
    regardless of which team it was. Same grouping as the Track Record
    tab's break-even odds table."""
    is_win = metric in ("team_win", "moneyline")
    level = "team" if (is_win or scope in ("home", "away")) else "match"
    return "|".join([
        sport or "football", metric, level,
        "" if is_win else (direction or ""),
        "" if is_win else _norm_line(line),
    ])


def build_calibration(log_path: Path) -> dict[str, dict]:
    """Per-market {"n", "hits", "avg_stated", "adjustment"} from every
    settled pick in the log (shown AND held-back ones - the whole point
    is that held-back markets keep being measured). Returns {} if there's
    no log yet."""
    if not Path(log_path).exists():
        return {}

    stats: dict[str, dict] = {}
    with open(log_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "settled" or row.get("direction") in ("Under", "No"):
                continue
            try:
                stated = float(row["confidence"])
            except (ValueError, TypeError, KeyError):
                continue
            key = market_key(row.get("sport") or "football", row["metric"], row.get("scope", ""),
                             row["direction"], row.get("line"))
            s = stats.setdefault(key, {"n": 0, "hits": 0, "stated_sum": 0.0})
            s["n"] += 1
            s["hits"] += row.get("result") == "hit"
            s["stated_sum"] += stated

    calibration = {}
    for key, s in stats.items():
        avg_stated = s["stated_sum"] / s["n"]
        gap = s["hits"] / s["n"] - avg_stated
        adjustment = gap * s["n"] / (s["n"] + SHRINKAGE_K)
        calibration[key] = {
            "n": s["n"], "hits": s["hits"],
            "avg_stated": round(avg_stated, 3),
            "adjustment": round(max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, adjustment)), 4),
        }
    return calibration


def load_calibration(log_path: Path) -> dict[str, dict]:
    return build_calibration(log_path)


def calibrated_confidence(calibration: dict | None, pick: dict) -> float:
    """The pick's raw confidence shifted by its market's adjustment
    (unchanged if the market has no settled history)."""
    raw = pick["confidence"]
    if not calibration:
        return raw
    key = market_key(pick.get("sport", "football"), pick["metric"], pick.get("scope", ""),
                     pick["direction"], pick.get("line"))
    adjustment = calibration.get(key, {}).get("adjustment", 0.0)
    return round(max(0.0, min(0.999, raw + adjustment)), 3)


def split_by_calibration(picks: list[dict], calibration: dict | None, bar_for) -> tuple[list[dict], list[dict]]:
    """Annotate every raw-qualifying pick with raw_confidence and its
    calibrated confidence (which replaces "confidence", so everything
    downstream - sorting, display - uses the honest number), then split
    into (shown, held_back) by whether the calibrated value still clears
    the pick's bar. bar_for(pick) returns that bar (e.g. 0.87, or the
    lower one for team-win / moneyline)."""
    shown, held_back = [], []
    for pick in picks:
        raw = pick["confidence"]
        pick["raw_confidence"] = raw
        pick["confidence"] = calibrated_confidence(calibration, pick)
        (shown if pick["confidence"] >= bar_for(pick) else held_back).append(pick)
    shown.sort(key=lambda p: p["confidence"], reverse=True)
    held_back.sort(key=lambda p: p["confidence"], reverse=True)
    return shown, held_back
