"""
nba_build_statpack.py

Orchestrates the NBA pipeline - the equivalent of nfl_build_statpack.py:
  1. Read fetch_nba.py's cached data (data/nba_*.csv)
  2. Fit team ratings + player form (nba_stats.py)
  3. For each upcoming game, run the model (nba_model.py): moneyline,
     spread, game total, team totals and player props
  4. Write nba_statpack.json / nba_statpack_data.js for the dashboard
     (const NBA_STATPACK_DATA) and pick Best Bets (nba_best_bets.py,
     shadow mode for now)

Run after fetch_nba.py. Needs pandas + numpy.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import nba_model as M
import nba_stats as S
from nba_best_bets import SHADOW_MODE, compute_nba_best_bets_split

LONDON = ZoneInfo("Europe/London")
EASTERN = ZoneInfo("America/New_York")

EARLY_SEASON_GAMES = 5      # a team with fewer games this season is flagged (ratings lean on last season)
PROP_TRIM = (0.04, 0.97)    # keep prop lines whose Over probability is in this band


def _trim(rows, key="over"):
    lo, hi = PROP_TRIM
    return [r for r in rows if lo <= r[key] <= hi]


def team_table(tg: pd.DataFrame, ratings, names: dict) -> dict:
    season = int(tg["season"].max())
    cur = tg[(tg["season"] == season) & (tg["stype"] == 2)]
    out = {}
    for t in ratings.teams:
        c = cur[cur["team"] == t]
        wins = int((c["pts"] > c["opp_pts"]).sum())
        losses = int((c["pts"] < c["opp_pts"]).sum())
        out[names[t]] = {
            "team_id": int(t),
            "games": int(len(c)),
            "record": f"{wins}-{losses}",
            "ppg": round(float(c["pts"].mean()), 1) if len(c) else None,
            "opp_ppg": round(float(c["opp_pts"].mean()), 1) if len(c) else None,
            "attack": round(ratings.attack[t], 2),
            "defence": round(ratings.defence[t], 2),
            "rating": round(ratings.attack[t] - ratings.defence[t], 2),   # points better than average per game
        }
    return out


def build_fixture(row, ratings, tg, forms, roster_by_team, names) -> dict:
    home_id, away_id = int(row["home_id"]), int(row["away_id"])
    start = datetime.strptime(row["start_uk"], "%Y-%m-%d %H:%M").replace(tzinfo=LONDON)
    game_day = start.astimezone(EASTERN).date()
    hb2b, ab2b = S.back_to_back(tg, home_id, game_day), S.back_to_back(tg, away_id, game_day)
    pred = M.predict_game(ratings, home_id, away_id, hb2b, ab2b)

    home, away = row["home"], row["away"]
    card = {
        "game_id": str(row["game_id"]),
        "home_team": home, "away_team": away,
        "home_id": home_id, "away_id": away_id,
        "date": row["date"], "time": row["time"],
        "back_to_back": {"home": bool(hb2b), "away": bool(ab2b)},
        "expected": {k: round(v, 1) for k, v in pred.items()},
        "model": {"margin_sd": M.MARGIN_SD, "total_sd": M.TOTAL_SD, "team_sd": M.TEAM_POINTS_SD},
        "predictions": {
            "moneyline": M.moneyline(pred["margin"]),
            "spread": M.spread_table(pred["margin"]),
            "total": _trim(M.total_table(pred["total"], M.TOTAL_SD, 22)),
            "team_totals": {
                "home": _trim(M.total_table(pred["home_pts"], M.TEAM_POINTS_SD, 20)),
                "away": _trim(M.total_table(pred["away_pts"], M.TEAM_POINTS_SD, 20)),
            },
        },
    }

    props = []
    for side, tid, tname in (("home", home_id, home), ("away", away_id, away)):
        for aid in roster_by_team.get(tid, []):
            f = forms.get(aid)
            if not f:
                continue
            lines, exp = {}, {}
            for stat in S.PROP_STATS:
                mean, sd = M.prop_dist(stat, f[stat]["mean"], f[stat]["sd"], f["stale_days"])
                exp[stat] = round(mean, 1)
                rows = _trim(M.prop_table(stat, mean, sd))
                if rows:
                    lines[stat] = rows
            props.append({"player": f["player"], "team": tname, "side": side,
                          "mins": f["mins"], "games": f["games"],
                          "expected": exp, "lines": lines})
    props.sort(key=lambda p: -p["mins"])
    card["props"] = props

    hg, ag = ratings.games_played.get(home_id, 0), ratings.games_played.get(away_id, 0)
    card["home_form_sample"], card["away_form_sample"] = hg, ag
    if hg < EARLY_SEASON_GAMES or ag < EARLY_SEASON_GAMES:
        card["early_season"] = True
        card["note"] = (f"Early season ({hg} home team / {ag} away team games played) - ratings still lean "
                        f"on last season, so rosters that changed over the summer can be mis-rated.")
    return card


def recent_results(tg: pd.DataFrame, pg: pd.DataFrame, names: dict, days: int = 2) -> list[dict]:
    if tg.empty:
        return []
    last = tg["date"].max()
    recent = tg[tg["date"] >= last - pd.Timedelta(days=days - 1)]
    out = []
    for gid, g in recent.groupby("game_id"):
        h, a = g[g["home"] == 1], g[g["home"] == 0]
        if h.empty or a.empty:
            continue
        h, a = h.iloc[0], a.iloc[0]
        gp = pg[pg["game_id"] == gid]

        def top(team_id):
            t = gp[gp["team_id"] == team_id]
            if t.empty:
                return None
            r = t.sort_values("points", ascending=False).iloc[0]
            return {"player": r["player"], "points": int(r["points"]), "rebounds": int(r["rebounds"]),
                    "assists": int(r["assists"])}
        out.append({"game_id": str(gid), "date": h["date"].strftime("%d/%m/%Y"),
                    "home_team": names[h["team"]], "away_team": names[a["team"]],
                    "home_score": int(h["pts"]), "away_score": int(a["pts"]),
                    "top_scorers": {"home": top(h["team"]), "away": top(a["team"])}})
    return sorted(out, key=lambda r: (r["date"][6:] + r["date"][3:5] + r["date"][:2], r["game_id"]), reverse=True)


def build_nba_statpack(data_dir: Path, now=None) -> dict:
    now = now or datetime.now(LONDON)
    tg_path = data_dir / "nba_team_games.csv"
    if not tg_path.exists():
        return {"upcoming_fixtures": [], "recent_results": [], "team_form": {}, "best_bets": [], "shadow_picks": []}
    tg = S.load_team_games(tg_path)
    pg = pd.read_csv(data_dir / "nba_player_games.csv", dtype={"game_id": str})
    names = pd.read_csv(tg_path).drop_duplicates("team_id").set_index("team_id")["team"].to_dict()
    up = pd.read_csv(data_dir / "nba_upcoming.csv", dtype={"game_id": str}) if (data_dir / "nba_upcoming.csv").exists() else pd.DataFrame()
    roster = pd.read_csv(data_dir / "nba_roster.csv") if (data_dir / "nba_roster.csv").exists() else pd.DataFrame()
    print(f"[debug] {len(tg)} team-game rows, {len(pg)} player-game rows, {len(up)} upcoming game(s), {len(roster)} roster rows")

    asof = pd.Timestamp(now.date()) + pd.Timedelta(days=1)       # every game dated up to today
    ratings = S.fit_ratings(tg, asof)
    forms = S.player_form(pg, asof)
    # Only the CURRENT roster counts: a player who moved in the summer is a prop for his new team.
    roster_by_team = roster.groupby("team_id")["athlete_id"].apply(list).to_dict() if len(roster) else {}
    print(f"[debug] ratings for {len(ratings.teams)} teams; prop form for {len(forms)} players")

    fixtures = []
    for _, row in up.iterrows():
        if int(row["home_id"]) not in ratings.attack or int(row["away_id"]) not in ratings.attack:
            print(f"[debug] skipping {row['away']} @ {row['home']}: no rating yet")
            continue
        fixtures.append(build_fixture(row, ratings, tg, forms, roster_by_team, names))
    teams = team_table(tg, ratings, names)
    pack = {
        "generated": now.strftime("%Y-%m-%d %H:%M"),
        "shadow_mode": SHADOW_MODE,
        "league_averages": {"points_per_team": round(ratings.mu, 1), "home_edge": round(ratings.home_edge, 2),
                            "back_to_back_penalty": round(ratings.b2b_self, 2)},
        "team_form": teams,
        "upcoming_fixtures": fixtures,
        "recent_results": recent_results(tg, pg, names),
    }
    pack["best_bets"], pack["shadow_picks"] = compute_nba_best_bets_split(pack, now=now)
    print(f"[debug] {len(fixtures)} fixture(s); {len(pack['best_bets'])} NBA best bet(s) shown, "
          f"{len(pack['shadow_picks'])} shadow pick(s)")
    return pack


if __name__ == "__main__":
    root = Path(__file__).parent
    pack = build_nba_statpack(root / "data")
    with open(root / "nba_statpack.json", "w") as f:
        json.dump(pack, f, indent=1, default=str)
    with open(root / "nba_statpack_data.js", "w") as f:
        f.write("const NBA_STATPACK_DATA = ")
        json.dump(pack, f, default=str, separators=(",", ":"))
        f.write(";\n")
    print("NBA stat pack written")
