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


def _ctx(ctx, kind, stat, team):
    return ctx[kind][stat].get(team, 0.0) if ctx else 0.0


def prop_features(ctx, f, side, team_id, opp_id, b2b, stat) -> dict:
    """The model's inputs for one player (see nba_model.PROP_COEFS)."""
    feat = {"home": 1.0 if side == "home" else -1.0, "b2b": float(b2b)}
    if ctx:
        net = ctx["net"]
        feat["absmargin"] = abs(net.get(team_id, 0.0) - net.get(opp_id, 0.0) + (2.0 if side == "home" else -2.0)) / 10.0
        feat["pace"] = (_ctx(ctx, "off", "poss", team_id) + _ctx(ctx, "off", "poss", opp_id)) / 2.0
        for c in ("pts", "fga", "tpa", "tpm", "reb", "ast", "miss"):
            feat["odef_" + c] = _ctx(ctx, "def", c, opp_id)
        for c in ("miss", "ast", "tpa"):
            feat["aoff_" + c] = _ctx(ctx, "off", c, team_id)
        for c in ("oreb", "tpa"):
            feat["ooff_" + c] = _ctx(ctx, "off", c, opp_id)
    m = f[stat]["mean"]
    feat["form40"] = (f[stat]["mean40"] - m) / max(m, 0.5)
    feat["mintrend"] = max(-0.5, min(0.5, f["mins5"] / max(f["mins40"], 1.0) - 1.0))
    return feat


def build_fixture(row, ratings, tg, forms, roster_by_team, names, ctx=None, appearances=None) -> dict:
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
        tid_team = tid
        for aid in roster_by_team.get(tid, []):
            f = forms.get(aid)
            if not f:
                continue
            if appearances is not None and appearances["inactive"].get((int(aid), tid_team)):
                continue   # on the roster but has not played a game this season (injured / not signed to play)
            lines, exp = {}, {}
            tid, oid = (home_id, away_id) if side == "home" else (away_id, home_id)
            form = {}
            for stat in S.PROP_STATS:
                feat = prop_features(ctx, f, side, tid, oid, hb2b if side == "home" else ab2b, stat)
                adj, _ = M.prop_adjust(stat, f[stat]["mean"], feat)
                mean, sd = M.prop_dist(stat, adj, f[stat]["sd"], f["stale_days"])
                exp[stat] = round(mean, 1)
                form[stat] = round(f[stat]["mean"], 1)
                rows = M.prop_ladder(stat, mean, sd)          # the fixed marks: 10/15/20/25+ points etc.
                if rows:
                    lines[stat] = rows
            prop = {"player": f["player"], "team": tname, "side": side,
                    "mins": f["mins"], "games": f["games"],
                    "expected": exp, "form": form, "lines": lines}
            if appearances is not None:
                # games played this season out of the team's games - lets the dashboard flag part-timers
                prop["season_gp"] = appearances["gp"].get(int(aid), 0)
                prop["season_team_gp"] = appearances["team_gp"].get(tid_team, 0)
            props.append(prop)
    props.sort(key=lambda p: -p["mins"])
    card["props"] = props
    if ctx:
        pct = lambda v: round(100 * v, 1)
        card["prop_context"] = {
            side: {"vs": opp_name,
                   "points": pct(_ctx(ctx, "def", "pts", oid)), "shots": pct(_ctx(ctx, "def", "fga", oid)),
                   "rebounds": pct(_ctx(ctx, "def", "reb", oid)), "assists": pct(_ctx(ctx, "def", "ast", oid)),
                   "threes": pct(_ctx(ctx, "def", "tpa", oid)),
                   "pace": pct((_ctx(ctx, "off", "poss", home_id) + _ctx(ctx, "off", "poss", away_id)) / 2)}
            for side, oid, opp_name in (("home", away_id, away), ("away", home_id, home))}

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


# A rostered player who has not appeared in any of his team's games this season is
# almost certainly out (injury, not yet returned), so he gets no props - the form the
# model has for him is last season's. Applies once a team has played this many games.
MIN_TEAM_GAMES_FOR_INACTIVE = 3


def season_appearances(pg: pd.DataFrame, roster: pd.DataFrame, asof) -> dict:
    """Regular-season appearances in the CURRENT season (the latest season in the data)."""
    out = {"gp": {}, "team_gp": {}, "inactive": {}}
    if pg.empty or "season" not in pg.columns:
        return out
    cur = pg[pg["season_type"] == 2]
    if cur.empty:
        return out
    cur = cur[cur["season"] == cur["season"].max()]
    cur = cur[pd.to_datetime(cur["date"]) < asof]
    out["gp"] = cur.groupby("athlete_id")["game_id"].nunique().to_dict()
    out["team_gp"] = cur.groupby("team_id")["game_id"].nunique().to_dict()
    for _, r in roster.iterrows() if len(roster) else []:
        aid, tid = int(r["athlete_id"]), int(r["team_id"])
        if out["team_gp"].get(tid, 0) >= MIN_TEAM_GAMES_FOR_INACTIVE and out["gp"].get(aid, 0) == 0:
            out["inactive"][(aid, tid)] = True
    print(f"[debug] {len(out['inactive'])} rostered player(s) with no appearance this season - no props for them")
    return out


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
    ctx = S.team_context(tg, asof)
    print(f"[debug] opposition context: {'on' if ctx else 'OFF (team box-score columns missing - props use form only)'}")
    # Only the CURRENT roster counts: a player who moved in the summer is a prop for his new team.
    roster_by_team = roster.groupby("team_id")["athlete_id"].apply(list).to_dict() if len(roster) else {}
    print(f"[debug] ratings for {len(ratings.teams)} teams; prop form for {len(forms)} players")

    appearances = season_appearances(pg, roster, asof)

    fixtures = []
    for _, row in up.iterrows():
        if int(row["home_id"]) not in ratings.attack or int(row["away_id"]) not in ratings.attack:
            print(f"[debug] skipping {row['away']} @ {row['home']}: no rating yet")
            continue
        fixtures.append(build_fixture(row, ratings, tg, forms, roster_by_team, names, ctx, appearances))
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
