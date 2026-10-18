"""
nba_stats.py - turns NBA box-score history into team ratings and player form.

TEAM RATINGS come from a time-decayed ridge regression over every
team-game:

    points_scored = league_mean + attack[team] + defence[opponent]
                    + home_edge * (+1 home / -1 away) + back-to-back effects

attack > 0 = scores more than average; defence > 0 = concedes more than
average (so lower is better). Older games count for less (half-life
below) and a ridge penalty pulls every team towards average, which keeps
early-season ratings sensible: with few games this season, last season's
(decayed) games carry the estimate. The half-life and ridge were chosen by
walk-forward testing on a full past season against closing lines.

PLAYER FORM is a decayed average of a player's last games (minutes,
points, rebounds, assists, threes) - see player_form().

Needs numpy + pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HALF_LIFE_DAYS = 60.0   # a game this many days old counts half as much
RIDGE = 4.0             # pull towards league average (per team parameter)

PROP_STATS = ("points", "rebounds", "assists", "threes")
PROP_GAMES = 15          # most recent games used
PROP_DECAY = 0.9         # weight multiplier per game further back
PROP_MIN_GAMES = 8       # fewer recorded games than this: no prop
PROP_MIN_MINUTES = 20.0  # only rotation players (decayed average minutes)
PROP_MAX_AGE_DAYS = 330  # ignore a player whose last game is older than this
LONG_FORM_GAMES = 40     # longer window for the "form40" blend
LONG_FORM_DECAY = 0.97
CONTEXT_HALF_LIFE = 45.0  # days: how fast opposition/pace numbers fade
CONTEXT_STATS = ("pts", "fga", "fta", "poss", "reb", "ast", "tpa", "tpm", "miss", "oreb")


def load_team_games(path) -> pd.DataFrame:
    g = pd.read_csv(path, dtype={"game_id": str})
    g["date"] = pd.to_datetime(g["date"])
    g = g.rename(columns={"team": "team_name"}).rename(columns={"team_id": "team", "opp_id": "opp", "season_type": "stype"})
    g = g.sort_values(["team", "date", "game_id"]).copy()
    g["rest"] = g.groupby("team")["date"].diff().dt.days
    g["b2b"] = (g["rest"] == 1).astype(int)
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)


class Ratings:
    def __init__(self, teams, mu, attack, defence, home_edge, b2b_self, b2b_opp, asof, games_played):
        self.teams, self.mu, self.attack, self.defence = teams, mu, attack, defence
        self.home_edge, self.b2b_self, self.b2b_opp, self.asof = home_edge, b2b_self, b2b_opp, asof
        self.games_played = games_played   # team -> games this season

    def expected_points(self, team: int, opp: int, home: int, team_b2b: int = 0, opp_b2b: int = 0) -> float:
        a = self.attack.get(team, 0.0)
        d = self.defence.get(opp, 0.0)
        side = 1 if home else -1
        return (self.mu + a + d + self.home_edge * side
                + self.b2b_self * team_b2b + self.b2b_opp * opp_b2b)


def fit_ratings(games: pd.DataFrame, asof, half_life: float = HALF_LIFE_DAYS, ridge: float = RIDGE) -> Ratings:
    """Fit on every game strictly before `asof`."""
    asof = pd.Timestamp(asof)
    g = games[games["date"] < asof]
    teams = sorted(set(g["team"]) | set(g["opp"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    age = (asof - g["date"]).dt.days.to_numpy(dtype=float)
    w = 0.5 ** (age / half_life)
    opp_b2b = g[["game_id", "team", "b2b"]].rename(columns={"team": "opp", "b2b": "opp_b2b"})
    rest_opp = g.merge(opp_b2b, on=["game_id", "opp"], how="left")["opp_b2b"].fillna(0).to_numpy()
    m = len(g)
    X = np.zeros((m, 2 * n + 3))
    rows = np.arange(m)
    X[rows, g["team"].map(idx).to_numpy()] = 1.0
    X[rows, n + g["opp"].map(idx).to_numpy()] = 1.0
    X[:, 2 * n] = np.where(g["home"].to_numpy() == 1, 1.0, -1.0)
    X[:, 2 * n + 1] = g["b2b"].to_numpy()
    X[:, 2 * n + 2] = rest_opp
    y = g["pts"].to_numpy(dtype=float)
    mu = float(np.average(y, weights=w))
    L = np.zeros(2 * n + 3)
    L[: 2 * n] = ridge
    L[2 * n:] = 1.0
    A = (X * w[:, None]).T @ X + np.diag(L)
    beta = np.linalg.solve(A, (X * w[:, None]).T @ (y - mu))
    season = int(g["season"].max()) if m else 0
    cur = g[g["season"] == season]
    played = cur.groupby("team").size().to_dict()
    return Ratings(teams, mu,
                   {t: float(beta[idx[t]]) for t in teams},
                   {t: float(beta[n + idx[t]]) for t in teams},
                   float(beta[2 * n]), float(beta[2 * n + 1]), float(beta[2 * n + 2]), asof, played)


def back_to_back(games: pd.DataFrame, team: int, game_day) -> int:
    """1 if `team` played the day before `game_day` (a calendar date)."""
    prev = pd.Timestamp(game_day) - pd.Timedelta(days=1)
    return int(((games["team"] == team) & (games["date"] == prev)).any())


def player_form(player_games: pd.DataFrame, asof) -> dict[int, dict]:
    """athlete_id -> decayed per-game averages over the last PROP_GAMES games.

    Regular-season games only (playoff minutes and usage are inflated).
    Skips anyone with too few games, too few minutes, or a last game too
    long ago. `sd` is the decayed standard deviation of each stat (the
    model widens it - see nba_model.PROP_SD_SCALE).
    """
    asof = pd.Timestamp(asof)
    pg = player_games.copy()
    pg["date"] = pd.to_datetime(pg["date"])
    pg = pg[(pg["date"] < asof) & (pg["season_type"] == 2)].sort_values(["athlete_id", "date", "game_id"])
    out = {}
    for aid, g in pg.groupby("athlete_id"):
        g = g.tail(PROP_GAMES)
        if len(g) < PROP_MIN_GAMES:
            continue
        if (asof - g["date"].iloc[-1]).days > PROP_MAX_AGE_DAYS:
            continue
        w = PROP_DECAY ** np.arange(len(g))[::-1]
        mins = float(np.average(g["mins"], weights=w))
        if mins < PROP_MIN_MINUTES:
            continue
        long = pg[pg["athlete_id"] == aid].tail(LONG_FORM_GAMES)
        lw = LONG_FORM_DECAY ** np.arange(len(long))[::-1]
        mins40 = float(np.average(long["mins"], weights=lw))
        rec = {"player": g["player"].iloc[-1], "team_id": int(g["team_id"].iloc[-1]),
               "mins5": float(g["mins"].tail(5).mean()), "mins40": mins40,
               "games": int(len(g)), "mins": round(mins, 1),
               "stale_days": int((asof - g["date"].iloc[-1]).days)}
        for s in PROP_STATS:
            mean = float(np.average(g[s], weights=w))
            sd = float(np.sqrt(np.average((g[s] - mean) ** 2, weights=w)))
            rec[s] = {"mean": mean, "sd": sd,
                      "mean40": float(np.average(long[s], weights=lw))}
        out[int(aid)] = rec
    return out


def team_context(games: pd.DataFrame, asof) -> dict | None:
    """Per-team scoring/shooting profile for the opposition adjustment of
    player props. For each team: what it does itself ("off") and what
    opponents do AGAINST it ("def"), as decayed per-game averages relative
    to the league (0.05 = 5% above average), plus its recent scoring margin.

    Needs the team box-score columns fetch_nba.py writes (fga, fta, tov,
    oreb, reb, ast, tpa, tpm, fgm); returns None if they're missing, in
    which case props fall back to form only.
    """
    need = {"fgm", "fga", "fta", "tov", "oreb", "reb", "ast", "tpa", "tpm"}
    if not need.issubset(games.columns):
        return None
    asof = pd.Timestamp(asof)
    g = games[(games["date"] < asof) & (games["stype"] == 2)].dropna(subset=list(need)).copy()
    if len(g) < 200:
        return None
    g["poss"] = g["fga"] + 0.44 * g["fta"] + g["tov"] - g["oreb"]
    g["miss"] = g["fga"] - g["fgm"]
    cols = list(CONTEXT_STATS)
    opp = g[["game_id", "team"] + cols].rename(columns={"team": "opp", **{c: "o_" + c for c in cols}})
    g = g.merge(opp, on=["game_id", "opp"], how="inner")
    w = 0.5 ** ((asof - g["date"]).dt.days.to_numpy(dtype=float) / CONTEXT_HALF_LIFE)
    g["w"] = w
    wsum = g.groupby("team")["w"].sum()
    out = {"off": {}, "def": {}, "net": {}}
    lg = {}
    for c in cols:
        off = (g[c] * g["w"]).groupby(g["team"]).sum() / wsum
        dfn = (g["o_" + c] * g["w"]).groupby(g["team"]).sum() / wsum
        lg[c] = float((g[c] * g["w"]).sum() / g["w"].sum())
        out["off"][c] = {int(t): float(v / lg[c] - 1) for t, v in off.items()}
        out["def"][c] = {int(t): float(v / lg[c] - 1) for t, v in dfn.items()}
    net = ((g["pts"] - g["o_pts"]) * g["w"]).groupby(g["team"]).sum() / wsum
    out["net"] = {int(t): float(v) for t, v in net.items()}
    return out
