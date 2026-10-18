"""
fetch_nba.py

Pulls NBA data from the sportsdataverse "hoopR-nba-data" repository on
GitHub (ESPN-derived box scores, schedule and rosters, published as
parquet files). Needs pandas + pyarrow (the workflow installs them).

Four files are written to data/:

  nba_team_games.csv   one row per team per game played - points for and
                        against, home/away, date. Last season + this
                        season. Feeds the team-ratings model AND settles
                        spread / total / moneyline picks.
  nba_player_games.csv  one row per player per game they actually played
                        (minutes, points, rebounds, assists, threes).
                        Feeds the player-prop model AND settles props.
  nba_upcoming.csv      scheduled games in the next few days, with the
                        tip-off time in UK time.
  nba_roster.csv        who is on each team THIS season (so a player who
                        moved in the summer is matched to his new team).

The season is named by the year it ENDS in (2026 = the 2025-26 season;
2027 = 2026-27, which starts on 20 Oct 2026). Files for a season that
hasn't started yet simply don't exist - that's handled, not an error.
"""

import io
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd

BASE = "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/nba"
UA = {"User-Agent": "Mozilla/5.0 (compatible; StatPackBot/1.0; personal use)"}
LONDON = ZoneInfo("Europe/London")

UPCOMING_DAYS = 4      # schedule rows kept: today + this many days ahead
LOOKBACK_START = "2025-09-01"   # team/player history starts here (the 2025-26 season onwards)


def current_season(today=None) -> int:
    """Season-ending year: Oct-Dec belong to next year's label."""
    today = today or datetime.now(LONDON).date()
    return today.year + 1 if today.month >= 9 else today.year


def read_parquet(url: str, retries: int = 3):
    """DataFrame from a parquet URL, or None if the file doesn't exist (404)."""
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers=UA), timeout=60) as r:
                return pd.read_parquet(io.BytesIO(r.read()))
        except HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except (URLError, TimeoutError, OSError) as e:
            err = e
        print(f"[warn] {url}: {err} (attempt {attempt + 1}/{retries})", file=sys.stderr)
        time.sleep(3 * (attempt + 1))
    return None


def load_seasons(kind: str, folder: str, seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        df = read_parquet(f"{BASE}/{folder}/parquet/{kind}_{s}.parquet")
        if df is None:
            print(f"[debug] {kind} {s}: not available (yet)")
            continue
        print(f"[debug] {kind} {s}: {len(df)} rows")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_team_games(tb: pd.DataFrame) -> pd.DataFrame:
    tb = tb.drop_duplicates(["game_id", "team_id"])
    out = pd.DataFrame({
        "game_id": tb["game_id"].astype(str),
        "date": tb["game_date"].astype(str),
        "season": tb["season"].astype(int),
        "season_type": tb["season_type"].astype(int),
        "team_id": tb["team_id"].astype(int),
        "team": tb["team_display_name"],
        "opp_id": tb["opponent_team_id"].astype(int),
        "home": (tb["team_home_away"] == "home").astype(int),
        "pts": tb["team_score"],
        "opp_pts": tb["opponent_team_score"],
    }).dropna(subset=["pts", "opp_pts"])
    out["pts"] = out["pts"].astype(int)
    out["opp_pts"] = out["opp_pts"].astype(int)
    out = out[out["date"] >= LOOKBACK_START]
    return out.sort_values(["date", "game_id", "home"]).reset_index(drop=True)


def build_player_games(pb: pd.DataFrame) -> pd.DataFrame:
    pb = pb.drop_duplicates(["game_id", "athlete_id"]).copy()
    pb["mins"] = pd.to_numeric(pb["minutes"], errors="coerce")
    pb = pb[(pb["mins"] > 0) & pb["athlete_id"].notna()]
    out = pd.DataFrame({
        "game_id": pb["game_id"].astype(str),
        "date": pb["game_date"].astype(str),
        "season": pb["season"].astype(int),
        "season_type": pb["season_type"].astype(int),
        "athlete_id": pb["athlete_id"].astype(int),
        "player": pb["athlete_display_name"],
        "team_id": pb["team_id"].astype(int),
        "team": pb["team_display_name"],
        "mins": pb["mins"].round(1),
        "points": pb["points"].fillna(0).astype(int),
        "rebounds": pb["rebounds"].fillna(0).astype(int),
        "assists": pb["assists"].fillna(0).astype(int),
        "threes": pb["three_point_field_goals_made"].fillna(0).astype(int),
    })
    out = out[out["date"] >= LOOKBACK_START]
    return out.sort_values(["date", "game_id", "athlete_id"]).reset_index(drop=True)


def build_upcoming(sched: pd.DataFrame, now: datetime) -> pd.DataFrame:
    if sched.empty:
        return pd.DataFrame(columns=["game_id", "date", "time", "start_uk", "home_id", "home", "away_id", "away", "season_type"])
    s = sched.drop_duplicates("game_id").copy()
    s = s[s["status_type_description"].isin(["Scheduled", "Postponed"]) & s["season_type"].isin([2, 3, 5])]
    start = pd.to_datetime(s["game_date_time"], utc=True).dt.tz_convert(LONDON)
    s = s.assign(start=start)
    # Keep games that haven't ended (started in the last ~4h still shown as
    # scheduled until the feed flips them) up to UPCOMING_DAYS ahead.
    s = s[(s["start"] >= now - timedelta(hours=6)) & (s["start"] <= now + timedelta(days=UPCOMING_DAYS))]
    return pd.DataFrame({
        "game_id": s["game_id"].astype(str),
        "date": s["start"].dt.strftime("%d/%m/%Y"),
        "time": s["start"].dt.strftime("%H:%M"),
        "start_uk": s["start"].dt.strftime("%Y-%m-%d %H:%M"),
        "home_id": s["home_id"].astype(int),
        "home": s["home_display_name"],
        "away_id": s["away_id"].astype(int),
        "away": s["away_display_name"],
        "season_type": s["season_type"].astype(int),
    }).sort_values(["start_uk", "game_id"]).reset_index(drop=True)


def main():
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    now = datetime.now(LONDON)
    season = current_season(now.date())
    seasons = [season - 1, season]

    tb = load_seasons("team_box", "team_box", seasons)
    pb = load_seasons("player_box", "player_box", seasons)
    sched = load_seasons("nba_schedule", "schedules", [season])
    roster = load_seasons("rosters", "rosters", [season])

    if tb.empty or pb.empty:
        sys.exit("No NBA box-score data could be downloaded - leaving the existing files untouched.")

    teams_df, players_df = build_team_games(tb), build_player_games(pb)
    if not roster.empty:
        # Drop exhibition games (All-Star teams etc.): keep only the 30 real franchises.
        real = set(roster["team_id"].astype(int))
        teams_df = teams_df[teams_df["team_id"].isin(real) & teams_df["opp_id"].isin(real)]
        players_df = players_df[players_df["team_id"].isin(real)]
    teams_df.to_csv(data_dir / "nba_team_games.csv", index=False)
    players_df.to_csv(data_dir / "nba_player_games.csv", index=False)
    up = build_upcoming(sched, now)
    up.to_csv(data_dir / "nba_upcoming.csv", index=False)
    if not roster.empty:
        r = roster.drop_duplicates("athlete_id")
        pd.DataFrame({
            "athlete_id": r["athlete_id"].astype(int),
            "player": r["display_name"],
            "team_id": r["team_id"].astype(int),
            "team": r["team_display_name"],
        }).to_csv(data_dir / "nba_roster.csv", index=False)
    print(f"[debug] {len(up)} upcoming game(s) in the next {UPCOMING_DAYS} days")


if __name__ == "__main__":
    main()
