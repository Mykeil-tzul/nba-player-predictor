"""Build a CourtIQ snapshot from the nba_data regular-season archive."""
import argparse
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.data import PLAYERS, SEASONS, SNAPSHOT, STATS, atomic_write, nba_today, normalize

BASE = "https://basketball-data.billmill.org/nba/stats"
BOX_COLUMNS = [
    "player_id", "game_id", "game_date", "team_id", "opp_team_id", "home",
    "min", *[stat for stat in STATS if stat != "minutes"],
]


def season_games(frame, season):
    game_ids = frame.game_id.astype(str)
    regular = game_ids.str.fullmatch(f"002{season[2:4]}[0-9]{{5}}")
    selected = frame.loc[regular & frame.player_id.astype(str).isin(map(str, PLAYERS))].copy()
    selected["season"] = season
    location = selected["home"].map({True: " vs. ", False: " @ "})
    selected["matchup"] = (
        selected.team_id.astype(str).map(TEAMS) + location
        + selected.opp_team_id.astype(str).map(TEAMS)
    )
    missing_matchup = selected["matchup"].isna()
    selected.loc[missing_matchup, "matchup"] = (
        selected.loc[missing_matchup, "team_id"].map(TEAMS).fillna("Unknown team") + " · NBA box score"
    )
    return normalize(selected.rename(columns={"min": "minutes"}))


TEAMS = {
    str(1610612737 + i): team for i, team in enumerate(
        "ATL BOS CLE NOP CHI DAL DEN GSW HOU LAC LAL MIA MIL MIN BKN NYK ORL IND PHI PHX POR SAC SAS OKC TOR UTA MEM WAS DET CHA".split()
    )
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, help="Read local files named 2023.parquet through 2027.parquet")
    parser.add_argument("--output", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    today = nba_today()
    frames = []
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
        )))
        for season in SEASONS:
            if (today.year, today.month) < (int(season[:4]), 10):
                continue
            year = int(season[:4]) + 1
            if args.archive_dir:
                source = args.archive_dir / f"{year}.parquet"
            else:
                response = session.get(
                    f"{BASE}/player_game_logs/season={year}/data.parquet", timeout=(5, 25),
                )
                response.raise_for_status()
                source = BytesIO(response.content)
            games = season_games(pd.read_parquet(source, columns=BOX_COLUMNS), season)
            if games.empty or games.game_date.dt.date.gt(today).any():
                raise ValueError(f"Missing or future-dated regular-season games for {season}")
            frames.append(games)
            print(f"{season}: {len(games)} appearances")
    snapshot = normalize(pd.concat(frames, ignore_index=True))
    atomic_write(args.output, snapshot.to_csv(index=False))


if __name__ == "__main__":
    main()
