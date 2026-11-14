"""Shared game schema and local snapshot utilities."""
import json
import unicodedata
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "games.csv"
CACHE = ROOT / "data" / "raw" / "nba"
ARTIFACT = ROOT / "artifacts" / "courtiq.json"
PLAYERS = {
    1629029: "Luka Dončić",
    202681: "Kyrie Irving",
    1628369: "Jayson Tatum",
    1628983: "Shai Gilgeous-Alexander",
    1628973: "Jalen Brunson",
}
SEASONS = tuple(f"{year}-{(year + 1) % 100:02d}" for year in range(2022, 2027))
TARGETS = ["pts", "reb", "ast"]
STATS = [*TARGETS, "minutes", "fga", "fta"]
COLUMNS = ["player_id", "game_id", "game_date", "season", "matchup", *STATS]


def nba_today():
    return datetime.now(ZoneInfo("America/New_York")).date()


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(content)
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def normalize(frame):
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    games = frame[COLUMNS].copy()
    games["game_date"] = pd.to_datetime(games["game_date"], format="ISO8601").dt.normalize()
    games["player_id"] = pd.to_numeric(games["player_id"], errors="raise")
    games["game_id"] = games["game_id"].astype("string").str.strip()
    games[STATS] = games[STATS].apply(pd.to_numeric, errors="raise")
    if games[COLUMNS].isna().any().any() or games["game_id"].eq("").any():
        raise ValueError("Missing game values")
    if not games["player_id"].isin(PLAYERS).all() or not games["season"].isin(SEASONS).all():
        raise ValueError("Unexpected player or season")
    if not np.isfinite(games[STATS].to_numpy(dtype=float)).all() or (games[STATS] < 0).any().any():
        raise ValueError("Invalid box-score numbers")
    # DNP records are neither labels nor prior appearances.
    games = games.loc[games["minutes"] > 0].drop_duplicates()
    if games.duplicated(["player_id", "game_id"]).any() or games.duplicated(["player_id", "game_date"]).any():
        raise ValueError("Conflicting game records or multiple appearances on the same date")
    return games.sort_values(["game_date", "player_id"]).reset_index(drop=True)


def minutes_played(value):
    value = str(value)
    if ":" in value:
        minutes, seconds = value.split(":")
        return float(minutes) + float(seconds) / 60
    return float(value)


def read_snapshot(path):
    return normalize(pd.read_csv(path, dtype={"game_id": "string", "season": "string"}, keep_default_na=False))


def legacy_games():
    """Explicit historical demo using the existing repo's CSV."""
    def plain(name):
        return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()

    frame = pd.read_csv(ROOT / "database_24_25.csv")
    player_ids = {plain(name): player_id for player_id, name in PLAYERS.items()}
    frame["player_id"] = frame["Player"].map(lambda name: player_ids.get(plain(name)))
    frame = frame.dropna(subset=["player_id", "MP", "PTS", "TRB", "AST", "FGA", "FTA"]).copy()
    frame["player_id"] = frame["player_id"].astype(int)
    frame["game_id"] = frame["Data"].astype(str)
    frame["season"] = "2024-25"
    frame["matchup"] = frame["Tm"] + " / " + frame["Opp"]
    frame["MP"] = frame["MP"].map(minutes_played)
    return normalize(frame.rename(columns={
        "Data": "game_date", "MP": "minutes", "PTS": "pts", "TRB": "reb",
        "AST": "ast", "FGA": "fga", "FTA": "fta",
    }))
