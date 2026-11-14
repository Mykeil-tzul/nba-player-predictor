"""Fetch NBA Stats regular-season player logs and publish a local snapshot."""
import argparse
import logging
import random
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from src.data import (
    CACHE, COLUMNS, PLAYERS, SEASONS, SNAPSHOT, atomic_write, minutes_played,
    nba_today, normalize, read_snapshot,
)

URL = "https://stats.nba.com/stats/playergamelog"
TIMEOUT = (5, 15)
ATTEMPTS = 3
RETRYABLE = {408, 425, 429, 500, 502, 503, 504}
HEADERS = {
    "Accept": "application/json",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": "Mozilla/5.0 (compatible; CourtIQ/1.0)",
}
NBA_COLUMNS = {
    "Player_ID": "player_id", "Game_ID": "game_id", "GAME_DATE": "game_date",
    "MATCHUP": "matchup", "PTS": "pts", "REB": "reb", "AST": "ast",
    "MIN": "minutes", "FGA": "fga", "FTA": "fta",
}


def decode(payload, player_id, season):
    result = next(item for item in payload["resultSets"] if item["name"] == "PlayerGameLog")
    games = pd.DataFrame(result["rowSet"], columns=result["headers"]).rename(columns=NBA_COLUMNS)
    games["game_date"] = pd.to_datetime(games["game_date"], format="%b %d, %Y").dt.strftime("%Y-%m-%d")
    games["minutes"] = games["minutes"].map(minutes_played)
    games["season"] = season
    if not pd.to_numeric(games["player_id"]).eq(player_id).all():
        raise ValueError("NBA Stats returned a different player")
    return normalize(games)


def download(session, player_id, season):
    for attempt in range(ATTEMPTS):
        time.sleep(1)
        response = None
        try:
            response = session.get(URL, params={
                "PlayerID": player_id, "Season": season, "SeasonType": "Regular Season",
                "LeagueID": "00", "DateFrom": "", "DateTo": "",
            }, timeout=TIMEOUT)
            response.raise_for_status()
            return decode(response.json(), player_id, season)
        except requests.HTTPError as error:
            if response.status_code not in RETRYABLE or attempt == ATTEMPTS - 1:
                raise
            if response.status_code == 429:
                logging.warning("NBA Stats is rate limiting requests")
            delay = response.headers.get("Retry-After", "")
            if delay.isdigit():
                if int(delay) > 60:
                    raise RuntimeError(f"NBA Stats requested a {delay}s pause; retry later") from error
                time.sleep(int(delay))
                continue
        except (requests.RequestException, ValueError, KeyError, TypeError, StopIteration):
            if attempt == ATTEMPTS - 1:
                raise
        time.sleep(2 ** attempt + random.random())


def collect(session, cache, output, *, refresh=False, offline=False, today=None):
    today = today or nba_today()
    frames = []
    failures = []
    stale = []
    consecutive_failures = 0
    for season in SEASONS:
        year = int(season[:4])
        if today < date(year, 10, 1):
            continue
        closed = today >= date(year + 1, 7, 1)
        for player_id, name in PLAYERS.items():
            path = cache / f"{player_id}_{season}.csv"
            saved = None
            if path.exists():
                try:
                    saved = read_snapshot(path)
                    if not saved["player_id"].eq(player_id).all() or not saved["season"].eq(season).all():
                        raise ValueError("Cache content does not match its filename")
                except ValueError as error:
                    logging.warning("Ignoring invalid cache for %s %s: %s", name, season, error)
                    saved = None
            # Old-season caches are permanent only if fetched after the season ended.
            now = time.time()
            final_cutoff = pd.Timestamp(date(year + 1, 7, 1), tz="America/New_York").timestamp()
            fresh = saved is not None and (
                (closed and path.stat().st_mtime >= final_cutoff)
                or (not closed and now - path.stat().st_mtime < 12 * 3600)
            )
            if saved is not None and (offline or (fresh and not refresh)):
                frames.append(saved)
                if not fresh:
                    stale.append(f"{name} {season}")
                continue
            try:
                if offline or consecutive_failures >= 3:
                    raise RuntimeError("Offline cache miss" if offline else "API paused after three failed requests")
                games = download(session, player_id, season)
                if games.game_date.dt.date.gt(today).any():
                    raise ValueError("NBA Stats returned a future appearance")
                if saved is not None and not saved.game_id.isin(games.game_id).all():
                    raise ValueError("Response is missing previously cached games. Inspect the response before replacing it.")
                atomic_write(path, games.to_csv(index=False))
                frames.append(games)
                consecutive_failures = 0
            except (requests.RequestException, ValueError, KeyError, TypeError, StopIteration, RuntimeError) as error:
                consecutive_failures += 1
                label = f"{name} {season}"
                logging.warning("%s: %s", label, error)
                if saved is not None:
                    frames.append(saved)
                    stale.append(label)
                else:
                    failures.append(label)
    if failures:
        raise RuntimeError("Snapshot kept as-is. Missing player-season caches: " + ", ".join(failures))
    if stale and not offline:
        raise RuntimeError(
            "Snapshot kept as-is. Refresh failed for: " + ", ".join(stale)
            + ". Use --offline to explicitly assemble from cached logs."
        )
    if not frames or not any(len(frame) for frame in frames):
        raise RuntimeError("No completed appearances available")
    games = normalize(pd.concat([frame for frame in frames if len(frame)], ignore_index=True))
    if games.game_date.dt.date.gt(today).any():
        raise ValueError("Snapshot has future game dates")
    atomic_write(output, games.to_csv(index=False))
    if stale:
        logging.warning("Used stale cache for: %s", ", ".join(stale))
    return games


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=SNAPSHOT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--refresh", action="store_true", help="Refresh historical caches too")
    modes.add_argument("--offline", action="store_true", help="Assemble from existing caches, including stale ones")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    with requests.Session() as session:
        session.headers.update(HEADERS)
        try:
            games = collect(session, args.cache, args.output, refresh=args.refresh, offline=args.offline)
        except (RuntimeError, ValueError, OSError) as error:
            parser.exit(1, f"{error}\n")
    print(f"Wrote {len(games)} appearances to {args.output.name}")


if __name__ == "__main__":
    main()
