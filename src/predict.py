"""Print a next-appearance projection from a trained CourtIQ artifact."""
import argparse
import json
from datetime import date
from pathlib import Path

from src.data import ARTIFACT, PLAYERS, SNAPSHOT, legacy_games, nba_today, read_snapshot
from src.model import predict, verify_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", type=int, required=True, choices=PLAYERS)
    parser.add_argument("--date", type=date.fromisoformat, required=True)
    parser.add_argument("--availability", choices=("expected", "unknown", "out"), default="unknown")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    path = args.artifact or (ARTIFACT.with_name("demo.json") if args.demo else ARTIFACT)
    try:
        if not args.demo and args.date < nba_today():
            raise ValueError("Enter a current or future target date")
        games = legacy_games() if args.demo else read_snapshot(args.data or SNAPSHOT)
        artifact = json.loads(path.read_text())
        verify_artifact(artifact, games)
        result = predict(artifact, games, args.player, args.date, args.availability, demo=args.demo)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps({
        "player": PLAYERS[args.player], "date": args.date.isoformat(), "demo": args.demo,
        "method": artifact["players"][str(args.player)]["method"],
        "assumption": "Player appears in the game", "data_through": artifact["data_through"],
        "projection": result,
    }, indent=2))


if __name__ == "__main__":
    main()
