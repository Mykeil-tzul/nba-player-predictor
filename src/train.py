"""Train and validate CourtIQ from a local NBA snapshot or the historical demo."""
import argparse
from pathlib import Path

from src.data import ARTIFACT, PLAYERS, SNAPSHOT, atomic_json, legacy_games, read_snapshot
from src.model import fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--data", type=Path)
    inputs.add_argument("--demo", action="store_true")
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    output = args.artifact or (ARTIFACT.with_name("demo.json") if args.demo else ARTIFACT)
    try:
        games = legacy_games() if args.demo else read_snapshot(args.data or SNAPSHOT)
        artifact = fit(games, "demo" if args.demo else "nba")
        atomic_json(output, artifact)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")

    print(f"Saved {output.name} using {len(games)} appearances through {artifact['data_through']}.")
    for player_id, name in PLAYERS.items():
        entry = artifact["players"].get(str(player_id))
        if entry is None:
            print(f"{name}: {artifact['skipped'][str(player_id)]}")
            continue
        print(f"\n{name} | {entry['validation_count']} validation appearances "
              f"({entry['validation_start']} to {entry['validation_end']})")
        for baseline, scores in entry["mae"].items():
            print(f"  {baseline:16} " + "  ".join(f"{stat} {mae:.2f}" for stat, mae in scores.items()))


if __name__ == "__main__":
    main()
