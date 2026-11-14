"""CourtIQ's local-data Streamlit dashboard."""
import json
from datetime import timedelta

import pandas as pd
import streamlit as st

from src.data import ARTIFACT, PLAYERS, SNAPSHOT, legacy_games, nba_today, read_snapshot
from src.model import fit, predict, verify_artifact

st.set_page_config(page_title="CourtIQ", page_icon="🏀", layout="wide")


@st.cache_data
def demo_data():
    games = legacy_games()
    return games, fit(games, "demo")


@st.cache_data
def nba_data(snapshot_version, artifact_version):
    games = read_snapshot(SNAPSHOT)
    artifact = json.loads(ARTIFACT.read_text())
    verify_artifact(artifact, games)
    return games, artifact


st.title("CourtIQ")
st.caption("Next-appearance projections · NBA regular season")
with st.sidebar:
    mode = st.radio(
        "Data source", ("NBA snapshot", "Historical demo"),
        index=0 if SNAPSHOT.exists() else 1,
    )
    name = st.selectbox("Player", list(PLAYERS.values()))
    player_id = next(key for key, label in PLAYERS.items() if label == name)
    availability = st.radio(
        "Availability", ("unknown", "expected", "out"),
        format_func={"unknown": "Unconfirmed", "expected": "Expected to play", "out": "Out / did not play"}.__getitem__,
        key=f"availability-{mode}-{player_id}",
    )

demo = mode == "Historical demo"
try:
    if demo:
        games, artifact = demo_data()
        st.info("Historical demo · Partial 2024–25 data bundled with the repository.")
    else:
        games, artifact = nba_data(SNAPSHOT.stat().st_mtime_ns, ARTIFACT.stat().st_mtime_ns)
except (OSError, ValueError, KeyError, TypeError) as error:
    st.error(f"Could not load game data and model. Run `python -m src.fetch_data` and `python -m src.train`. {error}")
    st.stop()

player = games.loc[games.player_id.eq(player_id)].copy()
entry = artifact["players"].get(str(player_id))
st.header(name)
if player.empty:
    st.warning("No completed appearances in this snapshot.")
    st.stop()

last = player.game_date.max().date()
data_through = games.game_date.max().date()
first_target = last + timedelta(days=1)
if not demo:
    first_target = max(nba_today(), last + timedelta(days=1))
game_date = st.sidebar.date_input(
    "Target game date", value=first_target, min_value=first_target,
    max_value=first_target + timedelta(days=60),
)
st.caption(f"Last recorded appearance {last:%b %d, %Y} · Snapshot through {data_through:%b %d, %Y}")
if demo:
    st.caption("Choose a hypothetical game date after this player's latest recorded appearance.")
    if entry and entry["model"] is None:
        st.info("This player's demo history supports a five-game average. Ridge requires more appearances.")
else:
    st.caption("Confirm the game date and availability independently. CourtIQ uses local game logs.")
if entry is None:
    st.warning(artifact["skipped"].get(str(player_id), "Insufficient history"))
elif availability == "unknown":
    st.info("Select Expected to play to see a projection conditional on an appearance.")
else:
    try:
        projection = predict(artifact, games, player_id, game_date, availability, demo=demo)
    except ValueError as error:
        st.warning(str(error))
    else:
        st.subheader("Projected box score")
        for column, (stat, value) in zip(st.columns(4), projection.items()):
            column.metric(stat, f"{value:.1f}")
        st.caption(f"{entry['method']} · Assumes the player appears. PRA sums the three unrounded projections.")

recent, validation = st.tabs(["Recent appearances", "Validation"])
with recent:
    player["pra"] = player[["pts", "reb", "ast"]].sum(axis=1)
    display = player.tail(15).rename(columns={
        "game_date": "Date", "matchup": "Matchup", "pts": "PTS",
        "reb": "REB", "ast": "AST", "pra": "PRA", "minutes": "Minutes",
    })
    st.line_chart(display.set_index("Date")[["PTS", "REB", "AST"]])
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        display[["Date", "Matchup", "Minutes", "PTS", "REB", "AST", "PRA"]].iloc[::-1],
        hide_index=True, use_container_width=True,
    )
with validation:
    if entry is None:
        st.info("Validation scores appear once there is sufficient history.")
    else:
        st.subheader("Mean absolute error")
        st.caption(
            f"{entry['validation_count']} validation appearances · {entry['validation_start']} to "
            f"{entry['validation_end']} · Lower is better"
        )
        st.dataframe(
            pd.DataFrame.from_dict(entry["mae"], orient="index"),
            column_config={stat: st.column_config.NumberColumn(format="%.2f") for stat in ("PTS", "REB", "AST", "PRA")},
        )
