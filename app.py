"""CourtIQ's local-data Streamlit dashboard."""
import json
from html import escape
from datetime import timedelta

import pandas as pd
import streamlit as st

from src.data import ARTIFACT, PLAYERS, SNAPSHOT, legacy_games, nba_today, read_snapshot
from src.model import predict, verify_artifact

st.set_page_config(page_title="CourtIQ", page_icon="🏀", layout="wide")


@st.cache_resource(max_entries=1)
def demo_data(artifact_version):
    games = legacy_games()
    artifact = json.loads(DEMO_ARTIFACT.read_text())
    verify_artifact(artifact, games)
    return games, artifact


@st.cache_resource(max_entries=1)
def nba_data(snapshot_version, artifact_version):
    games = read_snapshot(SNAPSHOT)
    artifact = json.loads(ARTIFACT.read_text())
    verify_artifact(artifact, games)
    return games, artifact


DEMO_ARTIFACT = ARTIFACT.with_name("demo.json")

SLUGS = {
    1629029: "luka-doncic", 202681: "kyrie-irving", 1628369: "jayson-tatum",
    1628983: "shai-gilgeous-alexander", 1628973: "jalen-brunson",
}
HEADSHOTS = "https://cdn.nba.com/headshots/nba/latest/1040x760"
st.title("CourtIQ")
st.caption("NBA regular season · Next-game forecasts")
with st.sidebar:
    nba_ready = SNAPSHOT.is_file() and ARTIFACT.is_file()
    mode = st.radio(
        "Data source", ("NBA snapshot", "Historical demo") if nba_ready else ("Historical demo",),
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
        games, artifact = demo_data(DEMO_ARTIFACT.stat().st_mtime_ns)
        st.info("Historical demo · Partial 2024–25 data bundled with the repository.")
    else:
        games, artifact = nba_data(SNAPSHOT.stat().st_mtime_ns, ARTIFACT.stat().st_mtime_ns)
except (OSError, ValueError, KeyError, TypeError) as error:
    st.error(f"Could not load game data and model. Run `python -m src.fetch_data` and `python -m src.train`. {error}")
    st.stop()

player = games.loc[games.player_id.eq(player_id)].copy()
entry = artifact["players"].get(str(player_id))
st.html(f"""
<style>
.ciq-hero {{
  display: flex; align-items: center; justify-content: space-between; gap: 1rem;
  padding: 1.25rem 1.5rem 0; margin-bottom: 1rem; overflow: hidden;
  border: 1px solid #4b3841; border-radius: 1rem;
  background: radial-gradient(ellipse at 85% 15%, #713f37, #242638 65%);
}}
.ciq-hero h2 {{ color: #fff; font-size: clamp(1.6rem, 5vw, 3.2rem); padding: 0; }}
.ciq-hero p {{ color: #ffd3bd; }}
.ciq-hero img {{ width: min(46%, 19rem); align-self: end; height: auto; }}
[data-testid="stMetric"] {{ background: #1b202c; border: 1px solid #353b49;
  border-radius: .8rem; padding: .9rem; }}
</style>
<section class="ciq-hero">
  <div><h2>{escape(name)}</h2>
  <p>{len(player)} recorded appearances</p></div>
  <img src="{HEADSHOTS}/{player_id}.png" alt="{escape(name)} NBA headshot">
</section>
""")
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
    st.caption("Enter the scheduled game date and confirm availability before requesting a forecast.")
    if (nba_today() - data_through).days > 3:
        st.warning("This snapshot is more than three days old. Refresh the game logs and retrain.")
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

recent, validation, film = st.tabs(["Recent form", "Model accuracy", "Film room"])
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
        st.write("This is the average size of a miss on later games. Lower is better.")
        st.caption(
            f"{entry['validation_count']} validation appearances · {entry['validation_start']} to "
            f"{entry['validation_end']} · Lower is better"
        )
        st.dataframe(
            pd.DataFrame.from_dict(entry["mae"], orient="index"),
            column_config={stat: st.column_config.NumberColumn(format="%.2f") for stat in ("PTS", "REB", "AST", "PRA")},
        )

with film:
    portrait, details = st.columns([1, 2])
    with portrait:
        st.image(f"{HEADSHOTS}/{player_id}.png", caption=f"{name} · NBA.com", width=280)
    with details:
        st.subheader("Player highlights")
        st.link_button("Watch on NBA.com ↗", f"https://www.nba.com/player/{player_id}/{SLUGS[player_id]}/videos")
        if not demo:
            st.write("Latest recorded game")
            st.link_button("Open NBA game center ↗", f"https://www.nba.com/game/{player.iloc[-1].game_id}")
