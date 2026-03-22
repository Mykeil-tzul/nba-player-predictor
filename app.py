import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Luka Doncic Predictor", page_icon="🏀", layout="wide")

st.title("🏀 Luka Dončić Points Predictor")
st.caption("A Streamlit app that uses rolling averages and advanced stats to predict Luka's next game points.")

DATA_FILE = "database_24_25.csv"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Keep only Luka
    luka = df[df["Player"] == "Luka Dončić"].copy()

    # Clean numeric columns used in the app/model
    numeric_cols = [
        "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "FT", "FTA", "FT%",
        "TRB", "AST", "STL", "BLK", "PTS", "GmSc"
    ]
    for col in numeric_cols:
        if col in luka.columns:
            luka[col] = pd.to_numeric(luka[col], errors="coerce")

    # Parse date if available and sort chronologically
    if "Data" in luka.columns:
        luka["Data"] = pd.to_datetime(luka["Data"], errors="coerce")
        luka = luka.sort_values("Data")
    else:
        luka = luka.sort_index()

    luka = luka.reset_index(drop=True)
    luka["Game_Number"] = range(1, len(luka) + 1)

    # Rolling features built from PRIOR games only to reduce leakage
    base_roll_cols = ["PTS", "MP", "FGA", "3PA", "FTA", "FG%", "3P%", "FT%", "AST", "TRB", "GmSc"]
    for col in base_roll_cols:
        if col in luka.columns:
            luka[f"rolling_{col.lower().replace('%', 'pct')}_3"] = luka[col].shift(1).rolling(3).mean()
            luka[f"rolling_{col.lower().replace('%', 'pct')}_5"] = luka[col].shift(1).rolling(5).mean()

    # Simple matchup/result encodings if present
    if "Res" in luka.columns:
        luka["win_flag"] = (luka["Res"] == "W").astype(int)
        luka["rolling_win_flag_5"] = luka["win_flag"].shift(1).rolling(5).mean()

    luka = luka.dropna().reset_index(drop=True)
    return luka


def build_model(luka_df: pd.DataFrame):
    feature_cols = [
        "MP", "FGA", "3PA", "FTA",
        "rolling_pts_3", "rolling_pts_5",
        "rolling_mp_5", "rolling_fga_5", "rolling_3pa_5", "rolling_fta_5",
        "rolling_fg_pct_5", "rolling_3p_pct_5", "rolling_ft_pct_5",
        "rolling_ast_5", "rolling_trb_5", "rolling_gmsc_5"
    ]

    feature_cols = [c for c in feature_cols if c in luka_df.columns]

    X = luka_df[feature_cols]
    y = luka_df["PTS"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, preds)

    comparison = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": preds
    }).reset_index(drop=True)

    return model, feature_cols, X_test, y_test, preds, comparison, mse, rmse, r2


try:
    luka_df = load_data(DATA_FILE)
except FileNotFoundError:
    st.error("database_24_25.csv was not found. Make sure it is in the same folder as this app.")
    st.stop()

if luka_df.empty:
    st.error("No Luka Dončić rows were found in the dataset.")
    st.stop()

model, feature_cols, X_test, y_test, preds, comparison, mse, rmse, r2 = build_model(luka_df)

latest = luka_df.iloc[-1]
last_5 = luka_df.tail(5)
last_10 = luka_df.tail(10)

# Sidebar prediction controls
st.sidebar.header("Predict Luka's Next Game")
st.sidebar.caption("Adjust the game volume inputs below.")

mp = st.sidebar.slider("Minutes Played (MP)", 28, 44, int(round(float(latest["MP"]))))
fga = st.sidebar.slider("Field Goal Attempts (FGA)", 12, 35, int(round(float(latest["FGA"]))))
three_pa = st.sidebar.slider("3-Point Attempts (3PA)", 3, 16, int(round(float(latest["3PA"]))))
fta = st.sidebar.slider("Free Throw Attempts (FTA)", 2, 16, int(round(float(latest["FTA"]))))

prediction_input = {
    "MP": mp,
    "FGA": fga,
    "3PA": three_pa,
    "FTA": fta,
}

# Fill model rolling/advanced features with latest known rolling values
for col in feature_cols:
    if col not in prediction_input:
        prediction_input[col] = float(latest[col])

prediction_df = pd.DataFrame([prediction_input])[feature_cols]
predicted_points = float(model.predict(prediction_df)[0])

# Top metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Predicted Next Game Points", f"{predicted_points:.1f}")
m2.metric("Luka Model RMSE", f"{rmse:.2f}")
m3.metric("Luka Model R²", f"{r2:.2f}")
m4.metric("Last Game Points", f"{float(last_10.iloc[-1]['PTS']):.0f}")

st.markdown("---")

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Recent Scoring Form")
    chart_df = luka_df[["Game_Number", "PTS", "rolling_pts_5"]].copy()
    chart_df = chart_df.set_index("Game_Number")
    st.line_chart(chart_df)

    st.subheader("Last 10 Games")
    show_cols = [c for c in ["Data", "Opp", "Res", "MP", "FGA", "3PA", "FTA", "AST", "TRB", "PTS", "GmSc"] if c in last_10.columns]
    st.dataframe(last_10[show_cols].reset_index(drop=True), use_container_width=True)

with right:
    st.subheader("Prediction Snapshot")
    st.write("**Input assumptions**")
    st.write({
        "MP": mp,
        "FGA": fga,
        "3PA": three_pa,
        "FTA": fta,
        "Rolling PTS (5)": round(float(latest.get("rolling_pts_5", np.nan)), 2),
        "Rolling AST (5)": round(float(latest.get("rolling_ast_5", np.nan)), 2),
        "Rolling TRB (5)": round(float(latest.get("rolling_trb_5", np.nan)), 2),
        "Rolling GmSc (5)": round(float(latest.get("rolling_gmsc_5", np.nan)), 2),
    })

    st.subheader("Advanced Stats (Last 5 Games Avg)")
    adv = pd.DataFrame({
        "Metric": ["PTS", "AST", "TRB", "FG%", "3P%", "FT%", "GmSc"],
        "Value": [
            round(float(last_5["PTS"].mean()), 2),
            round(float(last_5["AST"].mean()), 2),
            round(float(last_5["TRB"].mean()), 2),
            round(float(last_5["FG%"].mean()), 3),
            round(float(last_5["3P%"].mean()), 3),
            round(float(last_5["FT%"].mean()), 3),
            round(float(last_5["GmSc"].mean()), 2),
        ]
    })
    st.dataframe(adv, use_container_width=True, hide_index=True)

st.markdown("---")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Actual vs Predicted (Model Test Set)")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, preds, alpha=0.8)
    min_val = min(float(y_test.min()), float(preds.min()))
    max_val = max(float(y_test.max()), float(preds.max()))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    ax.set_xlabel("Actual Points")
    ax.set_ylabel("Predicted Points")
    ax.set_title("Luka Actual vs Predicted")
    st.pyplot(fig)

with c2:
    st.subheader("Prediction Errors")
    comparison_plot = comparison.copy()
    comparison_plot["Error"] = comparison_plot["Predicted"] - comparison_plot["Actual"]
    st.bar_chart(comparison_plot["Error"])

st.subheader("Model Comparison Table")
st.dataframe(comparison.round(2), use_container_width=True)

with st.expander("Show model features used"):
    st.write(feature_cols)

st.success("App ready. Save this file as app.py and run: streamlit run app.py")
