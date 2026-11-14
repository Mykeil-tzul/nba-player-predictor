"""Pregame features, fixed chronological evaluation and numeric model inference."""
import hashlib

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import PLAYERS, STATS, TARGETS

WINDOWS = (3, 5, 10)
FEATURES = [
    *[f"{stat}_lag1" for stat in STATS],
    *[f"{stat}_avg{window}" for window in WINDOWS for stat in STATS],
    "days_since_appearance",
]
MIN_TRAIN = 20
MIN_TEST = 5
MODEL_VERSION = 2


def features(games):
    ordered = games.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    by_player = ordered.groupby("player_id", sort=False)
    lagged = by_player[STATS].shift()
    values = {f"{stat}_lag1": lagged[stat] for stat in STATS}
    for window in WINDOWS:
        means = lagged.groupby(ordered["player_id"], sort=False).rolling(window).mean()
        means = means.reset_index(level=0, drop=True).reindex(ordered.index)
        values.update({f"{stat}_avg{window}": means[stat] for stat in STATS})
    values["days_since_appearance"] = by_player["game_date"].diff().dt.days.clip(upper=30)
    return ordered.join(pd.DataFrame(values))


def fingerprint(games):
    return hashlib.sha256(games.to_csv(index=False).encode()).hexdigest()


def bundle_model(rows):
    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
    pipeline.fit(rows[FEATURES], rows[TARGETS])
    scaler, ridge = pipeline.steps[0][1], pipeline.steps[1][1]
    return {
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": ridge.coef_.tolist(), "intercept": ridge.intercept_.tolist(),
    }


def estimate(model, rows):
    x = (rows[FEATURES].to_numpy() - model["mean"]) / model["scale"]
    return np.maximum(0, x @ np.asarray(model["coef"]).T + model["intercept"])


def with_pra(values):
    return np.column_stack([values, values.sum(axis=1)])


def evaluate(train, test):
    actual = with_pra(test[TARGETS].to_numpy())
    predictions = {
        "Last appearance": test[[f"{col}_lag1" for col in TARGETS]].to_numpy(),
        "Last 5 average": test[[f"{col}_avg5" for col in TARGETS]].to_numpy(),
    }
    if train is not None:
        predictions = {"Ridge": estimate(bundle_model(train), test), **predictions}
    labels = [col.upper() for col in [*TARGETS, "pra"]]
    return {
        name: dict(zip(labels, np.abs(with_pra(values) - actual).mean(axis=0).tolist()))
        for name, values in predictions.items()
    }


def fit(games, source):
    all_rows = features(games)
    rows = all_rows.dropna(subset=FEATURES)
    players = {}
    skipped = {}
    baseline_rows = all_rows.dropna(subset=[f"{stat}_avg5" for stat in TARGETS])
    for player_id in PLAYERS:
        player = rows.loc[rows.player_id.eq(player_id)]
        cutoff = int(len(player) * 0.8)
        train, test = player.iloc[:cutoff], player.iloc[cutoff:]
        model = None
        if len(train) >= MIN_TRAIN and len(test) >= MIN_TEST:
            model = bundle_model(player)
        elif source == "demo":
            available = baseline_rows.loc[baseline_rows.player_id.eq(player_id)]
            cutoff = int(len(available) * 0.8)
            train, test = available.iloc[:cutoff], available.iloc[cutoff:]
        minimum_validation = 4 if source == "demo" and model is None else MIN_TEST
        if len(test) < minimum_validation or (model is None and source != "demo"):
            skipped[str(player_id)] = f"Only {len(train)} training and {len(test)} validation appearances"
            continue
        players[str(player_id)] = {
            "mae": evaluate(train if model is not None else None, test),
            "train_count": len(train),
            "validation_count": len(test),
            "train_end": train.game_date.iloc[-1].date().isoformat(),
            "validation_start": test.game_date.iloc[0].date().isoformat(),
            "validation_end": test.game_date.iloc[-1].date().isoformat(),
            "model": model,
            "method": "Ridge" if model is not None else "Last 5 average",
        }
    if not players:
        raise ValueError("Insufficient history for every player")
    return {
        "version": MODEL_VERSION, "source": source, "data_hash": fingerprint(games),
        "features": FEATURES, "players": players, "skipped": skipped,
        "data_through": games.game_date.max().date().isoformat(),
    }


def verify_artifact(artifact, games):
    if artifact["version"] != MODEL_VERSION or artifact["features"] != FEATURES:
        raise ValueError("Model format has changed. Run training again.")
    if artifact["data_hash"] != fingerprint(games):
        raise ValueError("Game data has changed since training. Run training again.")


def predict(artifact, games, player_id, game_date, availability, *, demo=False):
    if availability != "expected":
        raise ValueError("Projection requires expected availability. No projection for unknown status or absences.")
    player = games.loc[games.player_id.eq(player_id)]
    entry = artifact["players"].get(str(player_id))
    if entry is None or player.empty:
        raise ValueError("Insufficient history for this player")
    game_date = pd.Timestamp(game_date)
    last = player.game_date.max()
    if game_date <= last:
        raise ValueError("Target date must be after the player's latest recorded appearance")
    if not demo and (game_date - last).days > 30:
        raise ValueError("More than 30 days since last appearance. Projection withheld.")
    # The placeholder has only pregame information, and never enters training.
    future = pd.DataFrame([{"player_id": player_id, "game_date": game_date}])
    upcoming = features(pd.concat([player, future], ignore_index=True)).tail(1)
    if entry["model"] is None:
        values = upcoming[[f"{stat}_avg5" for stat in TARGETS]].to_numpy()
    else:
        values = estimate(entry["model"], upcoming)
    prediction = with_pra(values)[0]
    return dict(zip(["PTS", "REB", "AST", "PRA"], prediction.tolist()))
