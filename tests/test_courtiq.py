import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

from src.data import COLUMNS, PLAYERS, normalize
from src.fetch_data import collect, decode, download
from src.import_archive import season_games
from src.model import FEATURES, evaluate, features, fit, predict, verify_artifact

LUKA, KYRIE, *_ = PLAYERS


def history():
    rows = [
        {
            "game_id": f"{i:010d}", "season": "2024-25", "matchup": "AAA vs. BBB",
            "player_id": player, "game_date": pd.Timestamp("2024-10-24") + pd.Timedelta(days=2 * i),
            "pts": offset + i, "reb": i % 8, "ast": i % 10,
            "minutes": 32, "fga": 15, "fta": 3,
        }
        for player, offset in [(LUKA, 0), (KYRIE, 100)]
        for i in range(50)
    ]
    return normalize(pd.DataFrame(rows[::-1]))


class ModelingTests(unittest.TestCase):
    def test_target_and_later_box_scores_do_not_enter_features(self):
        games = history()
        target = games.game_date.unique()[10]
        rows = features(games)
        altered = games.copy()
        altered.loc[altered.game_date.ge(target), "pts"] = 999
        pd.testing.assert_frame_equal(
            rows.loc[rows.game_date.eq(target), FEATURES],
            features(altered).loc[rows.game_date.eq(target), FEATURES],
        )
        self.assertEqual(rows.loc[rows.game_date.eq(target), "pts_avg10"].tolist(), [104.5, 4.5])
        self.assertEqual(rows.loc[rows.game_date.eq(target), "pts_lag1"].tolist(), [109, 9])

    def test_chronological_scores_and_pra(self):
        games = history()
        artifact = fit(games, "fixture")
        entry = artifact["players"][str(LUKA)]
        self.assertLess(entry["train_end"], entry["validation_start"])
        self.assertEqual((entry["train_count"], entry["validation_count"]), (32, 8))
        self.assertEqual(entry["mae"]["Last appearance"]["PTS"], 1)
        self.assertEqual(entry["mae"]["Last 5 average"]["PTS"], 3)
        self.assertEqual(artifact, fit(games, "fixture"))
        verify_artifact(artifact, games)
        with self.assertRaisesRegex(ValueError, "changed"):
            verify_artifact(artifact, games.iloc[:-1])
        upcoming = games.game_date.max() + pd.Timedelta(days=2)
        result = predict(artifact, games, LUKA, upcoming, "expected")
        self.assertAlmostEqual(result["PRA"], sum(result[k] for k in ("PTS", "REB", "AST")))
        for status in ("out", "unknown"):
            with self.assertRaisesRegex(ValueError, "expected availability"):
                predict(artifact, games, LUKA, upcoming, status)
        with self.assertRaisesRegex(ValueError, "30 days"):
            predict(artifact, games, LUKA, upcoming + pd.Timedelta(days=31), "expected")

    def test_pra_error_uses_signed_component_errors(self):
        rows = features(history()).dropna(subset=FEATURES)
        train = rows.iloc[:25]
        test = rows.iloc[25:26].copy()
        prediction = np.array([[30, 5, 5]])
        test[["pts", "reb", "ast"]] = [[35, 0, 5]]
        with patch("src.model.estimate", return_value=prediction):
            self.assertEqual(evaluate(train, test)["Ridge"], {"PTS": 5, "REB": 5, "AST": 0, "PRA": 0})

    def test_zero_minutes_are_skipped_and_conflicts_rejected(self):
        games = history()
        dnp = games.iloc[[0]].assign(minutes=0)
        self.assertEqual(len(normalize(pd.concat([games, dnp]))), len(games))
        conflicting = games.iloc[[0]].assign(pts=45)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            normalize(pd.concat([games, conflicting]))


class CollectionTests(unittest.TestCase):
    def test_archive_keeps_only_supported_regular_season_appearances(self):
        rows = []
        for game_id, player_id in [("0022600001", LUKA), ("0012600002", LUKA), ("0022600003", 99)]:
            rows.append(dict(
                game_id=game_id, player_id=player_id, game_date="2026-10-26",
                min=30, pts=23, reb=5, ast=7, fga=15, fta=2, home=True,
                team_id="1610612747", opp_team_id="1610612744",
            ))
        games = season_games(pd.DataFrame(rows), "2026-27")
        self.assertEqual(games.game_id.tolist(), ["0022600001"])
        self.assertEqual(games.matchup.tolist(), ["LAL vs. GSW"])


    def test_decode_nba_schema_and_leading_game_id_zeroes(self):
        payload = {"resultSets": [{
            "name": "PlayerGameLog",
            "headers": ["Player_ID", "Game_ID", "GAME_DATE", "MATCHUP", "PTS", "REB", "AST", "MIN", "FGA", "FTA"],
            "rowSet": [[LUKA, "0022400001", "OCT 24, 2024", "DAL vs. SAS", 28, 10, 8, "35:30", 21, 5]],
        }]}
        row = decode(payload, LUKA, "2024-25").iloc[0]
        self.assertEqual(row.game_id, "0022400001")
        self.assertEqual(row.minutes, 35.5)
        payload["resultSets"][0]["rowSet"] = []
        self.assertTrue(decode(payload, LUKA, "2024-25").empty)

    @patch("src.fetch_data.time.sleep")
    def test_rate_limit_then_success(self, sleep):
        session = Mock()
        limited = requests.Response()
        limited.status_code = 429
        limited.headers["Retry-After"] = "4"
        success = Mock()
        session.get.side_effect = [limited, success]
        with patch("src.fetch_data.decode", return_value="parsed"):
            self.assertEqual(download(session, LUKA, "2024-25"), "parsed")
        sleep.assert_any_call(4)
        self.assertEqual(session.get.call_count, 2)

    @patch("src.fetch_data.time.sleep")
    def test_access_denied_is_not_retried(self, sleep):
        session = Mock()
        denied = requests.Response()
        denied.status_code = 403
        session.get.return_value = denied
        with self.assertRaises(requests.HTTPError):
            download(session, LUKA, "2024-25")
        self.assertEqual(session.get.call_count, 1)

    @patch("src.fetch_data.SEASONS", ("2024-25", "2026-27"))
    @patch("src.fetch_data.PLAYERS", {LUKA: "Luka", KYRIE: "Kyrie"})
    def test_offline_uses_cache_and_preserves_snapshot_on_missing_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            output = directory / "snapshot.csv"
            output.write_text("previous snapshot")
            session = Mock()
            rows = history()
            rows.loc[rows.player_id.eq(LUKA), COLUMNS].to_csv(
                directory / f"{LUKA}_2024-25.csv", index=False
            )
            with self.assertRaisesRegex(RuntimeError, "Kyrie"):
                collect(session, directory, output, offline=True, today=date(2025, 11, 1))
            self.assertEqual(output.read_text(), "previous snapshot")
            pd.DataFrame(columns=COLUMNS).to_csv(directory / f"{KYRIE}_2024-25.csv", index=False)
            games = collect(session, directory, output, offline=True, today=date(2025, 11, 1))
            self.assertEqual(len(games), 50)
            session.get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
