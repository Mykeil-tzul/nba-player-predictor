import unittest
from datetime import date
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.data import ROOT


class DashboardTests(unittest.TestCase):
    @patch("src.data.nba_today", return_value=date(2026, 11, 14))
    def test_bundled_nba_models_forecast_all_five(self, today):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15).run()
        self.assertEqual(app.sidebar.radio[0].value, "NBA snapshot")
        for player in app.sidebar.selectbox[0].options:
            app.sidebar.selectbox[0].set_value(player).run()
            app.sidebar.radio[1].set_value("expected").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.metric), 4)
            self.assertTrue(any("Ridge" in item.value for item in app.caption))

    def test_demo_projection_and_availability(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=15)
        app.run()
        app.sidebar.radio[0].set_value("Historical demo").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 0)
        for player in app.sidebar.selectbox[0].options:
            app.sidebar.selectbox[0].set_value(player).run()
            self.assertEqual(len(app.metric), 0)
            app.sidebar.radio[1].set_value("expected").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual([metric.label for metric in app.metric], ["PTS", "REB", "AST", "PRA"])
            app.sidebar.radio[1].set_value("out").run()
            self.assertEqual(len(app.metric), 0)


if __name__ == "__main__":
    unittest.main()
