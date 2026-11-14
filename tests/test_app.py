import unittest

from streamlit.testing.v1 import AppTest

from src.data import ROOT


class DashboardTests(unittest.TestCase):
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
