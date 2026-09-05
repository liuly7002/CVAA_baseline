import unittest

from cvaa.metrics import compute_metric_pair, rank_actor_scores


class TestMetrics(unittest.TestCase):
    def test_ad_fd(self):
        original = [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]
        cf = [[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]]

        result = compute_metric_pair(
            original,
            cf,
            "pred_route",
        )

        self.assertAlmostEqual(result["AD"], 1.0, places=7)
        self.assertAlmostEqual(result["FD"], 2.0, places=7)

    def test_rank_ad_then_fd(self):
        rows = [
            {
                "route_id": "r",
                "frame": "0001",
                "actor_id": "1",
                "actor_class": "vehicle",
                "AD": 0.2,
                "FD": 0.1,
            },
            {
                "route_id": "r",
                "frame": "0001",
                "actor_id": "2",
                "actor_class": "vehicle",
                "AD": 0.3,
                "FD": 0.0,
            },
            {
                "route_id": "r",
                "frame": "0001",
                "actor_id": "3",
                "actor_class": "vehicle",
                "AD": 0.2,
                "FD": 0.5,
            },
        ]

        rankings = rank_actor_scores(rows)
        ids = [
            item["actor_id"]
            for item in rankings[0]["ranking"]
        ]
        self.assertEqual(ids, ["2", "3", "1"])


if __name__ == "__main__":
    unittest.main()
