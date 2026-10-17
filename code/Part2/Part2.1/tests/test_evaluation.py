import json
import math
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR))

from evaluation.accuracy import aggregate_accuracy, per_user_accuracy, user_accuracy
from evaluation.beyond_accuracy import (
    calibration_kl,
    catalog_coverage,
    gini_index,
    intra_list_diversity,
    mean_self_information,
    normalized_entropy,
    user_popularity_deviation,
)
from evaluation.cli import main, validate_protocol
from evaluation.data import age_group, load_item_genres, load_recommendations


class AccuracyTests(unittest.TestCase):
    def test_user_metrics(self):
        result = user_accuracy({"a", "c"}, ["a", "b", "c"], 3)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1"], 0.8)
        self.assertAlmostEqual(result["hit"], 1.0)
        self.assertAlmostEqual(result["mrr"], 1.0)
        expected_ndcg = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(result["ndcg"], expected_ndcg)
        self.assertAlmostEqual(result["map"], (1 + 2 / 3) / 2)

    def test_macro_average(self):
        rows = per_user_accuracy(
            {"u1": {"a"}, "u2": {"b"}},
            {"u1": ["a", "x"], "u2": ["x", "y"]},
            k=2,
        )
        result = aggregate_accuracy(rows)
        self.assertAlmostEqual(result["precision"], 0.25)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["hit"], 0.5)

    def test_user_sets_must_match(self):
        with self.assertRaises(ValueError):
            per_user_accuracy({"u1": {"a"}}, {"u2": ["a"]}, k=1)


class BeyondAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.genres = {
            "a": {"Action"},
            "b": {"Action"},
            "c": {"Comedy"},
        }

    def test_ild(self):
        self.assertAlmostEqual(intra_list_diversity(["a", "b"], self.genres), 0.0)
        self.assertAlmostEqual(intra_list_diversity(["a", "c"], self.genres), 1.0)

    def test_calibration(self):
        self.assertAlmostEqual(
            calibration_kl(
                ["a", "c"], ["a", "c"], self.genres, rank_weighted=False
            ),
            0.0,
        )
        self.assertGreater(calibration_kl(["a", "c"], ["a", "b"], self.genres), 0.0)

    def test_novelty_uses_laplace_smoothing(self):
        result = mean_self_information(["known", "unseen"], {"known": 2}, 2)
        expected = (-math.log2(3 / 4) - math.log2(1 / 4)) / 2
        self.assertAlmostEqual(result, expected)

    def test_coverage_and_exposure(self):
        recommendations = {"u1": ["a", "b"], "u2": ["b", "c"]}
        self.assertAlmostEqual(catalog_coverage(recommendations, self.genres), 1.0)
        self.assertAlmostEqual(gini_index([1, 1, 1]), 0.0)
        self.assertAlmostEqual(normalized_entropy([1, 1, 1]), 1.0)

    def test_popularity_deviation(self):
        groups = {"a": "head", "b": "mid", "c": "tail"}
        self.assertAlmostEqual(user_popularity_deviation(["a"], ["a"], groups), 0.0)
        self.assertGreater(user_popularity_deviation(["a"], ["c"], groups), 0.0)


class DataTests(unittest.TestCase):
    def test_teacher_item_field_and_age_bins(self):
        with tempfile.TemporaryDirectory() as directory_name:
            item_file = Path(directory_name) / "items.item"
            item_file.write_text(
                "item_id:token\tclass:token_seq\n1\tAction Comedy\n",
                encoding="utf-8",
            )
            self.assertEqual(load_item_genres(item_file)["1"], {"Action", "Comedy"})
        self.assertEqual(age_group("17"), "under_18")
        self.assertEqual(age_group("24"), "18_24")
        self.assertEqual(age_group("56"), "56_plus")

    def test_repository_recommendation_json(self):
        with tempfile.TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "recommendations.json"
            path.write_text(
                json.dumps(
                    {
                        "dataset": "ml-100k",
                        "model": "Example",
                        "split": "test",
                        "k": 2,
                        "recommendations": {
                            "u1": {"items": ["i1", "i2"], "scores": [1.0, 0.5]}
                        },
                    }
                ),
                encoding="utf-8",
            )
            metadata, recommendations = load_recommendations(path, 2)
            self.assertEqual(metadata["model"], "Example")
            self.assertEqual(recommendations["u1"], ["i1", "i2"])

    def test_protocol_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_protocol({"dataset": "ml-1m", "k": 10}, 10)
        with self.assertRaises(ValueError):
            validate_protocol({"dataset": "ml-100k", "seed": 7, "k": 10}, 10)


class CommandLineTests(unittest.TestCase):
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            train = directory / "ml-100k.train.tsv"
            valid = directory / "ml-100k.valid.tsv"
            test = directory / "ml-100k.test.tsv"
            recommendations = directory / "recommendations.json"
            items = directory / "ml-100k.item"
            users = directory / "ml-100k.user"
            output = directory / "result.json"
            per_user = directory / "per-user.tsv"

            train.write_text(
                "user_id\titem_id\nu1\ti1\nu2\ti2\n", encoding="utf-8"
            )
            valid.write_text(
                "user_id\titem_id\nu1\ti2\nu2\ti3\n", encoding="utf-8"
            )
            test.write_text(
                "user_id\titem_id\nu1\ti3\nu2\ti1\n", encoding="utf-8"
            )
            recommendations.write_text(
                json.dumps(
                    {
                        "dataset": "ml-100k",
                        "model": "Example",
                        "split": "test",
                        "k": 2,
                        "seed": 2020,
                        "eval_args": {
                            "split": {"RS": [0.8, 0.1, 0.1]},
                            "group_by": "user",
                            "order": "RO",
                            "mode": {"valid": "full", "test": "full"},
                        },
                        "recommendations": {
                            "u1": {"items": ["i3", "i2"]},
                            "u2": {"items": ["i1", "i3"]},
                        },
                    }
                ),
                encoding="utf-8",
            )
            items.write_text(
                "item_id:token\tclass:token_seq\n"
                "i1\tAction\ni2\tComedy\ni3\tAction Comedy\n",
                encoding="utf-8",
            )
            users.write_text(
                "user_id:token\tage:token\tgender:token\n"
                "u1\t24\tF\nu2\t35\tM\n",
                encoding="utf-8",
            )

            argv = [
                "evaluate.py",
                "--train", str(train),
                "--valid", str(valid),
                "--test", str(test),
                "--recommendations", str(recommendations),
                "--items", str(items),
                "--users", str(users),
                "--k", "2",
                "--output", str(output),
                "--per-user-output", str(per_user),
            ]
            with patch("sys.argv", argv), redirect_stdout(StringIO()):
                main()

            result = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["evaluated_users"], 2)
            self.assertAlmostEqual(result["accuracy"]["recall"], 1.0)
            self.assertAlmostEqual(result["accuracy"]["f1"], 2 / 3)
            self.assertIn("catalog_coverage", result["beyond_accuracy"])
            self.assertEqual(
                result["group_fairness"]["age"]["group_sizes"],
                {"18_24": 1, "35_44": 1},
            )
            self.assertTrue(per_user.exists())


if __name__ == "__main__":
    unittest.main()
