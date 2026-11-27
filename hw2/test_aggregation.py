import hashlib
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from aggregation import (
    estimate_weights,
    fit_dawid_skene,
    majority_vote,
    weighted_majority_vote,
)
from dataset import read_source
from experiment import paired_differences, quality, split_tasks


class AggregationTests(unittest.TestCase):
    def test_majority_ignores_missing_votes_and_breaks_ties(self) -> None:
        votes = np.array([[0, 1, -1], [1, -1, -1], [-1, 0, 0]])
        np.testing.assert_array_equal(majority_vote(votes), [0, 1, 0])
        result = weighted_majority_vote(votes, np.ones(3))
        np.testing.assert_array_equal(result, majority_vote(votes))

    def test_worker_weights_and_unseen_worker(self) -> None:
        gold = np.array([0, 1] * 10)
        missing = np.full_like(gold, -1)
        votes = np.column_stack([gold, gold, 1 - gold, missing])
        weights = estimate_weights(votes, gold)
        pooled = 41 / 62
        self.assertAlmostEqual(weights[3], np.log(pooled / (1 - pooled)))
        self.assertGreater(weights[0], weights[3])
        self.assertEqual(weights[2], 0)
        test = np.array([[1, -1, 0, 0]])
        np.testing.assert_array_equal(majority_vote(test), [0])
        np.testing.assert_array_equal(
            weighted_majority_vote(test, weights), [1]
        )

    def test_em_learns_a_reversed_worker_and_normalized_probabilities(
        self,
    ) -> None:
        gold = np.array([0, 1] * 15)
        votes = np.column_stack(
            [
                gold,
                gold,
                gold,
                1 - gold,
                np.ones_like(gold),
                np.full_like(gold, -1),
            ]
        )
        model = fit_dawid_skene(votes)
        self.assertGreater(model.confusion[3, 0, 1], 0.9)
        self.assertGreater(model.confusion[3, 1, 0], 0.9)
        np.testing.assert_allclose(model.confusion.sum(axis=2), 1)
        np.testing.assert_allclose(model.predict_proba(votes).sum(axis=1), 1)
        np.testing.assert_array_equal(model.predict(votes), gold)
        test = np.array([[-1] * 6, [-1, -1, -1, -1, -1, 1]])
        probabilities = model.predict_proba(test)
        np.testing.assert_allclose(probabilities[0], model.prior)
        self.assertGreater(probabilities[1, 1], model.prior[1])
        np.testing.assert_allclose(
            model.predict_proba(test[1:])[0], probabilities[1]
        )

    def test_split_is_reproducible_disjoint_and_stratified(self) -> None:
        gold = np.repeat([0, 1], 400)
        train, test = split_tasks(gold)
        np.testing.assert_array_equal(train, split_tasks(gold)[0])
        np.testing.assert_array_equal(np.bincount(gold[train]), [80, 80])
        np.testing.assert_array_equal(np.bincount(gold[test]), [320, 320])
        np.testing.assert_array_equal(
            np.sort(np.r_[train, test]), np.arange(800)
        )

    def test_metrics_count_both_classes(self) -> None:
        scores = quality(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
        self.assertEqual(scores["confusion"], [[1, 1], [0, 2]])
        self.assertEqual(scores["accuracy"], 0.75)
        self.assertAlmostEqual(scores["macro_f1"], (2 / 3 + 4 / 5) / 2)

    def test_bootstrap_pairs_the_same_tasks(self) -> None:
        gold = np.array([0, 0, 1, 1])
        repeated = np.array([0, 1, 1, 1])
        actual = paired_differences(gold, {"A": repeated, "B": repeated})
        self.assertEqual(actual["B minus A"]["ci95"], [0.0, 0.0])
        self.assertEqual(actual["B minus A"]["gains"], 0)

    def test_cache_verifies_bytes_before_replacing_a_file(self) -> None:
        content = b"0\n1\n"
        checksums = {"example": hashlib.sha256(content).hexdigest()}
        with (
            TemporaryDirectory() as directory,
            patch("dataset.CHECKSUMS", checksums),
        ):
            path = Path(directory)
            cached = path / "example"
            cached.write_bytes(b"corrupt")
            with patch("dataset.urlopen", return_value=BytesIO(b"incorrect")):
                with self.assertRaises(ValueError):
                    read_source(path, "example")
            self.assertEqual(cached.read_bytes(), b"corrupt")
            with patch(
                "dataset.urlopen", return_value=BytesIO(content)
            ) as request:
                self.assertEqual(read_source(path, "example"), "0\n1\n")
                self.assertEqual(read_source(path, "example"), "0\n1\n")
                self.assertEqual(request.call_count, 1)


if __name__ == "__main__":
    unittest.main()
