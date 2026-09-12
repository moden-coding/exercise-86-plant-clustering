#!/usr/bin/env python3

import unittest
from unittest.mock import patch

import sklearn
from sklearn.cluster import KMeans

from src.plant_clustering import plant_clustering


class TestPlantClustering(unittest.TestCase):

    def test_correctness(self):
        acc = plant_clustering()
        self.assertAlmostEqual(
            acc, 0.8933333333333333, places=5,
            msg="plant_clustering() should return an accuracy of about "
                "0.8933 on the iris dataset once cluster labels are aligned "
                "with the true species labels. Got %r." % (acc,))

    def test_accuracy_score_used(self):
        with patch("src.plant_clustering.accuracy_score",
                   wraps=sklearn.metrics.accuracy_score) as accuracy:
            plant_clustering()
            accuracy.assert_called()

    def test_kmeans_called_with_3_clusters_and_random_state_0(self):
        with patch("src.plant_clustering.KMeans", side_effect=KMeans) as mock:
            plant_clustering()
            mock.assert_called()
            args, kwargs = mock.call_args

            correct = (
                (len(args) > 0 and args[0] == 3)
                or ("n_clusters" in kwargs and kwargs["n_clusters"] == 3)
            )
            self.assertTrue(
                correct,
                msg="KMeans should be created with n_clusters=3, one cluster "
                    "per iris species.")
            self.assertIn(
                "random_state", kwargs,
                msg="KMeans should be called with an explicit random_state "
                    "argument so the clustering result is reproducible.")
            self.assertEqual(
                kwargs["random_state"], 0,
                msg="KMeans should be called with random_state=0. Got %r."
                    % (kwargs["random_state"],))


if __name__ == '__main__':
    unittest.main()
