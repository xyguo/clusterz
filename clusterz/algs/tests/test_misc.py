import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from clusterz.algs.misc import (
    DistQueryOracle, distributedly_estimate_diameter, estimate_diameter, farthest_neighbor
)


class MyTestCase(unittest.TestCase):

    def setUp(self):
        small_cluster = np.random.uniform(-1, 1, size=(5, 2))
        cluster = np.random.uniform(-1, 1, size=(40, 2))
        self.centers_ = np.array([
            [0, 30], [0, -30]
        ])
        self.outliers_ = np.array([
            [80, 0], [-80, 0]
        ])

        # data set on a single machine
        self.toyX_with_outliers = np.vstack(
            [self.centers_,
             self.outliers_,
             # clusters
             small_cluster + self.centers_[0] + np.array([5, 0]),
             small_cluster + self.centers_[0] + np.array([-5, 0]),
             small_cluster + self.centers_[1] + np.array([5, 0]),
             small_cluster + self.centers_[1] + np.array([-5, 0])])
        self.X_without_outliers_ = np.vstack(
            [self.centers_,
             # clusters
             cluster + self.centers_[0] + np.array([5, 0]),
             cluster + self.centers_[0] + np.array([-5, 0]),
             cluster + self.centers_[1] + np.array([5, 0]),
             cluster + self.centers_[1] + np.array([-5, 0])])
        self.X_with_outliers_ = np.vstack(
            [self.centers_,
             self.outliers_,
             # clusters
             cluster + self.centers_[0] + np.array([5, 0]),
             cluster + self.centers_[0] + np.array([-5, 0]),
             cluster + self.centers_[1] + np.array([5, 0]),
             cluster + self.centers_[1] + np.array([-5, 0])])
        self.random_weight_without_outliers_ = np.hstack(
            [np.ones(2) * 2,
             np.random.uniform(1, 2, len(cluster) * 4)])
        self.random_weight_ = np.hstack(
            [np.ones(2),
             np.ones(2),
             np.random.uniform(1, 2, len(cluster) * 4)])

        # data on 2 machines
        self.Xs_without_outliers_ = [
            np.vstack(
                [self.centers_[0],
                 # clusters
                 cluster + self.centers_[0] + np.array([5, 0]),
                 cluster + self.centers_[1] + np.array([-5, 0])]),
            np.vstack(
               [self.centers_[1],
                # clusters
                cluster + self.centers_[0] + np.array([-5, 0]),
                cluster + self.centers_[1] + np.array([5, 0]),
               ]
            )]
        self.Xs_with_outliers_ = [
            np.vstack(
                [self.centers_[0],
                 self.outliers_[0],
                 # clusters
                 cluster + self.centers_[0] + np.array([5, 0]),
                 cluster + self.centers_[1] + np.array([-5, 0])]),
            np.vstack(
               [self.centers_[1],
                self.outliers_[1],
                # clusters
                cluster + self.centers_[0] + np.array([-5, 0]),
                cluster + self.centers_[1] + np.array([5, 0]),
               ]
            )]

    def test_estimate_diameter(self):
        lb, ub = estimate_diameter(self.X_without_outliers_, n_estimation=10,
                                   metric='minkowski')
        assert lb >= 2 * np.sqrt(5 ** 2 + 30 ** 2) - 2 * np.sqrt(2)
        assert ub >= 2 * lb

        lb, ub = distributedly_estimate_diameter(Xs=self.Xs_without_outliers_,
                                                 n_estimation=1)
        assert lb >= 2 * np.sqrt(5 ** 2 + 30 ** 2) - 2 * np.sqrt(2)
        assert ub >= 2 * (2 * np.sqrt(5 ** 2 + 30 ** 2) - 2 * np.sqrt(2))

    def test_farthest_neighbor(self):
        d, idx = farthest_neighbor(self.centers_[0], self.X_with_outliers_)
        assert idx in [2, 3]
        assert_almost_equal(d, np.sqrt(30 ** 2 + 80 ** 2), decimal=3)

    def test_DistQueryOracle_ball_tree(self):

        # test oracle based on ball tree
        oc = DistQueryOracle(tree_algorithm='ball_tree',
                             leaf_size=5, precompute_distances='auto')
        oc.fit(self.X_with_outliers_, sample_weight=self.random_weight_)
        idxs_in_ball = oc.ball((self.centers_[0] + np.array([5, 0])).reshape(1, -1),
                               radius=2)[0]
        assert len(idxs_in_ball) == 40
        assert idxs_in_ball.max() < 44 and idxs_in_ball.min() > 3

        center, idxs_in_densest_ball = oc.densest_ball(radius=2)
        assert len(idxs_in_densest_ball) == 40
        assert np.linalg.norm(center - self.outliers_, axis=1).min() > \
               np.sqrt(75 ** 2 + 30 ** 2) - np.sqrt(2)

        dist, idxs = oc.kneighbors(
           (self.centers_[0] + np.array([5, 0])).reshape(1, -1), k=41)
        dist, idxs = dist[0], idxs[0]
        assert len(idxs) == 41
        dist.sort()
        assert dist.max() == 5
        assert dist[-2] < 1.5

    def test_DistQueryOracle_kd_tree(self):
        # test oracle based on KD tree
        oc = DistQueryOracle(tree_algorithm='kd_tree',
                             leaf_size=5, precompute_distances='auto')
        oc.fit(self.X_with_outliers_, sample_weight=self.random_weight_)
        idxs_in_ball = oc.ball((self.centers_[0] + np.array([5, 0])).reshape(1, -1),
                               radius=2)[0]
        assert len(idxs_in_ball) == 40
        assert idxs_in_ball.max() < 44 and idxs_in_ball.min() > 3

        center, idxs_in_densest_ball = oc.densest_ball(radius=2)
        assert len(idxs_in_densest_ball) == 40
        assert np.linalg.norm(center - self.outliers_, axis=1).min() > \
               np.sqrt(75 ** 2 + 30 ** 2) - np.sqrt(2)

        dist, idxs = oc.kneighbors(
           (self.centers_[0] + np.array([5, 0])).reshape(1, -1), k=41)
        dist, idxs = dist[0], idxs[0]
        assert len(idxs) == 41
        dist.sort()
        assert dist.max() == 5
        assert dist[-2] < 1.5

    def test_DistQueryOracle_brute(self):

        # test oracle based on small data set with brute force search
        oc = DistQueryOracle(tree_algorithm='brute',
                             leaf_size=1, precompute_distances='auto')
        oc.fit(self.toyX_with_outliers, sample_weight=self.random_weight_)
        idxs_in_ball = oc.ball((self.centers_[0] + np.array([5, 0])).reshape(1, -1),
                               radius=2)[0]
        assert len(idxs_in_ball) == 5
        assert idxs_in_ball.max() < 9 and idxs_in_ball.min() > 3

        center, idxs_in_densest_ball = oc.densest_ball(radius=2)
        assert len(idxs_in_densest_ball) == 5
        assert np.linalg.norm(center - self.outliers_, axis=1).min() > \
               np.sqrt(75 ** 2 + 30 ** 2) - np.sqrt(2)

        dist, idxs = oc.kneighbors(
           (self.centers_[0] + np.array([5, 0])).reshape(1, -1), k=6)
        dist, idxs = dist[0], idxs[0]
        assert len(idxs) == 6
        dist.sort()
        assert dist.max() == 5
        assert dist[-2] < 1.5


if __name__ == '__main__':
    unittest.main()
