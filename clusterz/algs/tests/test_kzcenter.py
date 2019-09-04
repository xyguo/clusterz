import unittest
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from numpy.testing import assert_array_equal
from clusterz.algs.kzcenter import (
    KZCenter, kzcenter_brute, kzcenter_charikar, kzcenter_charikar_eg, kcenter_greedy, DistributedKZCenter
)


class MyTestCase(unittest.TestCase):

    def setUp(self):
        small_cluster = np.random.uniform(-1, 1, size=(5, 2))
        large_cluster = np.random.uniform(-1, 1, size=(40, 2))
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
             large_cluster + self.centers_[0] + np.array([5, 0]),
             large_cluster + self.centers_[0] + np.array([-5, 0]),
             large_cluster + self.centers_[1] + np.array([5, 0]),
             large_cluster + self.centers_[1] + np.array([-5, 0])])
        self.X_with_outliers_ = np.vstack(
            [self.centers_,
             self.outliers_,
             # clusters
             large_cluster + self.centers_[0] + np.array([5, 0]),
             large_cluster + self.centers_[0] + np.array([-5, 0]),
             large_cluster + self.centers_[1] + np.array([5, 0]),
             large_cluster + self.centers_[1] + np.array([-5, 0])])
        self.toy_random_weight_ = np.hstack(
            [np.ones(2),
             np.ones(2),
             np.random.uniform(1, 2, len(small_cluster) * 4)])
        self.random_weight_ = np.hstack(
            [np.ones(2),
             np.ones(2),
             np.random.uniform(1, 2, len(large_cluster) * 4)])
        self.uniform_weight_ = np.ones(len(self.X_with_outliers_))

        # data on 2 machines
        self.Xs_without_outliers_ = [
            np.vstack(
                [self.centers_[0],
                 # clusters
                 large_cluster + self.centers_[0] + np.array([5, 0]),
                 large_cluster + self.centers_[1] + np.array([-5, 0])]),
            np.vstack(
               [self.centers_[1],
                # clusters
                large_cluster + self.centers_[0] + np.array([-5, 0]),
                large_cluster + self.centers_[1] + np.array([5, 0]),
               ]
            )]
        self.Xs_with_outliers_ = [
            np.vstack(
                [self.centers_[0],
                 self.outliers_[0],
                 # clusters
                 large_cluster + self.centers_[0] + np.array([5, 0]),
                 large_cluster + self.centers_[1] + np.array([-5, 0])]),
            np.vstack(
               [self.centers_[1],
                self.outliers_[1],
                # clusters
                large_cluster + self.centers_[0] + np.array([-5, 0]),
                large_cluster + self.centers_[1] + np.array([5, 0]),
               ]
            )]
        self.random_weights_ = [
            np.hstack(
            [np.ones(1),
             np.ones(1),
             np.random.uniform(1, 2, len(large_cluster) * 2)]),
            np.hstack(
            [np.ones(1),
             np.ones(1),
             np.random.uniform(1, 2, len(large_cluster) * 2)])
            ]
        self.uniform_weights_ = [np.ones(len(self.Xs_with_outliers_[0])),
                                 np.ones(len(self.Xs_with_outliers_[1]))]

    def is_centers_match(self, true_centers, centers, max_dist=0.001, match_shape=True):
        if match_shape:
            assert true_centers.shape == centers.shape
        _, dist = pairwise_distances_argmin_min(true_centers, centers)
        if dist.max() > max_dist:
            print("\ndist.max = {} but max_dist = {}".format(dist.max(), max_dist))
        return dist.max() <= max_dist

    def test_kzcenter_brute(self):
        clusters = kzcenter_brute(self.toyX_with_outliers,
                                  sample_weight=self.toy_random_weight_,
                                  n_clusters=2, n_outliers=2)
        centers = np.array([cw[0] for cw in clusters])
        assert self.is_centers_match(self.centers_, centers)

        clusters = kzcenter_brute(self.toyX_with_outliers,
                                  sample_weight=None,
                                  n_clusters=2, n_outliers=2)
        centers = np.array([cw[0] for cw in clusters])
        n_covered = np.array([cw[1] for cw in clusters])
        assert self.is_centers_match(self.centers_, centers)
        assert_array_equal(n_covered, np.ones(2) * 11)

    def test_kzcenter_charikar(self):
        clusters = kzcenter_charikar(self.X_with_outliers_,
                                     sample_weight=self.random_weight_,
                                     n_clusters=2, n_outliers=2)
        centers = np.array([cw[0] for cw in clusters])
        # charikar's method is a 3-approx
        assert self.is_centers_match(self.centers_, centers, max_dist=6.001)

    def test_kzcenter_charikar_eg(self):
        clusters = kzcenter_charikar_eg(self.X_with_outliers_,
                                        sample_weight=self.random_weight_, n_clusters=2, n_outliers=2)
        centers = np.array([cw[0] for cw in clusters])
        # charikar's method is a 3-approx
        assert self.is_centers_match(self.centers_, centers, max_dist=6.001)

    def test_kcenter_greedy(self):
        X = self.X_without_outliers_.copy()
        np.random.shuffle(X)
        centers, dist = kcenter_greedy(X, n_clusters=2, return_distance=True)
        assert centers.shape == (2, 2)
        assert dist.max() < 2 * 8.1

    def test_KZCenter(self):
        kzc = KZCenter(algorithm='greedy_covering',
                       n_clusters=2, n_outliers=2,
                       n_machines=2, epsilon=0.1,
                       debug=False)
        kzc.fit(self.X_with_outliers_,
                sample_weight=self.random_weight_,
                dist_oracle=None,
                guessed_opt=6.01)
        centers = kzc.cluster_centers
        assert self.is_centers_match(self.centers_, centers,
                                     max_dist=6.2, match_shape=False)

    def test_DistributedKZCenter(self):
        # test the main algorithm that achieves multiplicative error
        dkzc_m = DistributedKZCenter(algorithm='multiplicative',
                                     n_clusters=2, n_outliers=2, n_machines=2)
        dkzc_m.fit(Xs=self.Xs_with_outliers_,
                   sample_weights=self.random_weights_)
        assert self.is_centers_match(self.centers_, dkzc_m.cluster_centers, max_dist=6.001)

        # test the secondary algorithm that achieves additive error
        dkzc_a = DistributedKZCenter(algorithm='additive', sample_size=64,
                                     n_clusters=2, n_outliers=2, n_machines=2)
        dkzc_a.fit(Xs=self.Xs_with_outliers_,
                   sample_weights=self.uniform_weights_)
        assert self.is_centers_match(self.centers_, dkzc_a.cluster_centers, max_dist=6.001)

        # test the algorithm by moseley
        dkzc_moseley = DistributedKZCenter(algorithm='moseley',
                                           n_clusters=2, n_outliers=2, n_machines=2)
        dkzc_moseley.fit(Xs=self.Xs_with_outliers_,
                         sample_weights=self.random_weights_)
        assert self.is_centers_match(self.centers_, dkzc_moseley.cluster_centers, max_dist=6.001)

if __name__ == '__main__':
    unittest.main()
