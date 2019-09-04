import unittest
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from clusterz.algs.kzmeans import (
    DistributedKZMeans, BELDistributedKMeans, k_means_my, kz_means, KZMeans, KMeansWrapped
)


class MyTestCase(unittest.TestCase):
    def setUp(self):
        cluster = np.random.uniform(-1, 1, size=(40, 2))
        self.centers_ = np.array([
            [0, 30], [0, -30]
        ])
        self.outliers_ = np.array([
            [80, 0], [-80, 0]
        ])

        # data set on a single machine
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
        self.uniform_weight_ = np.ones(len(self.X_with_outliers_))

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
        self.random_weights_ = [
            np.hstack(
            [np.ones(1),
             np.ones(1),
             np.random.uniform(1, 2, len(cluster) * 2)]),
            np.hstack(
            [np.ones(1),
             np.ones(1),
             np.random.uniform(1, 2, len(cluster) * 2)])
            ]
        self.uniform_weights_ = [np.ones(len(self.Xs_with_outliers_[0])),
                                 np.ones(len(self.Xs_with_outliers_[1]))]

    def is_centers_match(self, true_centers, centers, max_dist=0.001, match_shape=True):
        if match_shape:
            assert true_centers.shape == centers.shape
        _, dist = pairwise_distances_argmin_min(true_centers, centers)
        if dist.max() > max_dist:
            print("\ndist.max = {} but max_dist = {}".format(dist.max(), max_dist))
        assert dist.max() <= max_dist

    def test_kmeans_my(self):
        centers = k_means_my(self.X_without_outliers_, n_clusters=2,
                             sample_weights=self.random_weight_without_outliers_)
        self.is_centers_match(self.centers_, centers, max_dist=3.001)

    def test_kzmeans(self):
        centers = kz_means(self.X_with_outliers_, n_clusters=2, n_outliers=2,
                           sample_weights=self.random_weight_)
        self.is_centers_match(self.centers_, centers, max_dist=3.001)

    def test_DistributedKZMeans(self):
        dkzm = DistributedKZMeans(
            n_clusters=2, n_outliers=2,
            n_machines=2, pre_clustering_routine=KZMeans,
            n_pre_clusters=2, epsilon=0.3,
            random_state=None, debug=False)
        dkzm.fit(self.Xs_with_outliers_)
        self.is_centers_match(self.centers_, dkzm.cluster_centers, max_dist=6.001)

    def test_BELDistributedKMeans(self):
        bel = BELDistributedKMeans(
            n_clusters=2, n_machines=2,
            pre_clustering_routine=KMeansWrapped, n_pre_clusters=2,
            epsilon=0.3, coreset_ratio=5,
            random_state=None, debug=False
        )
        bel.fit(self.Xs_without_outliers_)
        self.is_centers_match(self.centers_, bel.cluster_centers, max_dist=6.001)


if __name__ == '__main__':
    unittest.main()
