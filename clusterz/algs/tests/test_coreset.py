import unittest
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from clusterz.algs.kzmeans import KZMeans
from clusterz.algs.coreset import (
    Coreset, DistributedCoreset, DistributedSummary, SummaryOutliers
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

        # pre-clustered centers on a single machine
        self.pre_centers_ = np.array(
            [self.centers_[0] + np.array([5, 0]),
             self.centers_[0] + np.array([-5, 0]),
             self.centers_[1] + np.array([5, 0]),
             self.centers_[1] + np.array([-5, 0])]
        )
        _, self.costs_with_outliers_ = pairwise_distances_argmin_min(self.X_with_outliers_, self.pre_centers_)

    def is_centers_match(self, true_centers, centers, max_dist=0.001, match_shape=True):
        if match_shape:
            assert true_centers.shape == centers.shape
        _, dist = pairwise_distances_argmin_min(true_centers, centers)
        if dist.max() > max_dist:
            print("\ndist.max = {} but max_dist = {}".format(dist.max(), max_dist))
        return dist.max() <= max_dist

    def test_Coreset(self):
        cs = Coreset(30, cost_func=lambda _: self.costs_with_outliers_)
        cs.fit(self.X_with_outliers_)
        cs_idxs, cs_weights = cs.sample_indices, cs.sample_weights

        # outliers have the largest prob to be included in the coreset
        # while having the smallest weight
        assert 2 in cs_idxs or 3 in cs_idxs
        outlier_idx = 2 if 2 in cs_idxs else 3
        idx_of_outlier = np.where(cs_idxs == outlier_idx)[0][0]
        assert cs_weights.min() == cs_weights[idx_of_outlier]

    def test_DistributedCoreset(self):
        dcs = DistributedCoreset(
                sample_size=40,
                pre_clustering_method=KZMeans,
                n_pre_clusters=4,
                cost_func=lambda X, C, element_wise=True: pairwise_distances_argmin_min(X, C)[1])
        dcs.fit(self.Xs_with_outliers_)
        sp_idxs = dcs.sample_indices
        assert len(sp_idxs[0]) + len(sp_idxs[1]) < len(dcs.samples)

        # outliers have the largest prob to be included in the coreset
        # or appeared as the center of pre-clusters.
        assert self.is_centers_match(self.outliers_, dcs.samples, max_dist=0.1, match_shape=False)

    def test_SummaryOutliers(self):
        so = SummaryOutliers(n_clusters=2,
                                n_outliers=2,
                                alpha=2,
                                beta=0.2,
                                augmented=True)
        so.fit(self.X_with_outliers_)
        assert len(so.samples) == len(so.sample_indices) == len(so.weights)

    def test_DistributedSummary(self):
        ds = DistributedSummary(n_clusters=2,
                                n_outliers=2,
                                alpha=2,
                                beta=0.45,
                                adversary=False,
                                augmented=True)
        ds.fit(self.Xs_with_outliers_)
        sp_idxs = ds.sample_indices
        assert len(sp_idxs[0]) + len(sp_idxs[1]) == len(ds.samples)



if __name__ == '__main__':
    unittest.main()
