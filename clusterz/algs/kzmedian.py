# -*- coding: utf-8 -*-
"""Algorithm for distributed (k,z)-median"""

# Author: Xiangyu Guo     xiangyug@buffalo.edu
#         Yunus Esencayi  yunusese@buffalo.edu
#         Shi Li          shil@buffalo.edu

import numpy as np

from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin_min
from sklearn.utils import check_array

from .kz_lp_clustering import DistributedKZLpClustering, DistributedLpClustering


def kzmedian_cost_(X, C, sample_weights=None, n_outliers=0, L=None, element_wise=False):
    """
    :param X: array of shape=(n_samples, n_features)
    :param C: array of shape=(n_centers, n_features)
    :param sample_weights: array of shape=(n_samples,)
    :param n_outliers: int
    :param L:
    :param element_wise: bool, whether to return the cost for each element in X
    :return:
    """
    X = check_array(X, ensure_2d=True)
    C = check_array(C, ensure_2d=True)
    _, dists = pairwise_distances_argmin_min(X, C, axis=1)
    if L is None:
        if sample_weights is not None:
            dists *= sample_weights
        if element_wise:
            return dists
        dists.sort()
        return sum(dists[:-n_outliers]) if n_outliers > 0 else np.sum(dists)
    else:
        np.minimum(dists, L, out=dists)
        if sample_weights is not None:
            dists *= sample_weights
        if element_wise:
            return dists
        dist = dists.sum() - n_outliers * L
        return dist


class DistributedKZMedian(DistributedKZLpClustering):

    def __init__(self,
                 n_clusters=None, n_outliers=None, n_machines=None,
                 pre_clustering_routine=None, n_pre_clusters=None,
                 epsilon=0.1, delta=0.01, random_state=None, debug=False):
        """
        Serves as the master node.

        :param n_clusters: integer.
            Number of clusters.

        :param n_outliers: integer.
            Number of outliers

        :param pre_clustering_routine: function or None,
            Subroutine used for generating the pre-clusters for coreset sampling.

        :param n_machines: integer.
            Number of machines

        :param epsilon: float.
            error tolerance parameter for the number of outliers.

        :param delta: float.
            error tolerance parameter for the success probability

        :param random_state: numpy.RandomState

        :param debug: boolean, whether output debugging information
        """
        super().__init__(p=1,
                         cost_func=kzmedian_cost_, pairwise_dist_func=pairwise_distances,
                         n_clusters=n_clusters, n_outliers=n_outliers, n_machines=n_machines,
                         pre_clustering_routine=pre_clustering_routine,
                         n_pre_clusters=n_pre_clusters,
                         epsilon=epsilon, delta=delta, random_state=random_state, debug=debug)


def kmedian_(X, sample_weights):
    pass


class DistributedKMedian(DistributedLpClustering):

    def __init__(self,
                 n_clusters=None, n_machines=None,
                 pre_clustering_routine=None, n_pre_clusters=None,
                 epsilon=0.1, delta=0.01, random_state=None, debug=False):
        def kmedian_cost_no_outlier_(X, C, sample_weights=None, n_outliers=0, L=None):
            return kzmedian_cost_(X, C, sample_weights=sample_weights,
                                  n_outliers=0, L=None)
        super().__init__(p=1, local_clustering_method=kmedian_,
                         cost_func=kmedian_cost_no_outlier_, pairwise_dist_func=None,
                         n_clusters=n_clusters, n_machines=n_machines,
                         pre_clustering_routine=pre_clustering_routine,
                         n_pre_clusters=n_pre_clusters,
                         epsilon=epsilon, delta=delta,
                         random_state=random_state, debug=debug)





