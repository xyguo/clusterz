# -*- coding: utf-8 -*-
"""Algorithm for distributed and centralized (k,z)-median, k-median"""

# Author: Xiangyu Guo     xiangyug@buffalo.edu
#         Shi Li          shil@buffalo.edu

import numpy as np

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin_min
from sklearn.utils import check_array

from .kz_lp_clustering import DistributedKZLpClustering, DistributedLpClustering
from ..utils import compute_cost


def kzmedian_cost_(X, C, sample_weights=None, n_outliers=0, L=None, element_wise=False):
    """
    :param X: array of shape=(n_samples, n_features), data set
    :param C: array of shape=(n_centers, n_features), centers
    :param sample_weights: array of shape=(n_samples,), sample weights
    :param n_outliers: int, number of outliers
    :param L: None or float. if not None then all distances larger than L will be truncated
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


def kmedian_pp_(X, sample_weights, n_clusters):
    """
    (weighted) k-median++ initialization

    :param X: array of shape=(n_samples, n_features)
    :param sample_weights: array of shape=(n_samples,)
    :param n_clusters: int, number of clusters
    :return centers: array of shape=(n_clusters, n_features)
    """
    n_samples, _ = X.shape
    first_idx = np.random.randint(0, n_samples)
    centers = [X[first_idx]]
    for i in range(n_clusters - 1):
        _, dist = pairwise_distances_argmin_min(X, centers)
        probs = normalize((dist * sample_weights).reshape(1, -1), norm='l1')[0]
        next_idx = np.random.choice(n_samples, 1, replace=True, p=probs)[0]
        centers.append(X[next_idx])
    return np.array(centers)


def update_clusters_(X, centers, return_dist=False):
    """assign each point in X to its nearest center

    :param X: array of shape=(n_samples, n_features), data set.
    :param centers: array of shape=(n_centers, n_features), center points
    :param return_dist: whether to return the distances of each point to its nearest center
    :return clusters: list of arrays, each array consists of the indices
        for data in the same cluster. If some cluster has size less than 2 then
        it is ignored.

    """
    n_centers = len(centers)
    idxs, dists = pairwise_distances_argmin_min(X, centers)
    clusters = []
    for i in range(n_centers):
        clusters.append(np.where(idxs == i)[0])
    clusters = [c for c in clusters if len(c) >= 1]
    return (clusters, dists) if return_dist else clusters


def update_centers_(X, sample_weights, clusters, outliers=None):
    """Compute the new centers to be the weighted mean of each cluster

    :param X: array of shape=(n_samples, n_features), data set.
    :param sample_weights: array of shape=(n_samples,), weights
    :param clusters: list of arrays, each array of the indices for data in the same cluster.
    :param outliers: array of shape=(n_outliers,), indices of outliers
    :return centers: array of shape=(n_clusters, n_features)
    """
    _, n_features = X.shape
    centers = []
    for c in clusters:
        if outliers is not None:
            c = list(set(c).difference(outliers))
        # find the weighted median
        X_c, sw_c = X[c].T, sample_weights[c]
        sorted_ft_idxs = np.argsort(X_c, axis=1)
        sorted_sw = sw_c[sorted_ft_idxs]
        sorted_cum_sw = np.cumsum(sorted_sw, axis=1)
        wm_idx = np.apply_along_axis(lambda a: a.searchsorted(a[-1]/2),
                                     axis=1, arr=sorted_cum_sw)
        wm_idx_in_X_c = sorted_ft_idxs[(np.arange(n_features), wm_idx)]
        centers.append(X_c[(np.arange(n_features), wm_idx_in_X_c)])

    return np.array(centers)


def kmedian_mm_(X, sample_weights, n_clusters, n_outliers):
    """
    Weighted K-Median-- implementation.
    :param X:
    :param sample_weights:
    :param n_clusters:
    :param n_outliers:
    :return cluster_centers_:
    """
    n_samples, n_features = X.shape
    # TODO: find a better way to handle negtive weights
    centers_idxs = np.random.choice(n_samples, n_clusters, replace=False)
    cluster_centers_ = np.atleast_2d(X[centers_idxs])

    diff = np.inf
    while diff > 1e-3:
        clusters, dists = update_clusters_(X, cluster_centers_, return_dist=True)

        # ignore the outliers when updating centers
        if n_outliers > 0:
            outliers = np.argsort(dists)[-n_outliers:]
        else:
            outliers = None
        new_centers = update_centers_(X, sample_weights, clusters, outliers=outliers)
        diff = np.linalg.norm(new_centers - cluster_centers_)
        cluster_centers_ = new_centers

    # if the program finishes before finding k'<k centers, we use the FarthestNeighbor
    # method to produce the remained k-k' centers
    if len(cluster_centers_) < n_clusters:
        centers = [c for c in cluster_centers_]
        _, dists_to_centers = pairwise_distances_argmin_min(X, np.atleast_2d(centers))

        for i in range(0, n_clusters - len(cluster_centers_)):
            next_idx = np.argmax(dists_to_centers)
            centers.append(X[next_idx])
            _, next_dist = pairwise_distances_argmin_min(X, np.atleast_2d(centers[-1]))
            dists_to_centers = np.minimum(dists_to_centers, next_dist)
        cluster_centers_ = np.array(centers)

    return cluster_centers_


def kz_median(X, n_clusters, n_outliers, sample_weights=None):
    """ K-Median-- """
    n_samples, _ = X.shape
    if sample_weights is None:
        sample_weights = np.ones(n_samples)
    return kmedian_mm_(X=X, sample_weights=sample_weights,
                       n_clusters=n_clusters, n_outliers=n_outliers)


def kmedian_(X, sample_weights, n_clusters, init='kmedian++'):
    """
    Weighted K-Means implementation.
    :param X:
    :param sample_weights:
    :param n_clusters:
    :param init: string in {'random', 'kmeans++'}, default 'kmeans++'
    :return cluster_centers_:
    """
    n_samples, n_features = X.shape
    # TODO: find a better way to handle negtive weights

    cluster_centers_ = None
    if init == 'kmedian++':
        cluster_centers_ = kmedian_pp_(X, np.clip(sample_weights, 0, np.inf), n_clusters)
    elif init == 'random':
        centers_idxs = np.random.choice(n_samples, n_clusters, replace=False)
        cluster_centers_ = X[centers_idxs]
    elif isinstance(init, np.ndarray):
        cluster_centers_ = init

    diff = np.inf
    while diff > 1e-3:
        clusters = update_clusters_(X, cluster_centers_)
        new_centers = update_centers_(X, sample_weights, clusters)
        diff = np.linalg.norm(new_centers - cluster_centers_)
        cluster_centers_ = new_centers

    # if the program finishes before finding k'<k centers, we use the FarthestNeighbor
    # method to produce the remained k-k' centers
    if len(cluster_centers_) < n_clusters:
        centers = [c for c in cluster_centers_]
        _, dists_to_centers = pairwise_distances_argmin_min(X, np.atleast_2d(centers))

        for i in range(0, n_clusters - len(cluster_centers_)):
            next_idx = np.argmax(dists_to_centers)
            centers.append(X[next_idx])
            _, next_dist = pairwise_distances_argmin_min(X, np.atleast_2d(centers[-1]))
            dists_to_centers = np.minimum(dists_to_centers, next_dist)
        cluster_centers_ = np.array(centers)

    return cluster_centers_


def k_median_my(X, n_clusters, sample_weights=None):
    """ K-Median"""
    n_samples, _ = X.shape
    if sample_weights is None:
        sample_weights = np.ones(n_samples)
    return kmedian_(X=X, sample_weights=sample_weights,
                    n_clusters=n_clusters)


def kmedian_cost_no_outlier_(X, C, sample_weights=None,
                             n_outliers=0, L=None, element_wise=False):
    return kzmedian_cost_(X, C, sample_weights=sample_weights,
                          n_outliers=0, L=None, element_wise=element_wise)


class KZMedian(object):
    """ A wrapper for the kz_median function to support sklearn interface """
    def __init__(self, n_clusters, n_outliers=0):
        self.n_clusters_ = n_clusters
        self.n_outliers_ = n_outliers
        self.cluster_centers_ = None

    def fit(self, X):
        self.cluster_centers_ = kz_median(X, self.n_clusters_, self.n_outliers_)
        return self

    def predict(self, X):
        nearest, _ = pairwise_distances_argmin_min(X, self.cluster_centers_)
        return nearest


class KMedianWrapped(object):

    def __init__(self, n_clusters):
        self.n_clusters_ = n_clusters
        self.cluster_centers_ = None
        self.cost_func_ = kzmedian_cost_

    def cost(self, X, remove_outliers=True):
        """

        :param X: array of shape=(n_samples, n_features),
            data set
        :param remove_outliers: None or int, default None
            whether to remove outliers when computing the cost on X
        :return: float,
            actual cost
        """
        return compute_cost(X, cluster_centers=self.cluster_centers_,
                            cost_func=kmedian_cost_no_outlier_,
                            remove_outliers=remove_outliers)

    def fit(self, X):
        self.cluster_centers_ = k_median_my(X, self.n_clusters_)
        return self

    def predict(self, X):
        nearest, _ = pairwise_distances_argmin_min(X, self.cluster_centers_)
        return nearest


class BELDistributedKMedian(DistributedLpClustering):
    """
    Maria Florina Balcan, Steven Ehrlich, Yingyu Liang.
    Distributed k-Means and k-Median Clustering on General Topologies.
    NIPS'13.
    """
    def __init__(self,
                 n_clusters=None, n_machines=None,
                 pre_clustering_routine=None, n_pre_clusters=None,
                 epsilon=0.1, delta=0.01, coreset_ratio=10,
                 random_state=None, debug=False):
        super().__init__(p=1, local_clustering_method=kmedian_,
                         cost_func=kmedian_cost_no_outlier_, pairwise_dist_func=None,
                         n_clusters=n_clusters, n_machines=n_machines,
                         pre_clustering_routine=pre_clustering_routine,
                         n_pre_clusters=n_pre_clusters,
                         epsilon=epsilon, delta=delta, coreset_ratio=coreset_ratio,
                         random_state=random_state, debug=debug)





