# -*- coding: utf-8 -*-
"""Algorithm for distributed and centralized (k,z)-means, k-means"""

# Author: Xiangyu Guo     xiangyug@buffalo.edu
#         Shi Li          shil@buffalo.edu

import warnings
import numpy as np

import numpy as np

# from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances_argmin_min
from sklearn.utils import check_array
from sklearn.exceptions import NotFittedError

from .kz_lp_clustering import DistributedKZLpClustering, DistributedLpClustering
from ..utils import compute_cost


def kzmeans_cost_(X, C, sample_weights=None, n_outliers=0, L=None, element_wise=False):
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
    _, dists = pairwise_distances_argmin_min(X, C, axis=1, metric='euclidean',
                                             metric_kwargs={'squared': True})
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


def pairwise_squared_dist_(X, C):
    return euclidean_distances(X=X, Y=C, squared=True)


class DistributedKZMeans(DistributedKZLpClustering):

    def __init__(self,
                 n_clusters=None, n_outliers=None, n_machines=None,
                 pre_clustering_routine=None, n_pre_clusters=None,
                 epsilon=0.1, delta=0.01, coreset_ratio=10,
                 random_state=None, debug=False):
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

        :param coreset_ratio: how large should the coreset be,
            the sample size will be n_samples / coreset_ratio

        :param random_state: numpy.RandomState

        :param debug: boolean, whether output debugging information
        """
        super().__init__(p=2,
                         cost_func=kzmeans_cost_, pairwise_dist_func=pairwise_squared_dist_,
                         n_clusters=n_clusters, n_outliers=n_outliers, n_machines=n_machines,
                         pre_clustering_routine=pre_clustering_routine,
                         n_pre_clusters=n_pre_clusters, epsilon=epsilon, delta=delta,
                         coreset_ratio=coreset_ratio, random_state=random_state, debug=debug)


def kmeans_pp_(X, sample_weights, n_clusters):
    """
    (weighted) k-means++ initialization

    :param X: array of shape=(n_samples, n_features)
    :param sample_weights: array of shape=(n_samples,)
    :param n_clusters:
    :return centers: array of shape=(n_clusters, n_features)
    """
    n_samples, _ = X.shape
    first_idx = np.random.randint(0, n_samples)
    centers = [X[first_idx]]
    for i in range(n_clusters - 1):
        _, dist = pairwise_distances_argmin_min(X, centers)
        probs = normalize(((dist ** 2) * sample_weights).reshape(1, -1), norm='l1')[0]
        next_idx = np.random.choice(n_samples, 1, replace=True, p=probs)[0]
        centers.append(X[next_idx])
    return np.array(centers)


def update_clusters_(X, centers, return_dist=False):
    """

    :param X:
    :param centers:
    :param return_dist: whether to return the distances of each point to its nearest center
    :return clusters: list of arrays, each array consists of the indices
        for data in the same cluster. If some cluster has size less than 2 then
        it is ignored.

    """
    n_centers = len(centers)
    idxs, dists = pairwise_distances_argmin_min(X, centers)
    clusters = []
    # dists_collected = []
    for i in range(n_centers):
        clusters.append(np.where(idxs == i)[0])
        # dists_collected.append(dists[clusters[-1]])
    clusters = [c for c in clusters if len(c) >= 1]
    return (clusters, dists) if return_dist else clusters


def update_centers_(X, sample_weights, clusters, outliers=None):
    """

    :param X:
    :param sample_weights:
    :param clusters:
    :return centers:
    """
    centers = []
    for c in clusters:
        if outliers is not None:
            c = list(set(c).difference(outliers))
        if len(c) > 0:
            centers.append(np.average(X[c], axis=0, weights=sample_weights[c]))
    return np.array(centers)


def kmeans_(X, sample_weights, n_clusters, init='kmeans++', max_iter=300):
    """
    Weighted K-Means implementation (Lloyd's Algorithm).
    :param X:
    :param sample_weights:
    :param n_clusters:
    :param init: string in {'random', 'kmeans++'}, default 'kmeans++'
    :param max_iter: maximum number of iterations
    :return cluster_centers_:
    """
    n_samples, n_features = X.shape
    # TODO: find a better way to handle negtive weights

    cluster_centers_ = None
    if init == 'kmeans++':
        cluster_centers_ = kmeans_pp_(X, np.clip(sample_weights, 0, np.inf), n_clusters)
    elif init == 'random':
        centers_idxs = np.random.choice(n_samples, n_clusters, replace=False)
        cluster_centers_ = X[centers_idxs]
    elif isinstance(init, np.ndarray):
        cluster_centers_ = init

    diff = np.inf
    i = 0
    while diff > 1e-3 and i < max_iter:
        clusters = update_clusters_(X, cluster_centers_)
        new_centers = update_centers_(X, sample_weights, clusters)
        if len(new_centers) == len(cluster_centers_):
            diff = np.linalg.norm(new_centers - cluster_centers_)
        cluster_centers_ = new_centers
        i += 1

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


def k_means_my(X, n_clusters, sample_weights=None):
    """ K-Means """
    n_samples, _ = X.shape
    if sample_weights is None:
        sample_weights = np.ones(n_samples)
    return kmeans_(X=X, sample_weights=sample_weights,
                   n_clusters=n_clusters)


def kmeans_mm_(X, sample_weights, n_clusters, n_outliers, init='kmeans++', max_iter=300):
    """
    Weighted K-Means-- implementation.

    Sanjay Chawla and Aristides Gionis.
    k-means−−: A unified approach to clustering and outlier detection.
    In Proceedings of the 13th SIAM International Conference on Data Mining, 2013.
    :param X:
    :param sample_weights:
    :param n_clusters:
    :param n_outliers:
    :param init: string in {'random', 'kmeans++'}, default 'kmeans++'
    :param max_iter: maximum number of iterations
    :return cluster_centers_:
    """
    n_samples, n_features = X.shape
    # TODO: find a better way to handle negtive weights
    cluster_centers_ = None
    if init == 'kmeans++':
        cluster_centers_ = kmeans_pp_(X, np.clip(sample_weights, 0, np.inf), n_clusters)
    elif init == 'random':
        centers_idxs = np.random.choice(n_samples, n_clusters, replace=False)
        cluster_centers_ = X[centers_idxs]
    elif isinstance(init, np.ndarray):
        cluster_centers_ = init

    # centers_idxs = np.random.choice(n_samples, n_clusters, replace=False)
    # cluster_centers_ = X[centers_idxs]

    diff = np.inf
    i = 0
    while diff > 1e-3 and i < max_iter:
        clusters, dists = update_clusters_(X, cluster_centers_, return_dist=True)

        # ignore the outliers when updating centers
        outliers = np.argsort(dists)[-n_outliers:]
        new_centers = update_centers_(X, sample_weights, clusters, outliers=outliers)
        if len(new_centers) == len(cluster_centers_):
            diff = np.linalg.norm(new_centers - cluster_centers_)
        cluster_centers_ = new_centers
        i += 1

    # if the program finishes before finding k'<k centers, we use the FarthestNeighbor
    # method to produce the remained k-k' centers
    if len(cluster_centers_) < n_clusters:
        centers = [c for c in cluster_centers_]
        _, dists_to_centers = pairwise_distances_argmin_min(X, np.atleast_2d(centers))

        for i in range(0, n_clusters - len(cluster_centers_)):
            # next_idx = np.argmax(dists_to_centers)
            # Pick the (n_outliers + 1)-th farthest point as the new center
            far_to_nearest = np.argsort(dists_to_centers)[::-1]
            cum_weights = np.cumsum(sample_weights[far_to_nearest])
            next_idx = far_to_nearest[np.searchsorted(cum_weights, n_outliers)+1]
            ###
            centers.append(X[next_idx])
            _, next_dist = pairwise_distances_argmin_min(X, np.atleast_2d(centers[-1]))
            dists_to_centers = np.minimum(dists_to_centers, next_dist)
        cluster_centers_ = np.array(centers)

    return cluster_centers_


def kz_means(X, n_clusters, n_outliers, sample_weights=None):
    """ K-Means-- """
    n_samples, _ = X.shape
    if sample_weights is None:
        sample_weights = np.ones(n_samples)
    return kmeans_mm_(X=X, sample_weights=sample_weights, init='random',
                      n_clusters=n_clusters, n_outliers=n_outliers)


def kmeans_cost_no_outlier_(X, C, sample_weights=None, n_outliers=0, L=None, element_wise=False):
    return kzmeans_cost_(X, C, sample_weights=sample_weights,
                         n_outliers=0, L=None, element_wise=element_wise)


class KZMeans(object):
    """ A wrapper for the kz_means function to support sklearn interface """
    def __init__(self, n_clusters, n_outliers=0):
        self.n_clusters_ = n_clusters
        self.n_outliers_ = n_outliers
        self.cluster_centers_ = None
        self.cost_func_ = kzmeans_cost_

    def cost(self, X, remove_outliers=True):
        """

        :param X: array,
            data set
        :param remove_outliers: None or int, default None
            whether to remove outliers when computing the cost on X
        :return: float,
            actual cost
        """
        return compute_cost(X, cluster_centers=self.cluster_centers_,
                            cost_func=kzmeans_cost_,
                            remove_outliers=remove_outliers)

    def fit(self, X):
        self.cluster_centers_ = kz_means(X, self.n_clusters_, self.n_outliers_)
        return self

    def predict(self, X):
        nearest, _ = pairwise_distances_argmin_min(X, self.cluster_centers_)
        return nearest


class KMeansWrapped(object):

    def __init__(self, n_clusters):
        self.n_clusters_ = n_clusters
        self.cluster_centers_ = None
        self.cost_func_ = kzmeans_cost_
        # self.cluster_routine_ = KMeans(n_clusters)

    def cost(self, X, remove_outliers=True):
        """

        :param X: array,
            data set
        :param remove_outliers: None or int, default None
            whether to remove outliers when computing the cost on X
        :return: float,
            actual cost
        """
        return compute_cost(X, cluster_centers=self.cluster_centers_,
                            cost_func=kmeans_cost_no_outlier_,
                            remove_outliers=remove_outliers)

    def fit(self, X):
        self.cluster_centers_ = k_means_my(X, self.n_clusters_)
        return self

    def predict(self, X):
        nearest, _ = pairwise_distances_argmin_min(X, self.cluster_centers_)
        return nearest

    # def fit(self, X):
    #     self.cluster_routine_.fit(X)
    #     self.cluster_centers_ = self.cluster_routine_.cluster_centers_
    #     return self
    #
    # def predict(self, X):
    #     return self.cluster_routine_.predict(X)


class BEL_DistributedKMeans(DistributedLpClustering):
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

        super().__init__(p=2,
                         local_clustering_method=lambda X, sw, k: kmeans_(X, sample_weights=sw, n_clusters=k),
                         cost_func=kmeans_cost_no_outlier_, pairwise_dist_func=None,
                         n_clusters=n_clusters, n_machines=n_machines,
                         pre_clustering_routine=pre_clustering_routine,
                         n_pre_clusters=n_pre_clusters,
                         epsilon=epsilon, delta=delta, coreset_ratio=coreset_ratio,
                         random_state=random_state, debug=debug)


