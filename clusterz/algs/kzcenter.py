# -*- coding: utf-8 -*-
"""Algorithm for distributed (k,z)-center"""

# Author: Xiangyu Guo     xiangyug@buffalo.edu
#         Yunus Esencayi  yunusese@buffalo.edu
#         Shi Li          shil@buffalo.edu

import warnings
import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KDTree, BallTree, LSHForest

from .misc import DistQueryOracle


class DistributedKZCenter(object):

    def __init__(self,
                 n_clusters=None, n_outliers=None, n_machines=None,
                 epsilon=0.1, random_state=None):
        """
        Serves as the master node

        :param n_clusters: integer.
            Number of clusters.

        :param n_outliers: integer.
            Number of outliers

        :param n_machines: integer.
            Number of machines

        :param epsilon: float.
            error tolerance parameter.

        :param random_state: numpy.RandomState
        """
        self.n_clusters = n_clusters
        self.n_outliers = n_outliers
        self.n_machines = n_machines
        self.epsilon = epsilon
        self.random_state = random_state

        self.pre_clusters_ = None

    def fit(self, Xs, sample_weights=None):
        if sample_weights is None:
            sample_weights = [None] * len(Xs)

        self.n_machines = len(Xs)
        mappers = []
        for _ in range(self.n_machines):
            mappers.append(KZCenter(
                is_mapper=True, is_reducer=False,
                n_clusters=self.n_clusters, n_outliers=self.n_outliers,
                n_machines=self.n_machines, epsilon=self.epsilon,
                oracle=DistQueryOracle(tree_algorithm='ball_tree')
            ))

        # estimating the optimal radius using binary search
        total_iters_threshold = self.n_outliers * self.n_machines * (1 + 1 / self.epsilon)
        lb = 0
        ub = sum(m.estimate_diameter(n_estimation=10)[1] for m in mappers)
        guessed_opt = (lb + ub) / 2
        while ub > (1+self.epsilon) * lb:
            for i, m in enumerate(mappers):
                m.fit(guessed_opt, Xs[i], sample_weights[i])
            total_iters = sum(m.n_iters for m in mappers)
            if total_iters > total_iters_threshold:
                lb = guessed_opt
            else:
                ub = guessed_opt
            guessed_opt = (lb + ub) / 2

        # construct the centralized dataset
        for m in mappers:
            m.fit(guessed_opt=ub)
        X = np.array([m.results[0] for m in mappers])
        sample_weight = np.array([m.results[1] for m in mappers])

        reducer = KZCenter(is_mapper=False, is_reducer=True,
                           n_clusters=self.n_clusters,
                           n_outliers=(1 + self.epsilon) * self.n_outliers,
                           n_machines=self.n_machines, random_state=self.random_state)
        reducer.fit(5 * ub, X, sample_weight)

        return self


class KZCenter(object):

    def __init__(self,
                 is_mapper=True, is_reducer=False,
                 n_clusters=None, n_outliers=None, n_machines=None,
                 epsilon=0.1, random_state=None, oracle=None):
        """

        :param is_mapper: boolean.
            True if the current object act as a mapper.

        :param is_reducer: boolean.
            True if the current object act as a reducer.

        :param n_clusters: integer.
            Number of clusters.

        :param n_outliers: integer.
            Number of outliers

        :param n_machines: integer.
            Number of machines

        :param epsilon: float.
            error tolerance parameter.

        :param random_state: numpy.RandomState

        :param oracle: DistQueryOracle object,
            used for distance query
        """
        self.is_mapper = is_mapper
        self.is_reducer = is_reducer
        self.n_clusters = n_clusters
        self.n_outliers = n_outliers
        self.n_machines = n_machines
        self.epsilon = epsilon
        self.random_state = random_state
        self.dist_oracle = oracle

    def fit(self, guessed_opt, X, sample_weight=None):
        """

        :param X:
        :param sample_weight:
        :return:
        """
        if not self.dist_oracle.is_fitted:
            self.dist_oracle.fit(X, sample_weight)

        if self.is_mapper:
            self.pre_clustering(guessed_opt)
        elif self.is_reducer:
            kzcenter_charikar(X,
                              sample_weight=sample_weight,
                              n_clusters=self.n_clusters,
                              n_outliers=self.n_outliers,
                              dist_oracle=self.dist_oracle)
        else:
            raise AttributeError(
                "Unclear role: not sure whether current object is a mapper or a reducer.\n"
            )
        return self

    def pre_clustering(self, guessed_opt):
        """

        :param guessed_opt:
        :return:
        """
        threshold = self.epsilon * self.n_outliers / (self.n_clusters * self.n_machines)
        results = []
        removed = {}
        self.n_iters = 1
        while True:
            p, covered_pts = self.dist_oracle.densest_ball(2 * guessed_opt,
                                                           except_for=removed)
            if len(covered_pts) < threshold:
                break
            removed.update(self.dist_oracle.ball(p, 4 * guessed_opt))
            w_p = len(removed)
            results.append((p, w_p))
            self.n_iters += 1
        return None


def kzcenter_charikar(X, sample_weight=None, n_clusters=7, n_outliers=0, epsilon=0.1, dist_oracle=None):
    """

    :param X:
    :param sample_weight:
    :param n_clusters:
    :param n_outliers:
    :param epsilon:
    :param dist_oracle:
    :return:
    """
    if dist_oracle is None:
        dist_oracle = DistQueryOracle(tree_algorithm='ball_tree')
    if not dist_oracle.is_fitted:
        dist_oracle.fit(X, sample_weight)
    n_samples, _ = X.shape

    results = []

    # estimate the upperbound and lowerbound of the data set
    _, ub = dist_oracle.estimate_diameter(n_estimation=10)
    lb, _ = dist_oracle.nearest_neighbor(X[np.random.randint(0, n_samples)])
    removed = set()
    guessed_opt = lb
    while ub > (1+epsilon) * lb:
        for i in range(n_clusters):
            p, covered_pts = dist_oracle.find_densest_ball(X, 2 * guessed_opt, removed)
            removed.update(dist_oracle.ball(X, p, 4 * guessed_opt))
            w_p = len(removed)
            results.append((p, w_p))
        n_covered = sum(wp for _, wp in results)
        if n_covered >= n_samples - n_outliers:
            ub = guessed_opt
            guessed_opt = (lb + guessed_opt) / 2
        else:
            lb = guessed_opt
            guessed_opt = (guessed_opt + ub) / 2
    return results



