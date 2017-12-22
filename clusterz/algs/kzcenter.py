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
        for X, sw in zip(Xs, sample_weights):
            mappers.append(KZCenter(
                X, sample_weight=sw,
                is_mapper=True, is_reducer=False,
                n_clusters=self.n_clusters, n_outliers=self.n_outliers,
                n_machines=self.n_machines, epsilon=self.epsilon
            ))

        # estimating the optimal radius using binary search
        total_iters_threshold = self.n_outliers * self.n_machines * (1 + 1 / self.epsilon)
        lb, ub = None, None
        guessed_opt = (lb + ub) / 2
        while ub > (1+self.epsilon) * lb:
            for m in mappers:
                m.fit()
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

        reducer = KZCenter(
                           is_mapper=False, is_reducer=True,
                           n_clusters=self.n_clusters,
                           n_outliers=(1 + self.epsilon) * self.n_outliers,
                           n_machines=self.n_machines, random_state=self.random_state)
        reducer.fit(5 * ub)

        return self


class KZCenter(object):

    def __init__(self,
                 is_mapper=True, is_reducer=False,
                 n_clusters=None, n_outliers=None, n_machines=None,
                 epsilon=0.1, random_state=None):
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

        :param tree: string in {'auto', 'kd_tree', 'ball_tree', 'brute'}
            determines the

        :param leaf_size: int, default 40
            leaf size passed to BallTree or KDTree

        :param metric: string, default 'minkowski'
            the distance metric to use in the tree
        """
        self.data = X
        self.sample_weight = sample_weight
        self.is_mapper = is_mapper
        self.is_reducer = is_reducer
        self.n_clusters = n_clusters
        self.n_outliers = n_outliers
        self.n_machines = n_machines
        self.epsilon = epsilon
        self.random_state = random_state
        self.tree_algorithm = tree_algorithm

    def fit(self, guessed_opt, X, sample_weight=None):
        """

        :param X:
        :param sample_weight:
        :return:
        """
        if self.is_mapper:
            self.pre_clustering(guessed_opt, X, sample_weight)
        elif self.is_reducer:
            kzcenter_charikar(X,
                              sample_weight=sample_weight,
                              n_clusters=self.n_clusters,
                              n_outliers=self.n_outliers)
        else:
            raise AttributeError(
                "Unclear role: not sure whether current object is a mapper or a reducer.\n"
            )
        return self

    def pre_clustering(self, guessed_opt, X, sample_weight=None):
        """

        :param X:
        :param sample_weight:
        :return:
        """
        threshold = self.epsilon * self.n_outliers / (self.n_clusters * self.n_machines)
        results = []
        self.n_iters = 1
        while True:
            p, covered_pts = find_densest_ball(X, 2 * self.guessed_opt)
            if len(covered_pts) < threshold:
                break
            E = ball(X, p, 4 * self.guessed_opt)
            w_p = len(E)
            results.append((p, w_p))
            X = X - E
            self.n_iters += 1
        return None


def kzcenter_charikar(X, sample_weight=None, n_clusters=7, n_outliers=0, epsilon=0.1):
    """

    :param X:
    :param sample_weight:
    :param n_clusters:
    :param n_outliers:
    :param epsilon:
    :return:
    """
    n_samples, _ = X.shape
    results = []
    lb, ub = estimate_opt_range(X, sample_weight)
    guessed_opt = lb
    while ub > (1+epsilon) * lb:
        for i in range(n_clusters):
            p, covered_pts = find_densest_ball(X, 2 * guessed_opt)
            E = ball(X, p, 4 * guessed_opt)
            w_p = len(E)
            results.append((p, w_p))
            X = X - E
        n_covered = sum(wp for _, wp in results)
        if n_covered >= n_samples - n_outliers:
            ub = guessed_opt
            guessed_opt = (lb + L) / 2
        else:
            lb = guessed_opt
            guessed_opt = (L + ub) / 2
    return results



