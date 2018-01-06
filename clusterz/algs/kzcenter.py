# -*- coding: utf-8 -*-
"""Algorithm for distributed (k,z)-center"""

# Author: Xiangyu Guo     xiangyug@buffalo.edu
#         Yunus Esencayi  yunusese@buffalo.edu
#         Shi Li          shil@buffalo.edu

import warnings
import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array
from .misc import DistQueryOracle, distributedly_estimate_diameter
from ..utils import debug_print


class DistributedKZCenter(object):

    def __init__(self,
                 n_clusters=None, n_outliers=None, n_machines=None,
                 epsilon=0.1, delta=0.01, random_state=None, debug=False):
        """
        Serves as the master node.

        :param n_clusters: integer.
            Number of clusters.

        :param n_outliers: integer.
            Number of outliers

        :param n_machines: integer.
            Number of machines

        :param epsilon: float.
            error tolerance parameter for the number of outliers.

        :param delta: float.
            error tolerance parameter for estimating the optimal radius

        :param random_state: numpy.RandomState

        :param debug: boolean, whether output debugging information
        """
        self.n_clusters = n_clusters
        self.n_outliers = n_outliers
        self.n_machines = n_machines
        self.epsilon = epsilon
        self.delta = delta
        self.random_state = random_state
        self.debugging = debug

        self.cluster_centers_ = None
        self.n_samples_, self.n_features_ = None, None
        self.opt_ = None

    def cost(self, X, remove_outliers=True):
        """

        :param X: array,
            data set
        :param remove_outliers: boolean, default True
            whether to remove outliers when computing the cost on X
        :return: float,
            actual cost
        """
        if self.cluster_centers_ is None:
            raise NotFittedError("Model hasn't been fitted yet\n")
        X = check_array(X, ensure_2d=True)
        dists = np.fromiter((np.min(np.linalg.norm(x - self.cluster_centers_, axis=1))
                             for x in X), dtype=np.float64, count=X.shape[0])
        dists.sort()
        if remove_outliers:
            return dists[-int((1+self.epsilon) * self.n_outliers + 1)]
        else:
            return dists[-1]

    def fit(self, Xs, sample_weights=None):
        """

        :param Xs: list of arrays of shape=(n_samples_i, n_features),
            Divided data set. Each array in the list represents a bunch of data
            that has been partitioned onto one machine.
        :param sample_weights: list of arrays of shape=(n_samples_i,),
            sample weights for each machine's data
        :return self:
        """
        if sample_weights is None:
            sample_weights = [None] * len(Xs)

        self.n_machines = len(Xs)

        debug_print("Build distance query oracle ...", debug=self.debugging)
        oracles = []
        for i in range(self.n_machines):
            oracles.append(DistQueryOracle(tree_algorithm='auto',
                                           leaf_size=min(Xs[i].shape[0] // self.n_clusters, 30))
                           .fit(Xs[i]))

        debug_print("Initialize mappers ...", debug=self.debugging)
        mappers = []
        for i in range(self.n_machines):
            mappers.append(KZCenter(
                is_mapper=True, is_reducer=False,
                n_clusters=self.n_clusters, n_outliers=self.n_outliers,
                n_machines=self.n_machines, epsilon=self.epsilon,
                oracle=oracles[i]
            ))

        self.n_samples_ = sum(X.shape[0] for X in Xs)
        self.n_features_ = Xs[0].shape[1]
        # if the number of allowed outliers are more than n_samples then we
        # simply choose arbitrary n_clusters points in the data set as centers
        if self.n_samples_ <= (1 + self.epsilon) * self.n_outliers:
            while len(self.cluster_centers_) < min(self.n_clusters, self.n_samples_):
                for X in Xs:
                    for i in range(min(X.shape[0], self.n_clusters)):
                        self.cluster_centers_.append(X[i])
            return self

        # estimating the optimal radius using binary search
        n_guesses = 1  # used for debugging
        total_iters_at_most = self.n_clusters * self.n_machines * (1 + 1 / self.epsilon)
        total_covered_at_least = max(self.n_samples_ - (1+self.epsilon) * self.n_outliers, 1)
        lb = 0
        # upper bound is initialized as the sum of diameters across all machines
        _, ub = distributedly_estimate_diameter(Xs, n_estimation=10)
        debug_print("Start guessing optimal radius in [{},{}]...".format(lb, ub),
                    debug=self.debugging)
        guessed_opt = (lb + ub) / 2
        while ub > (1+self.delta) * lb and ub > 1e-3:
            debug_print("{}-th guess: trying with OPT = {}".format(n_guesses, guessed_opt), debug=self.debugging)
            for i, m in enumerate(mappers):
                m.fit(guessed_opt, Xs[i], sample_weights[i])
            total_iters = sum(m.n_iters for m in mappers)
            total_covered = sum(m.n_covered for m in mappers)

            # if the number of balls constructed is too large, or
            # the number of points covered is to small, then enlarge guessed opt
            if total_iters > total_iters_at_most or total_covered < total_covered_at_least:
                lb = guessed_opt
            else:
                ub = guessed_opt
            guessed_opt = (lb + ub) / 2
            n_guesses += 1

        if ub < 1e-3:
            warnings.warn("guessed OPT={} is smaller than 1e-3\n".format(ub), UserWarning)
        debug_print("Estimated optimal radius: {}".format(ub), debug=self.debugging)
        self.opt_ = ub

        # construct the centralized data set
        debug_print("Aggregate results on reducer ...", debug=self.debugging)
        for i, m in enumerate(mappers):
            m.fit(guessed_opt, Xs[i], sample_weights[i])
        X = np.vstack([m.cluster_centers for m in mappers])
        sample_weight = np.hstack([m.clusters_size for m in mappers])

        debug_print("Fit the aggregated data set ...", debug=self.debugging)
        reducer = KZCenter(is_mapper=False, is_reducer=True,
                           n_clusters=self.n_clusters,
                           n_outliers=(1 + self.epsilon) * self.n_outliers,
                           n_machines=self.n_machines, random_state=self.random_state,
                           oracle=DistQueryOracle(tree_algorithm='ball_tree'))
        reducer.fit(5 * ub, X, sample_weight)

        self.cluster_centers_ = list(reducer.cluster_centers)

        return self


class KZCenter(object):

    def __init__(self,
                 is_mapper=True, is_reducer=False,
                 n_clusters=None, n_outliers=None, n_machines=None,
                 epsilon=0.1, random_state=None, oracle=None):
        """
        Worker object on single machine. Can conduct pre-clustering or regular
        clustering tasks.

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

        self.n_iters_ = None
        self.n_covered_ = None
        self.results_ = None
        self.cluster_centers_ = None
        self.clusters_size_ = None
        self.n_samples_ = None
        self.n_features_ = None

    @property
    def n_covered(self):
        return self.n_covered_

    @property
    def clusters_size(self):
        return self.clusters_size_

    @property
    def fitted_results(self):
        return self.results_

    @property
    def cluster_centers(self):
        return self.cluster_centers_

    @property
    def n_iters(self):
        return self.n_iters_

    def fit(self, guessed_opt, X, sample_weight=None):
        """
        :param X:
        :param sample_weight:
        :return:
        """
        if not self.dist_oracle.is_fitted:
            self.dist_oracle.fit(X, sample_weight)

        self.n_samples_, self.n_features_ = X.shape
        if self.is_mapper:
            self.results_ = self.pre_clustering(guessed_opt)
        elif self.is_reducer:
            # if n^k / k! < 10^5
            if self.n_clusters * np.log10(self.n_samples_) - \
                    np.sum(np.log10(np.arange(1, self.n_clusters + 1))) < 6:
                self.results_ = kzcenter_brute(X, sample_weight,
                                               n_clusters=self.n_clusters,
                                               n_outliers=self.n_outliers)
            else:
                self.results_ = kzcenter_charikar(X, sample_weight=sample_weight, guessed_opt=guessed_opt,
                                                  n_clusters=self.n_clusters, n_outliers=self.n_outliers,
                                                  dist_oracle=self.dist_oracle)

        else:
            raise AttributeError(
                "Unclear role: not sure whether current object is a mapper or a reducer.\n"
            )
        self.cluster_centers_ = np.array([c for c, _ in self.results_])
        self.clusters_size_ = np.array([w for _, w in self.results_])
        self.n_covered_ = sum(self.clusters_size_)
        return self

    def pre_clustering(self, guessed_opt):
        """
        Greedily cover the data set with ball of specified size
        :param guessed_opt: the radius of the ball used for compressing the data set
        :return results: list of (array, int)
            List of (ball center, #points in the ball)
        """
        threshold = self.epsilon * self.n_outliers / (self.n_clusters * self.n_machines)
        results = []
        removed = set()
        to_be_updated = range(self.n_samples_)
        self.n_iters_ = 0
        while len(removed) < self.n_samples_:
            p, covered_pts = self.dist_oracle.densest_ball_faster_but_dirty(2 * guessed_opt,
                                                                            changed=to_be_updated,
                                                                            except_for=removed)

            if p is None:
                break
            if len(covered_pts) < threshold:
                break
            to_be_removed = self.dist_oracle.ball(p.reshape(1, -1), 4 * guessed_opt)[0]
            removed.update(to_be_removed)
            w_p = len(covered_pts)
            results.append((p, w_p))
            self.n_iters_ += 1

            # after removing ball(p, 4L), only the densest ball whose centers resides
            # in ball(p, 6L) \ ball(p, 4L) is affected. So we only need to consider
            to_be_updated = set(self.dist_oracle.ball(p.reshape(1, -1), 6 * guessed_opt)[0])
            to_be_updated.difference_update(to_be_removed)

        return results


def kzcenter_charikar(X, sample_weight=None, guessed_opt=None, n_clusters=7, n_outliers=0, delta=0.05,
                      dist_oracle=None):
    """
    Implementation of the algorithm proposed in Moses Charikar's SODA'01 paper
    :param X:
    :param sample_weight:
    :param guessed_opt:
    :param n_clusters:
    :param n_outliers:
    :param delta:
    :param dist_oracle:
    :return results: list of (array, int)
            List of (ball center, #points in the ball)
    """
    if dist_oracle is None:
        dist_oracle = DistQueryOracle(tree_algorithm='ball_tree')
    if not dist_oracle.is_fitted:
        dist_oracle.fit(X, sample_weight)
    n_distinct_points, _ = X.shape
    if sample_weight is None:
        sample_weight = np.ones(n_distinct_points)

    n_samples = sum(sample_weight)
    if n_distinct_points <= n_clusters:
        return [(c, w) for c, w in zip(X, sample_weight)]

    # estimate the upperbound and lowerbound of opt for the data set
    _, ub = dist_oracle.estimate_diameter(n_estimation=10)
    lb, _ = dist_oracle.kneighbors(X[np.random.randint(0, n_distinct_points)], k=2)
    lb = np.max(lb)
    if guessed_opt is not None:
        guessed_opt = min(guessed_opt, ub)
    if guessed_opt is None:
        guessed_opt = (lb + ub) / 2

    results = []
    while ub > (1+delta) * lb:
        removed = set()
        results = []

        for i in range(n_clusters):
            if len(removed) == n_distinct_points:
                break
            p, covered_pts = dist_oracle.densest_ball(2 * guessed_opt, removed)
            removed.update(dist_oracle.ball(p.reshape(1, -1), 4 * guessed_opt)[0])
            w_p = sum(sample_weight[covered_pts])
            results.append((p, w_p))
        n_covered = sum(wp for _, wp in results)
        if n_covered >= n_samples - n_outliers:
            ub = guessed_opt
            guessed_opt = (lb + guessed_opt) / 2
        else:
            lb = guessed_opt
            guessed_opt = (guessed_opt + ub) / 2
    return results


def kzcenter_brute(X, sample_weight=None, n_clusters=7, n_outliers=0):
    """
    Solve the (k,z)-center problem using brute-force
    :param X:
    :param sample_weight:
    :param n_clusters:
    :param n_outliers:
    :return: list of tuple of (array, int)
        List of (ball center, #points in the ball)
    """
    from itertools import combinations
    n_distinct_points, _ = X.shape
    if sample_weight is None:
        sample_weight = np.ones(n_distinct_points)
    n_samples = sum(sample_weight)

    if n_distinct_points <= n_clusters:
        return [(c, w) for c, w in zip(X, sample_weight)]

    costs = []
    # enumerate all possible k centers
    for sol in combinations(range(n_distinct_points), n_clusters):
        idx = list(sol)  # because sol is a tuple
        cs = X[list(idx)]
        dists = np.fromiter((np.min(np.linalg.norm(x - cs, axis=1)) for x in X),
                            dtype=np.float64,
                            count=X.shape[0])
        sorted_dist_idxs = np.argsort(dists)
        can_remove = 0
        farthest = 0

        # count how many weights can be removed
        while farthest < len(dists) and can_remove <= n_outliers:
            can_remove += sample_weight[sorted_dist_idxs[farthest]]
            farthest += 1

        costs.append((idx, dists[sorted_dist_idxs[farthest - 1]]))

    opt_centers, cost = min(costs, key=lambda c: c[1])
    opt_centers = X[opt_centers]
    n_covered = np.unique([np.argmin(np.linalg.norm(x - opt_centers, axis=1)) for x in X],
                          return_counts=True)
    return list(zip(opt_centers, n_covered[1]))


class DistributedKCenter(object):
    pass
