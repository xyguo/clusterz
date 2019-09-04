# -*- coding: utf-8 -*-
"""Miscellaneous helper functions for distance queries"""

# Author: Xiangyu Guo     xiangyug[at]buffalo.edu
#         Shi Li          shil[at]buffalo.edu

import warnings
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin_min
from sklearn.neighbors import KDTree, BallTree
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array


def _brute_force_knn(X, centers, k, return_distance=True):
    """
    :param X: array of shape=(n_samples, n_features)
    :param centers: array of shape=(n_centers, n_features)
    :param k: int, only looking for the nearest k points to each center.
    :param return_distance: bool, if True the return the distance along with the points
    :return:
    """
    if k == 1:
        nearest, dists = pairwise_distances_argmin_min(centers, X)
        return (dists, nearest) if return_distance else nearest
    else:
        dists = pairwise_distances(centers, X)
        nearest = np.argsort(dists, axis=1)[:, :k]
        return (np.vstack([dists[i, nearest[i]] for i in range(dists.shape[0])]),
                nearest) if return_distance else nearest


def _brute_force_ball_within_dataset(X, center_idxs, radius,
                                     sorted_distances, sorted_idxs,
                                     count_only=False):
    """Find the points in X that are within given distance `radius` to some centers, here the centers must be within X
    and are specified by `center_idxs`. Also allows for expoliting auxiliary information if available: specifically,
    if the distance of each point in X to the centers are already calculated out and sorted.
    :param X: array of shape=(n_samples, n_features),
        data set
    :param centers: array of shape=(n_centers, n_features),
        centers being queried
    :param radius: float, only count points within distance `radius` to its nearest center
    :param sorted_distances: array of shape=(n_samples,), a sorted array of each point's distance to its nearest center
    :param sorted_idxs: array of shape=(n_samples,), index of each point after it's sorted
    :param count_only: bool, if True then return only the number of points in each ball centered at the centers.
    :return: list of arrays
        list of data idxs
    """
    if sorted_distances is None:
        dists = pairwise_distances(X[center_idxs], X)
        if count_only:
            return list(np.sum(dists <= radius, axis=1))
        else:
            return list(np.where(d <= radius)[0] for d in dists)

    center_idxs = np.atleast_1d(center_idxs)
    count = np.apply_along_axis(lambda a: np.searchsorted(a, radius), axis=1, arr=sorted_distances[center_idxs])
    if count_only:
        return count
    else:
        balls = list(sorted_idxs[c, 0:count[i]] for i, c in enumerate(center_idxs))
        return balls


def _brute_force_ball(X, centers, radius, count_only=False):
    """Find the points in X that are within given distance `radius` to the `centers`
    :param X: array of shape=(n_samples, n_features),
        data set
    :param centers: array of shape=(n_centers, n_features),
        centers being queried
    :param radius: float, the queried distance
    :param count_only: bool, if True then only return the number of points
    :return: list of arrays
        list of data idxs
    """
    centers = check_array(centers, ensure_2d=True)
    dists = pairwise_distances(centers, X)
    if count_only:
        return list(np.sum(dists <= radius, axis=1))
    else:
        return list(np.where(d <= radius)[0] for d in dists)


def farthest_neighbor(c, X, return_distance=True):
    """Find the farthest point in X from c
    Time complexity: O(n_samples * n_features)
    :param c: array of shape=(n_features,) of (n_queries, n_features), the queried point(s)
    :param X: array of shape=(n_samples, n_features), the data set
    :param return_distance:
    :return (dist, idx) or (idx): (float, int) or int
        return the index of the point in X that is farthest to c, if return_distance is True,
        also return the correponding distance
    """
    X = check_array(X, ensure_2d=True)
    c = np.atleast_2d(c)
    dists = np.min(pairwise_distances(X, c), axis=1)
    farthest = np.argmax(dists)
    return (dists[farthest], farthest) if return_distance else farthest


def estimate_diameter(X, n_estimation=1, metric='minkowski'):
    """
    Pick an arbitrary point in the data set, suppose d is the largest distance between this
    point and any other points, then the diameter must be in [d, 2d]

    Time complexity: O(n_samples * n_estimations * n_features)
    :param X: array of shape=(n_samples, n_features),
        data set
    :param n_estimation: number of sampled estimation points
    :param metric: {'minkowski'}
    :return diam: tuple, (lower_bound, upper_bound)
    """
    X = check_array(X, ensure_2d=True)
    if metric == 'minkowski':
        n_samples, _ = X.shape
        estimations = [0] + list(np.random.choice(range(1, n_samples),
                                                  min(n_estimation - 1, n_samples - 1),
                                                  replace=False))
        lb = min(farthest_neighbor(X[i], X, return_distance=True)[0] for i in estimations)
        diam = (lb, 2 * lb)
    else:
        raise ValueError("metric `{}` currently not supported\n".format(metric))
    return diam


def distributedly_estimate_diameter(Xs, n_estimation=10):
    """
    Estimate diameter for distributed data set
    :param Xs: list of arrays, each array represents a subset of data partitioned on some machine
    :param n_estimation: number of estimation point
    :return (lb, ub): lower bound and upper bound of the diameter
    """
    n_machines = len(Xs)

    # sample base points.
    # each base points give a diameter estimations (lb, ub=2 * lb) for the whole data set,
    # the final estimation is simply the intersection of all such intervals.
    machines = np.random.choice(n_machines, n_estimation, replace=True)
    base_points = []
    for m in machines:
        idx = np.random.randint(0, Xs[m].shape[0])
        base_points.append(Xs[m][idx])

    lower_bounds = []
    for bp in base_points:
        lower_bounds.append(max(farthest_neighbor(bp, X, return_distance=True)[0] for X in Xs))
    return max(lower_bounds), min(map(lambda x: 2 * x, lower_bounds))


class DistQueryOracle(object):

    def __init__(self, tree_algorithm='auto', leaf_size=60,
                 metric='minkowski', precompute_distances='auto',
                 random_state=None, debugging=False):
        """
        :param tree_algorithm: string in {'auto', 'kd_tree', 'ball_tree', 'brute', 'lsh'}
            determines the

        :param leaf_size: int, default 40
            leaf size passed to BallTree or KDTree

        :param metric: string, default 'minkowski'
            the distance metric to use in the tree

        :param precompute_distances: {'auto', True, False}
            Precompute distances (faster but takes more memory).
            'auto': do not precompute distances if n_samples^2  > 1.2 million.
            This corresponds to about 10MB overhead per job using double precision.
            True: always precompute distances
            False: never precompute distances

        :param random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by np.random.

        :param debugging: bool, default False, whether to print debugging and User warning info
        """
        self.tree_algorithm = tree_algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        # TODO: add pre-computed distance matrix
        self.precompute_distances = precompute_distances
        self.random_state_ = random_state
        self.debugging_ = debugging

        self.nn_tree_ = None
        self.ball_oracle_ = None
        self.knn_oracle_ = None
        self.fitted_data_ = None
        self.data_weight_ = None
        self.diam_ = None
        self.n_samples_, self.n_features_ = None, None
        self.new_data = True

        # variables used for supporting fast densest-ball query and removing
        self.n_facilities_ = None
        self.facility_idxs_ = None
        self.sorted_distance_cache_ = None
        self.sorted_dist_idxs_ = None
        self.unweightted_ = None
        self.ball_weight_cache_ = None
        self.ball_size_cache_ = None
        self.ball_radius_cache_ = None
        self.ball_cache_ = None
        self.cache_inconsistent_ = None

    @property
    def is_fitted(self):
        return self.ball_oracle_ is not None

    def fit(self, X, sample_weight=None):
        """

        :param X: array of shape=(n_samples, n_features),
            Data set to be processed

        :param sample_weight: array of shape=(n_samples,),
            weight on each data instance. "None" means uniform weights.

        :return:
        """
        if self.tree_algorithm == 'auto':
            if X.shape[1] < 20:
                self.nn_tree_ = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
                self.ball_oracle_ = lambda cs, r: self.nn_tree_.query_radius(cs, r, return_distance=False)
                self.knn_oracle_ = lambda cs, k, rd: self.nn_tree_.query(cs, k, return_distance=rd)
            elif X.shape[0] < 40:
                self.sorted_distance_cache_, self.sorted_dist_idxs_ = self.precompute_distances_matrix_(X)
                self.ball_oracle_ = lambda cs, r: _brute_force_ball(X, cs, r)
                self.knn_oracle_ = lambda cs, k, rd: _brute_force_knn(X, cs, k, return_distance=rd)
                self.brute_force_ball_within_dataset_oracle_ = lambda cidx, r, co: \
                    _brute_force_ball_within_dataset(X, cidx, r,
                                                     sorted_distances=self.sorted_distance_cache_,
                                                     sorted_idxs=self.sorted_dist_idxs_, count_only=co)
            else:
                self.nn_tree_ = BallTree(X, leaf_size=self.leaf_size, metric=self.metric)
                self.ball_oracle_ = lambda cs, r: self.nn_tree_.query_radius(cs, r, return_distance=False)
                self.knn_oracle_ = lambda cs, k, rd: self.nn_tree_.query(cs, k, return_distance=rd)

        elif self.tree_algorithm == 'kd_tree':
            self.nn_tree_ = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
            self.ball_oracle_ = lambda cs, r: self.nn_tree_.query_radius(cs, r, return_distance=False)
            self.knn_oracle_ = lambda cs, k, rd: self.nn_tree_.query(cs, k, return_distance=rd)

        elif self.tree_algorithm == 'ball_tree':
            self.nn_tree_ = BallTree(X, leaf_size=self.leaf_size, metric=self.metric)
            self.ball_oracle_ = lambda cs, r: self.nn_tree_.query_radius(cs, r, return_distance=False)
            self.knn_oracle_ = lambda cs, k, rd: self.nn_tree_.query(cs, k, return_distance=rd)

        elif self.tree_algorithm == 'brute':
            self.sorted_distance_cache_, self.sorted_dist_idxs_ = self.precompute_distances_matrix_(X)
            self.ball_oracle_ = lambda cs, r: _brute_force_ball(X, cs, r)
            self.knn_oracle_ = lambda cs, k, rd: _brute_force_knn(X, cs, k, return_distance=rd)
            self.brute_force_ball_within_dataset_oracle_ = lambda cidx, r, co: \
                _brute_force_ball_within_dataset(X, cidx, r,
                                                 sorted_distances=self.sorted_distance_cache_,
                                                 sorted_idxs=self.sorted_dist_idxs_, count_only=co)

        else:
            raise ValueError("tree_algorithm \"{}\" not properly specified".
                             format(self.tree_algorithm))

        self.fitted_data_ = X
        self.n_samples_, self.n_features_ = X.shape

        if sample_weight is not None:
            self.data_weight_ = sample_weight
        else:
            self.data_weight_ = np.ones(X.shape[0], dtype=np.int)
            self.unweightted_ = True

        return self

    def precompute_distances_matrix_(self, X):
        if self.precompute_distances is True or \
                (self.precompute_distances == 'auto' and 8 * (X.shape[0] ** 2) < 20e6):
            distance_cache = pairwise_distances(X)
            distance_argsort = np.argsort(distance_cache, axis=1)
            return distance_cache[np.arange(distance_cache.shape[0])[:, None], distance_argsort],\
                   distance_argsort
        else:
            return None, None

    def ball(self, centers, radius):
        """Query the data points in X that are within distance `radius` to `centers`
        :param centers: array of shape=(n_points, n_features), queried centers
        :param radius: float, radius of the ball
        :return: an array of array,
            indices for each center in centers
        """
        if self.ball_oracle_ is None:
            raise NotFittedError("Tree hasn't been fitted yet\n")

        centers = check_array(centers, ensure_2d=True)
        return self.ball_oracle_(centers, radius)

    def densest_ball(self, radius, except_for=None):
        """Find the ball of given radius that covers the most data points
        :param radius: float, radius of the ball
        :param except_for: iterable or set,
            indices of points that should not be considered
        :return (densest_center, densest_ball): (array of shape=(n_features,), array of shape=(n_covered,)
            the center of the densest ball as well as the index of points the ball covers
        """
        if except_for is None or len(except_for) == 0:
            except_for = None

        if self.ball_oracle_ is None:
            raise NotFittedError("Tree hasn't been fitted yet\n")

        if except_for is not None and len(except_for) == self.n_samples_:
            return None, None

        if except_for is None:
            densest_idx, _ = max(
                ((i, self.data_weight_[self.ball_oracle_(x.reshape(1, -1), radius)[0]].sum())
                 for i, x in enumerate(self.fitted_data_)),
                key=lambda a: a[1]
            )
            densest_center = self.fitted_data_[densest_idx]
            densest_ball = self.ball_oracle_(densest_center.reshape(1, -1), radius)[0]
        else:
            densest_idx, _ = max(
                ((i, self.data_weight_[list(set(self.ball_oracle_(x.reshape(1, -1), radius)[0]).
                                            difference(except_for))].sum())
                 for i, x in enumerate(self.fitted_data_) if i not in except_for),
                key=lambda a: a[1]
            )
            densest_center = self.fitted_data_[densest_idx]
            densest_ball = np.array(list(set(self.ball_oracle_(densest_center.reshape(1, -1), radius)[0]).
                                         difference(except_for)))
        # assert len(densest_ball) > 0
        return densest_center, densest_ball

    def init_all_densest_ball_faster_but_dirty(self, radius):
        self.ball_radius_cache_ = radius
        balls = self.ball_oracle_(self.fitted_data_, radius)

    def dense_ball_(self, radius, except_for, changed, facility_idxs=None, return_idx=False, minimum_density=None):
        """
        When the radius is fixed and need to do a series of query, then this function
        will cache previous calculated balls for fast retrieving.

        Warning: This function implementation is coupled with the one that invokes it. Shouldn't
        be called by other functions except for KZCenter.fit.
        :param radius:
        :param except_for: iterable or set,
            indices of points that should not be considered
        :param changed: iterable or set,
            indices in cache that need to be updated
        :param return_idx: bool,
            whether to return the index of the densest ball center within the data set
        :param minimum_density: float,
            minimum density requirement for early returning

        :return :
            (densest_center, densest_ball, center_idx):
                (array of shape=(n_features,), array of shape=(n_covered,), int)
            (densest_center, densest_ball): (array of shape=(n_features,), array of shape=(n_covered,)
                the center of the densest ball as well as the index of points the ball covers:
        """
        # They said that if I can run fast enough, sadness wouldn't catch me up.
        if facility_idxs is None:
            self.facility_idxs_ = np.arange(self.fitted_data_.shape[0])
        else:
            self.facility_idxs_ = facility_idxs

        if self.ball_radius_cache_ != radius:
            # new search begins, should refresh all caches
            self.ball_radius_cache_ = radius
            self.ball_weight_cache_ = [None] * self.n_samples_
            self.ball_size_cache_ = [None] * self.n_samples_
            self.ball_cache_ = [None] * self.n_samples_
            self.cache_inconsistent_ = False

        if self.cache_inconsistent_ and self.debugging_:
            warnings.warn("Cache is inconsistent, may get outdated result\n", UserWarning)

        if self.ball_weight_cache_ is None:
            self.ball_weight_cache_ = [None] * self.n_samples_

        if except_for is None:
            except_for = {}
        if changed is None:
            changed = range(self.n_samples_)
        if len(except_for) == self.n_samples_:
            return (None, None, None) if return_idx else (None, None)

        for i in changed:
            x = self.fitted_data_[i]
            ball_i = set(self.ball_oracle_(x.reshape(1, -1), radius)[0])

            # update ball cache
            if len(except_for) / len(ball_i) > 10:
                ball_i.difference_update(except_for.intersection(ball_i))
            else:
                ball_i.difference_update(except_for)

            self.ball_size_cache_[i] = len(ball_i)
            self.ball_weight_cache_[i] = self.data_weight_[list(ball_i)].sum()
            # if a ball covers all points, then it must be the densest one
            if self.ball_size_cache_[i] >= min(self.n_samples_ - len(except_for), minimum_density):
                self.cache_inconsistent_ = True
                ball_i = np.array(list(ball_i))
                return (self.fitted_data_[i], ball_i, i) if return_idx \
                    else (self.fitted_data_[i], ball_i)

        dense_idx = None
        dense_ball_weight = 0
        dense_ball = None

        ball_i = None
        remained = set(self.facility_idxs_).difference(except_for)
        if len(remained) == 0:
            return (None, None, None) if return_idx else (None, None)

        for i in remained:
            # because ball_cache can become inconsistent due to early returning
            if self.ball_size_cache_[i] is None:
                ball_i = set(self.ball_oracle_(self.fitted_data_[i].reshape(1, -1), radius)[0])
                ball_i.difference_update(except_for)
                self.ball_size_cache_[i] = len(ball_i)
                self.ball_weight_cache_[i] = self.data_weight_[list(ball_i)].sum()

            # if a ball covers all points, then it must be the densest one
            # this serves as an early return, but
            if self.ball_size_cache_[i] >= min(self.n_samples_ - len(except_for), minimum_density):
                self.cache_inconsistent_ = True
                if not ball_i:
                    ball_i = set(self.ball_oracle_(self.fitted_data_[i].reshape(1, -1), radius)[0])
                    ball_i.difference_update(except_for)
                ball_i = np.array(list(ball_i))
                return (self.fitted_data_[i], ball_i, i) if return_idx \
                    else (self.fitted_data_[i], ball_i)

            if dense_ball_weight < self.ball_weight_cache_[i]:
                dense_ball_weight = self.ball_weight_cache_[i]
                dense_idx = i
        dense_ball = set(self.ball_oracle_(self.fitted_data_[dense_idx].reshape(1, -1), radius)[0])
        dense_ball.difference_update(except_for)
        dense_ball = np.array(list(dense_ball))

        return (self.fitted_data_[dense_idx], dense_ball, dense_idx) if return_idx \
            else (self.fitted_data_[dense_idx], dense_ball)

    def densest_ball_faster_but_dirty(self, radius, except_for, changed, within_idx=None, return_idx=False):
        """
        When the radius is fixed and need to do a series of query, then this function
        will cache previous calculated balls for fast retrieving.

        Warning: This function implementation is coupled with the one that invokes it. Shouldn't
        be called by other functions except for KZCenter.fit.
        :param radius:
        :param except_for: iterable or set,
            indices of points that should not be considered
        :param changed: iterable or set,
            indices in cache that need to be updated
        :param return_idx: bool,
            whether to return the index of the densest ball center within the data set
        :return :
            (densest_center, densest_ball, center_idx):
                (array of shape=(n_features,), array of shape=(n_covered,), int)
            (densest_center, densest_ball): (array of shape=(n_features,), array of shape=(n_covered,)
                the center of the densest ball as well as the index of points the ball covers:
        """
        # TODO: what is the actual complexity of radius_query for BallTree and KDTree?
        # They said that if I can run fast enough, sadness wouldn't catch me up.

        if self.ball_radius_cache_ != radius:
            # new search begins, should refresh all caches
            self.ball_radius_cache_ = radius
            self.ball_weight_cache_ = [None] * self.n_samples_
            self.ball_size_cache_ = [None] * self.n_samples_
            # self.ball_cache_ = [None] * self.n_samples_
            self.ball_cache_ = np.ones(self.n_samples_, dtype=object) * -1
            self.cache_inconsistent_ = False

        if self.cache_inconsistent_:
            warnings.warn("Cache is inconsistent, may get outdated result\n", UserWarning)

        if self.ball_weight_cache_ is None:
            self.ball_weight_cache_ = [None] * self.n_samples_

        if except_for is None:
            except_for = {}
        if changed is None:
            changed = range(self.n_samples_)
        if len(except_for) == self.n_samples_:
            return (None, None, None) if return_idx else (None, None)

        # the intersection between changed and except_for should be empty
        # changed = np.array(list(changed))
        # x = np.atleast_2d(self.fitted_data_[changed])
        # not_cached_yet = np.where(self.ball_cache_[changed] == -1)[0]
        # if len(not_cached_yet) > 0:
        #     balls = self.ball_oracle_(x[not_cached_yet], radius)
        # for i in not_cached_yet:
        #     self.ball_cache_[changed[i]] = set(balls[i])

        approx = 1.0
        for i in changed:
            if i in except_for:
                continue

            x = self.fitted_data_[i]
            if self.ball_cache_[i] == -1:
                self.ball_cache_[i] = set(self.ball_oracle_(x.reshape(1, -1), radius)[0])

            # update ball cache
            if len(except_for) / len(self.ball_cache_[i]) > 10:
                self.ball_cache_[i].difference_update(except_for.intersection(self.ball_cache_[i]))
            else:
                self.ball_cache_[i].difference_update(except_for)

            self.ball_size_cache_[i] = len(self.ball_cache_[i])
            self.ball_weight_cache_[i] = self.data_weight_[list(self.ball_cache_[i])].sum()
            # if a ball covers all points, then it must be the densest one
            if self.ball_size_cache_[i] >= approx * (self.n_samples_ - len(except_for)):
                self.cache_inconsistent_ = True
                return (self.fitted_data_[i], self.ball_cache_[i], i) if return_idx \
                    else (self.fitted_data_[i], self.ball_cache_[i])

        densest_idx = None
        densest_ball_weight = 0

        for i in range(self.n_samples_):
            if i in except_for:
                continue

            # because ball_cache can become inconsistent due to early returning
            if self.ball_cache_[i] == -1:
                self.ball_cache_[i] = set(self.ball_oracle_(x.reshape(1, -1), radius)[0])
                self.ball_cache_[i].difference_update(except_for)
                self.ball_size_cache_[i] = len(self.ball_cache_[i])
                self.ball_weight_cache_[i] = self.data_weight_[list(self.ball_cache_[i])].sum()

            # if a ball covers all points, then it must be the densest one
            # this serves as an early return, but
            if self.ball_size_cache_[i] >= approx * (self.n_samples_ - len(except_for)):
                self.cache_inconsistent_ = True
                return (self.fitted_data_[i], self.ball_cache_[i], i) if return_idx \
                    else (self.fitted_data_[i], self.ball_cache_[i])

            if densest_ball_weight < self.ball_weight_cache_[i]:
                densest_ball_weight = self.ball_weight_cache_[i]
                densest_idx = i

        return (self.fitted_data_[densest_idx], self.ball_cache_[densest_idx], densest_idx) if return_idx \
            else (self.fitted_data_[densest_idx], self.ball_cache_[densest_idx])

    def estimate_diameter(self, n_estimation=1):
        """
        Pick an arbitrary point in the data set, suppose d is the largest distance between this
        point and any other points, then the diameter must be in [d, 2d]

        Time complexity: O(n_samples * n_estimations * n_features)
        :param n_estimation: number of sampled estimation points
        :return: (lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise NotFittedError("Tree hasn't been fitted yet\n")

        if self.diam_ is not None:
            return self.diam_

        self.diam_ = estimate_diameter(X=self.fitted_data_,
                                       n_estimation=n_estimation,
                                       metric=self.metric)

        return self.diam_

    def farthest_neighbor(self, x, return_distance=True):
        """Find the farthest point away from x

        Time complexity: O(n_samples * n_features)
        :param x:
        :param return_distance:
        :return:
        """
        if not self.is_fitted:
            raise NotFittedError("Tree hasn't been fitted yet\n")
        return farthest_neighbor(x, X=self.fitted_data_, return_distance=return_distance)

    def kneighbors(self, centers, k=1, return_distance=True):
        """return the k nearest neighbors to centers

        Time complexity: depends on the tree alg used
            brute - O(n_samples * n_features)
            ball_tree - O(n_features * \log n_samples)
            kd_tree - O(n_features * \log n_samples) for small n_features and O(n_samples * n_features)
                      for large n_features
            lsh - o(n_samples * n_features)
        :param centers: array of shape=(n_queries, n_features)
        :param k: queried number of nearest neighbors
        :param return_distance: default True, whether to return the distance of the neighbors to its nearest centers
        :return: idxs or (dists, idxs)
            idx - the indices of the nearest neighbor in the fitted data set
            dists - the corresponding distance of nearest neighbors
        """
        if not self.is_fitted:
            raise NotFittedError("Tree hasn't been fitted yet\n")

        centers = check_array(centers, ensure_2d=True)
        return self.knn_oracle_(centers, k, return_distance)

