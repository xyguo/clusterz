import warnings
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.neighbors import KDTree, BallTree, LSHForest
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array


def _brute_force_ball(X, centers, radius, count_only=False):
    """
    :param X: array of shape=(n_samples, n_features),
        data set
    :param centers: array of shape=(n_centers, n_features),
        centers being queried
    :param radius: float
    :param count_only: bool
    :return: list of arrays
        list of data idxs
    """
    centers = check_array(centers, ensure_2d=True)
    if count_only:
        return list(np.sum(np.linalg.norm(X - c, axis=1) <= radius)
                    for c in centers)
    else:
        return list(np.where(np.linalg.norm(X - c, axis=1) <= radius)[0]
                    for c in centers)


def farthest_neighbor(c, X, return_distance=True):
    """
    Time complexity: O(n_samples * n_features)
    :param c: the queried point
    :param X: the data set
    :param return_distance:
    :return (dist, idx) or (idx): (float, int) or int
        return the index of the point in X that is farthest to c, if return_distance is True,
        also return the correponding distance
    """
    X = check_array(X, ensure_2d=True)
    dists = np.linalg.norm(X - c, axis=1)
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
    :return: (lower_bound, upper_bound)
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

    # sample base points
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

    def __init__(self,
                 tree_algorithm='auto', leaf_size=30,
                 metric='minkowski',
                 precompute_distances='auto',
                 # below are parameters for LSHForest specifically
                 n_estimators=10,
                 radius=1.0, n_candidates=50, n_neighbors=5,
                 min_hash_match=4, radius_cutoff_ratio=0.9,
                 random_state=None):
        """
        :param tree_algorithm: string in {'auto', 'kd_tree', 'ball_tree', 'brute', 'lsh'}
            determines the

        :param leaf_size: int, default 40
            leaf size passed to BallTree or KDTree

        :param metric: string, default 'minkowski'
            the distance metric to use in the tree

        :param precompute_distances: {'auto', True, False}
            Precompute distances (faster but takes more memory).
            'auto': do not precompute distances if n_samples^2  > 12 million.
            This corresponds to about 100MB overhead per job using double precision.
            True: always precompute distances
            False: never precompute distances

        Below are parameters specifically for LSHForest

        :param n_estimators : int (default = 10)
            Number of trees in the LSH Forest.
        :param min_hash_match : int (default = 4)
            lowest hash length to be searched when candidate
            selection is performed for nearest neighbors.
        :param n_candidates : int (default = 10)
            Minimum number of candidates evaluated per estimator,
            assuming enough items meet the min_hash_match constraint.
        :param n_neighbors : int (default = 5)
            Number of neighbors to be returned from query function
            when it is not provided to the kneighbors method.
        :param radius : float, optinal (default = 1.0)
            Radius from the data point to its neighbors. This is the
            parameter space to use by default for the :meth`radius_neighbors` queries.
        :param radius_cutoff_ratio : float, optional (default = 0.9)
            A value ranges from 0 to 1. Radius neighbors will be searched
            until the ratio between total neighbors within the radius and
            the total candidates becomes less than this value unless it is
            terminated by hash length reaching min_hash_match.
        :param random_state : int, RandomState instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used by np.random.
        """
        self.tree_algorithm = tree_algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        # TODO: add pre-computed distance matrix
        self.precompute_distances = precompute_distances

        # For LSHForest
        self.n_estimators = n_estimators
        self.radius = radius
        self.n_candidates = n_candidates
        self.n_neighbors = n_neighbors,
        self.min_hash_match = min_hash_match
        self.radius_cutoff_ratio = radius_cutoff_ratio
        self.random_state = random_state

        self.nn_tree_ = None
        self.ball_oracle_ = None
        self.fitted_data_ = None
        self.data_weight_ = None
        self.diam_ = None
        self.n_samples_, self.n_features_ = None, None
        self.new_data = True

        # variables used for supporting fast densest-ball query and removing
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
            elif X.shape[0] < 40:
                self.ball_oracle_ = lambda cs, r: _brute_force_ball(X, cs, r)
            else:
                self.nn_tree_ = BallTree(X, leaf_size=self.leaf_size, metric=self.metric)
                self.ball_oracle_ = lambda cs, r: self.nn_tree_.query_radius(cs, r, return_distance=False)

        elif self.tree_algorithm == 'kd_tree':
            self.nn_tree_ = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
            self.ball_oracle_ = lambda cs, r: self.nn_tree_.query_radius(cs, r, return_distance=False)

        elif self.tree_algorithm == 'ball_tree':
            self.nn_tree_ = BallTree(X, leaf_size=self.leaf_size, metric=self.metric)
            self.ball_oracle_ = lambda cs, r: self.nn_tree_.query_radius(cs, r, return_distance=False)

        elif self.tree_algorithm == 'lsh':
            self.nn_tree_ = LSHForest(n_estimators=self.n_estimators,
                                      radius=self.radius,
                                      n_candidates=self.n_candidates,
                                      n_neighbors=self.n_neighbors,
                                      min_hash_match=self.min_hash_match,
                                      radius_cutoff_ratio=self.radius_cutoff_ratio,
                                      random_state=self.random_state)
            self.nn_tree_.fit(X)
            self.ball_oracle_ = lambda cs, r: self.nn_tree_.radius_neighbors(cs, r, return_distance=False)

        elif self.tree_algorithm == 'brute':
            self.ball_oracle_ = lambda cs, r: _brute_force_ball(X, cs, r)

        else:
            raise ValueError("tree_algorithm \"{}\" not properly specified".
                             format(self.tree_algorithm))

        self.fitted_data_ = X
        self.n_samples_, self.n_features_ = X.shape
        self.data_weight_ = sample_weight if sample_weight is not None else np.ones(X.shape[0], dtype=np.int)

        return self

    def ball(self, centers, radius):
        """
        Query the data points in X that are within distance radius to centers
        :param centers: queried points
        :param radius: radius of the ball
        :return: an array of array,
            indices for each center in centers
        """
        if self.ball_oracle_ is None:
            raise NotFittedError("Tree hasn't been fitted yet\n")

        centers = check_array(centers, ensure_2d=True)
        return self.ball_oracle_(centers, radius)

    def densest_ball(self, radius, except_for=None):
        """

        :param radius:
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

    def densest_ball_faster_but_dirty(self, radius, except_for, changed):
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
        :return (densest_center, densest_ball): (array of shape=(n_features,), array of shape=(n_covered,)
            the center of the densest ball as well as the index of points the ball covers:
        """
        # TODO: what is the actual complexity of radius_query for BallTree, KDTree, or LSHForest?
        # They said that if I can run fast enough, sadness wouldn't catch me up.

        if self.ball_radius_cache_ != radius:
            # new search begins, should refresh all caches
            self.ball_radius_cache_ = radius
            self.ball_weight_cache_ = [None] * self.n_samples_
            self.ball_size_cache_ = [None] * self.n_samples_
            self.ball_cache_ = [None] * self.n_samples_
            self.cache_inconsistent_ = False

        if self.cache_inconsistent_:
            warnings.warn("Cache is inconsistent, may get outdated result\n", UserWarning)

        if self.ball_weight_cache_ is None:
            self.ball_weight_cache_ = [None] * self.n_samples_

        if except_for is None:
            except_for = {}
        if changed is None:
            changed = range(self.n_samples_)

        # the intersection between changed and except_for should be empty
        for i in changed:
            if i in except_for:
                continue

            x = self.fitted_data_[i]
            if self.ball_cache_[i] is None:
                self.ball_cache_[i] = set(self.ball_oracle_(x.reshape(1, -1), radius)[0])

            # update ball cache
            if len(except_for) / len(self.ball_cache_[i]) > 10:
                self.ball_cache_[i].difference_update(except_for.intersection(self.ball_cache_[i]))
            else:
                self.ball_cache_[i].difference_update(except_for)

            self.ball_size_cache_[i] = len(self.ball_cache_[i])
            self.ball_weight_cache_[i] = self.data_weight_[list(self.ball_cache_[i])].sum()
            # if a ball covers all points, then it must be the densest one
            if self.ball_size_cache_[i] == self.n_samples_ - len(except_for):
                self.cache_inconsistent_ = True
                return self.fitted_data_[i], self.ball_cache_[i]

        densest_idx = None
        densest_ball_weight = 0

        for i in range(self.n_samples_):
            if i in except_for:
                continue

            # because ball_cache can become inconsistent due to early returning
            if self.ball_cache_[i] is None:
                self.ball_cache_[i] = set(self.ball_oracle_(x.reshape(1, -1), radius)[0])
                self.ball_cache_[i].difference_update(except_for)
                self.ball_size_cache_[i] = len(self.ball_cache_[i])
                self.ball_weight_cache_[i] = self.data_weight_[list(self.ball_cache_[i])].sum()

            # if a ball covers all points, then it must be the densest one
            # this serves as an early return, but
            if self.ball_size_cache_[i] == self.n_samples_ - len(except_for):
                self.cache_inconsistent_ = True
                return self.fitted_data_[i], self.ball_cache_[i]

            if densest_ball_weight < self.ball_weight_cache_[i]:
                densest_ball_weight = self.ball_weight_cache_[i]
                densest_idx = i

        return self.fitted_data_[densest_idx], self.ball_cache_[densest_idx]

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
        """

        Time complexity: O(n_samples * n_features)
        :param x:
        :param return_distance:
        :return:
        """
        if not self.is_fitted:
            raise NotFittedError("Tree hasn't been fitted yet\n")
        return farthest_neighbor(x, X=self.fitted_data_, return_distance=return_distance)

    def kneighbors(self, centers, k=1, return_distance=True):
        """

        Time complexity: depends on the tree alg used
            brute - O(n_samples * n_features)
            ball_tree - O(n_features * \log n_samples)
            kd_tree - O(n_features * \log n_samples) for small n_features and O(n_samples * n_features)
                      for large n_features
            lsh - o(n_samples * n_features)
        :param centers: array of shape=(n_queries, n_features)
        :param k: queried number of nearest neighbors
        :param return_distance: default True
        :return: idxs or (dists, idxs)
            idx - the indices of the nearest neighbor in the fitted data set
            dists - the corresponding distance of nearest neighbors
        """
        if not self.is_fitted:
            raise NotFittedError("Tree hasn't been fitted yet\n")

        centers = check_array(centers, ensure_2d=True)
        if self.tree_algorithm == 'kd_tree' or self.tree_algorithm == 'ball_tree':
            return self.nn_tree_.query(centers, k, return_distance)
        elif self.tree_algorithm == 'lsh':
            return self.nn_tree_.kneighbors(centers, k, return_distance)
        elif self.tree_algorithm == 'brute':
            dists = np.linalg.norm(self.fitted_data_ - centers, axis=1)
            nearest = np.argsort(dists)[:k]
            return (dists[nearest], nearest) if return_distance else nearest
        else:
            raise ValueError("Tree algorithm `{}` unknown\n".format(self.tree_algorithm))

