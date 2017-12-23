import warnings
import numpy as np
from sklearn.neighbors import KDTree, BallTree, LSHForest
from sklearn.exceptions import NotFittedError


class DistQueryOracle(object):

    def __init__(self,
                 tree_algorithm='auto', leaf_size=40, p=2,
                 metric='minkowski',
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

        # For LSHForest
        self.n_estimators = n_estimators
        self.radius = radius
        self.n_candidates = n_candidates
        self.n_neighbors = n_neighbors,
        self.min_hash_match = min_hash_match
        self.radius_cutoff_ratio = radius_cutoff_ratio
        self.random_state = random_state

        self.nn_tree_ = None
        self.fitted_data_ = None
        self.data_weight_ = None
        self.diam_ = None
        self.new_data = True

    @property
    def is_fitted(self):
        return self.nn_tree_ is not None

    def fit(self, X, sampe_weight=None):
        """

        :param X: array of shape=(n_samples, n_features),
            Data set to be processed

        :param sample_weight: array of shape=(n_samples,),
            weight on each data instance. "None" means uniform weights.

        :return:
        """
        if self.tree_algorithm == 'auto':
            self.nn_tree_ = BallTree(X, leaf_size=self.leaf_size, metric=self.metric)
            warnings.warn("Currently `auto` defaults to KD-Tree\n", UserWarning)
        elif self.tree_algorithm == 'kd_tree':
            self.nn_tree_ = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
        elif self.tree_algorithm == 'ball_tree':
            self.nn_tree_ = BallTree(X, leaf_size=self.leaf_size, metric=self.metric)
        elif self.tree_algorithm == 'lsh':
            self.nn_tree_ = LSHForest(n_estimators=self.n_estimators,
                                      radius=self.radius,
                                      n_candidates=self.n_candidates,
                                      n_neighbors=self.n_neighbors,
                                      min_hash_match=self.min_hash_match,
                                      radius_cutoff_ratio=self.radius_cutoff_ratio,
                                      random_state=self.random_state)
            self.nn_tree_.fit(X)
        elif self.tree_algorithm == 'brute':
            pass
        else:
            raise ValueError("tree_algorithm \"{}\" not properly specified".
                             format(self.tree_algorithm))

        self.fitted_data_ = X
        self.data_weight_ = sampe_weight if sampe_weight is not None else np.ones(X.shape[0])

        return self

    def ball(self, center, radius):
        """
        Query the data points in X that are within distance radius to center
        :param center: queried point
        :param radius: radius of the ball
        :return: an array of index in the fitted dataset X
        """
        if self.nn_tree_ is None and self.tree_algorithm != 'brute':
            raise NotFittedError("Tree hasn't been fitted yet\n")
        if self.tree_algorithm == 'brute':
            return self._brute_force_ball(center, radius)
        elif self.tree_algorithm == 'kd_tree' or self.tree_algorithm == 'ball_tree':
            return self.nn_tree_.query_radius(center, radius, return_distance=False)
        elif self.tree_algorithm == 'lsh':
            return self.nn_tree_.radius_neighbors(center, radius, return_distance=False)
        else:
            raise ValueError("Tree algorithm `{}` unknown\n".format(self.tree_algorithm))

    def densest_ball(self, radius, except_for=None):
        """

        :param radius:
        :param except_for: array-like, indices of points that should not be considered
        :return: array of index of points in the fitted data set
        """
        densest_ball = None
        if except_for is None or len(except_for) == 0:
            except_for = None

        if self.nn_tree_ is None and self.tree_algorithm != 'brute':
            raise NotFittedError("Tree hasn't been fitted yet\n")

        if self.tree_algorithm == 'brute':
            densest_ball = self._brute_find_densest_ball(radius, except_for)
        elif self.tree_algorithm == 'kd_tree':
            densest_ball = self._kd_tree_find_densest_ball(radius, except_for)
        elif self.tree_algorithm == 'ball_tree':
            densest_ball = self._ball_tree_find_densest_ball(radius, except_for)
        elif self.tree_algorithm == 'lsh':
            densest_ball = self._lsh_find_densest_ball(radius, except_for)
        else:
            raise ValueError("Tree algorithm `{}` unknown\n".format(self.tree_algorithm))

        if except_for is not None:
            densest_ball = np.array(list(set(densest_ball).difference(except_for)))

        return densest_ball

    def _find_densest_ball(self, radius, except_for, tree_method, **kwargs):
        """
        
        :param x: 
        :param radius: 
        :param except_for: 
        :param tree_method:
            tree_method should return only the count of points in the ball if except_for is None,
            otherwise it should return the actual indices of points in the ball
        :param kwargs: 
        :return: the center of the densest ball
        """

        if except_for is None:
            densest_idx = np.argmax(
                self.data_weight_[tree_method(x, radius, **kwargs)].sum()
                for i, x in enumerate(self.fitted_data_))
        else:
            densest_idx = np.argmax(
                    self.data_weight_[list(set(tree_method(x, radius, **kwargs)).
                                           difference(except_for))].sum()
                    for i, x in enumerate(self.fitted_data_) if i not in except_for)
        densest_center = self.fitted_data_[densest_idx]
        return densest_center

    def _brute_find_densest_ball(self, radius, except_for):
        # TODO: estimate time complexity
        additional_kwargs = dict()
        center = self._find_densest_ball(radius, except_for, self._brute_force_ball, **additional_kwargs)
        return self._brute_force_ball(center, radius)

    def _kd_tree_find_densest_ball(self, radius, except_for):
        # TODO: estimate time complexity
        additional_kwargs = dict()
        center = self._find_densest_ball(radius, except_for, self.nn_tree_.query_radius, **additional_kwargs)
        return self.nn_tree_.query_radius(center, radius)

    def _ball_tree_find_densest_ball(self, radius, except_for):
        # TODO: estimate time complexity
        additional_kwargs = dict()
        center = self._find_densest_ball(radius, except_for, self.nn_tree_.query_radius, **additional_kwargs)
        return self.nn_tree_.query_radius(center, radius)

    def _lsh_find_densest_ball(self, radius, except_for):
        # TODO: estimate time complexity
        additional_kwargs = {'return_distance': False}
        center = self._find_densest_ball(radius, except_for, self.nn_tree_.radius_neighbors, **additional_kwargs)
        return self.nn_tree_.radius_neighbors(center, radius)

    def _brute_force_ball(self, center, radius, except_for=None, count_only=False):
        """

        :param center:
        :param radius:
        :param except_for: array-like
        :param count_only:
        :return:
        """
        if except_for is None:
            if count_only:
                return np.sum(np.linalg.norm(self.fitted_data_ - center, axis=1) <= radius)
            else:
                return np.where(np.linalg.norm(self.fitted_data_ - center, axis=1) <= radius)[0]
        else:
            idxs = np.where(np.linalg.norm(self.fitted_data_ - center, axis=1) <= radius)[0]
            if count_only:
                return len(set(idxs).difference(except_for))
            else:
                return np.array(list(set(idxs).difference(except_for)))

    def estimate_diameter(self, n_estimation=None):
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

        if self.metric is 'minkowski':
            if n_estimation is None:
                lb, _ = self.farthest_neighbor(self.fitted_data_[0], return_distance=True)
            else:
                n_samples, _ = self.fitted_data_.shape[0]
                estimations = [0] + list(np.random.randint(1, n_samples))
                lb = min(self.farthest_neighbor(self.fitted_data_[i], return_distance=True)
                         for i in estimations)
            self.diam_ = (lb, 2 * lb)
        else:
            raise ValueError("metric `{}` currently not supported\n".format(self.metric))
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
        dists = np.linalg.norm(self.fitted_data_ - x, axis=1)
        farthest = np.argmax(dists)
        return (dists[farthest], farthest) if return_distance else farthest

    def nearest_neighbor(self, x, return_distance=True):
        """

        Time complexity: depends on the tree alg used
            brute - O(n_samples * n_features)
            ball_tree - O(n_features * \log n_samples)
            kd_tree - O(n_features * \log n_samples) for small n_features and O(n_samples * n_features)
                      for large n_features
            lsh - o(n_samples * n_features)
        :param x:
        :param return_distance:
        :return:
        """
        if not self.is_fitted:
            raise NotFittedError("Tree hasn't been fitted yet\n")

        if self.tree_algorithm == 'kd_tree' or self.tree_algorithm == 'ball_tree':
            return self.nn_tree_.query(x, 1, return_distance)
        elif self.tree_algorithm == 'lsh':
            return self.nn_tree_.kneighbors(x, 1, return_distance)
        elif self.tree_algorithm == 'brute':
            dists = np.linalg.norm(self.fitted_data_ - x, axis=1)
            nearest = np.argmin(dists)
            return (dists[nearest], nearest) if return_distance else nearest
        else:
            raise ValueError("Tree algorithm `{}` unknown\n".format(self.tree_algorithm))
