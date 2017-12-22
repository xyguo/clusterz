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

    def fit(self, X):
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

    def find_densest_ball(self, radius, exceptfor=None):
        """

        :param radius:
        :param exceptfor: array-like, indices of points that should not be considered
        :return:
        """
        if exceptfor is None or len(exceptfor) == 0:
            if self.nn_tree_ is None and self.tree_algorithm != 'brute':
                raise NotFittedError("Tree hasn't been fitted yet\n")

            if self.tree_algorithm == 'brute':
                densest_idx = np.argmax(self._brute_force_ball(x, radius, count_only=True)
                                        for x in self.fitted_data_)
                densest_center = self.fitted_data_[densest_idx]
                return self._brute_force_ball(densest_center, radius)

            elif self.tree_algorithm == 'kd_tree' or self.tree_algorithm == 'ball_tree':
                densest_idx = np.argmax(self.nn_tree_.query_radius(x, radius, count_only=True)
                                        for x in self.fitted_data_)
                densest_center = self.fitted_data_[densest_idx]
                return self.nn_tree_.query_radius(densest_center, radius)

            elif self.tree_algorithm == 'lsh':
                densest_idx = np.argmax(
                    len(self.nn_tree_.radius_neighbors(x, radius, return_distance=False))
                    for x in self.fitted_data_)
                densest_center = self.fitted_data_[densest_idx]
                return self.nn_tree_.radius_neighbors(densest_center,
                                                      radius, return_distance=False)

            else:
                raise ValueError("Tree algorithm `{}` unknown\n".format(self.tree_algorithm))
        else:
            exceptfor = set(exceptfor)
            if self.nn_tree_ is None and self.tree_algorithm != 'brute':
                raise NotFittedError("Tree hasn't been fitted yet\n")

            if self.tree_algorithm == 'brute':
                densest_idx = np.argmax(
                    len(set(self._brute_force_ball(x, radius, count_only=True)).
                        difference(exceptfor))
                    for i, x in enumerate(self.fitted_data_) if i not in exceptfor)
                densest_center = self.fitted_data_[densest_idx]
                densest_ball = set(self._brute_force_ball(densest_center, radius)).\
                    difference(exceptfor)
                return np.array(list(densest_ball))

            elif self.tree_algorithm == 'kd_tree' or self.tree_algorithm == 'ball_tree':
                densest_idx = np.argmax(
                    len(set(self.nn_tree_.query_radius(x, radius, count_only=True)).
                        difference(exceptfor))
                    for i, x in enumerate(self.fitted_data_) if i not in exceptfor)
                densest_center = self.fitted_data_[densest_idx]
                densest_ball = set(self.nn_tree_.query_radius(densest_center, radius)).\
                    difference(exceptfor)
                return np.array(list(densest_ball))
            elif self.tree_algorithm == 'lsh':
                densest_idx = np.argmax(
                    len(set(self.nn_tree_.radius_neighbors(x, radius, return_distance=False)).
                        difference(exceptfor))
                    for i, x in enumerate(self.fitted_data_) if i not in exceptfor)
                densest_center = self.fitted_data_[densest_idx]
                densest_ball = set(self.nn_tree_.radius_neighbors(densest_center, radius)).\
                    difference(exceptfor)
                return np.array(list(densest_ball))
            else:
                raise ValueError("Tree algorithm `{}` unknown\n".format(self.tree_algorithm))

    def _brute_force_ball(self, center, radius, exceptfor=None, count_only=False):
        """

        :param center:
        :param radius:
        :param exceptfor: array-like
        :param count_only:
        :return:
        """
        if exceptfor is None:
            if count_only:
                return np.sum(np.linalg.norm(self.fitted_data_ - center, axis=1) <= radius)
            else:
                return np.where(np.linalg.norm(self.fitted_data_ - center, axis=1) <= radius)[0]
        else:
            idxs = np.where(np.linalg.norm(self.fitted_data_ - center, axis=1) <= radius)[0]
            if count_only:
                return len(set(idxs).difference(exceptfor))
            else:
                return np.array(list(set(idxs).difference(exceptfor)))

    def estimate_opt_range(self, sample_weight=None):
        """

        :param X:
        :param sample_weight:
        :param metric:
        :return:
        """
        if self.nn_tree_ is None and self.tree_algorithm != 'brute':
            raise NotFittedError("Tree hasn't been fitted yet\n")
        return None, None

