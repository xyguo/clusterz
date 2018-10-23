# -*- coding: utf-8 -*-
"""Algorithm for distributed (k,z)-center"""

# Author: Xiangyu Guo     xiangyug[at]buffalo.edu
#         Shi Li          shil[at]buffalo.edu

import warnings
import numpy as np
from time import time

from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin_min
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array
from .misc import DistQueryOracle, distributedly_estimate_diameter, farthest_neighbor
from ..utils import evenly_partition, debug_print


class DistributedKZCenter(object):

    def __init__(self, algorithm='multiplicative',
                 n_clusters=None, n_outliers=None, n_machines=None,
                 epsilon=0.1, delta=0.01, sample_size=None,
                 machine_multi=False, local_data_size=None,
                 random_state=None, debug=False):
        """
        Serves as the master node.

        :param algorithm: str from {'additive', 'multiplicative', 'moseley'}
            which algorithm to use.

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

        :param sample_size: int. default None.
            sample size for the 'additive' mode.

        :param random_state: numpy.RandomState

        :param debug: boolean, whether output debugging information
        """
        self.available_algorithms_ = {
            'multiplicative': self.fit_icml2018_multiplicative_,
            'additive': self.fit_icml2018_additive_,
            'moseley': self.fit_moseley_,
            'random': self.fit_randomly_,
            'random_charikar': self.fit_charikar_,
            'auto': self.fit_auto_
        }
        if algorithm in self.available_algorithms_:
            self.algorithm = algorithm
        else:
            raise ValueError("Unsupported algorithm {}\n".format(algorithm))

        self.n_clusters_ = n_clusters
        self.n_outliers_ = n_outliers
        self.n_machines_ = n_machines
        self.epsilon_ = epsilon
        self.delta_ = delta
        self.sample_size_ = sample_size
        self.random_state = random_state
        self.debugging = debug
        self.machine_multi_ = machine_multi
        self.local_data_size_ = local_data_size
        if self.machine_multi_ and local_data_size is None:
            self.local_data_size_ = 10000

        self.cluster_centers_ = None
        self.n_samples_, self.n_features_ = None, None
        self.opt_ = None
        self.communication_cost_ = None

    @property
    def communication_cost(self):
        return self.communication_cost_

    def cost(self, X, remove_outliers=None):
        """

        :param X: array,
            data set
        :param remove_outliers: None or int, default None
            whether to remove outliers when computing the cost on X
        :return: float,
            actual cost
        """
        if self.cluster_centers_ is None:
            raise NotFittedError("Model hasn't been fitted yet\n")
        X = check_array(X, ensure_2d=True)
        _, dists = pairwise_distances_argmin_min(X, self.cluster_centers_, axis=1)
        dists.sort()
        if remove_outliers is not None:
            assert remove_outliers >= 0
            return dists[-int(remove_outliers + 1)]
        else:
            return dists[-(int((1 + self.epsilon_) * self.n_outliers_) + 1)]

    def machine_multiplication_(self, Xs, shuffle=False):
        """make super machines"""
        if not self.machine_multi_:
            raise AttributeError("Machine multiplication hasn't been enabled.\n")
        multiplied_Xs = []

        for i, X in enumerate(Xs):
            if X.shape[0] > self.local_data_size_:
                multi = int(np.ceil(X.shape[0] / self.local_data_size_))
                multi_X = evenly_partition(X, n_machines=multi, shuffle=shuffle)
                multiplied_Xs += multi_X
            else:
                multiplied_Xs.append(X)
        return multiplied_Xs

    def fit(self, Xs, sample_weights=None, dist_oracles=None):
        """

        :param Xs: list of arrays of shape=(n_samples_i, n_features),
            Divided data set. Each array in the list represents a bunch of data
            that has been partitioned onto one machine.
        :param sample_weights: list of arrays of shape=(n_samples_i,),
            sample weights for each machine's data
        :param dist_oracles: list of distance oracles,
            If not provided then the class will create oracles by its own
        :return self:
        """
        self.n_machines_ = len(Xs)
        self.n_samples_ = sum(X.shape[0] for X in Xs)
        self.n_features_ = Xs[0].shape[1]

        return self.available_algorithms_[self.algorithm](Xs, sample_weights, dist_oracles)

    def fit_auto_(self, Xs, sample_weights=None, dist_oracles=None):
        if self.n_outliers_ > self.epsilon_ * self.n_samples_:
            return self.fit_icml2018_additive_(Xs, sample_weights, dist_oracles)
        else:
            return self.fit_icml2018_multiplicative_(Xs, sample_weights, dist_oracles)

    def fit_randomly_(self, Xs, sample_weights=None, dist_oracles=None):
        """Baseline method: randomly select k+z points from each machine,
        then randomly choose k points as final cluster centers from the m(k+z) sampled points.
        """
        aggregated_samples = [np.random.choice(X.shape[0], size=self.n_clusters_ + self.n_outliers_, replace=False)
                              for X in Xs]
        aggregated_samples = np.vstack(X[aggregated_samples[i]] for i, X in enumerate(Xs))
        assert aggregated_samples.shape[0] <= self.n_machines_ * (self.n_clusters_ + self.n_outliers_)
        assert aggregated_samples.shape[1] == self.n_features_

        final_samples = np.random.choice(aggregated_samples.shape[0],
                                         size=self.n_clusters_, replace=False)
        final_samples = aggregated_samples[final_samples]
        self.cluster_centers_ = final_samples
        self.communication_cost_ = aggregated_samples.shape[0] * aggregated_samples.shape[1]

        return self

    def fit_charikar_(self, Xs, sample_weights=None, dist_oracles=None):
        """Baseline method: randomly select k+z points from each machine,
        then run Charikar's method on these m(k+z) sampled points.
        """
        aggregated_samples = [np.random.choice(X.shape[0], size=self.n_clusters_ + self.n_outliers_, replace=False)
                              for X in Xs]
        aggregated_samples = np.vstack(X[aggregated_samples[i]] for i, X in enumerate(Xs))
        assert aggregated_samples.shape[0] <= self.n_machines_ * (self.n_clusters_ + self.n_outliers_)
        assert aggregated_samples.shape[1] == self.n_features_

        results = kzcenter_charikar(aggregated_samples, sample_weight=None, guessed_opt=None,
                                    n_clusters=self.n_clusters_, n_outliers=self.n_outliers_,
                                    dist_oracle=DistQueryOracle(tree_algorithm='ball_tree'),
                                    densest_ball_radius=1, removed_ball_radius=3)
        self.cluster_centers_ = np.array([c for c, _ in results])
        self.communication_cost_ = aggregated_samples.shape[0] * aggregated_samples.shape[1]
        return self

    def fit_moseley_(self, Xs, sample_weights=None, dist_oracles=None):
        """
        implementation for our Moseley's NIPS2015 paper
        """
        if sample_weights is None:
            sample_weights = [None] * len(Xs)

        self.n_machines_ = len(Xs)

        self.n_samples_ = sum(X.shape[0] for X in Xs)
        self.n_features_ = Xs[0].shape[1]
        # if the number of allowed outliers are more than n_samples then we
        # simply choose arbitrary n_clusters points in the data set as centers
        if self.n_samples_ <= (1 + self.epsilon_) * self.n_outliers_:
            self.cluster_centers_ = []
            while len(self.cluster_centers_) < min(self.n_clusters_, self.n_samples_):
                for X in Xs:
                    for i in range(min(X.shape[0], self.n_clusters_)):
                        self.cluster_centers_.append(X[i])
            self.cluster_centers_ = np.array(self.cluster_centers_)
            return self

        debug_print("Fit each machine's local data ...", debug=self.debugging)
        mappers = []
        for i in range(self.n_machines_):
            debug_print("\tFit {}th machine's local data ...".format(i), debug=self.debugging)
            mappers.append(KZCenter(algorithm='greedy',
                                    n_clusters=self.n_clusters_ + self.n_outliers_,
                                    n_outliers=0,
                                    n_machines=self.n_machines_, epsilon=self.epsilon_)
                           .fit(X=Xs[i], sample_weight=sample_weights[i],
                                dist_oracle=None,
                                guessed_opt=None))

        # construct the centralized data set
        debug_print("Aggregate results on reducer ...", debug=self.debugging)
        X = np.vstack([m.cluster_centers for m in mappers])
        sample_weight = np.hstack([m.clusters_size for m in mappers])
        assert X.shape[0] <= self.n_machines_ * (self.n_clusters_ + self.n_outliers_)
        assert X.shape[1] == self.n_features_

        debug_print("Fit the aggregated data set with shape={}...".format(X.shape), debug=self.debugging)
        results = kzcenter_charikar(X, sample_weight=sample_weight, guessed_opt=None,
                                    n_clusters=self.n_clusters_, n_outliers=self.n_outliers_,
                                    dist_oracle=DistQueryOracle(tree_algorithm='ball_tree'),
                                    densest_ball_radius=5, removed_ball_radius=11)
        self.cluster_centers_ = np.array([c for c, _ in results])
        self.communication_cost_ = X.shape[0] * X.shape[1]
        return self

    def fit_icml2018_additive_(self, Xs, sample_weights=None, dist_oracles=None):
        """
        implementation for the `distr-kz-ctl-additive` method in our ICML2018 paper
        """
        if sample_weights is None:
            sample_weights = [None] * len(Xs)

        # Round 1
        self.n_machines_ = len(Xs)
        self.n_samples_ = sum(X.shape[0] for X in Xs)
        self.n_features_ = Xs[0].shape[1]

        debug_print("Determing the size of samples", debug=self.debugging)
        # Round 2
        # compute the aggregated sample size: n'
        if not self.sample_size_:
            n_final_samples_ = (self.n_clusters_ * self.n_features_ * np.log(1 / self.delta_))
            n_final_samples_ = (n_final_samples_ / (self.epsilon_ ** 2)) * np.log(n_final_samples_ / self.epsilon_)
            n_final_samples_ = max(self.n_machines_ * self.n_clusters_, n_final_samples_)
            n_final_samples_ = min(self.n_machines_ * (self.n_clusters_ + self.n_outliers_) / 2,
                                   self.n_samples_ / self.n_machines_,
                                   self.n_machines_ * self.n_clusters_ * (1 + 1 / self.epsilon_),
                                   n_final_samples_)
            n_final_samples_ = int(n_final_samples_)
        else:
            n_final_samples_ = self.sample_size_

        # determine the final sample size on each machine, i.e. n'_i
        p = [X.shape[0] / self.n_samples_ for X in Xs]
        samples_sizes = np.random.choice(self.n_machines_, size=n_final_samples_, replace=True, p=p)
        _, samples_sizes_for_each_machine = np.unique(samples_sizes, return_counts=True)

        # Round 3
        # each machine creates its own final samples
        samples_for_each_machine = [np.random.choice(X.shape[0], size=samples_sizes_for_each_machine[i], replace=True)
                                    for i, X in enumerate(Xs)]
        # aggregate data
        debug_print("Aggregate results on reducer ...", debug=self.debugging)
        final_X = [X[samples_for_each_machine[i]] for i, X in enumerate(Xs)]
        final_X = np.vstack(final_X)
        n_final_outliers_ = self.n_outliers_ * n_final_samples_ / self.n_samples_ + self.epsilon_ * n_final_samples_ / 2

        debug_print("Fit the aggregated data set with shape {}...".format(final_X.shape), debug=self.debugging)
        reducer = KZCenter(algorithm='charikar', n_clusters=self.n_clusters_, n_outliers=n_final_outliers_,
                           n_machines=self.n_machines_, random_state=self.random_state)
        reducer.fit(X=final_X, sample_weight=None,
                    dist_oracle=DistQueryOracle(tree_algorithm='ball_tree'), guessed_opt=None)

        self.cluster_centers_ = np.atleast_2d(reducer.cluster_centers)
        self.communication_cost_ = final_X.shape[0] * final_X.shape[1]

        return self

    def fit_icml2018_multiplicative_(self, Xs, sample_weights=None, dist_oracles=None):
        """
        implementation for the `distr-kz-ctl-multiplicaive` method in our ICML2018 paper
        """
        if self.machine_multi_:
            debug_print("Machine multiplication ...", debug=self.debugging)
            Xs = self.machine_multiplication_(Xs)
            if sample_weights:
                sample_weights = self.machine_multiplication_(sample_weights)
            debug_print("\t got {} machines...".format(len(Xs)), debug=self.debugging)
        self.n_machines_ = len(Xs)
        if not sample_weights:
            sample_weights = [None] * len(Xs)

        t1 = time()
        debug_print("Build distance query oracle ...", debug=self.debugging)
        if dist_oracles is not None and len(dist_oracles) == self.n_machines_:
            oracles = dist_oracles
        else:
            oracles = []
            for i in range(self.n_machines_):
                oracles.append(DistQueryOracle(tree_algorithm='auto',
                                               leaf_size=min(Xs[i].shape[0] // self.n_clusters_, 60))
                               .fit(Xs[i]))
            print("Fitting oracles takes time {}".format(time()-t1))

        debug_print("Initialize mappers ...", debug=self.debugging)
        mappers = []
        for i in range(self.n_machines_):
            mappers.append(
                KZCenter(algorithm='greedy_covering', n_clusters=self.n_clusters_, n_outliers=self.n_outliers_,
                         n_machines=self.n_machines_, epsilon=self.epsilon_, debug=self.debugging))

        self.n_samples_ = sum(X.shape[0] for X in Xs)
        self.n_features_ = Xs[0].shape[1]
        # if the number of allowed outliers are more than n_samples then we
        # simply choose arbitrary n_clusters points in the data set as centers
        if self.n_samples_ <= (1 + self.epsilon_) * self.n_outliers_:
            self.cluster_centers_ = []
            while len(self.cluster_centers_) < min(self.n_clusters_, self.n_samples_):
                for X in Xs:
                    for i in range(min(X.shape[0], self.n_clusters_)):
                        self.cluster_centers_.append(X[i])
            self.cluster_centers_ = np.array(self.cluster_centers_)
            return self

        # estimating the optimal radius using binary search
        n_guesses = 1  # used for debugging
        total_iters_at_most = self.n_clusters_ * self.n_machines_ * (1 + 1 / self.epsilon_)
        total_covered_at_least = max(self.n_samples_ - (1 + self.epsilon_) * self.n_outliers_, 1)
        lb = 0
        # upper bound is initialized as the sum of diameters across all machines
        _, ub = distributedly_estimate_diameter(Xs, n_estimation=10)
        debug_print("Start guessing optimal radius in [{},{}]...".format(lb, ub),
                    debug=self.debugging)
        guessed_opt = (lb + ub) / 2
        opt_equal_ub = False
        t2 = time()
        while ub > (1 + self.delta_) * lb and ub > 1e-3:
            debug_print("{}-th guess: trying with OPT = {}".format(n_guesses, guessed_opt), debug=self.debugging)
            for i, m in enumerate(mappers):
                debug_print("\t{}th machine".format(i), debug=self.debugging)
                m.fit(X=Xs[i], sample_weight=sample_weights[i],
                      dist_oracle=oracles[i],
                      guessed_opt=guessed_opt)
            total_iters = sum(m.n_iters for m in mappers)
            total_covered = sum(m.n_covered for m in mappers)

            # if the number of balls constructed is too large, or
            # the number of points covered is to small, then enlarge guessed opt
            if total_iters > total_iters_at_most or total_covered < total_covered_at_least:
                lb = guessed_opt
            else:
                ub = guessed_opt
                if ub <= (1 + self.delta_) * lb or ub <= 1e-3:
                    opt_equal_ub = True
                    break
            guessed_opt = (lb + ub) / 2
            n_guesses += 1

        if ub < 1e-3:
            warnings.warn("guessed OPT={} is smaller than 1e-3\n".format(ub), UserWarning)
        debug_print("Estimated optimal radius: {}".format(ub), debug=self.debugging)
        self.opt_ = ub

        # construct the centralized data set
        debug_print("Aggregate results on reducer ...", debug=self.debugging)
        if not opt_equal_ub:
            for i, m in enumerate(mappers):
                m.fit(X=Xs[i], sample_weight=sample_weights[i], guessed_opt=self.opt_)
        X = np.vstack([m.cluster_centers for m in mappers])
        sample_weight = np.hstack([m.clusters_size for m in mappers])
        assert X.shape[0] <= total_iters_at_most

        print("Covering data on all the machines takes time {}".format(time()-t2))
        t3 = time()
        debug_print("Expected aggregated data size {}, got {}...".format(total_iters_at_most, X.shape),
                    debug=self.debugging)
        debug_print("Fit the aggregated data set with shape {}...".format(X.shape), debug=self.debugging)
        reducer = KZCenter(algorithm='charikar', n_clusters=self.n_clusters_,
                           n_outliers=(1 + self.epsilon_) * self.n_outliers_, n_machines=self.n_machines_,
                           random_state=self.random_state)
        reducer.fit(X=X, sample_weight=sample_weight,
                    dist_oracle=DistQueryOracle(tree_algorithm='ball_tree'),
                    guessed_opt=5 * ub)
        print("Fitting the aggregated data takes time {}".format(time()-t3))

        self.cluster_centers_ = np.atleast_2d(reducer.cluster_centers)
        self.communication_cost_ = X.shape[0] * self.n_features_

        return self


class KZCenter(object):

    def __init__(self, algorithm='greedy', n_clusters=None, n_outliers=None, n_machines=None, epsilon=0.1,
                 random_state=None, debug=False):
        """
        Worker object on single machine. Can conduct pre-clustering or regular
        clustering tasks.

        :param algorithm: {'greedy', 'charikar', 'greedy_covering'}
            Algorithm used for clustering.
            'greedy': The farthest-neighbor method
            'charikar': Charikar's densest-ball removing method.
            'greedy_covering': pre-clustering method, use dense-enough balls to cover as many as possible points

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
        self.available_algorithms_ = {
            'brute': self.fit_brute_,
            'greedy': self.fit_greedy_,
            'charikar': self.fit_charikar_,
            'greedy_covering': self.fit_greedy_covering_2_,
        }
        if algorithm in self.available_algorithms_:
            self.algorithm = algorithm
        else:
            raise ValueError("Unsupported mode {}\n".format(algorithm))

        self.n_clusters_ = n_clusters
        self.n_outliers_ = n_outliers
        self.n_machines_ = n_machines
        self.epsilon_ = epsilon
        self.random_state = random_state
        self.debugging = debug

        self.facilities_ = None
        self.facility_idxs_ = None

        self.n_iters_ = None
        self.n_covered_ = None
        self.results_ = None
        self.cluster_centers_ = None
        self.clusters_size_ = None
        self.n_samples_ = None
        self.n_features_ = None

    @property
    def communication_cost(self):
        return 0

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

    def cost(self, X, remove_outliers=None):
        """

        :param X: array,
            data set
        :param remove_outliers: None or int, default None
            whether to remove outliers when computing the cost on X
        :return: float,
            actual cost
        """
        if self.cluster_centers_ is None:
            raise NotFittedError("Model hasn't been fitted yet\n")
        X = check_array(X, ensure_2d=True)
        _, dists = pairwise_distances_argmin_min(X, self.cluster_centers_, axis=1)
        dists.sort()
        if remove_outliers is not None:
            assert remove_outliers >= 0
            return dists[-int(remove_outliers + 1)]
        else:
            return dists[-(int((1 + self.epsilon_) * self.n_outliers_) + 1)]

    def fit(self, X, sample_weight=None, dist_oracle=None, guessed_opt=None):
        """
        :param X:
        :param sample_weight:
        :return:
        """
        if self.algorithm not in ['greedy', 'brute']:
            if not dist_oracle:
                dist_oracle = DistQueryOracle(tree_algorithm='auto')
            if not dist_oracle.is_fitted:
                dist_oracle.fit(X, sample_weight)

        self.n_samples_, self.n_features_ = X.shape
        self.results_ = self.available_algorithms_[self.algorithm](X, sample_weight,
                                                                   dist_oracle=dist_oracle,
                                                                   guessed_opt=guessed_opt)

        self.cluster_centers_ = np.array([c for c, _ in self.results_])
        self.clusters_size_ = np.array([w for _, w in self.results_])
        self.n_covered_ = sum(self.clusters_size_)
        return self

    def fit_greedy_(self, X, sample_weight=None, dist_oracle=None, guessed_opt=None):
        centers = kcenter_greedy(X, n_clusters=self.n_clusters_)
        closest_center, _ = pairwise_distances_argmin_min(X, centers)
        cs, counts = np.unique(closest_center, return_counts=True)
        results = [(centers[cs[i]], wc) for i, wc in enumerate(counts)]
        return results

    def fit_charikar_(self, X, sample_weight=None, dist_oracle=None, guessed_opt=None):
        results = kzcenter_charikar(X, sample_weight=sample_weight, guessed_opt=guessed_opt,
                                    n_clusters=self.n_clusters_, n_outliers=self.n_outliers_,
                                    dist_oracle=dist_oracle)
        return results

    def fit_brute_(self, X, sample_weight=None, dist_oracle=None, guessed_opt=None):
        results = kzcenter_brute(X, sample_weight,
                                 n_clusters=self.n_clusters_,
                                 n_outliers=self.n_outliers_)
        return results

    def fit_greedy_covering_(self, X, sample_weight=None, dist_oracle=None, guessed_opt=None):
        """
        Greedily cover the data set with ball of specified size
        :param guessed_opt: the radius of the ball used for compressing the data set
        :return results: list of (array, int)
            List of (ball center, #points in the ball)
        """
        threshold = self.epsilon_ * self.n_outliers_ / (self.n_clusters_ * self.n_machines_)
        n_iters_upperbound = self.n_clusters_ * self.n_machines_ * (1 + 1 / self.epsilon_) - self.n_machines_ + 1
        results = []
        removed = set()
        to_be_updated = range(self.n_samples_)
        self.n_iters_ = 0
        debug_print("\tRemoving densest ball from data set with size {}...".format(self.n_samples_),
                    debug=self.debugging)
        while len(removed) < self.n_samples_ and self.n_iters_ <= n_iters_upperbound:
            # p, covered_pts = self.dist_oracle.densest_ball(2 * guessed_opt)
            p, covered_pts = dist_oracle.densest_ball_faster_but_dirty(2 * guessed_opt,
                                                                       changed=to_be_updated,
                                                                       except_for=removed)

            if p is None:
                break
            if len(covered_pts) < threshold:
                break
            to_be_removed = dist_oracle.ball(p.reshape(1, -1), 4 * guessed_opt)[0]
            removed.update(to_be_removed)
            w_p = len(covered_pts)
            results.append((p, w_p))
            self.n_iters_ += 1
            debug_print("\t\tremove {}th ball of size {}..."
                        .format(self.n_iters_, w_p), debug=self.debugging)

            # after removing ball(p, 4L), only the densest ball whose centers resides
            # in ball(p, 6L) \ ball(p, 4L) is affected. So we only need to consider
            to_be_updated = set(dist_oracle.ball(p.reshape(1, -1), 6 * guessed_opt)[0])
            to_be_updated.difference_update(removed)

        return results

    def fit_greedy_covering_2_(self, X, sample_weight=None, dist_oracle=None,
                               guessed_opt=None, reinitialize=False):
        """
        Greedily cover the data set with ball of specified size
        :param guessed_opt: the radius of the ball used for compressing the data set
        :param X:
        :param sample_weight:
        :param reinitialize:
        :return results: list of (array, int)
            List of (ball center, #points in the ball)
        """
        if reinitialize or self.facility_idxs_ is None:
            # if data is uniformly randomly and evenly partitioned
            # n_facilities = int(self.n_clusters_ + self.n_outliers_ / self.n_clusters_ * 2)
            n_facilities = min(self.n_clusters_ + self.n_outliers_, self.n_samples_)
            if n_facilities * self.n_samples_ * self.n_features_ < 5e8:
                _, self.facility_idxs_ = kcenter_greedy(
                    X,
                    n_clusters=n_facilities,
                    return_indices=True)
            else:
                self.facility_idxs_ = np.arange(self.n_samples_)
            debug_print("\tCreate {} initial facilities...".format(len(self.facility_idxs_)),
                        debug=self.debugging)
        threshold = self.epsilon_ * self.n_outliers_ / (self.n_clusters_ * self.n_machines_)
        n_iters_upperbound = self.n_clusters_ * self.n_machines_ * (1 + 1 / self.epsilon_) - self.n_machines_ + 1

        results = []
        removed = set()
        to_be_updated = self.facility_idxs_
        self.n_iters_ = 0
        debug_print("\tRemoving dense ball from data set with size {}...".format(self.n_samples_),
                    debug=self.debugging)
        while len(removed) < self.n_samples_ and self.n_iters_ <= n_iters_upperbound:
            # p, covered_pts = self.dist_oracle.densest_ball(2 * guessed_opt)
            p, covered_pts = dist_oracle.dense_ball_(2 * guessed_opt,
                                                     changed=to_be_updated,
                                                     facility_idxs=self.facility_idxs_,
                                                     except_for=removed,
                                                     minimum_density=threshold)

            if p is None:
                break
            if len(covered_pts) < threshold:
                break
            to_be_removed = dist_oracle.ball(p.reshape(1, -1), 4 * guessed_opt)[0]
            to_be_removed = set(to_be_removed).difference(removed)
            removed.update(to_be_removed)
            w_p = len(to_be_removed)
            results.append((p, w_p))
            self.n_iters_ += 1
            # debug_print("\t\tremove {}th ball of size {}..."
            #             .format(self.n_iters_, w_p), debug=self.debugging)

            # after removing ball(p, 4L), only the densest ball whose centers resides
            # in ball(p, 6L) \ ball(p, 4L) is affected. So we only need to consider
            to_be_updated = set(dist_oracle.ball(p.reshape(1, -1), 6 * guessed_opt)[0]). \
                intersection(self.facility_idxs_)
            to_be_updated.difference_update(removed)

        return results


def kzcenter_charikar_eg(X, sample_weight=None, guessed_opt=None,
                         n_clusters=7, n_outliers=0, delta=0.05,
                         dist_oracle=None, return_opt=False,
                         densest_ball_radius=2, removed_ball_radius=4):
    """
    Charikar's Method with epsilon-net support

    :param X:

    :param sample_weight:

    :param guessed_opt:

    :param n_clusters:

    :param n_outliers:

    :param delta:

    :param dist_oracle:

    :param densest_ball_radius: int, default 2,
        find the densest ball of radius densest_ball_radius * OPT

    :param removed_ball_radius: int, default 4,
        remove the ball of radius removed_ball_radius * OPT

    :return results: list of (array, int)
            List of (ball center, #points in the ball)
    """
    if dist_oracle is None:
        dist_oracle = DistQueryOracle(tree_algorithm='auto')
    if not dist_oracle.is_fitted:
        dist_oracle.fit(X, sample_weight)
    n_distinct_points, _ = X.shape
    if sample_weight is None:
        sample_weight = np.ones(n_distinct_points)

    n_samples = sum(sample_weight)
    if n_distinct_points <= n_clusters:
        warnings.warn("Number of total distinct data points is smaller than required number of clusters.")
        return [(c, w) for c, w in zip(X, sample_weight)]

    # estimate the upperbound and lowerbound of opt for the data set
    _, ub = dist_oracle.estimate_diameter(n_estimation=10)
    lb, _ = dist_oracle.kneighbors(X[np.random.randint(0, n_distinct_points)].reshape(1, -1), k=2)
    lb = np.max(lb)

    # creat a epsilon-net for faster searching for opt
    L = max(lb, 1e-2)
    L = min(L, ub / 100)
    U = ub
    radius_grid = np.power(1+delta, np.arange(0, int(np.log(U/L) / np.log(1+delta)) + 2)) * L
    radius_grid = list(radius_grid)
    radius_grid.append(U)

    L_idx, U_idx = 0, len(radius_grid)
    if guessed_opt is not None:
        guessed_opt = min(guessed_opt, U)
        guessed_opt_idx = np.where(np.array(radius_grid) >= guessed_opt)[0].min()
    if guessed_opt is None:
        guessed_opt_idx = (L_idx + U_idx) // 2
        guessed_opt = radius_grid[guessed_opt_idx]

    results = []
    facility_idxs = np.arange(n_distinct_points)
    n_facilities = len(facility_idxs)
    n_facilities_thresh = 1e4

    while L_idx < U_idx - 1:
        removed = set()
        results = []

        to_be_updated = facility_idxs
        for i in range(n_clusters):
            if len(removed) == n_distinct_points:
                break
            # When the number of available facilities is huge, use the dense_ball_ method that
            # has caching and early-returning
            if n_facilities > n_facilities_thresh:
                p, covered_pts = dist_oracle.dense_ball_(densest_ball_radius * guessed_opt,
                                                         changed=to_be_updated,
                                                         facility_idxs=facility_idxs,
                                                         except_for=removed,
                                                         minimum_density=np.inf)
            else:
                p, covered_pts = dist_oracle.densest_ball(densest_ball_radius * guessed_opt, removed)

            to_be_removed = dist_oracle.ball(p.reshape(1, -1), removed_ball_radius * guessed_opt)[0]
            to_be_removed = set(to_be_removed).difference(removed)
            to_be_removed = np.array(list(to_be_removed))
            removed.update(to_be_removed)
            w_p = sum(sample_weight[to_be_removed])
            results.append((p, w_p))

            # after removing ball(p, aL), only the densest ball whose centers resides
            # in ball(p, (a+b)L) \ ball(p, aL) is affected. So we only need to consider
            if n_facilities > n_facilities_thresh:
                to_be_updated = set(dist_oracle.ball(p.reshape(1, -1),
                                                     (densest_ball_radius + removed_ball_radius) * guessed_opt)[0]). \
                    intersection(facility_idxs)
                to_be_updated.difference_update(removed)

        n_covered = sum(wp for _, wp in results)
        if n_covered >= n_samples - n_outliers:
            U_idx = guessed_opt_idx
            guessed_opt_idx = (L_idx + guessed_opt_idx) // 2
        else:
            L_idx = guessed_opt_idx
            guessed_opt_idx = (guessed_opt_idx + U_idx) // 2
        guessed_opt = radius_grid[guessed_opt_idx]

    # if the program finishes before finding k'<k centers, we use the FarthestNeighbor
    # method to produce the remained k-k' centers
    if len(results) < n_clusters:
        centers = [c for c, _ in results]
        _, dists_to_centers = pairwise_distances_argmin_min(X, np.atleast_2d(centers))

        for i in range(0, n_clusters - len(results)):
            next_idx = np.argmax(dists_to_centers)
            centers.append(X[next_idx])
            # TODO: here the new center's weight is set to its own weight, this might be problematic(?)
            results.append((X[next_idx], sample_weight[next_idx]))
            _, next_dist = pairwise_distances_argmin_min(X, np.atleast_2d(centers[-1]))
            dists_to_centers = np.minimum(dists_to_centers, next_dist)

    return (results, guessed_opt) if return_opt else results


def kzcenter_charikar(X, sample_weight=None, guessed_opt=None,
                      n_clusters=7, n_outliers=0, delta=0.05,
                      dist_oracle=None, return_opt=False,
                      densest_ball_radius=2, removed_ball_radius=4):
    """
    Implementation of the algorithm proposed in Moses Charikar's SODA'01 paper:

    Moses Charikar, Samir Khuller, David M. Mount, and Giri Narasimhan.
    Algorithms for facility location problems with outliers. SODA'2001

    :param X:

    :param sample_weight:

    :param guessed_opt:

    :param n_clusters:

    :param n_outliers:

    :param delta:

    :param dist_oracle:

    :param densest_ball_radius: int, default 2,
        find the densest ball of radius densest_ball_radius * OPT

    :param removed_ball_radius: int, default 4,
        remove the ball of radius removed_ball_radius * OPT

    :return results: list of (array, int)
            List of (ball center, #points in the ball)
    """
    if dist_oracle is None:
        dist_oracle = DistQueryOracle(tree_algorithm='auto')
    if not dist_oracle.is_fitted:
        dist_oracle.fit(X, sample_weight)
    n_distinct_points, _ = X.shape
    if sample_weight is None:
        sample_weight = np.ones(n_distinct_points)

    n_samples = sum(sample_weight)
    if n_distinct_points <= n_clusters:
        warnings.warn("Number of total distinct data points is smaller than required number of clusters.")
        return [(c, w) for c, w in zip(X, sample_weight)]

    # estimate the upperbound and lowerbound of opt for the data set
    _, ub = dist_oracle.estimate_diameter(n_estimation=10)
    lb, _ = dist_oracle.kneighbors(X[np.random.randint(0, n_distinct_points)].reshape(1, -1), k=2)
    lb = np.max(lb)

    if guessed_opt is not None:
        guessed_opt = min(guessed_opt, ub)
    if guessed_opt is None:
        guessed_opt = (lb + ub) / 2

    results = []
    facility_idxs = np.arange(n_distinct_points)
    n_facilities = len(facility_idxs)
    n_facilities_thresh = 1e4

    while ub > (1 + delta) * lb:
        removed = set()
        results = []

        to_be_updated = facility_idxs
        for i in range(n_clusters):
            if len(removed) == n_distinct_points:
                break
            # When the number of available facilities is huge, use the dense_ball_ method that
            # has caching and early-returning
            if n_facilities > n_facilities_thresh:
                p, covered_pts = dist_oracle.dense_ball_(densest_ball_radius * guessed_opt,
                                                         changed=to_be_updated,
                                                         facility_idxs=facility_idxs,
                                                         except_for=removed,
                                                         minimum_density=np.inf)
            else:
                p, covered_pts = dist_oracle.densest_ball(densest_ball_radius * guessed_opt, removed)

            to_be_removed = dist_oracle.ball(p.reshape(1, -1), removed_ball_radius * guessed_opt)[0]
            to_be_removed = set(to_be_removed).difference(removed)
            to_be_removed = np.array(list(to_be_removed))
            removed.update(to_be_removed)
            w_p = sum(sample_weight[to_be_removed])
            results.append((p, w_p))

            # after removing ball(p, aL), only the densest ball whose centers resides
            # in ball(p, (a+b)L) \ ball(p, aL) is affected. So we only need to consider
            if n_facilities > n_facilities_thresh:
                to_be_updated = set(dist_oracle.ball(p.reshape(1, -1),
                                                     (densest_ball_radius + removed_ball_radius) * guessed_opt)[0]). \
                    intersection(facility_idxs)
                to_be_updated.difference_update(removed)

        n_covered = sum(wp for _, wp in results)
        if n_covered >= n_samples - n_outliers:
            ub = guessed_opt
            guessed_opt = (lb + guessed_opt) / 2
        else:
            lb = guessed_opt
            guessed_opt = (guessed_opt + ub) / 2

    # if the program finishes before finding k'<k centers, we use the FarthestNeighbor
    # method to produce the remained k-k' centers
    if len(results) < n_clusters:
        centers = [c for c, _ in results]
        _, dists_to_centers = pairwise_distances_argmin_min(X, np.atleast_2d(centers))

        for i in range(0, n_clusters - len(results)):
            next_idx = np.argmax(dists_to_centers)
            centers.append(X[next_idx])
            # TODO: here the new center's weight is set to its own weight, this might be problematic(?)
            results.append((X[next_idx], sample_weight[next_idx]))
            _, next_dist = pairwise_distances_argmin_min(X, np.atleast_2d(centers[-1]))
            dists_to_centers = np.minimum(dists_to_centers, next_dist)

    return (results, guessed_opt) if return_opt else results


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
        _, dists = pairwise_distances_argmin_min(X, cs, axis=1)
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


def kcenter_greedy(X, n_clusters=7, random_state=None,
                   return_indices=False, return_distance=False):
    """
    The 2-approx algorithm for k-center.
    Dorit S. Hochbaum and David B. Shmoys.
    A best possible heuristic for the k-center problem.
    Math. Oper. Ues., 10(2):180–184, 1985.
    :param X:

    :param n_clusters:

    :param random_state:

    :param return_indices: bool, default False
        whether to return the indices of the choosed centers

    :param return_distance: bool, default False,
        whether to return the distance between each point and the choosed center set

    :return:
        (centers, center_idxs, distances):
        (centers, distances):
        (centers, center_idxs):
        centers:
    """
    np.random.seed(random_state)
    n_samples, _ = X.shape

    # indices of points that are selected as centers
    center_idxs = [np.random.randint(0, n_samples)]
    # the distance between each data point to the center set
    dists_to_centers = pairwise_distances(X, X[center_idxs[-1]].reshape(1, -1))

    for i in range(1, n_clusters):
        next_idx = np.argmax(dists_to_centers)
        center_idxs.append(next_idx)
        next_dist = pairwise_distances(X, X[center_idxs[-1]].reshape(1, -1))
        dists_to_centers = np.minimum(dists_to_centers, next_dist)

    dists_to_centers = dists_to_centers.ravel()

    # in case that n_clusters = 1
    center_idxs = np.array(center_idxs)
    center_idxs = center_idxs[:n_clusters]

    if return_indices or return_distance:
        results = [X[center_idxs]]
        if return_indices:
            results.append(center_idxs)
        if return_distance:
            results.append(dists_to_centers)
        return tuple(results)
    else:
        return X[center_idxs]


