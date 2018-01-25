# -*- coding: utf-8 -*-
"""Algorithm for distributed (k,z)-median"""

# Author: Xiangyu Guo     xiangyug@buffalo.edu
#         Yunus Esencayi  yunusese@buffalo.edu
#         Shi Li          shil@buffalo.edu

import numpy as np

from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin_min
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array

from .robust_facility_location import robust_facility_location
from .misc import distributedly_estimate_diameter
from .coreset import DistributedCoreset
from ..utils import debug_print


def lp_cost_(p, X, C, sample_weights=None, n_outliers=0, L=None, element_wise=False):
    """
    compute the cost of \sum_{x\in X} \min(d(x, C), L)^p - L^p
    :param p: norm type
    :param X: array of shape=(n_samples, n_features)
    :param C: array of shape=(n_centers, n_features)
    :param sample_weights: array of shape=(n_samples,)
    :param n_outliers: int
    :param L: threshold for every d(x, C)^p
    :param element_wise: bool, whether to return the cost for each element in X
    :return:
    """
    assert 1 <= p <= 2
    X = check_array(X, ensure_2d=True)
    C = check_array(C, ensure_2d=True)
    _, dists = pairwise_distances_argmin_min(X, C, axis=1)
    dists **= p
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


def pairwise_lp_dist_(p, X, C):
    """
    return the p-th elementwise power of the pairwise distance matrix between X and C
    :param p: float between 1 and 2
    :param X: array of shape=(n_samples, n_features)
    :param C: array of shape=(n_centers, n_features)
    :return: array of shape=(n_samples, n_centers)
    """
    assert 1 <= p <= 2
    return pairwise_distances(X, C) ** p


class DistributedKZLpClustering(object):

    def __init__(self, p,
                 cost_func=None, pairwise_dist_func=None,
                 n_clusters=None, n_outliers=None, n_machines=None,
                 pre_clustering_routine=None, n_pre_clusters=None,
                 epsilon=0.1, delta=0.01, random_state=None, debug=False):
        """
        Serves as the master node.

        :param p: float between 1 and 2
            The cost(X, C) will be \sum_{x\in X) d(x, C)^p

        :param cost_func: cost_func(X, C, sample_weights=None, n_outliers=0, L=None)
            return the cost of center set C on data set X with sample_weights, number of outliers n_outliers,
             and threshold distance L.

        :param pairwise_dist_func: pairwise_dist_func(X, C)
            return the pairwise distance matrix which contains the distance from each point in data set X to
            each center in C

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
        self.p = p
        if cost_func is None:
            cost_func = lambda X, C, **kwargs: lp_cost_(self.p, X, C, **kwargs)
        self.cost_func = cost_func
        if pairwise_dist_func is None:
            pairwise_dist_func = lambda X, C: pairwise_lp_dist_(self.p, X, C)
        self.pairwise_dist_func = pairwise_dist_func

        self.n_clusters = n_clusters
        self.n_outliers = n_outliers
        self.n_machines = n_machines
        self.epsilon = epsilon
        self.delta = delta
        self.random_state = random_state
        self.debugging = debug
        self.pre_clustering_routine = pre_clustering_routine
        self.n_pre_clusters = n_pre_clusters

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
        if remove_outliers:
            cost = self.cost_func(X, self.cluster_centers_,
                                  n_outliers=int((1+self.epsilon) * self.n_outliers), L=None)
            return cost
        else:
            cost = self.cost_func(X, self.cluster_centers_, n_outliers=0, L=None)
            return cost

    def fit(self, Xs, opt_radius_lb=None, opt_radius_ub=None):
        """

        :param Xs: list of arrays of shape=(n_samples_i, n_features),
            Divided data set. Each array in the list represents a bunch of data
            that has been partitioned onto one machine.
        :param opt_radius_lb: float,
            lower bound for the optimal radius
        :param opt_radius_ub: float,
            upper bound for the optimal radius
        :return self:
        """
        self.n_machines = len(Xs)
        self.n_samples_ = sum(X.shape[0] for X in Xs)
        self.n_features_ = Xs[0].shape[1]
        debug_print("n_samples={}, n_features={}, n_machines={}".format(self.n_samples_, self.n_features_, self.n_machines))

        # if the number of allowed outliers are more than n_samples then we
        # simply choose arbitrary n_clusters points in the data set as centers
        if self.n_samples_ <= (1 + self.epsilon) * self.n_outliers:
            while len(self.cluster_centers_) < min(self.n_clusters, self.n_samples_):
                for X in Xs:
                    for i in range(min(X.shape[0], self.n_clusters)):
                        self.cluster_centers_.append(X[i])
            return self

        # estimating the optimal radius with an (1+epsilon) grid
        guessed_radius = self.estimate_opt_radius_range_(Xs, opt_radius_lb, opt_radius_ub)

        debug_print("Start sampling coreset for a range of guessed optimal radius in [{},{}] ({} guesses)...".
                    format(guessed_radius[0], guessed_radius[-1], len(guessed_radius)),
                    debug=self.debugging)
        coresets = self.coresets_collector_(Xs, guessed_radius)

        # construct the centralized data set and solve for final solution
        debug_print("Solve the robust k-median problem ...", debug=self.debugging)
        self.cluster_centers_ = self.centralized_solver_(coresets, Xs)
        assert self.cluster_centers_.shape == (self.n_clusters, self.n_features_)
        return self

    def estimate_opt_radius_range_(self, Xs, opt_radius_lb=None, opt_radius_ub=None):
        """compute the (1+epsilon)-grid range that contains the optimal radius for a
        distributed data set
        """
        if not opt_radius_lb or opt_radius_lb == 0:
            opt_radius_lb = 1
        # upper bound is initialized as the sum of diameters across all machines
        if not opt_radius_ub or opt_radius_ub == np.inf:
            _, opt_radius_ub = distributedly_estimate_diameter(Xs, n_estimation=1)

        n_radius_tried = int(np.log(opt_radius_ub / opt_radius_lb) / np.log(1 + self.epsilon))
        guessed_radius = [(opt_radius_lb * ((1 + self.epsilon) ** i)) ** self.p
                          for i in range(n_radius_tried)]
        return guessed_radius

    def coresets_collector_(self, Xs, guessed_radiuses):
        """
        sampling coresets from a distributed data set according to some threshold distance set
        :param Xs: list of array, distributed data set
        :param guessed_radiuses: list of float, threshold distance
        :return coresets: list of tuples (samples, weights, guessed_opt, coreset_indices)
            each tuple contains the information for one coreset with one threshold distance.

            samples: array of shape=(n_coreset_size, n_features),

            weights: array of shape=(n_coreset_size,)

            guessed_opt: the corresponding threshold distance

            coreset_indices: list of length=(n_coreset_size,), where each element is a tuple
            (machine_id, sample_id), represents the origin of the corresponding sample
        """
        # construct pre-clusterings as base points for sampling coresets
        pre_cluster_results = [
            self.pre_clustering_routine(n_clusters=self.n_pre_clusters).fit(X) for X in Xs
        ]
        pre_cluster_centers = [pr.cluster_centers_ for pr in pre_cluster_results]
        pre_clusters = [pr.predict(X) for pr, X in zip(pre_cluster_results, Xs)]
        pre_clustering_costs_cache = [self.cost_func(X, C, element_wise=True, n_outliers=0, L=None)
                                      for X, C in zip(Xs, pre_cluster_centers)]

        # This is a hand pick coreset size, which is based on the theoretical results as well as the
        # actual data size. The size is determined such that in the robust_kzmedian step the total number
        # of clients is no more than n_samples / 10
        coreset_size = (self.n_clusters * self.n_features_ + np.log(1 / self.delta)) / (self.epsilon ** 2)
        coreset_size = max(min(coreset_size, self.n_samples_ / (100 * len(guessed_radiuses))), self.n_clusters * 5)
        debug_print("Coreset size = {}".format(coreset_size), debug=self.debugging)

        coresets = []
        n_guesses = 1  # used for debugging
        for guessed_opt in guessed_radiuses:
            debug_print("{}-th guess: trying with optimal radius = {}".format(n_guesses, guessed_opt), debug=self.debugging)
            n_guesses += 1

            # construct coreset w.r.t. the threshold distance d_L
            coreset_sampler = DistributedCoreset(
                sample_size=coreset_size,
                pre_clustering_method=self.pre_clustering_routine,
                n_pre_clusters=self.n_pre_clusters,
                cost_func=lambda X, C: self.cost_func(X, C, n_outliers=0, L=guessed_opt)
            )
            # cache data for efficient coreset sampling
            pre_clustering_costs = [np.minimum(c, guessed_opt) for c in pre_clustering_costs_cache]

            # sampling coreset
            coreset_sampler.fit(Xs,
                                pre_cluster_centers=pre_cluster_centers,
                                pre_clusters=pre_clusters,
                                pre_clustering_costs=pre_clustering_costs)
            samples, weights = coreset_sampler.coreset
            coreset_indices = [idxs for machine in coreset_sampler.sample_indices for idxs in machine]
            coresets.append((samples, weights, guessed_opt, coreset_indices))

        return coresets

    def centralized_solver_(self, coresets, Xs):
        """Solve the centralized clustering problem on the collected coresets
        :param coresets: list of tuples (samples, weights, guessed_opt, coreset_indices)
            each tuple contains the information for one coreset with one threshold distance.

            samples: array of shape=(n_coreset_size, n_features),

            weights: array of shape=(n_coreset_size,)

            guessed_opt: the corresponding threshold distance

            coreset_indices: list of length=(n_coreset_size,), where each element is a tuple
            (machine_id, sample_id), represents the origin of the corresponding sample
        :param Xs: the distributed data set where the coreset is sampled from
        :return cluster_centers: array of shape=(n_clusters, n_features)
        """
        final_samples, final_weights, final_cost, final_indices = zip(*coresets)
        final_samples, final_weights = list(final_samples), list(final_weights)
        facility_idxs = list(set([idxs for sample in final_indices for idxs in sample]))
        facilities = np.vstack([Xs[i][j] for i, j in facility_idxs])
        cluster_centers = robust_facility_location(client_sets=final_samples, facilities=facilities,
                                                   sample_weights=final_weights,
                                                   radiuses=final_cost,
                                                   threshold_cost=self.cost_func,
                                                   pairwise_dist=self.pairwise_dist_func,
                                                   n_clusters=self.n_clusters,
                                                   n_outliers=(1+self.epsilon) * self.n_outliers,
                                                   return_cost=False)
        return cluster_centers


class DistributedLpClustering(DistributedKZLpClustering):

    def __init__(self, p, local_clustering_method,
                 cost_func=None, pairwise_dist_func=None,
                 n_clusters=None, n_machines=None,
                 pre_clustering_routine=None, n_pre_clusters=None,
                 epsilon=0.1, delta=0.01, random_state=None, debug=False):
        """
        Serves as the master node.

        :param p: float between 1 and 2
            The cost(X, C) will be \sum_{x\in X) d(x, C)^p

        :param local_clustering_method: clustering method
            clustering method used for solving the weighted L_p clustering problem on the coreset

        :param cost_func: cost_func(X, C, sample_weights=None)
            return the cost of center set C on data set X with sample_weights

        :param pairwise_dist_func: pairwise_dist_func(X, C)
            return the pairwise distance matrix which contains the distance from each point in data set X to
            each center in C

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
        if cost_func:
            def lp_cost_no_outlier_(X, C, sample_weights=None, n_outliers=0, L=None):
                return cost_func(X, C, sample_weights=sample_weights)
        else:
            def lp_cost_no_outlier_(X, C, sample_weights=None, n_outliers=0, L=None):
                return lp_cost_(p, X, C, sample_weights=sample_weights, n_outliers=0, L=None)
        cost_func = lp_cost_no_outlier_

        super().__init__(p=p,
                         cost_func=cost_func, pairwise_dist_func=pairwise_dist_func,
                         n_clusters=n_clusters, n_outliers=0, n_machines=n_machines,
                         pre_clustering_routine=pre_clustering_routine,
                         n_pre_clusters=n_pre_clusters,
                         epsilon=epsilon, delta=delta, random_state=random_state, debug=debug)
        self.local_clustering_method = local_clustering_method

    def estimate_opt_radius_range_(self, Xs, opt_radius_lb=None, opt_radius_ub=None):
        return [np.inf]

    def centralized_solver_(self, coresets, Xs):
        final_samples, final_weights, *_ = coresets[0]
        return self.local_clustering_method(final_samples, final_weights)