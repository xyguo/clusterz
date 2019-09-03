# -*- coding: utf-8 -*-
"""Algorithm for sampling coreset:
Based on the following two papers
M. F. Balcan, S. Ehrlich, Y. Liang.
Distributed k-Means and k-Median Clustering on General Topologies. NIPS'13

J. Chen, E. S. Azer, Q. Zhang,
A Practical Algorithm for Distributed Clustering and Outlier Detection, NIPS'18
"""

# Author: Xiangyu Guo     xiangyug[at]buffalo.edu
#         Shi Li          shil[at]buffalo.edu

import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from ..utils import debug_print


class DistributedCoreset(object):
    """M. F. Balcan, S. Ehrlich, Y. Liang.
    Distributed k-Means and k-Median Clustering on General Topologies. NIPS'13
    """
    def __init__(self, sample_size, cost_func=None, debug=False,
                 pre_clustering_method=KMeans,
                 n_pre_clusters=None, **kwargs):
        """
        :param sample_size: int,
            total size of the final coreset
        :param cost_func: function of type costs=cost_func(X, C),
            where X is an array of shape=(n_samples, n_features) that represents the data set,
        and C is an array of shape=(n_centers, n_features) that represents the centers.
        The return value is an array of shape=(n_samples,) that contains the corresponding cost
        for each data point in X.
        :param debug: bool,
            whether to toggle on debug output
        :param pre_clustering_method: clustering method,
            used for providing a initial pre-clustering on each machine
        :param n_pre_clusters: None or int,
            number of pre-clusters created on each machine
        :param kwargs:
            additional arguments for the pre-clustering method
        """
        self.sample_size = sample_size
        self.cost_func = cost_func
        self.n_pre_clusters = n_pre_clusters
        self.pre_clustering_method = pre_clustering_method
        self.additional_kwargs = kwargs

        self.n_machines = None
        self.debugging = debug
        self.samples_ = None
        self.weights_ = None
        self.total_sensitivity_ = None
        self.sample_indices_ = None

    @property
    def coreset(self):
        """return sampled coreset as (samples, weights)"""
        return self.samples_, self.weights_

    @property
    def sample_indices(self):
        return self.sample_indices_

    def fit(self, Xs, pre_cluster_centers=None, pre_clusters=None, pre_clustering_costs=None):
        """

        :param Xs: list of arrays of shape=(n_samples_i, n_features),
            Divided data set. Each array in the list represents a bunch of data
            that has been partitioned onto one machine.
        :param pre_cluster_centers: None or list of arrays of shape=(n_pre_clusters_i, n_features),
            given pre-cluster centers for coreset sampling. If None, then the fit method will
            construct pre-clusters via self.pre_clustering_method.
        :param pre_clusters: list of arrays of shape=(n_pre_clusters_i, cluster_size_i)
            indices of data that are within each pre-clusters
        :param pre_clustering_costs: None or list of arrays of shape=(n_samples_i, )
            pre-clustering cost of each samples
        :return self:
        """
        self.n_machines = len(Xs)
        actual_coreset_size = max(self.n_machines, self.sample_size - self.n_pre_clusters * self.n_machines)

        if (pre_clusters or pre_clustering_costs) and not pre_cluster_centers:
            raise ValueError("You can't provide pre_clusters or pre_clustering_costs"
                             " without specifying pre_cluster_centers\n")

        debug_print("Conducting pre-clustering ...", debug=self.debugging)
        # initialize pre-cluster centers
        if not pre_cluster_centers:
            pre_clustering_results = []
            for i in range(self.n_machines):
                pre_clustering_results.append(self.pre_clustering_method(n_clusters=self.n_pre_clusters,
                                                                         **self.additional_kwargs)
                                              .fit(Xs[i]))
            pre_cluster_centers = [pre_clustering_results[i].cluster_centers_ for i in range(self.n_machines)]

        # initialize pre-cluster affiliation
        if not pre_clusters:
            dists = [np.array([[self.cost_func(Xs[i][j].reshape(-1, 1), c.reshape(-1,1))
                                for c in pre_cluster_centers[i]]
                               for j in range(Xs[i].shape[0])])
                     for i in range(self.n_machines)]
            pre_clusters = [np.argmin(d, axis=1) for d in dists]

        # initialize pre-cluster costs
        if pre_clustering_costs:
            costs = pre_clustering_costs
        else:
            costs = [self.cost_func(Xs[i], pre_cluster_centers[i], element_wise=True)
                     for i in range(self.n_machines)]

        # compute the coreset size
        local_total_costs = [c.sum() for c in costs]
        total_cost = sum(local_total_costs)
        weight_normalizers = [total_cost / (local_total_costs[i] * actual_coreset_size)
                              for i in range(self.n_machines)]
        coreset_sizes = [int(actual_coreset_size * lc // total_cost) for lc in local_total_costs]

        debug_print("Initialize mappers ...", debug=self.debugging)
        mappers = []
        for i in range(self.n_machines):
            mappers.append(
                Coreset(
                    sample_size=coreset_sizes[i],
                    cost_func=lambda _: costs[i],
                    weight_normalizer=weight_normalizers[i]
                ).fit(Xs[i])
            )

        all_samples, all_weights = [], []
        for i in range(self.n_machines):
            samples, weights = mappers[i].coreset
            all_samples.append(samples)
            all_weights.append(weights)

            # Note that the sampling_prob for any center is zero, i.e.,
            # they won't appear in the coresets, so we add them manually
            coreset_idxs = mappers[i].sample_indices
            center_weights = []
            centers = pre_cluster_centers[i]
            affiliation = pre_clusters[i]
            # TODO: check here
            for j, c in enumerate(centers):
                wt = np.where(affiliation == j)[0]
                wb = len(wt) - mappers[i].all_weights[list(set(coreset_idxs).intersection(wt))].sum()
                center_weights.append(wb)
            all_samples.append(centers)
            all_weights.append(np.array(center_weights))

        self.samples_ = np.vstack(all_samples)
        self.weights_ = np.hstack(all_weights)

        self.sample_indices_ = [list(map(lambda x: (i, x), mappers[i].sample_indices))
                                for i in range(self.n_machines)]
        return self


class Coreset(object):

    def __init__(self, sample_size, cost_func=None, weight_normalizer=1):
        """
        :param sample_size:
        :param cost_func: function of type costs=cost_func(X),
            accepts an array of shape=(n_samples, n_features) that represents the data set,
        and return an array of shape=(n_samples,) that contains the corresponding cost.
        :param weight_normalizer: float,
            by default, the weight of each point in the coreset is 1 / sample_prob, but user can
            multiply it with some normalization number
        """
        self.sample_size = sample_size
        self.cost_func = cost_func
        self.weight_normalizer = weight_normalizer
        self.samples_ = None
        self.sample_weights_ = None
        self.weights_ = None
        self.sample_indices_ = None
        self.total_sensitivity_ = None

    @property
    def all_weights(self):
        return self.weights_

    @property
    def sample_indices(self):
        return self.sample_indices_

    @property
    def coreset(self):
        """return sampled coreset as (samples, weights)"""
        return self.samples_, self.sample_weights_

    def fit(self, X):
        """

        :param X:
        :return:
        """
        n_samples, _ = X.shape

        # calculate sensitivity of each point (actually an upper bound)
        sensitivity = self.cost_func(X)
        self.total_sensitivity_ = sensitivity.sum()
        sampling_prob = sensitivity / self.total_sensitivity_

        indices = np.arange(n_samples)
        coreset_idxs = np.random.choice(indices, size=self.sample_size, replace=True,
                                        p=sampling_prob)
        self.samples_ = X[coreset_idxs]
        self.weights_ = np.ones(sampling_prob.shape) * np.inf
        self.weights_[coreset_idxs] = self.weight_normalizer / sampling_prob[coreset_idxs]
        self.sample_weights_ = self.weights_[coreset_idxs]
        self.sample_indices_ = coreset_idxs

        return self


class DistributedSummary(object):
    """
    Implements the algorithm in the following paper:
    J. Chen, E. S. Azer, Q. Zhang,
    A Practical Algorithm for Distributed Clustering and Outlier Detection,
    NIPS'18
    """
    def __init__(self, n_clusters, n_outliers, alpha, beta,
                 augmented=False, adversary=False, debug=False):
        """
        Distributed driver of Algorithm 1: Summary-Outliers
        :param n_clusters:
        :param n_outliers:
        :param alpha: parameter, determine the sample size
        :param beta: parameter, determine the ball radius
        :param augmented: bool, whether to use Algorithm 2 to augment the summary
        :param debug: debugging flag
        """
        self.n_clusters_ = n_clusters
        self.n_outliers_ = n_outliers
        self.alpha_ = alpha
        self.beta_ = min(beta, 1.0)
        self.samples_ = None
        self.weights_ = None
        self.sample_indices_ = None
        self.augmented_ = augmented
        self.adversary_ = adversary
        self.n_outliers_per_machine_ = None

        self.n_machines_ = None
        self.debugging = debug
        self.samples_ = None
        self.weights_ = None
        self.sample_indices_ = None

    @property
    def samples(self):
        return self.samples_

    @property
    def weights(self):
        return self.weights_

    @property
    def sample_indices(self):
        return self.sample_indices_

    def fit(self, Xs):
        """

        :param Xs: list of arrays of shape=(n_samples_i, n_features),
            Divided data set. Each array in the list represents a bunch of data
            that has been partitioned onto one machine.
        :return self:
        """
        self.n_machines_ = len(Xs)
        mappers = []
        all_samples, all_weights, all_indices = [], [], []
        self.n_outliers_per_machine_ = self.n_outliers_ if self.adversary_ else 2 * self.n_outliers_ // self.n_machines_

        for X in Xs:
            m = SummaryOutliers(n_clusters=self.n_clusters_,
                                n_outliers=self.n_outliers_per_machine_,
                                alpha=self.alpha_,
                                beta=self.beta_,
                                augmented=self.augmented_)
            m.fit(X)
            all_samples.append(m.samples)
            all_weights.append(m.weights)
            mappers.append(m)

        self.samples_ = np.vstack(all_samples)
        self.weights_ = np.hstack(all_weights)

        self.sample_indices_ = [list(map(lambda x: (i, x), mappers[i].sample_indices))
                                for i in range(self.n_machines_)]
        return self


class SummaryOutliers(object):

    def __init__(self, n_clusters, n_outliers, alpha, beta,
                 augmented=False):
        """
        Algorithm 1: Summary-Outliers
        :param n_clusters: int, determine the summary size along with n_outliers and alpha. See the paper for details.
        :param n_outliers: int, determine the summary size along with n_clusters and alpha. See the paper for details.
        :param alpha: float > 0, scale of the sample size. The large alpha the larger the summary size.
            See the paper for details.
        :param beta: float between 0 and 1, determine the ball radius. Each point in the summary will cover a ball
            that contains at least beta * n_samples points in the original data set.
        :param augmented: bool, whether to use Algorithm 2 to augment the summary
        """
        self.n_clusters_ = n_clusters
        self.n_outliers_ = n_outliers
        self.alpha_ = alpha
        self.beta_ = min(beta, 1.0)
        self.samples_ = None
        self.weights_ = None
        self.sample_indices_ = None
        self.augmented_ = augmented

    @property
    def samples(self):
        return self.samples_

    @property
    def weights(self):
        return self.weights_

    @property
    def sample_indices(self):
        return self.sample_indices_

    def fit(self, X):
        """
        :param X: array of shape=(n_samples, n_features)
        :return self:
        """
        n_samples, _ = X.shape
        samples_ = []
        weights_ = []
        sample_indices_ = []
        X_i = np.arange(0, n_samples)  # indices for remained data points

        while len(X_i) > max(8 * self.n_outliers_, 0):
            kappa = max(np.log(len(X_i)), self.n_clusters_)
            S_i_size = np.int(self.alpha_ * kappa)
            # 6. construct a set S_i of size \alpha\kappa by random sampling (with replacement) from X_i
            S_i = np.random.choice(X_i, size=S_i_size, replace=True)
            S_i = np.unique(S_i)
            w_i = np.ones((len(S_i),))
            # 6. for each point in X_i, compute the distance to its nearest point in S_i
            # each value in nearest would range from 0 to len(S_i)
            nearest, distance = pairwise_distances_argmin_min(X[X_i], X[S_i])
            # 8. let rho_i be the smallest radius s.t. |B(S_i, X_i, rho_i)| >= beta|X_i|.
            rho_i = np.sort(distance)[np.int(np.ceil((len(X_i) - 1) * self.beta_))]
            # Let C_i = B(S_i, X_i, rho_i)
            # foreach x\in X_i, assign sigma(x)=x
            # for each x \in X_i, assign weight w_x = |\sigma^{−1}(x)| and add (x, w_x) into Q
            idxs, counts = np.unique(nearest[distance <= rho_i], return_counts=True)
            w_i[idxs] = counts

            samples_.append(X[S_i])
            weights_.append(w_i)
            sample_indices_.append(S_i)

            # 9. for each x \in C_i, choose the point y \in S_i that minimizes d(x, y) and assign \sigma(x) = y
            X_i = X_i[distance > rho_i]

        # Augmented-Summary-Outliers (Algorithm 2 in the paper)
        if self.augmented_ and len(sample_indices_) > 0:
            self.sample_indices_, self.weights_, self.samples_ = self.augmenting_(X, X_i, np.hstack(sample_indices_))
            return self

        # append the remained "outliers"
        samples_.append(X[X_i])
        weights_.append(np.ones(len(X_i)))
        sample_indices_.append(X_i)

        # concatenate all S_i's to build the final summary
        self.samples_ = np.vstack(samples_)
        self.weights_ = np.hstack(weights_)
        self.sample_indices_ = np.hstack(sample_indices_)

        assert len(self.samples_) == len(self.weights_)
        return self

    def augmenting_(self, X, X_r, S):
        """
        Augmented-Summary-Outliers (Algorithm 2 in the paper).
        :param X: data set, array of shape=(n_samples, n_features)
        :param X_r: array of int, the indices of remained "outliers" after SummaryOutliers
        :param S: array of int, indices of the collected "inliers" after SummaryOutliers
        :return (augmented_sample_idxs, augmented_weights, augmented_samples):
            augmented_sample_idxs: array of int, the indices of augmented summary in X
            augmented_weights: array of float, the weight of each point in the augmented summary
            augmented_samples: array of shape(n_summary, n_features), equals X[augmented_sample_idxs].
        """
        S_p_size = len(X_r) - len(S)
        X_idxs_set = set(np.arange(X.shape[0]))
        X_r_set = set(X_r)
        S_set = set(S)
        X_minus_X_r_and_S = list(X_idxs_set.difference(X_r_set.union(S_set)))

        # 2. Sample S' of size |X_r|-|S| from X\(X_r\cup S)
        S_p = np.random.choice(X_minus_X_r_and_S,
                               size=S_p_size, replace=True)
        S_p = np.unique(S_p)

        # 3. construct \pi(x)
        S_and_S_p = list(S_set.union(S_p))
        X_minus_X_r = list(X_idxs_set.difference(X_r_set))

        # each value in nearest would range from 0 to len(S_and_S_p)
        nearest, distance = pairwise_distances_argmin_min(X[X_minus_X_r], X[S_and_S_p])

        # 4. count the weights
        idxs, weights = np.unique(nearest, return_counts=True)

        S_and_S_p = np.array(S_and_S_p)
        augmented_sample_idxs = S_and_S_p[idxs]
        augmented_weights = weights
        augmented_samples = X[augmented_sample_idxs]

        return augmented_sample_idxs, augmented_weights, augmented_samples
