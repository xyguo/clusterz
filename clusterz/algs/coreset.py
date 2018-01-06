# -*- coding: utf-8 -*-
"""Algorithm for sampling coreset:
Based on the following paper
M. F. Balcan, S. Ehrlich, Y. Liang.
Distributed k-Means and k-Median Clustering on General Topologies. NIPS'13
"""

# Author: Xiangyu Guo     xiangyug@buffalo.edu
#         Yunus Esencayi  yunusese@buffalo.edu
#         Shi Li          shil@buffalo.edu

import warnings
import numpy as np

from sklearn.cluster import KMeans
from .misc import DistQueryOracle
from ..utils import debug_print


class DistributedCoreset(object):
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
        self.samples_size = sample_size
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
        actual_coreset_size = max(self.n_machines, self.samples_size - self.n_pre_clusters * self.n_machines)

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
        self.samples_size = sample_size
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
        coreset_idxs = np.random.choice(indices, size=self.samples_size, replace=False,
                                        p=sampling_prob)
        self.samples_ = X[coreset_idxs]
        self.weights_ = self.weight_normalizer / sampling_prob
        self.sample_weights_ = self.weights_[coreset_idxs]
        self.sample_indices_ = coreset_idxs

        return self


