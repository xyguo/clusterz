# -*- coding: utf-8 -*-
"""Utility functions"""
# Author: Xiangyu Guo     xiangyug[at]buffalo.edu
#         Shi Li          shil[at]buffalo.edu

import numpy as np
from time import time
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array
from sklearn.metrics import pairwise_distances_argmin_min


def compute_cost(X, cluster_centers, cost_func, remove_outliers=True):
    """
    :param X: array,
        data set
    :param remove_outliers: None or int, default None
        whether to remove outliers when computing the cost on X
    :return: float,
        actual cost
    """
    if cluster_centers is None:
        raise NotFittedError("Model hasn't been fitted yet\n")
    X = check_array(X, ensure_2d=True)
    _, dists = pairwise_distances_argmin_min(X, cluster_centers, axis=1)
    dist_idxs = np.argsort(dists)
    if remove_outliers is not None:
        assert remove_outliers >= 0
        dist_idxs = dist_idxs if remove_outliers == 0 else dist_idxs[:-int(remove_outliers)]

        return cost_func(X[dist_idxs], cluster_centers)
    else:
        return cost_func(X, cluster_centers)


def arbitrary_partition(X, n_machines, random_state=None):
    """partition the data arbitrarily into n_machines parts"""
    n_samples, _ = X.shape
    np.random.seed(random_state)

    partition = np.arange(n_samples)
    np.random.shuffle(partition)

    # generate random cuts
    division = list(np.random.choice(n_samples, n_machines - 1, replace=False))
    division.sort()
    division = [0] + division + [n_samples]

    Xs = [X[partition[division[i]:division[i+1]]] for i in range(n_machines)]
    return Xs


def evenly_partition(X, n_machines, random_state=None, shuffle=True):
    """partition the data evenly into n_machines parts"""
    n_samples, _ = X.shape

    partition = np.arange(n_samples)
    if shuffle:
        np.random.seed(random_state)
        np.random.shuffle(partition)

    # generate even cuts
    division = np.arange(0, n_samples + 1, n_samples // n_machines)
    division[-1] = n_samples

    Xs = [X[partition[division[i]:division[i+1]]] for i in range(n_machines)]
    return Xs


def debug_print(s, debug=True):
    if debug:
        print(s)


def measure_method(name, model, data, distributed_data, n_outliers, dist_oracles=None, seq=False):
    print("\nFit model {}:".format(name))
    t_start = time()
    if not dist_oracles:
        if seq:
            model.fit(data)
        else:
            model.fit(distributed_data)
    else:
        if seq:
            model.fit(data, sample_weight=None, dist_oracle=dist_oracles)
        else:
            model.fit(distributed_data, sample_weights=None, dist_oracles=dist_oracles)
    t_elapsed = time() - t_start

    comm_cost = 0 if seq else model.communication_cost
    cost_with_outliers = model.cost(data, remove_outliers=0)
    cost_without_outliers = model.cost(data, remove_outliers=n_outliers)
    print("Result for {}:".format(name))
    print("Communication cost (#points * #features): {0:e}".format(comm_cost))
    print("SOL cost (remove {0} outliers): {1:e}".
          format(n_outliers, cost_without_outliers))
    print("SOL cost (with all outliers): {0:e}".
          format(cost_with_outliers))
    print("Time used: {}s".format(t_elapsed))

    return comm_cost, cost_with_outliers, cost_without_outliers
