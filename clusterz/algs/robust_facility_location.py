# -*- coding: utf-8 -*-
"""robust facility location with outliers"""

# Author: Xiangyu Guo     xiangyug@buffalo.edu
#         Yunus Esencayi  yunusese@buffalo.edu
#         Shi Li          shil@buffalo.edu

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array
from ..utils import debug_print


def reverse_greedy_(distances, weights, client_set_residence, client_set_weight,
                    n_client_sets, n_clusters, B):
    """
    Helper function for robust facility location algorithm. Implement the
    reverse-greedy-and-reweighting process
    :param distances: array of shape=(n_clients, n_facilities)
    :param weights: array of shape=(n_clients,)
    :param client_set_residence: array of shape=(n_clients,)
    :param client_set_weight: array of shape=(n_client_sets,)
    :param n_client_sets:
    :param n_clusters:
    :param B: a guessed number between 4OPT and 8OPT
    :return:
    """
    n_clients, n_facilities = distances.shape

    first_and_second_nearest_idxs = (np.argpartition(distances, 1, axis=1)[:, 0:2]).T
    # cache the nearest and second nearest center for each client to facilitate fast error increase calculation.
    # if the client is also a open facility itself, then nearest[i] = i
    nearest = first_and_second_nearest_idxs[0]
    second_nearest = first_and_second_nearest_idxs[1]

    initial_affiliation = nearest
    # cluster[i] is the list of points assigned to center i,
    # if cluster[i] == [], then i must be unavailable;
    # if j \in cluster[i] for some i, then j must be unavailable;
    clusters = [list(np.where(initial_affiliation == i)[0]) for i in range(n_facilities)]
    error_increase = np.zeros((n_facilities, n_client_sets))  # cache the error increase

    available = np.ones(n_facilities, dtype=bool)  # record whether a facility has been removed
    affected_clients = np.arange(n_clients)  # cache the indices of centers whose error increase needs update
    affected_facilities = set(range(n_facilities))  # cache the indices of centers whose error increase needs update
    debug_print("\tCompute solution for B={} ...".format(B), True)
    for t in range(n_facilities - n_clusters):
        # update the cache for nearest and second nearest point for each client
        if len(affected_clients) > 0:
            first_and_second_nearest_idxs = (np.argpartition(distances[affected_clients], 1, axis=1)[:, 0:2]).T
            nearest[affected_clients] = first_and_second_nearest_idxs[0]
            second_nearest[affected_clients] = first_and_second_nearest_idxs[1]
        # compute the resulted objective increase on each client set for removing each center (facility)
        for i in affected_facilities:
            cs = clusters[i]
            if not available[i] or len(cs) == 0:
                continue
            current_dists = distances[cs, nearest[cs]]
            future_dists = distances[cs, second_nearest[cs]]
            # assert np.all(future_dists < np.inf)
            # assert np.all(current_dists < np.inf)
            costs_inc = (future_dists - current_dists) * weights[cs]
            error_increase[i, client_set_residence[cs]] = 0
            np.add.at(error_increase, [i, client_set_residence[cs]], costs_inc)

        # gather the points that increases error not so much
        can_remove = np.where(np.logical_and(available, np.all(error_increase <= B / 2, axis=1)))[0]
        if len(can_remove) == 0:
            debug_print("\t\tB={0:.3f} is too small, there're still {1} points remained to be removed, skip...".
                        format(B, n_facilities - n_clusters - t), True)
            return None
        # find the one with minimum weighted error increase
        vt_idx = np.argmin(error_increase[can_remove].dot(client_set_weight))
        vt_idx = can_remove[vt_idx]
        # debug_print("\t\tremove {}th center {}".format(t+1, vt_idx), True)

        # update cache info
        available[vt_idx] = False
        distances[:, vt_idx] = np.inf
        # all clients in the removed cluster needs update
        affected_clients = set(clusters[vt_idx])
        # all clients that have vt_idx as their second nearest neighbor needs update
        affected = np.where(second_nearest == vt_idx)[0]
        affected_clients.update(affected)
        affected_clients = np.array(list(affected_clients))

        # Two kinds of facilities are affected:
        # 1. Those that accept the clients from the removed cluster
        affected_facilities = set()
        for e in clusters[vt_idx]:
            new_center = second_nearest[e]
            # assert new_center != vt_idx and available[new_center]
            clusters[new_center].append(e)
            affected_facilities.add(new_center)
        # 2. Those that have clients which has the removed cluster as the second nearest center
        affected_facilities.update(nearest[affected])
        affected_facilities = np.array(list(affected_facilities))

        # empty the removed cluster
        clusters[vt_idx] = []

        # update client sets weight
        client_set_weight *= (1 + 1 / B) ** error_increase[vt_idx]

    remained_centers_idxs = np.where(available)[0]
    return remained_centers_idxs


def robust_facility_location(client_sets, facilities, sample_weights, radiuses,
                             threshold_cost, pairwise_dist,
                             n_clusters=8, n_outliers=0, return_cost=False):
    """
    solve \min_C \sup_L (cost(X_L, d_L, C) - (1+\epsilon)zL)

    :param client_sets: list of arrays of shape=(n_samples_L, n_features)
    :param facilities: array of shape=(n_facilities, n_features)
    :param sample_weights: list of arrays of shape=(n_samples_L,)
    :param radiuses: list of length n_client_sets
    :param threshold_cost: func(X, C, sample_weights=None, n_outliers=0, L=None, element_wise=False)
    :param pairwise_dist: func(X, C)
    :param n_clusters: int, number of clusters
    :param n_outliers: int, number of outliers
    :param return_cost: bool, whether to return the cost for the corresponding data set
    :return centers or (centers, cost):
        centers: array of shape=(n_centers, n_features)
        cost: corresponding cost of the selected centers.

    :reference: B. Anthony, V. Goyal, A. Gupta and V. Nagarajan.
        A Plant Location Guide for the Unsure: Approximation Algorithms for Min-Max Location Problems.
        Mathematics of Operations Research, Vol. 35, No. 1 (Feb., 2010), pp. 79-101
    """
    clients = np.vstack(client_sets)
    weights = np.hstack(sample_weights)

    n_clients = clients.shape[0]
    n_facilities = facilities.shape[0]
    n_client_sets = len(client_sets)
    client_set_weight = np.ones(n_client_sets)
    debug_print("{} client sets with {} clients in total, and {} available facilities, of which {} will be selected ..."
                .format(n_client_sets, n_clients, n_facilities, n_clusters), True)

    # pre-compute distances
    client_set_sizes = [X.shape[0] for X in client_sets]
    client_set_idxs = np.cumsum([0] + client_set_sizes)
    client_set_residence = np.zeros(n_clients, dtype=np.int)
    cached_distances = pairwise_dist(clients, facilities)

    # threshold distances
    for i in range(1, len(client_set_sizes)):
        cached_distances[client_set_idxs[i-1]:client_set_idxs[i], :] = \
            np.minimum(cached_distances[client_set_idxs[i-1]:client_set_idxs[i], :], radiuses[i - 1])
        client_set_residence[client_set_idxs[i-1]:client_set_idxs[i]] = i - 1

    # perturbe the distances such that no clients has the same distance towards different facilities
    perturbation_extent = 0.1 * np.min(cached_distances[cached_distances > 1e-5])
    perturbation = np.random.uniform(0, perturbation_extent, size=cached_distances.shape)
    cached_distances += perturbation

    # distance[i, j] is the distance from client i to facility j under threshold distance L[i]
    distances = cached_distances.copy()

    results = []
    lb = 4 * min(radiuses[i] * sample_weights[i][sample_weights[i] > 0].sum() for i in range(n_client_sets))
    ub = 8 * max(radiuses[i] * sample_weights[i][sample_weights[i] > 0].sum() for i in range(n_client_sets))
    debug_print("\tCompute solution for B in [{},{}] ...".format(lb, ub), True)
    B = (lb + ub) / 2
    while ub > 1.01 * lb:
        distances[:] = cached_distances
        remained_centers_idxs = reverse_greedy_(distances=distances, weights=weights,
                                                client_set_residence=client_set_residence,
                                                client_set_weight=client_set_weight,
                                                n_client_sets=n_client_sets, n_clusters=n_clusters, B=B)

        if remained_centers_idxs is None:
            lb = B
            B = (lb + ub) / 2
            continue

        # compute the minimax cost
        remained_facilities = facilities[remained_centers_idxs]
        final_cost = max(threshold_cost(X=client_sets[i], C=remained_facilities,
                                        sample_weights=sample_weights[i],
                                        n_outliers=n_outliers, L=radiuses[i], element_wise=False)
                         for i in range(n_client_sets))
        debug_print("\t\tFor B={}, the cost is {}.".format(B, final_cost))
        results.append((remained_centers_idxs, final_cost))

        ub = B
        B = (lb + ub) / 2

    best_centers_idxs, minimax_cost = min(results, key=lambda a: a[1])
    return (facilities[best_centers_idxs], minimax_cost) if return_cost else facilities[best_centers_idxs]
