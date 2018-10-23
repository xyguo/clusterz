# -*- coding: utf-8 -*-
"""Helper functions for creating synthesized data"""
# Author: Xiangyu Guo     xiangyug[at]buffalo.edu
#         Shi Li          shil[at]buffalo.edu

import numpy as np
import scipy as sp


def gaussian_mixture(n_samples, n_clusters, n_outliers, n_features,
                     means=None, covar=None, outliers_dist_factor=50,
                     random_state=None):
    """
    generate data according to gaussian mixture model
    :param n_samples:
    :param n_clusters:
    :param n_outliers:
    :param n_features:
    :param means: array of shape=(n_clusters, n_features)
        mean for each cluster
    :param covar: list of arrays, each of shape=(n_features, n_features)
        covariance matrix for each cluster
    :param outliers_dist_factor:
    :return X: array of shape=(n_samples, n_features)
    """
    np.random.seed(random_state)
    if means is None:
        means = np.random.randn(n_clusters, n_features) * (outliers_dist_factor / 5)
    assert means.shape == (n_clusters, n_features)

    if covar is None:
        covar = [np.identity(n_features) * (outliers_dist_factor / 15) for _ in range(n_clusters)]

    X = np.zeros((n_samples, n_features))

    # arbitrarily determine the size of each cluster
    division = list(np.random.choice(n_samples, n_clusters - 1, replace=False))
    division.sort()
    division = [0] + division + [n_samples]

    # each cluster is generated according to a multivariate Gaussian distribution
    for i in range(n_clusters):
        X[division[i]:division[i+1]] = np.random.multivariate_normal(means[i], covar[i], division[i+1] - division[i])

    # randomly select some points as outliers
    shift = max(np.mean(np.linalg.norm(X, axis=1)), 1.0)
    outliers = np.random.choice(n_samples, n_outliers, replace=False)
    outliers_shift = np.random.multivariate_normal(np.zeros(n_features),
                                                   np.identity(n_features) * outliers_dist_factor * shift,
                                                   n_outliers)
    X[outliers] += outliers_shift
    return X


def add_outliers(X, n_outliers, dist_factor=10, random_state=None, return_index=False):
    """
    add outliers to a given data set: add large random shifts to some of the data in X

    :param X: array of shape=(n_samples, n_features)
    :param n_outliers:
    :param dist_factor: int,
        affects how far away the outliers distributed
    :param random_state:
    :param return_index: bool, whether to return
    :return X or (X, outliers):
        X: array of shape=(n_samples, n_features);
        outliers: array of shape=(n_outliers,), indices of outlier points in the returned data set.
    """
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    assert n_outliers < n_samples
    shift = max(np.mean(np.linalg.norm(X, axis=1)), 1.0)
    outlier_idxs = np.random.choice(n_samples, n_outliers, replace=False)
    outliers_shift = np.random.multivariate_normal(np.zeros(n_features),
                                                   np.identity(n_features) * dist_factor * shift,
                                                   n_outliers)
    outliers = X[outlier_idxs] + outliers_shift
    X = np.vstack((X, outliers))
    return (X, outlier_idxs) if return_index else X
