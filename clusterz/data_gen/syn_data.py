import numpy as np
import scipy as sp


def gaussian_mixture(n_samples, n_clusters, n_outliers, n_features,
                     means=None, covar=None, outliers_dist_factor=50,
                     random_state=None):
    """
    gaussian_mixture(n_samples, n_clusters, n_outliers, n_features,
                     means=None, covar=None, outliers_dist_factor=50,
                     random_state=None)

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


def add_outliers(X, n_outliers, dist_factor=10, random_state=None):
    """
    add_outliers(X, n_outliers, dist_factor=10, random_state=None)

    add outliers to given data set.

    :param X:
    :param n_outliers:
    :param dist_factor: int,
        affects how far away the outliers distributed
    :param random_state:
    :return:
    """
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    assert n_outliers < n_samples
    shift = max(np.mean(np.linalg.norm(X, axis=1)), 1.0)
    outliers = np.random.choice(n_samples, n_outliers, replace=False)
    outliers_shift = np.random.multivariate_normal(np.zeros(n_features),
                                                   np.identity(n_features) * dist_factor * shift,
                                                   n_outliers)
    X[outliers] += outliers_shift
    return X
