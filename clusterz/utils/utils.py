import numpy as np


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


def evenly_partition(X, n_machines, random_state=None):
    """partition the data evenly into n_machines parts"""
    n_samples, _ = X.shape
    np.random.seed(random_state)

    partition = np.arange(n_samples)
    np.random.shuffle(partition)

    # generate even cuts
    division = np.arange(0, n_samples + 1, n_samples // n_machines)
    division[-1] = n_samples

    Xs = [X[partition[division[i]:division[i+1]]] for i in range(n_machines)]
    return Xs


def debug_print(s, debug=True):
    if debug:
        print(s)