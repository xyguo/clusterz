import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from sklearn.metrics import pairwise_distances_argmin_min
from ..utils import evenly_partition, arbitrary_partition, compute_cost


class MyTestCase(unittest.TestCase):

    def test_evenly_partition(self):
        X = np.random.randn(20, 20)
        X_copy = X.copy()
        m = 5
        blocksize = X.shape[0] // m

        # no shuffle
        Xs = evenly_partition(X, n_machines=m, shuffle=False)
        assert_array_equal(X_copy, X)
        assert len(Xs) == m
        for i in range(m):
            assert Xs[i].shape == (4, 20)
            assert_array_equal(Xs[i],
                               X[(i * blocksize):((i + 1) * blocksize)])

        # shuffled
        Xs = evenly_partition(X, n_machines=m, shuffle=True)
        assert_array_equal(X_copy, X)
        assert len(Xs) == m
        for i in range(m):
            assert Xs[i].shape == (4, 20)
            diff = Xs[i] - X[(i * blocksize):((i + 1) * blocksize)]
            assert(np.linalg.norm(diff, ) > 0.05)

    def test_arbitrary_partition(self):
        X = np.random.randn(20, 20)
        X_copy = X.copy()
        m = 5

        Xs = arbitrary_partition(X, n_machines=m)
        assert_array_equal(X_copy, X)
        blocksizes = [x.shape[0] for x in Xs]
        assert len(Xs) == m
        assert X.shape[0] == sum(blocksizes)
        sizediff = np.array(blocksizes) - np.ones(m) * (X.shape[0] / m)
        assert np.linalg.norm(sizediff) > 1

    def test_compute_cost(self):
        X = np.array(
            [[0, 1], [0, -1], [1, 0], [-1, 0],
             [0, 6], [0, 4], [1, 5], [-1, 5],
             [5, 1], [5, -1], [6, 0], [4, 0],
             [5, 6], [5, 4], [6, 5], [4, 5]]
        )
        C = np.array(
            [[0, 0], [0, 5],
             [5, 0], [5, 5]]
        )

        def kmeans_cost_mock(data, centers):
            _, dists = pairwise_distances_argmin_min(data, centers,
                                                     axis=1, metric='euclidean',
                                                     metric_kwargs={'squared': True})
            return np.linalg.norm(dists) ** 2

        assert_almost_equal(compute_cost(X, C, kmeans_cost_mock, remove_outliers=None),
                            16)

if __name__ == '__main__':
    unittest.main()
