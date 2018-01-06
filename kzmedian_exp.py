import numpy as np
from time import time
from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_digits
from sklearn.cluster import KMeans
from clusterz.algs import DistributedKZMedian, DistributedKMedian
from clusterz.data_gen import gaussian_mixture, add_outliers
from clusterz.utils import arbitrary_partition, evenly_partition

# synthesized data set
n_machines = 20
n_samples = 100000
n_features = 20
n_outliers = int(0.1 * n_samples)
n_clusters = 4
epsilon = 0.1
random_state = 13

X = gaussian_mixture(n_samples=n_samples, n_clusters=n_clusters,
                     n_outliers=0, n_features=n_features,
                     random_state=random_state)
original_X = X.copy()
X = add_outliers(X, n_outliers, dist_factor=100, random_state=random_state)

# Xs = arbitrary_partition(X, n_machines)
Xs = evenly_partition(X, n_machines)


# real data set
realworld = True
if realworld:
    X, y = load_boston(return_X_y=True)
    n_machines = 6
    n_clusters = len(np.unique(y, return_counts=True))
    n_samples, n_features = X.shape
    n_outliers = n_samples // 5
    epsilon = 0.1
    original_X = X.copy()
    X = add_outliers(X, n_outliers, dist_factor=100, random_state=random_state)
    Xs = evenly_partition(X, n_machines, random_state=random_state)

learner = DistributedKZMedian(n_machines=n_machines, n_clusters=n_clusters,
                              pre_clustering_routine=KMeans, n_pre_clusters=n_clusters,
                              n_outliers=n_outliers, epsilon=epsilon, debug=True)

t_start = time()
# learner.fit(Xs, opt_radius_lb=20, opt_radius_ub=500)
learner.fit(Xs)
t_elapsed = time() - t_start

cost_original = learner.cost(original_X, remove_outliers=False)
cost_no_outliers = learner.cost(X, remove_outliers=True)
cost_with_outliers = learner.cost(X, remove_outliers=False)

print("\nSOL cost on the original data set (before adding outliers): {}".format(cost_original))
print("SOL cost (remove the (1+eps)z farthest points): {}".format(cost_no_outliers))
print("SOL cost (with outliers): {}".format(cost_with_outliers))
print("\ncost-original / cost-with-outlier = {0:.3f}"
      .format(cost_original / cost_with_outliers))
print("cost-without-outlier / cost-with-outlier = {0:.3f}".format(cost_no_outliers / cost_with_outliers))
print("cost-without-outlier / cost-original = {0:.3f}".format(cost_no_outliers / cost_original))
print("\nTime used: {}s".format(t_elapsed))
