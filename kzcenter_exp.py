import numpy as np
from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_digits
from clusterz.algs import DistributedKZCenter
from clusterz.data_gen import gaussian_mixture, add_outliers
from clusterz.utils import arbitrary_partition, evenly_partition

# synthesized data set
n_machines = 10
n_samples = 10000
n_features = 2
n_outliers = 800
n_clusters = 4
epsilon = 0.1
random_state = None

X = gaussian_mixture(n_samples=n_samples, n_clusters=n_clusters,
                     n_outliers=n_outliers, n_features=n_features,
                     outliers_dist_factor=100)

Xs = arbitrary_partition(X, n_machines)


# real data set
realworld = True
if realworld:
    X, y = load_breast_cancer(return_X_y=True)
    n_machines = 3
    n_clusters = len(np.unique(y, return_counts=True))
    n_samples, n_features = X.shape
    n_outliers = n_samples // 5
    epsilon = 0.1
    X = add_outliers(X, n_outliers, dist_factor=50)
    Xs = evenly_partition(X, n_machines)

learner = DistributedKZCenter(n_machines=n_machines, n_clusters=n_clusters,
                              n_outliers=n_outliers, epsilon=epsilon, debug=True)
learner.fit(Xs)

print("estimated OPT cost: {}".format(learner.opt_))
print("SOL cost: {}".format(learner.cost(X)))
print("outliers' cost in SOL: {}".format(learner.cost(X, consider_outliers=False)))
