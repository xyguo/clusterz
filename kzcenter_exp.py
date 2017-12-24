import numpy as np
from sklearn.datasets import load_iris
from clusterz.algs import DistributedKZCenter

n_machines = 2
n_samples = 1000
n_features = 10
n_outliers = 80
n_clusters = 3
epsilon = 0.1

partition = np.arange(n_samples)
np.random.shuffle(partition)
division = list(np.random.choice(n_samples, n_machines - 1, replace=False))
division.sort()
division = [0] + division + [n_samples]

X = np.random.randn(n_samples, n_features) * 10
X = np.vstack((X, np.random.randn(n_outliers, n_features) * 100))
Xs = [X[partition[division[i]:division[i+1]]] for i in range(n_machines)]

learner = DistributedKZCenter(n_machines=n_machines, n_clusters=n_clusters,
                              n_outliers=n_outliers, epsilon=epsilon)
learner.fit(Xs)

print("estimated OPT cost: {}".format(learner.opt_))
print("SOL cost: {}".format(learner.cost(X)))
print("outliers' cost in SOL: {}".format(learner.cost(X, consider_outliers=False)))
