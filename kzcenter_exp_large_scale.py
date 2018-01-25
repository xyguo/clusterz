import numpy as np
import pandas as pd
from itertools import product
from time import time
from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_digits
from clusterz.algs import DistributedKZCenter, KZCenter
from clusterz.data_gen import gaussian_mixture, add_outliers, get_realworld_data
from clusterz.utils import arbitrary_partition, evenly_partition


def measure_method(name, model, data, distributed_data, n_outliers):
    print("\nFit model {}:".format(name))
    t_start = time()
    model.fit(distributed_data)
    t_elapsed = time() - t_start

    comm_cost = model.communication_cost
    cost_with_outliers = model.cost(data, remove_outliers=0)
    cost_without_outliers = model.cost(data, remove_outliers=n_outliers)
    print("Result for {}:".format(name))
    print("Communication cost (#points * #features): {}".format(comm_cost))
    print("SOL cost (remove {} outliers): {}".
          format(n_outliers, cost_without_outliers))
    print("SOL cost (with all outliers): {}".
          format(cost_with_outliers))
    print("Time used: {}s".format(t_elapsed))

    return comm_cost, cost_with_outliers, cost_without_outliers


# real data set

parameters = {
    'skin': {'n_machines': [i for i in range(10, 21, 2)],
                 'n_clusters': [i for i in range(10, 101, 10)],
                 'n_outliers': [2 ** i for i in range(5, 10)]
                 },
    'covertype': {'n_machines': [i for i in range(10, 21, 2)],
                   'n_clusters': [i for i in range(10, 101, 10)],
                   'n_outliers': [2 ** i for i in range(5, 10)]
                   },
    'power': {'n_machines': [i for i in range(2, 21, 2)],
                  'n_clusters': [i for i in range(10, 20, 1)],#[i for i in range(10, 20, 2)],
                  'n_outliers': [128],#[2 ** i for i in range(7, 12)]
                },
    'higgs': {'n_machines': [20],#[i for i in range(10, 21, 2)],
               'n_clusters': [i for i in range(10, 101, 10)],
               'n_outliers': [2 ** i for i in range(7, 12)]
               },
}
realworld = True
if realworld:
    dataset = 'higgs'
    # X, y = load_digits(return_X_y=True)
    X = get_realworld_data(dataset)
    z_star = 0
    random_state = 13
    X = add_outliers(X, z_star, dist_factor=50, random_state=random_state)
else:
    # synthesized data set
    dataset = 'synthesized data'
    n_machines = 50
    n_samples = 100000
    n_features = 10
    n_outliers = 10000
    n_clusters = 4
    epsilon = 0.1
    random_state = 13

    X = gaussian_mixture(n_samples=n_samples, n_clusters=n_clusters,
                         n_outliers=n_outliers, n_features=n_features,
                         outliers_dist_factor=100,
                         random_state=random_state)


n_machines_range = parameters[dataset]['n_machines']
n_clusters_range = parameters[dataset]['n_clusters']
n_outliers_range = parameters[dataset]['n_outliers']

n_repeat = 5

results = {'n_machines_range': np.array(n_machines_range),
           'n_clusters_range': np.array(n_clusters_range),
           'n_outliers_range': np.array(n_outliers_range),
           'dkzc-m-sz-comm-cost': [[] for _ in range(n_repeat)],
           'dkzc-m-sz-cost': [[] for _ in range(n_repeat)],
           'dkzc-m-lz-comm-cost': [[] for _ in range(n_repeat)],
           'dkzc-m-lz-cost': [[] for _ in range(n_repeat)],
           'dkzc-a-comm-cost': [[] for _ in range(n_repeat)],
           'dkzc-a-cost': [[] for _ in range(n_repeat)],
           'moseley-comm-cost': [[] for _ in range(n_repeat)],
           'moseley-cost': [[] for _ in range(n_repeat)],
           'random-random-comm-cost': [[] for _ in range(n_repeat)],
           'random-random-cost': [[] for _ in range(n_repeat)],
           'random-charikar-comm-cost': [[] for _ in range(n_repeat)],
           'random-charikar-cost': [[] for _ in range(n_repeat)],
           }

for r in range(n_repeat):
    print("\n" + "==============" * 8)
    print("\nRepeat {}: ".format(r))
    for n_machines, n_clusters, n_outliers in product(n_machines_range,
                                                      n_clusters_range,
                                                      n_outliers_range):
        print("\n" + "==============" * 6)
        print("Repeat {}: n_machines={}, n_clusters={}, n_outliers={}"
              .format(r, n_machines, n_clusters, n_outliers))
        n_samples, n_features = X.shape
        epsilon_sz = 1
        epsilon = 0.1
        Xs = evenly_partition(X, n_machines, random_state=random_state)

        learner_mul_sz = DistributedKZCenter(algorithm='multiplicative',
                                             n_machines=n_machines, n_clusters=n_clusters,
                                             n_outliers=n_outliers, epsilon=epsilon_sz, debug=False)

        learner_mul_lz = DistributedKZCenter(algorithm='multiplicative',
                                             n_machines=n_machines, n_clusters=n_clusters,
                                             n_outliers=n_outliers, epsilon=epsilon, debug=False)

        learner_add = DistributedKZCenter(algorithm='additive',
                                          n_machines=n_machines, n_clusters=n_clusters,
                                          n_outliers=n_outliers, epsilon=epsilon, debug=False)

        distr_baseline_moseley = DistributedKZCenter(algorithm='moseley',
                                                     n_machines=n_machines, n_clusters=n_clusters,
                                                     n_outliers=n_outliers, epsilon=epsilon, debug=False)

        distr_baseline_random = DistributedKZCenter(algorithm='random',
                                                    n_machines=n_machines, n_clusters=n_clusters,
                                                    n_outliers=n_outliers, epsilon=epsilon, debug=False)

        distr_baseline_charikar = DistributedKZCenter(algorithm='random_charikar',
                                                      n_machines=n_machines, n_clusters=n_clusters,
                                                      n_outliers=n_outliers, epsilon=epsilon, debug=False)

        distr_models = {'dkzc-m-sz': learner_mul_sz,
                        'dkzc-m-lz': learner_mul_lz,
                        'dkzc-a': learner_add,
                        'moseley': distr_baseline_moseley,
                        'random-random': distr_baseline_random,
                        'random-charikar': distr_baseline_charikar}

        print("On data set {} with shape {}:".format(dataset, (n_samples, n_features)))

        print("\n" + "=====" * 8)
        print("\nFit the distributed models:")
        for nm, md in distr_models.items():
            comm, _, cost = measure_method(name=nm,
                                           model=md,
                                           data=X,
                                           distributed_data=Xs,
                                           n_outliers=n_outliers)
                                       # n_outliers=(1+learner_mul.epsilon_) * learner_mul.n_outliers_)
            results[nm + '-comm-cost'][r].append(comm)
            results[nm + '-cost'][r].append(cost)


for k, v in results.items():
    results[k] = np.array(v)

np.savez("kzm_{}_{}_{}_z_star_{}_large_scale_exp_results_for_dataset_{}_20170123".
         format(len(n_clusters_range), len(n_outliers_range), len(n_machines_range),
                z_star, dataset), **results)