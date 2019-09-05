# Distributed Clustering with Outliers
This is the experiment code for our paper

> X. Guo, S. Li. Distributed _k_-Clustering for Data with Heavy Noise. _NIPS'18_

## Requirement
* Python >= 3.4
* `numpy` >= 1.15 and `scipy` >= 1.2.0
* `scikit-learn` >= 17.0


## Testing
To run the code, please see the `kzcenter_exp.py` file as example.


## Progress

1. [x] Data preprocessing
2. [x] _(k,z)_-center algorithm
3. [x] coreset
4. [x] _(k,z)_-median
5. [x] _(k,z)_-means


## Summary

All the algorithm implementations are under `clusterz/algs`:

* `kzcenter.py`: includes the implementation of our distributed _(k,z)_-center algorithm,
the algorithm by Malkomes et al.[1], the centralized _(k,z)_-center algorithm by
Charikar et al.[2], and the 2-approx _k_-center algorithm by Hochbaum and Shmoys[3]

* `kz_clustering_from_others.py`: contains the implementation for some _(k,z)_-clustering
algorithms proposed recently, including Guha et al.[4] and Chen et al.[5]

* `kz_lp_clustering.py`: contains the implementation for our distributed _(k,z)_-median/means
algorithm. Actually this implementation works for any _L_p_ norms.

* `kzmeans.py`: instantiate the distributed _L_p_ clustering routine defined in `kz_lp_clustering.py`
with _L_2_ norm, thus obtain a distributed _(k,z)_-means routine. Also contains implementations
for centralized _k/(k,z)_-means algorithm based on Lloyd[6] and Chawla and Gionis[7]

* `kzmedian.py`: instantiate the distributed _L_p_ clustering routine defined in `kz_lp_clustering.py`
with _L_1_ norm, thus obtain a distributed _(k,z)_-median routine. Also contains implementations
for centralized _k/(k,z)_-means algorithm based on Lloyd[6] and Chawla and Gionis[7]

* `misc.py`: utility functions for the classes in `kzcenter.py`. Maily facilitate the
ball-cover step.

* `robust_facility_location.py`: centralized solver for the min-max _k_-clustering problem based on Anthony et al.[8].
Will be invoked by our distributed _(k,z)_-means/median algorithms as the final step.

> [1] Gustavo Malkomes, Matt J. Kusner, Wenlin Chen,Kilian Q. Weinberger, and Benjamin Moseley.
Fast distributed k-center clustering with outliers on massive data. _NIPS'15_

> [2] Moses Charikar, Samir Khuller, David M. Mount, and Giri Narasimhan.
Algorithms for facility location problems with outliers. _SODA'01_

> [3] Dorit S. Hochbaum and David B. Shmoys.
A best possible heuristic for the k-center problem. _Math. Oper. Res._, 1985

> [4] Sudipto Guha, Yi Li, and Qin Zhang.
Distributed partial clustering. _SPAA'17_

> [5] Jiecao Chen, Erfan Sadeqi Azer, and Qin Zhang.
A practical algorithm for distributed clustering and outlier detection. _NIPS'18_

> [6] Stuart P. Lloyd:
Least squares quantization in PCM. _IEEE Trans. Information Theory_. 1982

> [7] Sanjay Chawla and Aristides Gionis.
k-means−−: A unified approach to clustering and outlier detection. _ICDM'13_

> [8] Barbara M. Anthony, Vineet Goyal, Anupam Gupta, and Viswanath Nagarajan.
A plant location guide for the unsure: Approximation algorithms for min-max location problems.
_Math. Oper. Res._, 2010.
