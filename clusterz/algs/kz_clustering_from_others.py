"""
Implementation for the algorithm of Guha et al's SPAA'17 paper and [CAZ'18] NIPS'18 paper.
"""
import numpy as np

from sklearn.metrics.pairwise import pairwise_distances, pairwise_distances_argmin_min
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array
from scipy.spatial import ConvexHull

from .kzcenter import kcenter_greedy, kzcenter_charikar, kzcenter_charikar_eg
from .kzmeans import kz_means, kzmeans_cost_
from .kzmedian import kz_median, kzmedian_cost_
from .misc import DistQueryOracle
from .coreset import DistributedSummary
from ..utils import debug_print


class GuhaDistributedKZCenter(object):
    """
    Sudipto Guha, Yi Li, and Qin Zhang.
    Distributed partial clustering.
    In Proceedings of the 29th ACM Symposium on Parallelism in Algorithms and Architectures, SPAA 2017,
    """
    def __init__(self, return_outliers=False, use_epsilon_net=False,
                 n_clusters=None, n_outliers=None, n_machines=None,
                 epsilon=0.1, rho=2, random_state=None, debug=False):
        """
        Serves as the master node.

        :param p: float between 1 and 2
            The cost(X, C) will be \sum_{x\in X) d(x, C)^p

        :param cost_func: cost_func(X, C, sample_weights=None, n_outliers=0, L=None, element_wise=False)
            return the cost of center set C on data set X with sample_weights, number of outliers n_outliers,
             and threshold distance L.

        :param pairwise_dist_func: pairwise_dist_func(X, C)
            return the pairwise distance matrix which contains the distance from each point in data set X to
            each center in C

        :param n_clusters: integer.
            Number of clusters.

        :param n_outliers: integer.
            Number of outliers

        :param pre_clustering_routine: function or None,
            Subroutine used for generating the pre-clusters for coreset sampling.

        :param n_machines: integer.
            Number of machines

        :param epsilon: float.
            error tolerance parameter for the number of outliers.

        :param rho: float > 1.
            parameter for pre-clustering.

        :param random_state: numpy.RandomState

        :param debug: boolean, whether output debugging information
        """
        self.n_clusters_ = n_clusters
        self.n_outliers_ = n_outliers
        self.n_machines_ = n_machines
        self.epsilon_ = epsilon
        self.rho_ = rho
        self.random_state = random_state
        self.debugging = debug
        self.return_outliers_ = return_outliers
        self.use_epsilon_net_ = use_epsilon_net

        self.dist_oracles_ = None
        self.cluster_centers_ = None
        self.n_samples_, self.n_features_ = None, None
        self.opt_ = None
        self.communication_cost_ = None

    def cost(self, X, remove_outliers=True):
        """

        :param X: array of shape=(n_samples, n_features),
            data set
        :param remove_outliers: None or int, default None
            whether to remove outliers when computing the cost on X
        :return: float,
            actual cost
        """
        if self.cluster_centers_ is None:
            raise NotFittedError("Model hasn't been fitted yet\n")
        X = check_array(X, ensure_2d=True)
        _, dists = pairwise_distances_argmin_min(X, self.cluster_centers_, axis=1)
        dists.sort()
        if remove_outliers is not None:
            assert remove_outliers >= 0
            return dists[-int(remove_outliers + 1)]
        else:
            return dists[-(int((1 + self.epsilon_) * self.n_outliers_) + 1)]

    @property
    def communication_cost(self):
        return self.communication_cost_

    def local_solver_(self, X, n_clusters, n_outliers, sample_weight=None,
                      dist_oracle=None, return_opt=False):
        return kzcenter_charikar_eg(X, n_clusters=n_clusters, n_outliers=n_outliers,
                                 sample_weight=sample_weight, return_opt=return_opt,
                                 dist_oracle=dist_oracle,
                                 densest_ball_radius=1, removed_ball_radius=3)

    def fit(self, Xs, sample_weights=None, dist_oracles=None):
        """

        :param Xs: list of arrays of shape=(n_samples_i, n_features),
            Divided data set. Each array in the list represents a bunch of data
            that has been partitioned onto one machine.
        :param sample_weights: Only for interface compatibility, not used.
        :param dist_oracles: DistQueryOracle object
        :return self:
        """
        self.n_machines_ = len(Xs)
        self.n_samples_ = sum(X.shape[0] for X in Xs)
        self.n_features_ = Xs[0].shape[1]
        debug_print("n_samples={}, n_features={}, n_machines={}".format(self.n_samples_, self.n_features_, self.n_machines_))

        if not dist_oracles:
            self.dist_oracles_ = [DistQueryOracle(tree_algorithm='auto', leaf_size=60).fit(x) for x in Xs]
        else:
            self.dist_oracles_ = dist_oracles
        # if the number of allowed outliers are more than n_samples then we
        # simply choose arbitrary n_clusters points in the data set as centers
        if self.n_samples_ <= (1 + self.epsilon_) * self.n_outliers_:
            if not self.cluster_centers_:
                self.cluster_centers_ = []
            while len(self.cluster_centers_) < min(self.n_clusters_, self.n_samples_):
                for X in Xs:
                    for i in range(min(X.shape[0], self.n_clusters_)):
                        self.cluster_centers_.append(X[i])
            return self

        # estimate the number of outliers on each local data set
        # grid, hull, cost_grid, local_results = self.gather_local_info_1_(Xs)
        if self.use_epsilon_net_:
            grid, local_centers, cost_grid = self.gather_local_info_alg_1_and_2_(Xs)
        else:
            grid, local_centers = self.gather_local_info_(Xs)

        i0, q0 = self.determine_threshold_(grid)

        if self.return_outliers_:
            results = self.collect_local_results_with_outliers_(Xs, grid, local_centers,
                                                                i0, q0)
        else:
            results = self.collect_local_results_without_outliers_(Xs, grid, i0, q0)
        # results = self.collect_local_results_1_(Xs, grid, hull, cost_grid, local_results,
        #                                         i0, q0, return_outliers=False)
        X_agg = [c for c, _ in results]
        X_agg = np.vstack(X_agg)
        sample_weights_agg = np.array([t for _, t in results])

        #TODO: measure comm cost
        self.communication_cost_ = 0
        if self.use_epsilon_net_:
            self.communication_cost_ += cost_grid[1].shape[0] * cost_grid[1].shape[1]
        else:
            self.communication_cost_ += grid.shape[0] * grid.shape[1]
        self.communication_cost_ += X_agg.shape[0] * self.n_features_
        self.communication_cost_ += sample_weights_agg.shape[0]

        # construct the centralized data set and solve for final solution
        agg_n_outliers = self.n_outliers_
        # agg_n_outliers = 0 if outliers_eliminated else self.n_outliers_
        results, est_opt = self.local_solver_(X_agg, sample_weight=sample_weights_agg, return_opt=True,
                                              n_clusters=self.n_clusters_, n_outliers=agg_n_outliers)
        self.cluster_centers_ = np.array([c for (c, _) in results])
        self.opt_ = est_opt
        assert self.cluster_centers_.shape == (self.n_clusters_, self.n_features_)
        return self

    def gather_local_info_alg_1_and_2_(self, Xs):
        """compute the (1+epsilon)-grid range that contains the optimal radius for a
        distributed data set
        """
        vertices_q = np.power(self.rho_,
                              np.arange(1, np.int(np.log(self.n_outliers_) / np.log(self.rho_))))
        vertices_q = vertices_q.astype(np.int)
        vertices_q = list(vertices_q)
        if vertices_q[-1] != self.n_outliers_:
            vertices_q.append(self.n_outliers_)
        vertices_q.append(0)
        vertices_q.sort()
        vertices_q = np.array(vertices_q)

        # pre-clustering using Farthest Neighbor
        n_local_clusters = self.n_clusters_ + self.n_outliers_ + 1
        grid = np.zeros((self.n_machines_, self.n_outliers_ + 1))
        cost_grid = [vertices_q, []]

        cluster_centers = []  # store the solution sol(A_i, 2k, q) for each i and q
        for i in range(self.n_machines_):
            centers = kcenter_greedy(Xs[i], n_clusters=n_local_clusters)
            costs_q = np.zeros(vertices_q.shape[0])

            for j, q in enumerate(vertices_q):
                _, d = pairwise_distances_argmin_min(centers[self.n_clusters_ + q, np.newaxis],
                                                     centers[:self.n_clusters_ + q])
                costs_q[j] = d[0]

            # Find the convex hull
            vertices = np.vstack((vertices_q, costs_q)).T
            grid[i, :] = self.lower_convex_hull_(vertices)

            cluster_centers.append(centers)
            cost_grid[1].append(costs_q)  # corresponds to original points (not the hull) in \mathbb{I}
        cost_grid[1] = np.array(cost_grid[1])
        return grid, cluster_centers, cost_grid

    def gather_local_info_(self, Xs):
        """compute the (1+epsilon)-grid range that contains the optimal radius for a
        distributed data set
        """

        # pre-clustering using Farthest Neighbor
        n_local_clusters = self.n_clusters_ + self.n_outliers_ + 1
        grid = np.zeros((self.n_machines_, self.n_outliers_ + 1))

        cluster_centers = []  # store the solution sol(A_i, 2k, q) for each i and q
        for i in range(self.n_machines_):
            centers = kcenter_greedy(Xs[i], n_clusters=n_local_clusters)

            for q in range(1, self.n_outliers_ + 1):
                _, d = pairwise_distances_argmin_min(centers[self.n_clusters_ + q, np.newaxis],
                                                     centers[:self.n_clusters_ + q])
                grid[i, q] = d[0]
            cluster_centers.append(centers)
        return grid, cluster_centers

    def gather_local_info_shi_(self, Xs):
        """

        :param Xs:
        :return (grid, hull_table, opt_grid, local_results):
        """
        vertices_q = np.power(self.rho_,
                              np.arange(1, np.int(np.log(self.n_outliers_) / np.log(self.rho_))))
        vertices_q = vertices_q.astype(np.int)
        vertices_q = list(vertices_q)
        if vertices_q[-1] != self.n_outliers_:
            vertices_q.append(self.n_outliers_)
        vertices_q.append(0)
        vertices_q.sort()
        vertices_q = np.array(vertices_q)

        hull_table = np.zeros((self.n_machines_, self.n_outliers_ + 1))  # corresponds to table f(i,q) in the paper
        grid = np.zeros((self.n_machines_, self.n_outliers_ + 1))  # corresponds to table l(i,q) in the paper
        local_results = []  # store the solution sol(A_i, 2k, q) for each i and q
        cost_grid = (vertices_q, [])
        for i in range(self.n_machines_):
            opt_q = np.zeros(vertices_q.shape[0])
            local_results.append([])
            for j, q in enumerate(vertices_q):
                assert q + self.n_clusters_ * 2 < Xs[i].shape[0]
                results, opt = self.local_solver_(Xs[i], n_clusters=self.n_clusters_ * 2, n_outliers=q,
                                                  dist_oracle=self.dist_oracles_[i], return_opt=True)
                local_results[-1].append((q, results))
                opt_q[j] = opt

            # same effect of the lower convex hull
            for j in range(1, len(opt_q)):
                opt_q[j] = min(opt_q[j-1], opt_q[j])

            hull_table[i, :] = np.interp(np.arange(0, self.n_outliers_ + 1),
                                         vertices_q, opt_q)
            cost_grid[1].append(opt_q)  # corresponds to original points (not the hull) in \mathbb{I}
        grid[:, 1:] = hull_table[:, :self.n_outliers_] - hull_table[:, 1:self.n_outliers_ + 1]
        return grid, hull_table, cost_grid, local_results

    def gather_local_info_alg1_(self, Xs):
        # local convex hulls
        vertices_q = np.power(self.rho_,
                              np.arange(1, np.int(np.log(self.n_outliers_) / np.log(self.rho_))))
        vertices_q = vertices_q.astype(np.int)
        vertices_q = list(vertices_q)
        if vertices_q[-1] != self.n_outliers_:
            vertices_q.append(self.n_outliers_)
        vertices_q.append(0)
        vertices_q.sort()
        vertices_q = np.array(vertices_q)

        hull_table = np.zeros((self.n_machines_, self.n_outliers_ + 1))
        local_results = []  # store the solution sol(A_i, 2k, q) for each i and q
        cost_grid = (vertices_q, [])
        for i in range(self.n_machines_):
            costs_q = np.zeros(vertices_q.shape[0])
            local_results.append([])
            for j,q in enumerate(vertices_q):
                assert q + self.n_clusters_ * 2 < Xs[i].shape[0]
                results = self.local_solver_(Xs[i], n_clusters=self.n_clusters_ * 2, n_outliers=q,
                                             dist_oracle=self.dist_oracles_[i])
                centers = np.array([c for (c,_) in results])
                _, dists = pairwise_distances_argmin_min(Xs[i], centers, axis=1)
                costs_q[j] = np.sort(dists)[-q-1]
                local_results[-1].append((q, results))

            # Find the convex hull
            vertices = np.vstack((vertices_q, costs_q)).T
            hull_table[i, :] = self.lower_convex_hull_(vertices)

            cost_grid[1].append(costs_q)  # corresponds to original points (not the hull) in \mathbb{I}
        grid = np.zeros((self.n_machines_, self.n_outliers_ + 1))
        grid[:, 1:] = hull_table[:, :self.n_outliers_] - hull_table[:, 1:self.n_outliers_ + 1]
        return grid, hull_table, cost_grid, local_results

    def lower_convex_hull_(self, vertices):
        # after adding these two extreme points, we're sure that
        # 1. the convex hull must contain these two points;
        # 2. If we remove these two points from the convex hull, we get the lower convex hull
        leftmost, rightmost = vertices[:, 0].min()-1, vertices[:, 0].max()+1
        max_cost = vertices[:, 1].max() * 10
        vertices = np.vstack((np.array([leftmost, max_cost]),
                              vertices,
                              np.array([rightmost, max_cost])))

        # find the vertices of the convex hull
        hull = ConvexHull(vertices)
        hull_vertex_idxs = hull.vertices
        # remove the two extreme points to get the lower convex hull
        hull_vertex_idxs = set(hull_vertex_idxs).difference([0, len(vertices)-1])
        hull_vertex_idxs = list(hull_vertex_idxs)
        hull_vertex_idxs.sort()
        hull_vertices = vertices[hull_vertex_idxs]
        hull_vertices = hull_vertices[hull_vertices[:, 0].argsort()]

        hull_interp = np.interp(np.arange(0, self.n_outliers_ + 1),
                                hull_vertices[:, 0], hull_vertices[:, 1])
        return hull_interp

    def determine_threshold_(self, grid):
        """
        :param Xs: list of array, distributed data set
        :param guessed_radiuses: list of float, threshold distance
        :return coresets: list of tuples (samples, weights, guessed_opt, coreset_indices)
            each tuple contains the information for one coreset with one threshold distance.

            samples: array of shape=(n_coreset_size, n_features),

            weights: array of shape=(n_coreset_size,)

            guessed_opt: the corresponding threshold distance

            coreset_indices: list of length=(n_coreset_size,), where each element is a tuple
            (machine_id, sample_id), represents the origin of the corresponding sample
        """
        assert self.rho_ < self.n_machines_
        thresh_idx = min(np.int(self.rho_ * self.n_outliers_), self.n_machines_ * self.n_outliers_)
        idx = np.argsort(-grid.ravel())[thresh_idx]

        i0 = idx // self.n_outliers_
        q0 = idx % self.n_outliers_
        return i0, q0

    def collect_local_results_without_outliers_(self, Xs, grid, i0, q0):
        """Solve the centralized clustering problem on the collected coresets
        :param Xs: the distributed data set where the coreset is sampled from
        :param local_results: list of list of tuple
            local_results[i] = [(q0, res0), (q1, res1), ..., (qn, res n)],
            where each res_i is a list of tuples of (center, weight)
        :return results: list of tuples of (center, weights)
        """
        thresh = grid[i0, q0]
        results = []
        for i in range(self.n_machines_):
            available = np.where(grid[i] >= thresh)[0]
            t_i = np.max(available) if len(available) > 0 else 0
            r = self.local_solver_(Xs[i], n_clusters=self.n_clusters_,
                                   dist_oracle=self.dist_oracles_[i],
                                   n_outliers=t_i, return_opt=False)
            results += r
        return results

    def collect_local_results_with_outliers_(self, Xs, grid, local_centers,
                                             i0, q0, return_outliers=True):
        """Solve the centralized clustering problem on the collected coresets
        :param Xs: the distributed data set where the coreset is sampled from
        :param local_results: list of list of tuple
            local_results[i] = [(q0, res0), (q1, res1), ..., (qn, res n)],
            where each res_i is a list of tuples of (center, weight)
        :return results: list of tuples of (center, weights)
        """
        thresh = grid[i0, q0]
        results = []
        for i in range(self.n_machines_):
            available = np.where(grid[i] >= thresh)[0]
            t_i = np.max(available) if len(available) > 0 else 0
            centers = local_centers[i][:self.n_clusters_ * 2 + t_i]
            closest, _ = pairwise_distances_argmin_min(Xs[i], centers)
            cs, counts = np.unique(closest, return_counts=True)
            r = [(centers[cs[i]], wc) for i, wc in enumerate(counts)]
            results += r
        return results

    def collect_local_results_1_(self, Xs, grid, hull, cost_grid, local_results,
                                 i0, q0, return_outliers=True):
        """Solve the centralized clustering problem on the collected coresets
        :param Xs: the distributed data set where the coreset is sampled from
        :param local_results: list of list of tuple
            local_results[i] = [(q0, res0), (q1, res1), ..., (qn, res n)],
            where each res_i is a list of tuples of (center, weight)
        :return results: list of tuples of (center, weights)
        """
        thresh = grid[i0, q0]
        results = []
        for i in range(self.n_machines_):
            available = np.where(grid[i] >= thresh)[0]
            t_i = np.max(available) if len(available) > 0 else 0
            if i == i0:
                # if return_outliers:
                t_i = min(q for q, c in zip(cost_grid[0], cost_grid[1][i])
                          if q >= q0 and c <= hull[i0, q0])
                # else:
                #     t_i1 = max(q for q, c in cost_grid
                #                if q <= t_i and c >= hull[i0, q0])
                #     t_i2 = min(q for q, c in cost_grid
                #                if q >= t_i and c >= hull[i0, q0])
            found = False
            local_result = local_results[i]
            for q, r in local_result:
                if q == t_i:
                    results += r
                    found = True
                    break
            if not found:
                r = self.local_solver_(Xs[i], n_clusters=self.n_clusters_ * 2,
                                       dist_oracle=self.dist_oracles_[i],
                                       n_outliers=t_i, return_opt=False)
                results += r
        return results


class CAZDistributedKZClustering(object):
    """
    Jiecao Chen, Erfan Sadeqi Azer, Qin Zhang.
    A Practical Algorithm for Distributed Clustering and Outlier Detection.
    NIPS'18
    """
    def __init__(self, central_solver, cost_func,
                 n_clusters=None, n_outliers=None, n_machines=None,
                 alpha=2, beta=0.45, random_state=None, debug=False):
        """

        :param n_clusters:
        :param n_outliers:
        :param n_machines:
        :param alpha:
        :param beta:
        :param random_state:
        :param debug:
        """
        self.central_solver_ = central_solver
        self.cost_func_ = cost_func
        self.n_clusters_ = n_clusters
        self.n_outliers_ = n_outliers
        self.n_machines_ = n_machines
        self.alpha_ = alpha
        self.beta_ = beta
        self.random_state = random_state
        self.debugging = debug

        self.summary_ = None
        self.cluster_centers_ = None
        self.communication_cost_ = None

    @property
    def communication_cost(self):
        return self.communication_cost_

    def cost(self, X, remove_outliers=True):
        """

        :param X: array,
            data set
        :param remove_outliers: None or int, default None
            whether to remove outliers when computing the cost on X
        :return: float,
            actual cost
        """
        if self.cluster_centers_ is None:
            raise NotFittedError("Model hasn't been fitted yet\n")
        X = check_array(X, ensure_2d=True)
        _, dists = pairwise_distances_argmin_min(X, self.cluster_centers_, axis=1)
        dist_idxs = np.argsort(dists)
        if remove_outliers is not None:
            assert remove_outliers >= 0
            dist_idxs = dist_idxs if remove_outliers == 0 else dist_idxs[:-int(remove_outliers)]

            return self.cost_func_(X[dist_idxs], self.cluster_centers_)
        else:
            return self.cost_func_(X, self.cluster_centers_)

    def fit(self, Xs):
        self.n_machines_ = len(Xs)
        ds = DistributedSummary(n_clusters=self.n_clusters_,
                                n_outliers=self.n_outliers_,
                                alpha=self.alpha_,
                                beta=self.beta_,
                                adversary=False,
                                augmented=False)
        ds.fit(Xs)

        X_summary = ds.samples
        wt_summary = ds.weights
        print("Summary size = {}".format(X_summary.shape[0]))
        self.communication_cost_ = X_summary.shape[0] * X_summary.shape[1]
        self.communication_cost_ += wt_summary.shape[0]

        self.cluster_centers_ = self.central_solver_(X=X_summary, sample_weights=wt_summary,
                                                     n_clusters=self.n_clusters_,
                                                     n_outliers=self.n_outliers_)
        return self


class CAZDistributedKZMeans(CAZDistributedKZClustering):
    """
    Jiecao Chen, Erfan Sadeqi Azer, Qin Zhang.
    A Practical Algorithm for Distributed Clustering and Outlier Detection.
    NIPS'18
    """
    def __init__(self,
                 n_clusters=None, n_outliers=None, n_machines=None,
                 alpha=2, beta=0.45, random_state=None, debug=False):
        """
        :param n_clusters:
        :param n_outliers:
        :param n_machines:
        :param alpha:
        :param beta:
        :param random_state:
        :param debug:
        """
        super().__init__(central_solver=kz_means, cost_func=kzmeans_cost_,
                         n_clusters=n_clusters, n_outliers=n_outliers,
                         n_machines=n_machines, alpha=alpha, beta=beta,
                         random_state=random_state, debug=debug)


class CAZDistributedKZMedian(CAZDistributedKZClustering):
    """
    Jiecao Chen, Erfan Sadeqi Azer, Qin Zhang.
    A Practical Algorithm for Distributed Clustering and Outlier Detection.
    NIPS'18
    """
    def __init__(self,
                 n_clusters=None, n_outliers=None, n_machines=None,
                 alpha=2, beta=0.45, random_state=None, debug=False):
        """
        :param n_clusters:
        :param n_outliers:
        :param n_machines:
        :param alpha:
        :param beta:
        :param random_state:
        :param debug:
        """
        super().__init__(central_solver=kz_median, cost_func=kzmedian_cost_,
                         n_clusters=n_clusters, n_outliers=n_outliers,
                         n_machines=n_machines, alpha=alpha, beta=beta,
                         random_state=random_state, debug=debug)
