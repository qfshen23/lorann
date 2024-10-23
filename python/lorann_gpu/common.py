import numpy as np

import lorann


def run_kmeans(train, n_clusters, euclidean=False, balanced=True):
    kmeans = lorann.KMeans(
        n_clusters=n_clusters,
        iters=10,
        euclidean=euclidean,
        balanced=balanced,
        max_balance_diff=16,
        verbose=False,
    )

    cluster_map = kmeans.train(train)
    return kmeans, kmeans.get_centroids(), cluster_map


class LorannBase:

    def __init__(
        self, data, n_clusters, global_dim, rank=24, train_size=5, euclidean=False, approximate=True
    ):
        self.data = data
        self.euclidean = euclidean
        n_samples, dim = data.shape

        if global_dim < dim:
            Y_global = data.T @ data
            _, v = np.linalg.eigh(Y_global)
            self.global_transform = v[:, -global_dim:]

            reduced_train_mat = data @ self.global_transform
            kmeans, self.centroids, self.cluster_map = run_kmeans(
                reduced_train_mat, n_clusters, euclidean, True
            )
            centroid_train_map = kmeans.assign(reduced_train_mat, train_size)
        else:
            kmeans, self.centroids, self.cluster_map = run_kmeans(data, n_clusters, euclidean, True)
            centroid_train_map = kmeans.assign(data, train_size)
            self.global_transform = None

        self.max_cluster_size = max(len(c) for c in self.cluster_map)

        if euclidean:
            self.global_centroid_norms = np.linalg.norm(self.centroids, ord=2, axis=1) ** 2
            self.data_norms = np.linalg.norm(data, ord=2, axis=1) ** 2
            self.cluster_norms = []
        else:
            self.global_centroid_norms = None
            self.data_norms = None
            self.cluster_norms = None

        self.A, self.B = [], []
        for i in range(n_clusters):
            if len(self.cluster_map[i]) == 0:
                self.cluster_map[i] = np.zeros(self.max_cluster_size, dtype=np.int32)
                continue

            sz = len(self.cluster_map[i])
            pts = data[self.cluster_map[i]]
            Q = data[centroid_train_map[i]]

            if global_dim < dim:
                if approximate:
                    beta_hat = (pts @ self.global_transform).T
                    Y_hat = (Q @ self.global_transform) @ beta_hat
                else:
                    Y = Q @ pts.T
                    X = Q @ self.global_transform
                    beta_hat = np.linalg.lstsq(X, Y, rcond=None)[0]
                    Y_hat = X @ beta_hat
            else:
                beta_hat = pts.T
                Y_hat = Q @ pts.T

            V = lorann.compute_V(Y_hat, rank, approximate)

            A = beta_hat @ V.T
            B = np.hstack((V, np.zeros((V.shape[0], self.max_cluster_size - sz), dtype=np.float32)))
            self.A.append(A)
            self.B.append(B)

            if self.cluster_norms is not None:
                self.cluster_norms.append(
                    np.concatenate(
                        (
                            self.data_norms[self.cluster_map[i]],
                            np.zeros(self.max_cluster_size - sz, dtype=np.float32),
                        )
                    )
                )

            self.cluster_map[i] = np.concatenate(
                (self.cluster_map[i], np.zeros(self.max_cluster_size - sz, dtype=np.int32))
            )


class IVFBase:

    def __init__(self, data, n_clusters, euclidean=False):
        self.data = data
        self.euclidean = euclidean

        kmeans, self.centroids, self.cluster_map = run_kmeans(data, n_clusters, euclidean, True)
        self.max_cluster_size = max(len(c) for c in self.cluster_map)

        if euclidean:
            self.global_centroid_norms = np.linalg.norm(self.centroids, ord=2, axis=1) ** 2
            self.data_norms = np.linalg.norm(data, ord=2, axis=1) ** 2
            self.cluster_norms = []
        else:
            self.global_centroid_norms = None
            self.data_norms = None
            self.cluster_norms = None

        self.A, self.B = [], []
        for i in range(n_clusters):
            sz = len(self.cluster_map[i])
            A = data[self.cluster_map[i]].T
            self.A.append(
                np.hstack((A, np.zeros((A.shape[0], self.max_cluster_size - sz), dtype=np.float32)))
            )

            if self.cluster_norms is not None:
                self.cluster_norms.append(
                    np.concatenate(
                        (
                            self.data_norms[self.cluster_map[i]],
                            np.zeros(self.max_cluster_size - sz, dtype=np.float32),
                        )
                    )
                )

            self.cluster_map[i] = np.concatenate(
                (self.cluster_map[i], np.zeros(self.max_cluster_size - sz, dtype=np.int32))
            )
