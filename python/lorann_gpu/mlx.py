import numpy as np
import mlx.core as mx

from .common import LorannBase, IVFBase


class Lorann(LorannBase):

    def __init__(
        self,
        data,
        n_clusters,
        global_dim,
        rank=24,
        train_size=5,
        euclidean=False,
        approximate=True,
        dtype=mx.float32,
    ):
        super().__init__(data, n_clusters, global_dim, rank, train_size, euclidean, approximate)

        self.dtype = dtype
        self.data = mx.array(self.data, dtype=self.dtype)
        self.centroids = mx.array(self.centroids, dtype=self.dtype)
        self.A = mx.array(np.array(self.A), dtype=self.dtype)
        self.B = mx.array(np.array(self.B), dtype=self.dtype)
        self.cluster_map = mx.array(np.array(self.cluster_map, dtype=np.int32))

        if self.global_transform is not None:
            self.global_transform = mx.array(self.global_transform, dtype=self.dtype)
        if euclidean:
            self.global_centroid_norms = mx.array(self.global_centroid_norms, dtype=self.dtype)
            self.data_norms = mx.array(self.data_norms, dtype=self.dtype)
            self.cluster_norms = mx.array(np.array(self.cluster_norms), dtype=self.dtype)

    def search(self, q, k, clusters_to_search, points_to_rerank):
        q = mx.array(q, dtype=self.dtype)
        if self.euclidean:
            q = -2 * q
        else:
            q = -q

        batch_size = q.shape[0]
        clusters_to_search = min(clusters_to_search, self.centroids.shape[0])
        estimate_size = (batch_size, clusters_to_search * self.max_cluster_size)

        if self.global_transform is not None:
            global_tmp = mx.matmul(q, self.global_transform)
        else:
            global_tmp = q

        d = mx.matmul(global_tmp, self.centroids.T)
        if self.euclidean:
            d += self.global_centroid_norms

        I = mx.argpartition(d, clusters_to_search, axis=1)[:, :clusters_to_search]

        global_tmp = global_tmp.reshape((batch_size, 1, 1, -1))
        res = mx.matmul(mx.matmul(global_tmp, self.A[I]), self.B[I]).reshape(estimate_size)
        if self.euclidean:
            res += self.cluster_norms[I].reshape(estimate_size)
        idx = self.cluster_map[I].reshape(estimate_size)

        if points_to_rerank >= clusters_to_search * self.max_cluster_size:
            cs = idx
            points_to_rerank = clusters_to_search * self.max_cluster_size
        else:
            idx_cs = mx.argpartition(res, points_to_rerank, axis=1)
            cs = mx.take_along_axis(idx, idx_cs[:, :points_to_rerank], axis=1)

        if points_to_rerank <= k:
            return cs

        final_dists = mx.matmul(self.data[cs], q[:, :, None]).reshape(
            (batch_size, points_to_rerank)
        )
        if self.euclidean:
            final_dists += self.data_norms[cs]

        k = min(k, final_dists.shape[1])
        idx_final = mx.argpartition(final_dists, k, axis=1)
        return np.array(mx.take_along_axis(cs, idx_final[:, :k], axis=1))


class IVF(IVFBase):

    def __init__(self, data, n_clusters, euclidean, dtype=mx.float32):
        super().__init__(data, n_clusters, euclidean)

        self.dtype = dtype
        self.data = mx.array(self.data, dtype=dtype)
        self.centroids = mx.array(self.centroids, dtype=dtype)
        self.A = mx.array(np.array(self.A), dtype=dtype)
        self.cluster_map = mx.array(np.array(self.cluster_map, dtype=np.int32))

        if euclidean:
            self.global_centroid_norms = mx.array(self.global_centroid_norms, dtype=dtype)
            self.data_norms = mx.array(self.data_norms, dtype=dtype)
            self.cluster_norms = mx.array(np.array(self.cluster_norms), dtype=dtype)

    def search(self, q, k, clusters_to_search):
        if self.euclidean:
            q = -2 * mx.array(q, dtype=self.dtype)
        else:
            q = -mx.array(q, dtype=self.dtype)

        batch_size = q.shape[0]
        estimate_size = (batch_size, clusters_to_search * self.max_cluster_size)

        d = mx.matmul(q, self.centroids.T)
        if self.euclidean:
            d += self.global_centroid_norms

        I = mx.argpartition(d, clusters_to_search, axis=1)[:, :clusters_to_search]

        q = q.reshape(batch_size, 1, 1, -1)
        final_dists = mx.matmul(q, self.A[I]).reshape(estimate_size)
        if self.euclidean:
            final_dists += self.cluster_norms[I].reshape(estimate_size)
        idx = self.cluster_map[I].reshape(estimate_size)

        k = min(k, final_dists.shape[1])
        idx_final = mx.argpartition(final_dists, k, axis=1)
        return np.array(mx.take_along_axis(idx, idx_final[:, :k], axis=1))
