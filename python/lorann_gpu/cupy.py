import cupy as cp
import numpy as np
from pylibraft.matrix import select_k

from .common import LorannBase


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
        dtype=cp.float32,
    ):
        super().__init__(data, n_clusters, global_dim, rank, train_size, euclidean, approximate)

        self.dtype = dtype
        self.data = cp.array(self.data, dtype=self.dtype)
        self.centroids = cp.array(self.centroids, dtype=self.dtype)
        self.A = cp.array(np.array(self.A), dtype=self.dtype)
        self.B = cp.array(np.array(self.B), dtype=self.dtype)
        self.cluster_map = cp.array(np.array(self.cluster_map, dtype=np.int32))

        if self.global_transform is not None:
            self.global_transform = cp.array(self.global_transform, dtype=self.dtype)
        if euclidean:
            self.global_centroid_norms = cp.array(self.global_centroid_norms, dtype=self.dtype)
            self.data_norms = cp.array(self.data_norms, dtype=self.dtype)
            self.cluster_norms = cp.array(np.array(self.cluster_norms), dtype=self.dtype)

    def search(self, q, k, clusters_to_search, points_to_rerank):
        if self.euclidean:
            q = -2 * cp.array(q, dtype=self.dtype)
        else:
            q = -cp.array(q, dtype=self.dtype)

        batch_size = q.shape[0]
        clusters_to_search = min(clusters_to_search, self.centroids.shape[0])
        estimate_size = (batch_size, clusters_to_search * self.max_cluster_size)

        if self.global_transform is not None:
            global_tmp = q @ self.global_transform
        else:
            global_tmp = q

        d = global_tmp @ self.centroids.T
        if self.euclidean:
            d += self.global_centroid_norms

        _, I = select_k(d, clusters_to_search)

        global_tmp = global_tmp.reshape((batch_size, 1, 1, -1))
        res = (global_tmp @ self.A[I] @ self.B[I]).reshape(estimate_size)
        if self.euclidean:
            res += self.cluster_norms[I].reshape(estimate_size)
        idx = self.cluster_map[I].reshape(estimate_size)

        if points_to_rerank >= clusters_to_search * self.max_cluster_size:
            cs = idx
            points_to_rerank = clusters_to_search * self.max_cluster_size
        else:
            _, idx_cs = select_k(res, points_to_rerank)
            cs = cp.take_along_axis(idx, cp.asarray(idx_cs), axis=1)

        if points_to_rerank <= k:
            return cs

        final_dists = (self.data[cs] @ q[:, :, None]).reshape((batch_size, points_to_rerank))
        if self.euclidean:
            final_dists += self.data_norms[cs]

        k = min(k, final_dists.shape[1])
        _, idx_final = select_k(final_dists, k)
        return cp.take_along_axis(cs, cp.asarray(idx_final), axis=1).get()
