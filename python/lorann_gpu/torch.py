import torch
import numpy as np

from .common import LorannBase


torch.set_float32_matmul_precision("high")


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
        dtype=torch.float32,
        device="cuda",
    ):
        super().__init__(data, n_clusters, global_dim, rank, train_size, euclidean, approximate)

        self.dtype = dtype
        self.data = torch.tensor(self.data, dtype=self.dtype, device=device)
        self.centroids = torch.tensor(self.centroids, dtype=self.dtype, device=device)
        self.A = torch.tensor(np.array(self.A), dtype=self.dtype, device=device)
        self.B = torch.tensor(np.array(self.B), dtype=self.dtype, device=device)
        self.cluster_map = torch.tensor(np.array(self.cluster_map, dtype=np.int32), device=device)

        if self.global_transform is not None:
            self.global_transform = torch.tensor(
                self.global_transform, dtype=self.dtype, device=device
            )
        if euclidean:
            self.global_centroid_norms = torch.tensor(
                self.global_centroid_norms, dtype=self.dtype, device=device
            )
            self.data_norms = torch.tensor(self.data_norms, dtype=self.dtype, device=device)
            self.cluster_norms = torch.tensor(
                np.array(self.cluster_norms), dtype=self.dtype, device=device
            )

    @torch.compile(fullgraph=True)
    def compiled_search(self, q, k, clusters_to_search, points_to_rerank):
        if self.euclidean:
            q = -2 * q
        else:
            q = -q

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

        _, I = torch.topk(d, clusters_to_search, largest=False)

        global_tmp = global_tmp.reshape((batch_size, 1, 1, -1))
        res = (global_tmp @ self.A[I] @ self.B[I]).reshape(estimate_size)
        if self.euclidean:
            res += self.cluster_norms[I].reshape(estimate_size)
        idx = self.cluster_map[I].reshape(estimate_size)

        if points_to_rerank >= clusters_to_search * self.max_cluster_size:
            cs = idx
            points_to_rerank = clusters_to_search * self.max_cluster_size
        else:
            _, idx_cs = torch.topk(res, points_to_rerank, largest=False)
            cs = torch.gather(idx, 1, idx_cs)

        if points_to_rerank <= k:
            return cs

        final_dists = (self.data[cs] @ q[:, :, None]).reshape((batch_size, points_to_rerank))
        if self.euclidean:
            final_dists += self.data_norms[cs]

        k = min(k, final_dists.shape[1])
        _, idx_final = torch.topk(final_dists, k, largest=False)
        return torch.gather(cs, 1, idx_final)

    def search(self, q, k, clusters_to_search, points_to_rerank, device="cuda"):
        q = torch.tensor(q, dtype=self.dtype, device=device)
        return self.compiled_search(q, k, clusters_to_search, points_to_rerank).to("cpu").numpy()
