import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
from functools import partial

from .common import LorannBase, IVFBase


class Lorann:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def build(
        cls,
        data,
        n_clusters,
        global_dim,
        rank=24,
        train_size=5,
        euclidean=False,
        approximate=True,
        data_dtype=jnp.float32,
        dtype=jnp.float32,
    ):
        n = LorannBase(data, n_clusters, global_dim, rank, train_size, euclidean, approximate)

        data = jnp.array(n.data, dtype=data_dtype)
        centroids = jnp.array(n.centroids, dtype=dtype)
        A = jnp.array(np.array(n.A), dtype=dtype)
        B = jnp.array(np.array(n.B), dtype=dtype)
        cluster_map = jnp.array(np.array(n.cluster_map, dtype=np.int32))
        max_cluster_size = n.max_cluster_size

        if n.global_transform is not None:
            global_transform = jnp.array(n.global_transform, dtype=dtype)
        else:
            global_transform = None

        if euclidean:
            global_centroid_norms = jnp.array(n.global_centroid_norms, dtype=dtype)
            data_norms = jnp.array(n.data_norms, dtype=dtype)
            cluster_norms = jnp.array(np.array(n.cluster_norms), dtype=dtype)

            index_data = [
                "data",
                "global_transform",
                "centroids",
                "global_centroid_norms",
                "data_norms",
                "A",
                "B",
                "cluster_norms",
                "max_cluster_size",
                "cluster_map",
                "euclidean",
                "dtype",
            ]
        else:
            index_data = [
                "data",
                "global_transform",
                "centroids",
                "A",
                "B",
                "max_cluster_size",
                "cluster_map",
                "euclidean",
                "dtype",
            ]

        variable_dict = dict(globals(), **locals())
        return cls(**{v: variable_dict[v] for v in index_data})

    @partial(jax.jit, static_argnames=["k", "clusters_to_search", "points_to_rerank"])
    def compiled_search(self, q, k, clusters_to_search, points_to_rerank):
        if self.euclidean:
            q = -2 * jnp.array(q, dtype=self.dtype)
        else:
            q = -jnp.array(q, dtype=self.dtype)

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

        _, I = jax.lax.top_k(-d, k=clusters_to_search)

        global_tmp = global_tmp.reshape((batch_size, 1, 1, -1))
        res = (global_tmp @ self.A[I] @ self.B[I]).reshape(estimate_size)
        if self.euclidean:
            res += self.cluster_norms[I].reshape(estimate_size)
        idx = self.cluster_map[I].reshape(estimate_size)

        if points_to_rerank >= clusters_to_search * self.max_cluster_size:
            cs = idx
            points_to_rerank = clusters_to_search * self.max_cluster_size
        else:
            _, cs_idx = jax.lax.top_k(-res, points_to_rerank)
            cs = jnp.take_along_axis(idx, cs_idx, axis=1)

        if points_to_rerank <= k:
            return cs

        final_dists = (self.data[cs] @ q[:, :, None]).reshape((batch_size, points_to_rerank))
        if self.euclidean:
            final_dists += self.data_norms[cs]

        k = min(k, final_dists.shape[1])
        _, res_idx = jax.lax.top_k(-final_dists, k)
        return jnp.take_along_axis(cs, res_idx, axis=1)

    def search(self, q, k, clusters_to_search, points_to_rerank):
        return np.asarray(self.compiled_search(q, k, clusters_to_search, points_to_rerank))

    def _tree_flatten(self):
        return (tuple(), self.__dict__)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


class IVF:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def build(cls, data, n_clusters, euclidean, data_dtype=jnp.float32, dtype=jnp.float32):
        n = IVFBase(data, n_clusters, euclidean)

        data = jnp.array(n.data, dtype=data_dtype)
        centroids = jnp.array(n.centroids, dtype=dtype)
        A = jnp.array(np.array(n.A), dtype=dtype)
        cluster_map = jnp.array(np.array(n.cluster_map, dtype=np.int32))
        max_cluster_size = n.max_cluster_size

        if euclidean:
            global_centroid_norms = jnp.array(n.global_centroid_norms, dtype=dtype)
            data_norms = jnp.array(n.data_norms, dtype=dtype)
            cluster_norms = jnp.array(np.array(n.cluster_norms), dtype=dtype)

            index_data = [
                "data",
                "centroids",
                "global_centroid_norms",
                "data_norms",
                "A",
                "cluster_norms",
                "max_cluster_size",
                "cluster_map",
                "euclidean",
                "dtype",
            ]
        else:
            index_data = [
                "data",
                "centroids",
                "A",
                "max_cluster_size",
                "cluster_map",
                "euclidean",
                "dtype",
            ]

        variable_dict = dict(globals(), **locals())
        return cls(**{v: variable_dict[v] for v in index_data})

    @partial(jax.jit, static_argnames=["k", "clusters_to_search"])
    def compiled_search(self, q, k, clusters_to_search):
        if self.euclidean:
            q = -2 * jnp.array(q, dtype=self.dtype)
        else:
            q = -jnp.array(q, dtype=self.dtype)

        batch_size = q.shape[0]
        estimate_size = (batch_size, clusters_to_search * self.max_cluster_size)

        d = q @ self.centroids.T
        if self.euclidean:
            d += self.global_centroid_norms

        _, I = jax.lax.top_k(-d, k=clusters_to_search)

        res = (q.reshape(batch_size, 1, 1, -1) @ self.A[I]).reshape(estimate_size)
        if self.euclidean:
            res += self.cluster_norms[I].reshape(estimate_size)
        idx = self.cluster_map[I].reshape(estimate_size)

        _, res_idx = jax.lax.top_k(-res, k)
        return jnp.take_along_axis(idx, res_idx, axis=1)

    def search(self, q, k, clusters_to_search):
        return np.asarray(self.compiled_search(q, k, clusters_to_search))

    def _tree_flatten(self):
        return (tuple(), self.__dict__)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


jax.tree_util.register_pytree_node(Lorann, Lorann._tree_flatten, Lorann._tree_unflatten)
jax.tree_util.register_pytree_node(IVF, IVF._tree_flatten, IVF._tree_unflatten)
