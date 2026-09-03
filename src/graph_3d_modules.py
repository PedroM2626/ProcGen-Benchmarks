import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple


class Graph3DEntityExtractor(nn.Module):
    """
    Extrator Relacional 3D (GNN / GAT para Física e Controle 3D).
    Converte nós com coordenadas (x, y, z) e velocidades em embeddings relacionais 3D
    com matriz de adjacência Euclideana tridimensional contínua.
    """
    num_nodes: int = 6
    node_dim: int = 16
    embed_dim: int = 64

    @nn.compact
    def __call__(self, obs_3d: jnp.ndarray) -> jnp.ndarray:
        # obs_3d: (batch, obs_len)
        batch = obs_3d.shape[0]
        
        # Projeção de nós 3D
        h_nodes = nn.Dense(self.num_nodes * self.node_dim)(obs_3d)
        nodes = jnp.reshape(h_nodes, (batch, self.num_nodes, self.node_dim))

        # Posições latentes 3D (x, y, z)
        pos_3d = nodes[..., :3]
        
        # Matriz de Distância Euclideana 3D: dist_ij = ||p_i - p_j||_2
        diff = pos_3d[:, :, None, :] - pos_3d[:, None, :, :] # (batch, N, N, 3)
        dist_3d = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-6) # (batch, N, N)

        # Adjacência contínua por afinidade espacial Gaussiana RBF em 3D
        adj_3d = jnp.exp(- (dist_3d**2) / 0.5)

        # 1. Message Passing Relacional 3D
        message = nn.Dense(self.node_dim)(nodes)
        aggregated = jnp.einsum('bij,bjd->bid', adj_3d, message)
        nodes_updated = nn.relu(nodes + aggregated)

        # 2. Multi-Head Graph Attention (GAT 3D)
        q = nn.Dense(self.node_dim)(nodes_updated)
        k = nn.Dense(self.node_dim)(nodes_updated)
        v = nn.Dense(self.node_dim)(nodes_updated)
        
        attn_logits = jnp.einsum('bid,bjd->bij', q, k) / jnp.sqrt(self.node_dim)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        gat_out = jnp.einsum('bij,bjd->bid', attn_weights, v)

        # 3. Readout Pooling Tridimensional (Invariante à Permutação)
        mean_pool = jnp.mean(gat_out, axis=1)
        max_pool = jnp.max(gat_out, axis=1)
        global_context = jnp.concatenate([mean_pool, max_pool], axis=-1)
        
        out = nn.Dense(self.embed_dim)(global_context)
        return nn.relu(out)
