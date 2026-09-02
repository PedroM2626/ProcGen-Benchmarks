import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple


# =====================================================================
# 1. GRAPH ENTITY CONVERTER
# Converts spatial observations into an Entity Graph G = (V, E)
# Nodes V: Agent (1), Closest Enemies (K_enemies), Nearby Resources (K_resources)
# Edges E: Adjacency matrix based on spatial distance threshold
# =====================================================================
class GraphEntityExtractor:
    """
    Extracts structured entity nodes and adjacency matrices from procedural state.
    Provides permutation invariance and relational inductive bias.
    """
    def __init__(self, num_entities: int = 16, entity_feat_dim: int = 8, radius: float = 0.5):
        self.num_entities = num_entities
        self.entity_feat_dim = entity_feat_dim
        self.radius = radius

    @staticmethod
    def extract_graph_from_obs(obs: jnp.ndarray, num_nodes: int = 16, feat_dim: int = 8) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # obs: (batch, H, W, C) or (batch, features)
        batch = obs.shape[0]
        # In JAX functional pipeline: projects visual feature grid or symbolic state into entity nodes
        # Each node represents an active entity: [x_pos, y_pos, entity_type_id, health, distance_to_agent, vel_x, vel_y, active_mask]
        obs_flat = obs.reshape((batch, -1))
        # Linear entity projection
        node_features = nn.Dense(num_nodes * feat_dim)(obs_flat)
        node_features = node_features.reshape((batch, num_nodes, feat_dim))
        
        # Spatial positions from first 2 coordinates of node features
        pos = node_features[:, :, :2]  # (batch, num_nodes, 2)
        diff = pos[:, :, None, :] - pos[:, None, :, :]  # (batch, num_nodes, num_nodes, 2)
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-6)  # (batch, num_nodes, num_nodes)
        
        # Continuous adjacency matrix (Gaussian affinity based on Euclidean distance)
        adj_matrix = jnp.exp(-dist**2 / (2.0 * (0.4**2)))  # (batch, num_nodes, num_nodes)
        return node_features, adj_matrix


# =====================================================================
# 2. GRAPH ATTENTION NETWORK (GAT / GNN) EXTRACTOR
# Message passing over entity graph with multi-head attention and global readout
# =====================================================================
class FeatureExtractorGNN(nn.Module):
    """
    Graph Neural Network (GNN) with Graph Attention (GAT) layers and Graph Readout (Pooling).
    Inherently permutation-invariant across entities.
    """
    num_nodes: int = 16
    node_feat_dim: int = 8
    hidden_dim: int = 128
    out_dim: int = 512
    num_heads: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (batch, ...)
        batch = x.shape[0]
        x_flat = x.astype(jnp.float32).reshape((batch, -1))

        # 1. Project input to Entity Graph Nodes
        nodes = nn.Dense(self.num_nodes * self.node_feat_dim)(x_flat)
        nodes = nodes.reshape((batch, self.num_nodes, self.node_feat_dim))
        nodes = nn.relu(nodes)

        # 2. Compute Relational Adjacency
        pos = nodes[:, :, :2]
        diff = pos[:, :, None, :] - pos[:, None, :, :]
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-6)
        adj = jnp.exp(-dist**2 / 0.32)  # (batch, num_nodes, num_nodes)

        # 3. Graph Convolution / GAT Layer 1
        h = nn.Dense(self.hidden_dim)(nodes)
        h = nn.relu(h)
        # Message Passing: Aggregate neighbor features weighted by adjacency
        # Adjacency normalized by degree
        deg = jnp.sum(adj, axis=-1, keepdims=True) + 1e-6
        norm_adj = adj / deg
        messages = jnp.matmul(norm_adj, h)  # (batch, num_nodes, hidden_dim)
        h = nn.LayerNorm()(h + messages)

        # 4. Multi-Head Node Self-Attention Layer (Relational Reasoning)
        attn_out = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(h, h)
        h = nn.LayerNorm()(h + attn_out)

        # 5. Graph Readout / Global Pooling (Permutation Invariant)
        # Combines Mean Pooling (global context) + Max Pooling (most salient entity/threat)
        mean_pool = jnp.mean(h, axis=1)  # (batch, hidden_dim)
        max_pool = jnp.max(h, axis=1)    # (batch, hidden_dim)
        graph_repr = jnp.concatenate([mean_pool, max_pool], axis=-1)  # (batch, 2 * hidden_dim)

        # 6. Final Projector to match standard 512D latent feature space
        out = nn.Dense(self.out_dim)(graph_repr)
        out = nn.relu(out)
        return out
