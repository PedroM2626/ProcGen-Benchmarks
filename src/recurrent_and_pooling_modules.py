import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple


# =====================================================================
# 1. LSTM + ATTENTION EXTRACTOR (Temporal Recurrent Memory)
# Matches models/combined_extractors.py: LSTMAttentionExtractor
# =====================================================================
class FeatureExtractorLSTMAttention(nn.Module):
    """
    CNN feature extractor followed by Spatial Attention and recurrent GRU/LSTM cell
    for partially observable procedural environments.
    """
    latent_dim: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray, hidden_state: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # x: (batch, H, W, C)
        batch = x.shape[0]
        # Visual CNN
        h = nn.Conv(32, kernel_size=(3, 3), strides=(2, 2))(x)
        h = nn.relu(h)
        h = nn.Conv(64, kernel_size=(3, 3), strides=(2, 2))(h)
        h = nn.relu(h)
        
        # Spatial Attention
        attn = nn.Conv(1, kernel_size=(3, 3), padding="SAME")(h)
        attn = nn.sigmoid(attn)
        h = h * attn + h  # Residual connection
        
        feat = h.reshape((batch, -1))
        feat = nn.Dense(self.latent_dim)(feat)
        feat = nn.relu(feat)
        
        # Recurrent Cell (GRU / LSTM equivalent in JAX)
        if hidden_state is None:
            hidden_state = jnp.zeros((batch, self.latent_dim))
        
        gru_cell = nn.GRUCell(features=self.latent_dim)
        new_hidden, out = gru_cell(hidden_state, feat)
        return out, new_hidden


# =====================================================================
# 2. IMPOOLA CNN: GLOBAL AVERAGE POOLING (GAP 64D)
# Matches models/combined_extractors.py: ImpoolaCNNExtractor
# Replaces parameter-heavy Flatten -> FC 512 with Parameter-free GAP
# =====================================================================
class FeatureExtractorImpoola(nn.Module):
    """
    Impoola: Convolutional stack with Global Average Pooling (GAP)
    drastically reduces parameter count and prevents spatial overfitting.
    """
    out_dim: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (batch, H, W, C)
        h = nn.Conv(32, kernel_size=(3, 3), strides=(2, 2))(x)
        h = nn.relu(h)
        h = nn.Conv(64, kernel_size=(3, 3), strides=(2, 2))(h)
        h = nn.relu(h)
        h = nn.Conv(self.out_dim, kernel_size=(3, 3), strides=(1, 1))(h)
        h = nn.relu(h)
        
        # Global Average Pooling over spatial dimensions (H, W)
        gap = jnp.mean(h, axis=(1, 2))  # (batch, out_dim)
        return gap


# =====================================================================
# 3. NGU (NEVER GIVE UP) EPISODIC MEMORY BONUS
# Matches compare_maze_heist.py: NGUWrapper
# Extends RND with an episodic novelty memory buffer: r_intr = r_RND * L_episodic
# =====================================================================
class NGUEpisodicMemory:
    """
    Episodic memory bonus that scales intrinsic reward based on within-episode visitation counts.
    """
    def __init__(self, memory_size: int = 200, beta: float = 0.5):
        self.memory_size = memory_size
        self.beta = beta

    @staticmethod
    def compute_bonus(rnd_bonus: jnp.ndarray, visit_counts: jnp.ndarray) -> jnp.ndarray:
        # Episodic factor: 1 / sqrt(N(s)) + 1
        episodic_factor = 1.0 / jnp.sqrt(visit_counts.astype(jnp.float32) + 1.0)
        # NGU modulated bonus
        ngu_bonus = rnd_bonus * (1.0 + episodic_factor)
        return ngu_bonus
