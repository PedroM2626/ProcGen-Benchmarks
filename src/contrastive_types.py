import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Tuple


# =====================================================================
# 1. SPATIAL CONTRASTIVE (CURL / SimCLR style)
# Pair: view_1(s_t) vs view_2(s_t) via random augmentations
# =====================================================================
class SpatialContrastiveHead(nn.Module):
    proj_dim: int = 64

    @nn.compact
    def __call__(self, feat: jnp.ndarray) -> jnp.ndarray:
        z = nn.Dense(128)(feat)
        z = nn.relu(z)
        z = nn.Dense(self.proj_dim)(z)
        return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)


# =====================================================================
# 2. TEMPORAL CONTRASTIVE (CPC - Contrastive Predictive Coding)
# Pair: s_t vs s_{t+k} across trajectory
# =====================================================================
class TemporalContrastiveHead(nn.Module):
    proj_dim: int = 64
    k_steps: int = 3

    @nn.compact
    def __call__(self, feat_t: jnp.ndarray, feat_future: jnp.ndarray) -> jnp.ndarray:
        # Predicts transformation W_k such that z_t * W_k matches z_{t+k}
        z_t = nn.Dense(self.proj_dim)(feat_t)
        z_fut = nn.Dense(self.proj_dim)(feat_future)
        
        z_t = z_t / (jnp.linalg.norm(z_t, axis=-1, keepdims=True) + 1e-8)
        z_fut = z_fut / (jnp.linalg.norm(z_fut, axis=-1, keepdims=True) + 1e-8)
        
        # Bilinear mapping
        w = self.param("w_bilinear", nn.initializers.normal(0.02), (self.proj_dim, self.proj_dim))
        pred_fut = jnp.matmul(z_t, w)
        return pred_fut, z_fut


# =====================================================================
# 3. ACTION-CONDITIONAL CONTRASTIVE (ACL)
# Pair: (s_t, a_t) vs true s_{t+1} (positive) vs other transitions (negatives)
# =====================================================================
class ActionConditionalContrastiveHead(nn.Module):
    proj_dim: int = 64
    action_dim: int = 17

    @nn.compact
    def __call__(self, feat_t: jnp.ndarray, action: jnp.ndarray, feat_next: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        act_one_hot = jax.nn.one_hot(action, self.action_dim)
        in_joint = jnp.concatenate([feat_t, act_one_hot], axis=-1)
        z_trans = nn.Dense(self.proj_dim)(nn.relu(nn.Dense(128)(in_joint)))
        z_next = nn.Dense(self.proj_dim)(feat_next)
        
        z_trans = z_trans / (jnp.linalg.norm(z_trans, axis=-1, keepdims=True) + 1e-8)
        z_next = z_next / (jnp.linalg.norm(z_next, axis=-1, keepdims=True) + 1e-8)
        return z_trans, z_next


# =====================================================================
# 4. SELF-PREDICTIVE REPRESENTATIONS (SPR / BYOL style)
# Non-contrastive / Latent prediction with cosine loss and stop-gradient
# =====================================================================
class SPRPredictorHead(nn.Module):
    proj_dim: int = 64

    @nn.compact
    def __call__(self, online_feat: jnp.ndarray, target_feat: jnp.ndarray) -> jnp.ndarray:
        # Online predictor
        pred = nn.Dense(128)(online_feat)
        pred = nn.relu(pred)
        pred = nn.Dense(self.proj_dim)(pred)
        
        target = nn.Dense(self.proj_dim)(target_feat)
        # Cosine similarity loss (no negative pairs required)
        pred_norm = pred / (jnp.linalg.norm(pred, axis=-1, keepdims=True) + 1e-8)
        targ_norm = target / (jnp.linalg.norm(target, axis=-1, keepdims=True) + 1e-8)
        
        # Stop gradient on target network (BYOL principle)
        targ_norm = jax.lax.stop_gradient(targ_norm)
        cos_sim = jnp.sum(pred_norm * targ_norm, axis=-1)
        return 1.0 - jnp.mean(cos_sim)


def info_nce_similarity(queries: jnp.ndarray, keys: jnp.ndarray, temperature: float = 0.1) -> jnp.ndarray:
    """Computes InfoNCE loss with dot-product logits."""
    logits = jnp.matmul(queries, keys.T) / temperature
    labels = jnp.arange(queries.shape[0])
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
