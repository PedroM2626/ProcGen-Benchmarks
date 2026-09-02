import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Tuple


# =====================================================================
# 1. VISION TRANSFORMER (ViT) vs CNN
# =====================================================================
class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) in Flax for pixel-based RL.
    Converts 63x63x3 image into a sequence of patch tokens with Self-Attention.
    """
    patch_size: int = 7
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    action_dim: int = 17

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # x: (Batch, 63, 63, 3)
        b, h, w, c = x.shape
        x = x.astype(jnp.float32)
        x = jnp.where(jnp.max(x) > 1.0, x / 255.0, x)

        # Linear projection of flattened patches (9x9 = 81 patches of 7x7x3)
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size
        patches = x.reshape((b, num_patches_h, self.patch_size, num_patches_w, self.patch_size, c))
        patches = patches.transpose((0, 1, 3, 2, 4, 5))
        patches = patches.reshape((b, num_patches_h * num_patches_w, -1))

        # Patch embedding
        tokens = nn.Dense(self.embed_dim)(patches)

        # Learnable positional embedding
        pos_embed = self.param("pos_embed", nn.initializers.normal(0.02), (1, tokens.shape[1], self.embed_dim))
        tokens = tokens + pos_embed

        # Transformer Encoder Blocks
        for _ in range(self.num_layers):
            norm1 = nn.LayerNorm()(tokens)
            attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(norm1, norm1)
            tokens = tokens + attn
            norm2 = nn.LayerNorm()(tokens)
            mlp = nn.Dense(self.embed_dim * 2)(norm2)
            mlp = nn.relu(mlp)
            mlp = nn.Dense(self.embed_dim)(mlp)
            tokens = tokens + mlp

        # Global average pooling
        cls_rep = tokens.mean(axis=1)

        actor_logits = nn.Dense(self.action_dim)(cls_rep)
        critic_value = nn.Dense(1)(cls_rep)
        return actor_logits, jnp.squeeze(critic_value, axis=-1)


# =====================================================================
# 2. INTRINSIC CURIOSITY MODULE (ICM - Pathak et al.)
# =====================================================================
class IntrinsicCuriosityModule(nn.Module):
    """
    Intrinsic Curiosity Module (ICM):
    Computes intrinsic exploration rewards based on prediction error of forward dynamics.
    - Inverse Model: (phi(s), phi(s')) -> a_hat
    - Forward Model: (phi(s), a) -> phi_hat(s')
    - Intrinsic Reward: 0.5 * || phi_hat(s') - phi(s') ||^2
    """
    feature_dim: int = 128
    action_dim: int = 17

    @nn.compact
    def __call__(self, state: jnp.ndarray, next_state: jnp.ndarray, action: jnp.ndarray):
        s_flat = state.astype(jnp.float32).reshape((state.shape[0], -1))
        ns_flat = next_state.astype(jnp.float32).reshape((next_state.shape[0], -1))

        # Shared feature encoder
        encoder = nn.Sequential([
            nn.Dense(256),
            nn.relu,
            nn.Dense(self.feature_dim)
        ])
        phi_s = encoder(s_flat)
        phi_next = encoder(ns_flat)

        # Inverse model: predicts action taken between s and s'
        inv_in = jnp.concatenate([phi_s, phi_next], axis=-1)
        pred_action_logits = nn.Dense(self.action_dim)(nn.relu(nn.Dense(128)(inv_in)))

        # Forward model: predicts next latent feature phi(s') from phi(s) and action
        act_one_hot = jax.nn.one_hot(action, self.action_dim)
        fwd_in = jnp.concatenate([phi_s, act_one_hot], axis=-1)
        pred_phi_next = nn.Dense(self.feature_dim)(nn.relu(nn.Dense(128)(fwd_in)))

        # Intrinsic reward (surprise / curiosity)
        intrinsic_reward = 0.5 * jnp.sum(jnp.square(pred_phi_next - phi_next), axis=-1)
        return pred_action_logits, pred_phi_next, phi_next, intrinsic_reward


# =====================================================================
# 3. CONTRASTIVE REPRESENTATION LEARNING (CURL / InfoNCE)
# =====================================================================
class ContrastiveEncoder(nn.Module):
    """
    Contrastive Representation Learning (InfoNCE / CURL):
    Maximizes agreement between representations of the same state under random transformations.
    """
    latent_dim: int = 128

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_flat = x.astype(jnp.float32).reshape((x.shape[0], -1))
        h = nn.Dense(256)(x_flat)
        h = nn.relu(h)
        z = nn.Dense(self.latent_dim)(h)
        # Normalize to unit hypersphere
        z = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)
        return z


def info_nce_loss(query: jnp.ndarray, key: jnp.ndarray, temperature: float = 0.1) -> jnp.ndarray:
    """
    Computes InfoNCE contrastive loss over a batch of queries and keys.
    """
    logits = jnp.matmul(query, key.T) / temperature
    labels = jnp.arange(query.shape[0])
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss


# =====================================================================
# 4. LATENT WORLD MODEL (Dynamics & Auxiliary Reward Predictor)
# =====================================================================
class LatentWorldModel(nn.Module):
    """
    Latent World Model (RSSM-Lite style):
    Encodes states into latent space and predicts both future latent transition and reward.
    """
    latent_dim: int = 128
    action_dim: int = 17

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray, next_state: jnp.ndarray):
        s_flat = state.astype(jnp.float32).reshape((state.shape[0], -1))
        ns_flat = next_state.astype(jnp.float32).reshape((next_state.shape[0], -1))

        enc = nn.Sequential([
            nn.Dense(256),
            nn.relu,
            nn.Dense(self.latent_dim)
        ])
        z_curr = enc(s_flat)
        z_next = enc(ns_flat)

        act_one_hot = jax.nn.one_hot(action, self.action_dim)
        dyn_in = jnp.concatenate([z_curr, act_one_hot], axis=-1)
        pred_z_next = nn.Dense(self.latent_dim)(nn.relu(nn.Dense(128)(dyn_in)))
        pred_reward = nn.Dense(1)(nn.relu(nn.Dense(64)(dyn_in)))

        dyn_loss = jnp.mean(jnp.square(pred_z_next - jax.lax.stop_gradient(z_next)))
        return z_curr, pred_z_next, jnp.squeeze(pred_reward, -1), dyn_loss
