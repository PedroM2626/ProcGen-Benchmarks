import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Sequence


# =====================================================================
# 1. ATTENTION MODULES (CBAM & SPATIAL ATTENTION WITH RESIDUAL)
# =====================================================================
class ChannelAttention(nn.Module):
    channels: int
    reduction: int = 16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, H, W, C)
        b, h, w, c = x.shape
        # Global Avg and Max Pooling over spatial dimensions
        avg_pool = jnp.mean(x, axis=(1, 2))  # (B, C)
        max_pool = jnp.max(x, axis=(1, 2))   # (B, C)

        # Shared MLP
        mlp = nn.Sequential([
            nn.Dense(self.channels // self.reduction),
            nn.relu,
            nn.Dense(self.channels)
        ])

        scale = jax.nn.sigmoid(mlp(avg_pool) + mlp(max_pool))
        scale = scale[:, None, None, :]  # Broadcast (B, 1, 1, C)
        return x * scale


class SpatialAttention(nn.Module):
    kernel_size: int = 7

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, H, W, C)
        avg_pool = jnp.mean(x, axis=-1, keepdims=True)  # (B, H, W, 1)
        max_pool = jnp.max(x, axis=-1, keepdims=True)   # (B, H, W, 1)
        feat = jnp.concatenate([avg_pool, max_pool], axis=-1)  # (B, H, W, 2)
        
        attn_map = nn.Conv(features=1, kernel_size=(self.kernel_size, self.kernel_size), padding="SAME")(feat)
        attn_map = jax.nn.sigmoid(attn_map)
        
        # Residual connection as designed in ProcGen-Benchmarks: x * attn + x
        return x * attn_map + x


class AttentionCNN(nn.Module):
    """
    NatureCNN equipped with Spatial or CBAM attention (Wood et al. & ProcGen-Benchmarks).
    """
    action_dim: int = 17
    use_cbam: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = x.astype(jnp.float32)
        x = jnp.where(jnp.max(x) > 1.0, x / 255.0, x)

        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)

        # Apply Attention
        if self.use_cbam:
            x = ChannelAttention(channels=64)(x)
            x = SpatialAttention()(x)
        else:
            x = SpatialAttention()(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)

        actor_logits = nn.Dense(features=self.action_dim)(x)
        critic_value = nn.Dense(features=1)(x)
        return actor_logits, jnp.squeeze(critic_value, axis=-1)


# =====================================================================
# 2. QUANTILE REGRESSION DQN (QR-DQN - Dabney et al. 2018)
# =====================================================================
class QRDQNNetwork(nn.Module):
    action_dim: int = 17
    num_quantiles: int = 50

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (B, D) or (B, H, W, C)
        if x.ndim == 4:
            x = x.astype(jnp.float32)
            x = jnp.where(jnp.max(x) > 1.0, x / 255.0, x)
            x = nn.Conv(32, (8, 8), (4, 4))(x)
            x = nn.relu(x)
            x = nn.Conv(64, (4, 4), (2, 2))(x)
            x = nn.relu(x)
            x = nn.Conv(64, (3, 3), (1, 1))(x)
            x = nn.relu(x)
            x = x.reshape((x.shape[0], -1))
        else:
            x = x.astype(jnp.float32)
            x = nn.Dense(256)(x)
            x = nn.relu(x)
            x = nn.Dense(256)(x)
            x = nn.relu(x)

        x = nn.Dense(512)(x)
        x = nn.relu(x)
        # Emits (B, action_dim * num_quantiles) -> (B, action_dim, num_quantiles)
        quantiles = nn.Dense(self.action_dim * self.num_quantiles)(x)
        return quantiles.reshape((x.shape[0], self.action_dim, self.num_quantiles))


def quantile_huber_loss(quantiles: jnp.ndarray, target_quantiles: jnp.ndarray, kappa: float = 1.0) -> jnp.ndarray:
    """
    Quantile Huber Loss for QR-DQN.
    quantiles: (B, num_quantiles)
    target_quantiles: (B, num_quantiles)
    """
    # pairwise diff: (B, num_quantiles, num_quantiles)
    diff = target_quantiles[:, None, :] - quantiles[:, :, None]
    abs_diff = jnp.abs(diff)
    huber = jnp.where(abs_diff <= kappa, 0.5 * jnp.square(diff), kappa * (abs_diff - 0.5 * kappa))
    
    num_quantiles = quantiles.shape[1]
    tau = (jnp.arange(num_quantiles, dtype=jnp.float32) + 0.5) / num_quantiles
    tau = tau[None, :, None]
    loss = jnp.abs(tau - (diff < 0).astype(jnp.float32)) * huber
    return jnp.mean(loss)


# =====================================================================
# 3. RANDOM NETWORK DISTILLATION (RND - Burda et al.)
# =====================================================================
class RNDTargetNetwork(nn.Module):
    feature_dim: int = 128

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_flat = x.astype(jnp.float32).reshape((x.shape[0], -1))
        h = nn.Dense(256)(x_flat)
        h = nn.relu(h)
        return nn.Dense(self.feature_dim)(h)


class RNDPredictorNetwork(nn.Module):
    feature_dim: int = 128

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_flat = x.astype(jnp.float32).reshape((x.shape[0], -1))
        h = nn.Dense(256)(x_flat)
        h = nn.relu(h)
        h = nn.Dense(256)(h)
        h = nn.relu(h)
        return nn.Dense(self.feature_dim)(h)


# =====================================================================
# 4. DATA AUGMENTATION FOR JAX (Crop, Color, Noise)
# =====================================================================
def augment_crop(rng: jax.random.PRNGKey, imgs: jnp.ndarray, pad: int = 4) -> jnp.ndarray:
    """Random crop with zero-padding (matching ProcGen-Benchmarks ContrastiveCrop)."""
    # imgs: (B, H, W, C)
    b, h, w, c = imgs.shape
    padded = jnp.pad(imgs, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
    rng, sub1, sub2 = jax.random.split(rng, 3)
    crop_y = jax.random.randint(sub1, (b,), minval=0, maxval=2 * pad + 1)
    crop_x = jax.random.randint(sub2, (b,), minval=0, maxval=2 * pad + 1)

    def _crop_single(img, cy, cx):
        return jax.lax.dynamic_slice(img, (cy, cx, 0), (h, w, c))

    return jax.vmap(_crop_single)(padded, crop_y, crop_x)


def augment_color(rng: jax.random.PRNGKey, imgs: jnp.ndarray) -> jnp.ndarray:
    """Random brightness/contrast scale (matching ProcGen-Benchmarks ContrastiveColor)."""
    b = imgs.shape[0]
    factors = jax.random.uniform(rng, (b, 1, 1, 1), minval=0.8, maxval=1.2)
    return jnp.clip(imgs * factors, 0.0, 255.0)


def augment_noise(rng: jax.random.PRNGKey, imgs: jnp.ndarray, std: float = 0.01) -> jnp.ndarray:
    """Additive Gaussian noise (matching ProcGen-Benchmarks ContrastiveNoise)."""
    noise = jax.random.normal(rng, imgs.shape) * (std * 255.0)
    return jnp.clip(imgs + noise, 0.0, 255.0)
