import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Any, Callable


# =====================================================================
# UNIVERSAL FEATURE EXTRACTOR FACTORY
# =====================================================================
class FeatureExtractorNatureCNN(nn.Module):
    features_dim: int = 512

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
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
        return nn.Dense(self.features_dim)(nn.relu(x))


class FeatureExtractorSpatialAttention(nn.Module):
    features_dim: int = 512

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim == 4:
            x = x.astype(jnp.float32)
            x = jnp.where(jnp.max(x) > 1.0, x / 255.0, x)
            x = nn.Conv(32, (8, 8), (4, 4))(x)
            x = nn.relu(x)
            x = nn.Conv(64, (4, 4), (2, 2))(x)
            x = nn.relu(x)
            x = nn.Conv(64, (3, 3), (1, 1))(x)
            x = nn.relu(x)
            # Spatial Attention map
            avg_p = jnp.mean(x, axis=-1, keepdims=True)
            max_p = jnp.max(x, axis=-1, keepdims=True)
            feat = jnp.concatenate([avg_p, max_p], axis=-1)
            attn = jax.nn.sigmoid(nn.Conv(1, (7, 7), padding="SAME")(feat))
            x = x * attn + x  # Residual connection
            x = x.reshape((x.shape[0], -1))
        else:
            x = x.astype(jnp.float32)
            x = nn.Dense(256)(x)
            x = nn.relu(x)
            attn = jax.nn.sigmoid(nn.Dense(256)(x))
            x = x * attn + x
        return nn.Dense(self.features_dim)(nn.relu(x))


class FeatureExtractorCBAM(nn.Module):
    features_dim: int = 512

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim == 4:
            x = x.astype(jnp.float32)
            x = jnp.where(jnp.max(x) > 1.0, x / 255.0, x)
            x = nn.Conv(32, (8, 8), (4, 4))(x)
            x = nn.relu(x)
            x = nn.Conv(64, (4, 4), (2, 2))(x)
            x = nn.relu(x)
            x = nn.Conv(64, (3, 3), (1, 1))(x)
            x = nn.relu(x)
            # Channel Attention
            avg_c = jnp.mean(x, axis=(1, 2))
            max_c = jnp.max(x, axis=(1, 2))
            mlp = nn.Sequential([nn.Dense(16), nn.relu, nn.Dense(64)])
            scale_c = jax.nn.sigmoid(mlp(avg_c) + mlp(max_c))[:, None, None, :]
            x = x * scale_c
            # Spatial Attention
            avg_p = jnp.mean(x, axis=-1, keepdims=True)
            max_p = jnp.max(x, axis=-1, keepdims=True)
            attn_s = jax.nn.sigmoid(nn.Conv(1, (7, 7), padding="SAME")(jnp.concatenate([avg_p, max_p], axis=-1)))
            x = x * attn_s + x
            x = x.reshape((x.shape[0], -1))
        else:
            x = x.astype(jnp.float32)
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        return nn.Dense(self.features_dim)(nn.relu(x))


class FeatureExtractorImpalaResNet(nn.Module):
    features_dim: int = 512

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim == 4:
            x = x.astype(jnp.float32)
            x = jnp.where(jnp.max(x) > 1.0, x / 255.0, x)
            for ch in [32, 64]:
                x = nn.Conv(ch, (3, 3), padding="SAME")(x)
                x = nn.max_pool(x, (3, 3), (2, 2), padding="SAME")
                # Residual 1
                res = x
                x = nn.Conv(ch, (3, 3), padding="SAME")(nn.relu(x))
                x = res + nn.Conv(ch, (3, 3), padding="SAME")(nn.relu(x))
            x = x.reshape((x.shape[0], -1))
        else:
            x = x.astype(jnp.float32)
            res = nn.Dense(256)(x)
            x = res + nn.Dense(256)(nn.relu(res))
        return nn.Dense(self.features_dim)(nn.relu(x))


class FeatureExtractorViT(nn.Module):
    features_dim: int = 512
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim == 4:
            b, h, w, c = x.shape
            x = x.astype(jnp.float32)
            x = jnp.where(jnp.max(x) > 1.0, x / 255.0, x)
            p_h, p_w = h // 7, w // 7
            patches = x.reshape((b, p_h, 7, p_w, 7, c)).transpose((0, 1, 3, 2, 4, 5)).reshape((b, p_h * p_w, -1))
            tokens = nn.Dense(self.embed_dim)(patches)
        else:
            b = x.shape[0]
            x = x.astype(jnp.float32)
            tokens = nn.Dense(self.embed_dim * 4)(x).reshape((b, 4, self.embed_dim))

        pos = self.param("pos", nn.initializers.normal(0.02), (1, tokens.shape[1], self.embed_dim))
        tokens = tokens + pos
        for _ in range(self.num_layers):
            norm = nn.LayerNorm()(tokens)
            tokens = tokens + nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(norm, norm)
            tokens = tokens + nn.Dense(self.embed_dim)(nn.relu(nn.Dense(self.embed_dim * 2)(nn.LayerNorm()(tokens))))
        cls = tokens.mean(axis=1)
        return nn.Dense(self.features_dim)(nn.relu(cls))


# =====================================================================
# UNIVERSAL ACTOR-CRITIC (PPO / A2C) WITH PLUGGABLE BACKBONE
# =====================================================================
class UniversalActorCritic(nn.Module):
    extractor_cls: Any
    action_dim: int = 17

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        features = self.extractor_cls()(x)
        logits = nn.Dense(self.action_dim)(features)
        value = nn.Dense(1)(features)
        return logits, jnp.squeeze(value, axis=-1)


# =====================================================================
# UNIVERSAL Q-NETWORK (DQN / QR-DQN) WITH PLUGGABLE BACKBONE
# =====================================================================
class UniversalQNetwork(nn.Module):
    extractor_cls: Any
    action_dim: int = 17
    is_quantile: bool = False
    num_quantiles: int = 50

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        features = self.extractor_cls()(x)
        if self.is_quantile:
            out = nn.Dense(self.action_dim * self.num_quantiles)(features)
            return out.reshape((features.shape[0], self.action_dim, self.num_quantiles))
        else:
            return nn.Dense(self.action_dim)(features)
