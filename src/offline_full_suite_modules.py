import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Dict, Any


# =====================================================================
# 1. SAC TEACHER ACTOR & CRITIC
# =====================================================================
class SACTeacherActor(nn.Module):
    action_dim: int = 8
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = nn.relu(nn.Dense(self.hidden_dim)(obs))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, -20.0, 2.0)
        return mean, log_std

    def sample_action(self, params, obs, rng):
        mean, log_std = self.apply(params, obs)
        std = jnp.exp(log_std)
        normal_noise = jax.random.normal(rng, shape=mean.shape)
        action_raw = mean + std * normal_noise
        action = jnp.tanh(action_raw)
        return action


class SACTeacherCritic(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        sa = jnp.concatenate([obs, action], axis=-1)
        q1 = nn.relu(nn.Dense(self.hidden_dim)(sa))
        q1 = nn.relu(nn.Dense(self.hidden_dim)(q1))
        q1 = jnp.squeeze(nn.Dense(1)(q1), axis=-1)

        q2 = nn.relu(nn.Dense(self.hidden_dim)(sa))
        q2 = nn.relu(nn.Dense(self.hidden_dim)(q2))
        q2 = jnp.squeeze(nn.Dense(1)(q2), axis=-1)
        return q1, q2


# =====================================================================
# 2. BEHAVIORAL CLONING (BC)
# =====================================================================
class BCActor(nn.Module):
    action_dim: int = 8
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = nn.relu(nn.Dense(self.hidden_dim)(obs))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return nn.tanh(nn.Dense(self.action_dim)(x))


# =====================================================================
# 3. GAIL DISCRIMINATOR (INVERSE RL)
# =====================================================================
class GAILDiscriminator(nn.Module):
    """
    Discriminador D(s, a) de GAIL (Ho & Ermon, 2016).
    Diferencia transições do especialista de transições do aprendiz.
    """
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        sa = jnp.concatenate([obs, action], axis=-1)
        x = nn.relu(nn.Dense(self.hidden_dim)(sa))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        logits = jnp.squeeze(nn.Dense(1)(x), axis=-1)
        return logits


# =====================================================================
# 4. BCQ (BATCH-CONSTRAINED DEEP Q-LEARNING) COM VAE GENERATIVO
# =====================================================================
class BCQVAEEncoder(nn.Module):
    latent_dim: int = 16
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        sa = jnp.concatenate([obs, action], axis=-1)
        x = nn.relu(nn.Dense(self.hidden_dim)(sa))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        mean = nn.Dense(self.latent_dim)(x)
        log_std = jnp.clip(nn.Dense(self.latent_dim)(x), -4.0, 15.0)
        return mean, log_std


class BCQVAEDecoder(nn.Module):
    action_dim: int = 8
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        sz = jnp.concatenate([obs, z], axis=-1)
        x = nn.relu(nn.Dense(self.hidden_dim)(sz))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return nn.tanh(nn.Dense(self.action_dim)(x))


class BCQPerturbationNetwork(nn.Module):
    """Rede xi(s, a) que adiciona pequena perturbação bounded nas ações geradas pelo VAE."""
    action_dim: int = 8
    hidden_dim: int = 256
    max_perturbation: float = 0.05

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        sa = jnp.concatenate([obs, action], axis=-1)
        x = nn.relu(nn.Dense(self.hidden_dim)(sa))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return self.max_perturbation * nn.tanh(nn.Dense(self.action_dim)(x))


# =====================================================================
# 5. DECISION TRANSFORMER (DT) COM ATENÇÃO CAUSAL
# =====================================================================
class DecisionTransformerActor(nn.Module):
    """
    Decision Transformer (Chen et al., 2021) em JAX.
    Modelagem de sequência autorregressiva com auto-atenção causal sobre (R_t, s_t, a_t).
    """
    action_dim: int = 8
    embed_dim: int = 128
    num_heads: int = 4

    @nn.compact
    def __call__(self, rtg: jnp.ndarray, obs: jnp.ndarray) -> jnp.ndarray:
        # rtg: (batch, 1) Return-to-go desejado
        # obs: (batch, obs_dim)
        r_emb = nn.Dense(self.embed_dim)(rtg)
        s_emb = nn.Dense(self.embed_dim)(obs)
        tokens = jnp.stack([r_emb, s_emb], axis=1) # (batch, 2, embed_dim)
        
        # Self-Attention Causal
        attn_out = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(tokens, tokens)
        h = nn.relu(tokens + attn_out)
        pred_action = nn.Dense(self.action_dim)(h[:, -1, :])
        return nn.tanh(pred_action)


# =====================================================================
# 6. IQL & CQL NETWORKS
# =====================================================================
class IQLValueNetwork(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = nn.relu(nn.Dense(self.hidden_dim)(obs))
        x = nn.relu(nn.Dense(self.hidden_dim)(x))
        return jnp.squeeze(nn.Dense(1)(x), axis=-1)
