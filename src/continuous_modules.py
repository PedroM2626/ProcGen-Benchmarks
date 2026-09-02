import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple


# =====================================================================
# 1. CONTINUOUS GAUSSIAN ACTOR (PPO Continuous)
# Emits mean mu(s) and learned log_std for continuous action spaces
# =====================================================================
class ContinuousGaussianActor(nn.Module):
    action_dim: int = 2
    init_std: float = 0.5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # x: (batch, obs_dim)
        batch = x.shape[0]
        h = nn.Dense(128)(x)
        h = nn.relu(h)
        h = nn.Dense(128)(h)
        h = nn.relu(h)
        
        # Mean bounded in [-1, 1] via tanh
        mu = nn.tanh(nn.Dense(self.action_dim)(h))
        
        # Trainable log_std parameter
        log_std = self.param(
            "log_std", 
            lambda rng, shape: jnp.full(shape, jnp.log(self.init_std)), 
            (self.action_dim,)
        )
        log_std = jnp.broadcast_to(log_std, (batch, self.action_dim))
        return mu, log_std

    @staticmethod
    def sample_and_log_prob(rng: jax.Array, mu: jnp.ndarray, log_std: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        std = jnp.exp(log_std)
        noise = jax.random.normal(rng, shape=mu.shape)
        action = mu + std * noise
        # Gaussian log-likelihood: -0.5 * ( (a - mu)^2 / std^2 + 2*log_std + log(2*pi) )
        log_prob = -0.5 * jnp.sum(
            ((action - mu) / (std + 1e-8))**2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi), 
            axis=-1
        )
        return jnp.clip(action, -1.0, 1.0), log_prob


# =====================================================================
# 2. SOFT ACTOR-CRITIC (SAC) CONTINUOUS NETWORK
# Dual Q-networks + Squashed Tanh Gaussian Actor with Maximum Entropy RL
# =====================================================================
class SACCritic(nn.Module):
    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        sa = jnp.concatenate([state, action], axis=-1)
        
        # Q1
        q1 = nn.Dense(256)(sa)
        q1 = nn.relu(q1)
        q1 = nn.Dense(256)(q1)
        q1 = nn.relu(q1)
        q1_out = jnp.squeeze(nn.Dense(1)(q1), axis=-1)
        
        # Q2 (Clipped Double Q-Learning)
        q2 = nn.Dense(256)(sa)
        q2 = nn.relu(q2)
        q2 = nn.Dense(256)(q2)
        q2 = nn.relu(q2)
        q2_out = jnp.squeeze(nn.Dense(1)(q2), axis=-1)
        return q1_out, q2_out


class SACTanhGaussianActor(nn.Module):
    action_dim: int = 2
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        h = nn.Dense(256)(state)
        h = nn.relu(h)
        h = nn.Dense(256)(h)
        h = nn.relu(h)
        
        mu = nn.Dense(self.action_dim)(h)
        log_std = nn.Dense(self.action_dim)(h)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    @staticmethod
    def sample_squashed(rng: jax.Array, mu: jnp.ndarray, log_std: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        std = jnp.exp(log_std)
        epsilon = jax.random.normal(rng, shape=mu.shape)
        pre_tanh = mu + std * epsilon
        action = jnp.tanh(pre_tanh)
        
        # Enforcing Action Bounds / Change of Variables formula:
        # log pi(a|s) = log mu(u|s) - sum(log(1 - tanh(u)^2))
        log_prob_u = -0.5 * jnp.sum(((pre_tanh - mu) / (std + 1e-8))**2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi), axis=-1)
        log_prob = log_prob_u - jnp.sum(jnp.log(1.0 - action**2 + 1e-6), axis=-1)
        return action, log_prob
