import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple


class BCActor(nn.Module):
    """
    Ator de Behavioral Cloning (Imitation Learning Supervisionado).
    Aprende diretamente a imitar as ações do especialista: L_BC = ||pi(s) - a_expert||^2
    """
    action_dim: int = 8
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        # Saída contínua limitada em [-1, 1] com Tanh
        return nn.tanh(nn.Dense(self.action_dim)(x))


class IQLValueNetwork(nn.Module):
    """
    Rede de Valor V(s) para Implicit Q-Learning (Kostrikov et al., 2021).
    Treinada via regressão expectil assimétrica: L_2^tau(Q(s, a) - V(s)).
    """
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(obs)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return jnp.squeeze(nn.Dense(1)(x), axis=-1)


class OfflineDoubleCritic(nn.Module):
    """
    Crítico Duplo Q(s, a) usado no IQL e no Conservative Q-Learning (CQL).
    """
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        sa = jnp.concatenate([obs, action], axis=-1)
        
        # Q1
        q1 = nn.Dense(self.hidden_dim)(sa)
        q1 = nn.relu(q1)
        q1 = nn.Dense(self.hidden_dim)(q1)
        q1 = nn.relu(q1)
        q1 = jnp.squeeze(nn.Dense(1)(q1), axis=-1)

        # Q2
        q2 = nn.Dense(self.hidden_dim)(sa)
        q2 = nn.relu(q2)
        q2 = nn.Dense(self.hidden_dim)(q2)
        q2 = nn.relu(q2)
        q2 = jnp.squeeze(nn.Dense(1)(q2), axis=-1)

        return q1, q2


def expectile_loss(diff: jnp.ndarray, expectile: float = 0.7) -> jnp.ndarray:
    """Perda assimétrica de expectil para IQL."""
    weight = jnp.where(diff > 0, expectile, 1.0 - expectile)
    return jnp.mean(weight * (diff ** 2))
