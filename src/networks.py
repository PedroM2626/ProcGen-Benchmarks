import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Tuple


class NatureCNN(nn.Module):
    """
    Standard NatureCNN architecture (Mnih et al. 2015):
    3 Conv2D layers + Dense 512, with Actor and Critic heads.
    """
    action_dim: int = 17

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

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)

        actor_logits = nn.Dense(features=self.action_dim)(x)
        critic_value = nn.Dense(features=1)(x)
        return actor_logits, jnp.squeeze(critic_value, axis=-1)


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        res = x
        x = nn.relu(x)
        x = nn.Conv(features=self.channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.channels, kernel_size=(3, 3), padding="SAME")(x)
        return res + x


class ImpalaBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Conv(features=self.channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = ResidualBlock(channels=self.channels)(x)
        x = ResidualBlock(channels=self.channels)(x)
        return x


class ImpalaCNN(nn.Module):
    """
    IMPALA CNN architecture (Espeholt et al. 2018):
    Deeper residual convolutional architecture used in Procgen benchmarks.
    """
    action_dim: int = 17
    channel_sequence: Sequence[int] = (16, 32, 32)
    dense_dim: int = 512

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = x.astype(jnp.float32)
        x = jnp.where(jnp.max(x) > 1.0, x / 255.0, x)

        for ch in self.channel_sequence:
            x = ImpalaBlock(channels=ch)(x)

        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.dense_dim)(x)
        x = nn.relu(x)

        actor_logits = nn.Dense(features=self.action_dim)(x)
        critic_value = nn.Dense(features=1)(x)
        return actor_logits, jnp.squeeze(critic_value, axis=-1)


class SymbolicActorCritic(nn.Module):
    """
    Fast MLP Actor-Critic for Craftax Symbolic representation (1345 features).
    Enables ultra-fast benchmarking (>200.000 FPS).
    """
    action_dim: int = 17
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = x.astype(jnp.float32)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)

        actor_logits = nn.Dense(features=self.action_dim)(x)
        critic_value = nn.Dense(features=1)(x)
        return actor_logits, jnp.squeeze(critic_value, axis=-1)


class QNetwork(nn.Module):
    """
    Q-Network for DQN / QR-DQN.
    """
    action_dim: int = 17
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.float32)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        q_values = nn.Dense(features=self.action_dim)(x)
        return q_values


class HierarchicalLearnedActorCritic(nn.Module):
    """
    2-Level Hierarchical RL Architecture:
    1. Meta-Controller: Selects latent skill z in {0..num_skills-1}
    2. Low-Level Policy: Conditioned on obs + one-hot(z), outputs primitive action logits.
    """
    action_dim: int = 17
    num_skills: int = 6
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray, z: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x_flat = x.astype(jnp.float32).reshape((x.shape[0], -1))
        
        # Meta-trunk
        m = nn.Dense(features=self.hidden_dim)(x_flat)
        m = nn.relu(m)
        meta_logits = nn.Dense(features=self.num_skills)(m)
        meta_value = nn.Dense(features=1)(m)
        
        # If z is not provided, use argmax/sample of meta_logits
        if z is None:
            z = jnp.argmax(meta_logits, axis=-1)
            
        z_one_hot = jax.nn.one_hot(z, self.num_skills)
        low_input = jnp.concatenate([x_flat, z_one_hot], axis=-1)
        
        # Low-level trunk
        l = nn.Dense(features=self.hidden_dim)(low_input)
        l = nn.relu(l)
        low_logits = nn.Dense(features=self.action_dim)(l)
        
        return meta_logits, jnp.squeeze(meta_value, axis=-1), low_logits
