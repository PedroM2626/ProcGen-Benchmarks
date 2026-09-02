import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple


# =====================================================================
# ACTOR AND CRITIC NETWORKS FOR MARL (IPPO vs MAPPO)
# =====================================================================
class MARLActor(nn.Module):
    action_dim: int = 5

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        # obs: (..., obs_dim)
        x = nn.Dense(128)(obs)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)


class DecentralizedCritic(nn.Module):
    """Local Critic used in Independent PPO (IPPO): evaluates only local observation o_i."""
    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(128)(obs)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        return jnp.squeeze(nn.Dense(1)(x), axis=-1)


class CentralizedCritic(nn.Module):
    """
    Centralized Critic used in MAPPO (CTDE Paradigm):
    Takes the joint global state s = (o_1, o_2, ..., o_N) to evaluate joint value.
    Eliminates environment non-stationarity during training.
    """
    @nn.compact
    def __call__(self, global_state: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(256)(global_state)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        return jnp.squeeze(nn.Dense(1)(x), axis=-1)


# =====================================================================
# VALUE FACTORIZATION (VDN & QMIX)
# =====================================================================
class MARLQNetwork(nn.Module):
    action_dim: int = 5

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(128)(obs)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)


class QMIXMixingNetwork(nn.Module):
    """
    QMIX Monotonic Mixing Network (Rashid et al. 2018).
    Hypernetworks generate positive weights |W_1|, |W_2| conditioning on global state s.
    Ensures d(Q_tot) / d(Q_i) >= 0.
    """
    num_agents: int = 3
    embed_dim: int = 64

    @nn.compact
    def __call__(self, q_values: jnp.ndarray, global_state: jnp.ndarray) -> jnp.ndarray:
        # q_values: (batch, num_agents)
        # global_state: (batch, state_dim)
        batch = q_values.shape[0]

        # Hypernetwork 1: generates W1 (num_agents, embed_dim)
        w1 = nn.Dense(self.num_agents * self.embed_dim)(global_state)
        w1 = jnp.abs(w1.reshape((batch, self.num_agents, self.embed_dim)))
        b1 = nn.Dense(self.embed_dim)(global_state)[:, None, :]

        # First hidden layer
        h = nn.elu(jnp.matmul(q_values[:, None, :], w1) + b1)  # (batch, 1, embed_dim)

        # Hypernetwork 2: generates W2 (embed_dim, 1)
        w2 = nn.Dense(self.embed_dim)(global_state)
        w2 = jnp.abs(w2.reshape((batch, self.embed_dim, 1)))

        # State bias
        b2 = nn.Sequential([nn.Dense(self.embed_dim), nn.relu, nn.Dense(1)])(global_state)[:, None, :]

        q_tot = jnp.matmul(h, w2) + b2  # (batch, 1, 1)
        return jnp.squeeze(q_tot, axis=(1, 2))


def vdn_mix(q_values: jnp.ndarray) -> jnp.ndarray:
    """Additive value factorization (Sunehag et al. 2017)."""
    return jnp.sum(q_values, axis=-1)


# =====================================================================
# MA-POCA: MULTI-AGENT POSTHUMOUS CREDIT ASSIGNMENT (Cohen et al. 2021)
# =====================================================================
class MAPOCACritic(nn.Module):
    """
    MA-POCA Centralized Critic with Multi-Head Self-Attention and Counterfactual Credit Assignment.
    Solves the multi-agent credit assignment problem by comparing team value with and without agent i.
    """
    embed_dim: int = 128
    num_heads: int = 4

    @nn.compact
    def __call__(self, obs_all: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # obs_all: (batch, num_agents, obs_dim)
        batch, num_agents, _ = obs_all.shape

        # 1. Project each agent's observation into embedding space
        tokens = nn.Dense(self.embed_dim)(obs_all)
        tokens = nn.relu(tokens)

        # 2. Multi-Head Self-Attention over all agents in the team
        norm_tokens = nn.LayerNorm()(tokens)
        attn_out = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(norm_tokens, norm_tokens)
        tokens = tokens + attn_out  # (batch, num_agents, embed_dim)

        # 3. Global team context
        team_context = jnp.mean(tokens, axis=1, keepdims=True)  # (batch, 1, embed_dim)
        team_context_expanded = jnp.repeat(team_context, num_agents, axis=1)

        # Joint features for each agent: [agent_features, team_features]
        joint_features = jnp.concatenate([tokens, team_context_expanded], axis=-1)
        h = nn.Dense(128)(joint_features)
        h = nn.relu(h)

        # V_i(s): Individualized team value for agent i
        v_team = jnp.squeeze(nn.Dense(1)(h), axis=-1)  # (batch, num_agents)

        # Counterfactual baseline V_{-i}(s): Team value excluding agent i's contribution
        # In MA-POCA, calculated by evaluating team value without agent i's token
        v_counterfactual = jnp.squeeze(nn.Dense(1)(nn.relu(nn.Dense(128)(team_context_expanded))), axis=-1)

        return v_team, v_counterfactual
