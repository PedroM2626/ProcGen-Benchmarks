import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple


class LearnedCommChannel(nn.Module):
    """
    Canal de Comunicação Explícita Neural (CommNet / TarMAC com Graph Attention).
    Cada agente i emite uma mensagem m_i in R^msg_dim.
    Uma camada de multi-head attention calcula o fluxo de mensagens m_{i->j}
    durante a própria inferência/execução em campo.
    """
    msg_dim: int = 16
    hidden_dim: int = 64
    num_heads: int = 2

    @nn.compact
    def __call__(self, obs_all_agents: jnp.ndarray, mask: jnp.ndarray = None) -> jnp.ndarray:
        # obs_all_agents: (batch, num_agents, obs_dim)
        batch, num_agents, _ = obs_all_agents.shape
        
        # 1. Emissor de Mensagem Local por Agente
        h_local = nn.Dense(self.hidden_dim)(obs_all_agents)
        h_local = nn.relu(h_local)
        msgs = nn.Dense(self.msg_dim)(h_local) # (batch, num_agents, msg_dim)

        # 2. Canal de Atenção Multi-Head (Quem ouve quem?)
        q = nn.Dense(self.msg_dim)(msgs)
        k = nn.Dense(self.msg_dim)(msgs)
        v = nn.Dense(self.msg_dim)(msgs)

        scale = jnp.sqrt(self.msg_dim)
        attn_logits = jnp.einsum('bmd,bnd->bmn', q, k) / scale # (batch, num_agents, num_agents)

        if mask is not None:
            # Máscara de conectividade de rádio / alcance
            attn_logits = jnp.where(mask, attn_logits, -1e9)

        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        # Mensagem agregada recebida por cada agente dos seus pares
        comm_context = jnp.einsum('bmn,bnd->bmd', attn_weights, v) # (batch, num_agents, msg_dim)

        # 3. Fusão: Observação Própria + Mensagens Recebidas
        combined = jnp.concatenate([h_local, comm_context], axis=-1)
        out = nn.Dense(self.hidden_dim)(combined)
        return nn.relu(out)


class CommActorCritic(nn.Module):
    """
    Ator-Crítico com Comunicação Explícita em Execução.
    """
    action_dim: int
    msg_dim: int = 16

    def setup(self):
        self.comm_channel = LearnedCommChannel(msg_dim=self.msg_dim)
        self.actor_head = nn.Dense(self.action_dim)
        self.critic_head = nn.Dense(1)

    def __call__(self, obs_all_agents: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # obs_all_agents: (batch, num_agents, obs_dim)
        features = self.comm_channel(obs_all_agents)
        logits = self.actor_head(features)
        values = self.critic_head(features).squeeze(-1)
        return logits, values
