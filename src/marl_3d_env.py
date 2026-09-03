import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple


class MultiAgent3DEnvState(NamedTuple):
    agent_pos: jnp.ndarray      # (N, 3) posições contínuas [x, y, z] em [-1, 1]^3
    agent_vel: jnp.ndarray      # (N, 3) velocidades contínuas [vx, vy, vz]
    landmark_pos: jnp.ndarray   # (L, 3) metas no espaço tridimensional
    step_count: int


class MultiAgent3DCooperativeEnv:
    """
    Ambiente Multi-Agente 3D em JAX Puro (3D Cooperative Navigation / Drones Flocking).
    N agentes e L alvos no espaço contínuo [-1, 1]^3.
    Ações contínuas a_i in [-1, 1]^3 (vetores de empuxo tridimensional).
    """
    def __init__(self, num_agents: int = 3, num_landmarks: int = 3, max_steps: int = 100):
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.max_steps = max_steps
        self.action_dim = 3  # Força contínua [Fx, Fy, Fz]
        # Obs local por agente: pos (3) + vel (3) + rel_landmarks (3 * L) + rel_other_agents (3 * (N-1))
        self.obs_dim = 3 + 3 + (3 * num_landmarks) + (3 * (num_agents - 1))
        self.state_dim = self.num_agents * self.obs_dim

    def reset(self, rng: jax.Array) -> Tuple[jnp.ndarray, MultiAgent3DEnvState]:
        rng, k1, k2 = jax.random.split(rng, 3)
        agent_pos = jax.random.uniform(k1, (self.num_agents, 3), minval=-0.8, maxval=0.8)
        agent_vel = jnp.zeros((self.num_agents, 3))
        landmark_pos = jax.random.uniform(k2, (self.num_landmarks, 3), minval=-0.8, maxval=0.8)
        
        state = MultiAgent3DEnvState(
            agent_pos=agent_pos,
            agent_vel=agent_vel,
            landmark_pos=landmark_pos,
            step_count=0
        )
        obs = self._get_obs(state)
        return obs, state

    def step(self, rng: jax.Array, state: MultiAgent3DEnvState, actions: jnp.ndarray):
        # actions: (N, 3) forças contínuas em [-1, 1]^3
        actions = jnp.clip(actions, -1.0, 1.0)
        
        # Física inercial 3D com atrito e gravidade/amortecimento
        new_vel = state.agent_vel * 0.82 + actions * 0.08
        new_pos = jnp.clip(state.agent_pos + new_vel, -1.0, 1.0)

        # 1. Recompensa de cobertura 3D: menor distância de cada alvo para o agente mais próximo
        diff = state.landmark_pos[:, None, :] - new_pos[None, :, :]
        dist_matrix = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-6)
        min_dist_per_landmark = jnp.min(dist_matrix, axis=1) # (L,)
        team_reward = -jnp.mean(min_dist_per_landmark)

        # 2. Penalidade por colisão esférica 3D entre agentes (raio = 0.15)
        agent_diff = new_pos[:, None, :] - new_pos[None, :, :]
        agent_dists = jnp.sqrt(jnp.sum(agent_diff**2, axis=-1) + 1e-6)
        mask = 1.0 - jnp.eye(self.num_agents)
        collision_matrix = (agent_dists < 0.15) * mask
        num_collisions = jnp.sum(collision_matrix) / 2.0
        team_reward = team_reward - 0.2 * num_collisions

        new_step = state.step_count + 1
        done = new_step >= self.max_steps

        new_state = MultiAgent3DEnvState(
            agent_pos=new_pos,
            agent_vel=new_vel,
            landmark_pos=state.landmark_pos,
            step_count=new_step
        )
        obs = self._get_obs(new_state)
        rewards = jnp.full((self.num_agents,), team_reward)
        return obs, new_state, rewards, done

    def _get_obs(self, state: MultiAgent3DEnvState) -> jnp.ndarray:
        obs_list = []
        for i in range(self.num_agents):
            p_i = state.agent_pos[i]
            v_i = state.agent_vel[i]
            rel_landmarks = (state.landmark_pos - p_i).flatten()
            
            other_indices = [j for j in range(self.num_agents) if j != i]
            rel_agents = (state.agent_pos[jnp.array(other_indices)] - p_i).flatten()
            
            obs_i = jnp.concatenate([p_i, v_i, rel_landmarks, rel_agents])
            obs_list.append(obs_i)
        return jnp.stack(obs_list)
