import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple


class MARLEnvState(NamedTuple):
    agent_pos: jnp.ndarray      # (num_agents, 2)
    agent_vel: jnp.ndarray      # (num_agents, 2)
    landmark_pos: jnp.ndarray   # (num_landmarks, 2)
    step_count: int


class MultiAgentParticleEnv:
    """
    Pure JAX Multi-Agent Cooperative Navigation Environment (matching OpenAI MPE / JaxMARL).
    N agents must cooperate to cover L landmarks while avoiding collisions.
    """
    def __init__(self, num_agents: int = 3, num_landmarks: int = 3, max_steps: int = 50):
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.max_steps = max_steps
        self.num_actions = 5  # 0: None, 1: Left, 2: Right, 3: Down, 4: Up
        # Obs dim: self_vel (2) + landmark_rel (num_landmarks*2) + other_agents_rel ((num_agents-1)*2)
        self.obs_dim = 2 + (num_landmarks * 2) + ((num_agents - 1) * 2)
        self.global_state_dim = num_agents * self.obs_dim

    def reset(self, rng: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray, MARLEnvState]:
        rng, k1, k2 = jax.random.split(rng, 3)
        agent_pos = jax.random.uniform(k1, (self.num_agents, 2), minval=-0.8, maxval=0.8)
        agent_vel = jnp.zeros((self.num_agents, 2))
        landmark_pos = jax.random.uniform(k2, (self.num_landmarks, 2), minval=-0.8, maxval=0.8)
        
        state = MARLEnvState(
            agent_pos=agent_pos,
            agent_vel=agent_vel,
            landmark_pos=landmark_pos,
            step_count=0
        )
        obs, global_state = self._get_obs(state)
        return obs, global_state, state

    def step(self, rng: jax.Array, state: MARLEnvState, actions: jnp.ndarray):
        # actions: (num_agents,) in {0..4}
        action_vectors = jnp.array([
            [0.0, 0.0],   # Stay
            [-0.1, 0.0],  # Left
            [0.1, 0.0],   # Right
            [0.0, -0.1],  # Down
            [0.0, 0.1]    # Up
        ])
        moves = action_vectors[actions]
        new_vel = state.agent_vel * 0.5 + moves
        new_pos = jnp.clip(state.agent_pos + new_vel, -1.0, 1.0)

        # Cooperative reward: distance of each landmark to the closest agent
        # pairwise_dist: (num_landmarks, num_agents)
        diff = state.landmark_pos[:, None, :] - new_pos[None, :, :]
        dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-6)
        min_dists = jnp.min(dists, axis=-1)  # (num_landmarks,)
        reward = -jnp.sum(min_dists)

        # Collision penalty between agents
        agent_diff = new_pos[:, None, :] - new_pos[None, :, :]
        agent_dists = jnp.sqrt(jnp.sum(agent_diff**2, axis=-1) + 1e-6)
        # Exclude diagonal
        collisions = jnp.sum((agent_dists < 0.1) & ~jnp.eye(self.num_agents, dtype=bool)) / 2.0
        reward = reward - collisions * 0.5

        new_step = state.step_count + 1
        done = new_step >= self.max_steps

        new_state = MARLEnvState(
            agent_pos=new_pos,
            agent_vel=new_vel,
            landmark_pos=state.landmark_pos,
            step_count=new_step
        )
        obs, global_state = self._get_obs(new_state)
        return obs, global_state, new_state, reward, done

    def _get_obs(self, state: MARLEnvState) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Returns local obs for each agent: (num_agents, obs_dim)
        obs_list = []
        for i in range(self.num_agents):
            self_vel = state.agent_vel[i]
            # Landmarks relative to agent i
            landmark_rel = (state.landmark_pos - state.agent_pos[i]).flatten()
            # Other agents relative to agent i
            other_indices = jnp.array([j for j in range(self.num_agents) if j != i])
            other_rel = (state.agent_pos[other_indices] - state.agent_pos[i]).flatten()
            agent_obs = jnp.concatenate([self_vel, landmark_rel, other_rel])
            obs_list.append(agent_obs)

        obs = jnp.stack(obs_list, axis=0)  # (num_agents, obs_dim)
        global_state = obs.flatten()       # (num_agents * obs_dim,)
        return obs, global_state
