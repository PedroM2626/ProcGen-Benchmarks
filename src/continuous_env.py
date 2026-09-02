import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple


class ContinuousEnvState(NamedTuple):
    pos: jnp.ndarray       # (batch, 2)
    vel: jnp.ndarray       # (batch, 2)
    target_pos: jnp.ndarray  # (batch, 2)
    step_count: int


class ContinuousSingleAgentNavigationEnv:
    """
    Pure JAX Continuous Control Environment.
    Agent applies continuous force vector a in [-1, 1]^2 to navigate to procedural targets with inertia.
    """
    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps
        self.action_dim = 2
        self.obs_dim = 6  # pos (2) + vel (2) + target_rel (2)

    def reset(self, rng: jax.Array) -> Tuple[jnp.ndarray, ContinuousEnvState]:
        rng, k1, k2 = jax.random.split(rng, 3)
        pos = jax.random.uniform(k1, (2,), minval=-0.8, maxval=0.8)
        vel = jnp.zeros((2,))
        target_pos = jax.random.uniform(k2, (2,), minval=-0.8, maxval=0.8)
        
        state = ContinuousEnvState(pos=pos, vel=vel, target_pos=target_pos, step_count=0)
        obs = self._get_obs(state)
        return obs, state

    def step(self, rng: jax.Array, state: ContinuousEnvState, action: jnp.ndarray):
        # action: (2,) continuous force in [-1, 1]
        action = jnp.clip(action, -1.0, 1.0)
        # Physics update with mass and friction
        new_vel = state.vel * 0.85 + action * 0.05
        new_pos = jnp.clip(state.pos + new_vel, -1.0, 1.0)
        
        # Reward: negative Euclidean distance to target + control penalty
        dist = jnp.sqrt(jnp.sum((target_rel := state.target_pos - new_pos)**2) + 1e-6)
        reward = -dist - 0.01 * jnp.sum(action**2)
        
        # Target reached bonus
        reached = dist < 0.1
        reward = reward + jnp.where(reached, 5.0, 0.0)

        new_step = state.step_count + 1
        done = (new_step >= self.max_steps) | reached

        new_state = ContinuousEnvState(pos=new_pos, vel=new_vel, target_pos=state.target_pos, step_count=new_step)
        obs = self._get_obs(new_state)
        return obs, new_state, reward, done

    def _get_obs(self, state: ContinuousEnvState) -> jnp.ndarray:
        target_rel = state.target_pos - state.pos
        return jnp.concatenate([state.pos, state.vel, target_rel])
