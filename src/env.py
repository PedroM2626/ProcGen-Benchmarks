import jax
import jax.numpy as jnp
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv


class CraftaxLevelManager:
    """
    Manages procedural levels (train vs unseen), replicating ProcGen's protocol.
    """
    def __init__(self, use_pixels: bool = False, num_train_levels: int = 200, eval_seed_offset: int = 1000):
        self.use_pixels = use_pixels
        self.env = CraftaxClassicPixelsEnv() if use_pixels else CraftaxClassicSymbolicEnv()
        self.params = self.env.default_params
        self.num_actions = self.env.num_actions
        self.num_train_levels = num_train_levels
        self.eval_seed_offset = eval_seed_offset

    def reset_train(self, rng: jax.Array, num_envs: int):
        """
        Resets environments by randomly sampling from the fixed train level pool (0..num_train_levels-1).
        """
        rng, subkey = jax.random.split(rng)
        level_ids = jax.random.randint(subkey, shape=(num_envs,), minval=0, maxval=self.num_train_levels)
        keys = jax.vmap(jax.random.PRNGKey)(level_ids)
        obs, states = jax.vmap(self.env.reset, in_axes=(0, None))(keys, self.params)
        return obs, states, rng

    def reset_unseen(self, rng: jax.Array, num_envs: int):
        """
        Resets environments for evaluation on unseen procedural seeds (eval_seed_offset..).
        """
        rng, subkey = jax.random.split(rng)
        level_ids = jax.random.randint(
            subkey, 
            shape=(num_envs,), 
            minval=self.eval_seed_offset, 
            maxval=self.eval_seed_offset + 100
        )
        keys = jax.vmap(jax.random.PRNGKey)(level_ids)
        obs, states = jax.vmap(self.env.reset, in_axes=(0, None))(keys, self.params)
        return obs, states, rng

    def step(self, rng: jax.Array, states, actions):
        """
        Vectorized step. Auto-resets done environments with a newly sampled train level seed.
        """
        rng, step_key, reset_key = jax.random.split(rng, 3)
        num_envs = actions.shape[0]
        step_keys = jax.random.split(step_key, num_envs)
        
        # Step environment
        next_obs, next_states, rewards, dones, info = jax.vmap(
            self.env.step, in_axes=(0, 0, 0, None)
        )(step_keys, states, actions, self.params)
        
        # For done envs, auto-reset to a new level
        reset_level_ids = jax.random.randint(reset_key, shape=(num_envs,), minval=0, maxval=self.num_train_levels)
        reset_keys = jax.vmap(jax.random.PRNGKey)(reset_level_ids)
        reset_obs, reset_states = jax.vmap(self.env.reset, in_axes=(0, None))(reset_keys, self.params)
        
        # Where done, replace obs and state
        obs = jax.tree_util.tree_map(
            lambda r, o: jnp.where(jnp.expand_dims(dones, tuple(range(1, r.ndim))), r, o),
            reset_obs, next_obs
        )
        states = jax.tree_util.tree_map(
            lambda r, s: jnp.where(jnp.expand_dims(dones, tuple(range(1, r.ndim))), r, s),
            reset_states, next_states
        )
        
        return obs, states, rewards, dones, info, rng
