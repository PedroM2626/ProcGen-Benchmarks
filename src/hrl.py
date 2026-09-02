import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from typing import NamedTuple, Any

from src.env import CraftaxLevelManager
from src.networks import HierarchicalLearnedActorCritic, SymbolicActorCritic


# Fixed macro-actions mapping for Craftax (17 actions total):
# 0: NOOP, 1: LEFT, 2: RIGHT, 3: UP, 4: DOWN, 5: DO (mine/attack), 6: SLEEP, ...
FIXED_SKILL_TO_ACTION = jnp.array([
    0,  # Skill 0: Wait / NOOP
    1,  # Skill 1: Move Left
    2,  # Skill 2: Move Right
    3,  # Skill 3: Move Up
    4,  # Skill 4: Move Down
    5,  # Skill 5: Mine / Attack (DO action)
], dtype=jnp.int32)


class HRLTrainer:
    """
    Implements the 4 exact HRL & temporal abstraction paradigms from ProcGen-Benchmarks:
    1. 'flat': Standard PPO on 17 primitive actions.
    2. 'skip4': Action-repeat 4 (temporal abstraction without hierarchy).
    3. 'hrl': PPO meta-controller selecting 6 fixed macro-skills repeated for 4 frames.
    4. 'hrl_learned': Co-trained 2-level hierarchy: Meta selects latent skill z every 4 frames,
       low-level p(a|obs, z) executes primitive actions per frame.
    """
    def __init__(
        self,
        mode: str,  # 'flat', 'skip4', 'hrl', 'hrl_learned'
        env_manager: CraftaxLevelManager,
        num_envs: int = 64,
        num_steps: int = 64,
        skip_k: int = 4,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2
    ):
        assert mode in ['flat', 'skip4', 'hrl', 'hrl_learned'], f"Unknown mode: {mode}"
        self.mode = mode
        self.env_manager = env_manager
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.skip_k = skip_k
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        
        if self.mode == 'hrl_learned':
            self.model = HierarchicalLearnedActorCritic(action_dim=self.env_manager.num_actions, num_skills=6)
        elif self.mode == 'hrl':
            self.model = SymbolicActorCritic(action_dim=6)
        else:
            self.model = SymbolicActorCritic(action_dim=self.env_manager.num_actions)

    def create_state(self, rng, input_shape):
        if self.mode == 'hrl_learned':
            init_x = jnp.zeros((1, *input_shape))
            variables = self.model.init(rng, init_x, jnp.zeros((1,), dtype=jnp.int32))
        else:
            init_x = jnp.zeros((1, *input_shape))
            variables = self.model.init(rng, init_x)

        tx = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(self.learning_rate, eps=1e-5)
        )
        train_state = TrainState.create(apply_fn=self.model.apply, params=variables['params'], tx=tx)
        return train_state

    def step_temporal_abstraction(self, train_state, env_state, obs, rng):
        """
        Executes a decision step with temporal abstraction (skip4 / hrl / hrl_learned).
        """
        rng, subkey = jax.random.split(rng)
        
        batch_size = obs.shape[0]
        if self.mode == 'flat':
            logits, value = self.model.apply({'params': train_state.params}, obs)
            action = jax.random.categorical(subkey, logits)
            log_prob = jax.nn.log_softmax(logits)[jnp.arange(batch_size), action]
            next_obs, next_env_state, reward, done, info, rng = self.env_manager.step(rng, env_state, action)
            return next_obs, next_env_state, reward, done, action, value, log_prob, rng

        elif self.mode == 'skip4':
            logits, value = self.model.apply({'params': train_state.params}, obs)
            action = jax.random.categorical(subkey, logits)
            log_prob = jax.nn.log_softmax(logits)[jnp.arange(batch_size), action]

            # Repeat action 4 times
            def _substep(s_state, _):
                curr_obs, curr_env_state, accum_rew, done_any, r = s_state
                n_obs, n_env, r_rew, d_done, _, r = self.env_manager.step(r, curr_env_state, action)
                return (n_obs, n_env, accum_rew + r_rew, jnp.logical_or(done_any, d_done), r), None

            init_sub = (obs, env_state, jnp.zeros(batch_size), jnp.zeros(batch_size, dtype=bool), rng)
            (next_obs, next_env_state, total_reward, total_done, rng), _ = jax.lax.scan(_substep, init_sub, None, length=self.skip_k)
            return next_obs, next_env_state, total_reward, total_done, action, value, log_prob, rng

        elif self.mode == 'hrl':
            # Meta chooses skill in {0..5}
            meta_logits, value = self.model.apply({'params': train_state.params}, obs)
            skill = jax.random.categorical(subkey, meta_logits)
            log_prob = jax.nn.log_softmax(meta_logits)[jnp.arange(batch_size), skill]
            primitive_action = FIXED_SKILL_TO_ACTION[skill]

            # Repeat skill 4 times
            def _substep(s_state, _):
                curr_obs, curr_env_state, accum_rew, done_any, r = s_state
                n_obs, n_env, r_rew, d_done, _, r = self.env_manager.step(r, curr_env_state, primitive_action)
                return (n_obs, n_env, accum_rew + r_rew, jnp.logical_or(done_any, d_done), r), None

            init_sub = (obs, env_state, jnp.zeros(batch_size), jnp.zeros(batch_size, dtype=bool), rng)
            (next_obs, next_env_state, total_reward, total_done, rng), _ = jax.lax.scan(_substep, init_sub, None, length=self.skip_k)
            return next_obs, next_env_state, total_reward, total_done, skill, value, log_prob, rng

        elif self.mode == 'hrl_learned':
            # Meta chooses latent skill z in {0..5}
            meta_logits, value, _ = self.model.apply({'params': train_state.params}, obs)
            z = jax.random.categorical(subkey, meta_logits)
            log_prob = jax.nn.log_softmax(meta_logits)[jnp.arange(batch_size), z]

            # Low-level executes 4 steps conditioned on z
            def _substep(s_state, _):
                curr_obs, curr_env_state, accum_rew, done_any, r = s_state
                r, sub_r = jax.random.split(r)
                _, _, low_logits = self.model.apply({'params': train_state.params}, curr_obs, z)
                act = jax.random.categorical(sub_r, low_logits)
                n_obs, n_env, r_rew, d_done, _, r = self.env_manager.step(r, curr_env_state, act)
                return (n_obs, n_env, accum_rew + r_rew, jnp.logical_or(done_any, d_done), r), None

            init_sub = (obs, env_state, jnp.zeros(batch_size), jnp.zeros(batch_size, dtype=bool), rng)
            (next_obs, next_env_state, total_reward, total_done, rng), _ = jax.lax.scan(_substep, init_sub, None, length=self.skip_k)
            return next_obs, next_env_state, total_reward, total_done, z, value, log_prob, rng
