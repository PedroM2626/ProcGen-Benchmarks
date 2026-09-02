import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from typing import NamedTuple

from src.env import CraftaxLevelManager


class ReplayBuffer(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_obs: jnp.ndarray
    dones: jnp.ndarray
    idx: int
    size: int
    capacity: int


def init_replay_buffer(capacity: int, obs_shape: tuple) -> ReplayBuffer:
    return ReplayBuffer(
        obs=jnp.zeros((capacity, *obs_shape), dtype=jnp.float32),
        actions=jnp.zeros((capacity,), dtype=jnp.int32),
        rewards=jnp.zeros((capacity,), dtype=jnp.float32),
        next_obs=jnp.zeros((capacity, *obs_shape), dtype=jnp.float32),
        dones=jnp.zeros((capacity,), dtype=jnp.float32),
        idx=0,
        size=0,
        capacity=capacity
    )


def push_buffer(buffer: ReplayBuffer, obs, actions, rewards, next_obs, dones) -> ReplayBuffer:
    batch_size = actions.shape[0]
    indices = (buffer.idx + jnp.arange(batch_size)) % buffer.capacity
    
    new_obs = buffer.obs.at[indices].set(obs)
    new_actions = buffer.actions.at[indices].set(actions)
    new_rewards = buffer.rewards.at[indices].set(rewards)
    new_next_obs = buffer.next_obs.at[indices].set(next_obs)
    new_dones = buffer.dones.at[indices].set(dones)
    
    new_idx = (buffer.idx + batch_size) % buffer.capacity
    new_size = jnp.minimum(buffer.size + batch_size, buffer.capacity)
    
    return buffer._replace(
        obs=new_obs,
        actions=new_actions,
        rewards=new_rewards,
        next_obs=new_next_obs,
        dones=new_dones,
        idx=new_idx,
        size=new_size
    )


class DQNTrainer:
    """
    Vectorized Deep Q-Network in JAX with circular replay buffer and target network.
    """
    def __init__(
        self,
        model,
        env_manager: CraftaxLevelManager,
        num_envs: int = 64,
        buffer_size: int = 50000,
        batch_size: int = 64,
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        target_update_interval: int = 500,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 20000
    ):
        self.model = model
        self.env_manager = env_manager
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.target_update_interval = target_update_interval
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

    def create_state(self, rng, input_shape):
        init_x = jnp.zeros((1, *input_shape))
        params = self.model.init(rng, init_x)['params']
        tx = optax.adam(self.learning_rate)
        train_state = TrainState.create(apply_fn=self.model.apply, params=params, tx=tx)
        target_params = params
        buffer = init_replay_buffer(self.buffer_size, input_shape)
        return train_state, target_params, buffer

    def train_step(self, step_idx, runner_state):
        train_state, target_params, buffer, env_state, last_obs, rng = runner_state
        rng, subkey1, subkey2, sample_key = jax.random.split(rng, 4)

        # 1. EPSILON GREEDY ACTION
        eps = jnp.maximum(
            self.eps_end,
            self.eps_start - (self.eps_start - self.eps_end) * (step_idx * self.num_envs / self.eps_decay_steps)
        )
        q_vals = self.model.apply({'params': train_state.params}, last_obs)
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        random_actions = jax.random.randint(subkey1, shape=(self.num_envs,), minval=0, maxval=self.env_manager.num_actions)
        
        do_random = jax.random.uniform(subkey2, shape=(self.num_envs,)) < eps
        actions = jnp.where(do_random, random_actions, greedy_actions)

        # 2. STEP ENVIRONMENT
        next_obs, next_env_state, rewards, dones, info, rng = self.env_manager.step(
            rng, env_state, actions
        )

        # 3. STORE IN REPLAY BUFFER
        buffer = push_buffer(buffer, last_obs, actions, rewards, next_obs, dones)

        # 4. SAMPLE BATCH AND UPDATE Q-NETWORK
        def _update(t_state, key):
            sample_indices = jax.random.randint(key, shape=(self.batch_size,), minval=0, maxval=buffer.size)
            b_obs = buffer.obs[sample_indices]
            b_act = buffer.actions[sample_indices]
            b_rew = buffer.rewards[sample_indices]
            b_next_obs = buffer.next_obs[sample_indices]
            b_done = buffer.dones[sample_indices]

            # Bellman Target (Double DQN style)
            next_q = self.model.apply({'params': t_state.params}, b_next_obs)
            next_actions = jnp.argmax(next_q, axis=-1)
            target_next_q = self.model.apply({'params': target_params}, b_next_obs)
            target_val = target_next_q[jnp.arange(self.batch_size), next_actions]
            y = b_rew + self.gamma * (1.0 - b_done) * target_val

            def _loss_fn(params):
                current_q = self.model.apply({'params': params}, b_obs)
                chosen_q = current_q[jnp.arange(self.batch_size), b_act]
                loss = optax.huber_loss(chosen_q - y).mean()
                return loss

            loss, grads = jax.value_and_grad(_loss_fn)(t_state.params)
            t_state = t_state.apply_gradients(grads=grads)
            return t_state, loss

        # Only update if buffer has enough samples
        train_state, loss = jax.lax.cond(
            buffer.size >= self.batch_size,
            lambda ts: _update(ts, sample_key),
            lambda ts: (ts, jnp.array(0.0, dtype=jnp.float32)),
            train_state
        )

        # 5. PERIODIC TARGET UPDATE
        target_params = jax.lax.cond(
            (step_idx % (self.target_update_interval // self.num_envs + 1)) == 0,
            lambda _: train_state.params,
            lambda _: target_params,
            operand=None
        )

        mean_reward = rewards.mean()
        metrics = {"loss": loss, "mean_reward": mean_reward, "eps": eps}
        return (train_state, target_params, buffer, next_env_state, next_obs, rng), metrics
