import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from typing import NamedTuple, Any
import functools

from src.env import CraftaxLevelManager


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Any


def create_train_state(model, rng, input_shape, learning_rate=3e-4):
    init_x = jnp.zeros((1, *input_shape))
    variables = model.init(rng, init_x)
    tx = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adam(learning_rate, eps=1e-5)
    )
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx
    )


class PPOTrainer:
    """
    PureJaxRL style PPO & A2C implementation.
    Fully jitted and vectorized with zero host-accelerator memory copies.
    """
    def __init__(
        self,
        model,
        env_manager: CraftaxLevelManager,
        num_envs: int = 64,
        num_steps: int = 128,
        update_epochs: int = 4,
        num_minibatches: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        learning_rate: float = 3e-4,
        is_a2c: bool = False
    ):
        self.model = model
        self.env_manager = env_manager
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.update_epochs = 1 if is_a2c else update_epochs
        self.num_minibatches = 1 if is_a2c else num_minibatches
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        # For A2C, set huge clipping to make clipping inactive
        self.clip_eps = 1e9 if is_a2c else clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.learning_rate = learning_rate
        self.is_a2c = is_a2c

    def train_step(self, runner_state):
        train_state, env_state, last_obs, rng = runner_state

        # 1. COLLECT ROLLOUT (lax.scan over num_steps)
        def _env_step(step_state, _):
            t_state, e_state, obs, r = step_state
            r, subkey = jax.random.split(r)

            # Forward pass
            logits, value = self.model.apply({'params': t_state.params}, obs)
            action = jax.random.categorical(subkey, logits)
            log_prob = jax.nn.log_softmax(logits)[jnp.arange(self.num_envs), action]

            # Step environment
            next_obs, next_e_state, reward, done, info, r = self.env_manager.step(
                r, e_state, action
            )
            transition = Transition(done, action, value, reward, log_prob, obs, info)
            return (t_state, next_e_state, next_obs, r), transition

        (train_state, env_state, last_obs, rng), traj_batch = jax.lax.scan(
            _env_step, (train_state, env_state, last_obs, rng), None, length=self.num_steps
        )

        # 2. CALCULATE ADVANTAGES VIA GAE
        _, last_val = self.model.apply({'params': train_state.params}, last_obs)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = transition.done, transition.value, transition.reward
                delta = reward + self.gamma * next_value * (1.0 - done) - value
                gae = delta + self.gamma * self.gae_lambda * (1.0 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            returns = advantages + traj_batch.value
            return advantages, returns

        advantages, returns = _calculate_gae(traj_batch, last_val)

        # 3. UPDATE NETWORKS (EPOCHS & MINIBATCHES)
        def _update_epoch(update_state, _):
            def _update_minibatch(t_state, batch_info):
                traj, gae, ret = batch_info

                def _loss_fn(params):
                    logits, value = self.model.apply({'params': params}, traj.obs)
                    log_probs = jax.nn.log_softmax(logits)
                    log_prob = log_probs[jnp.arange(len(traj.action)), traj.action]

                    # Entropy
                    probs = jax.nn.softmax(logits)
                    entropy = -jnp.sum(probs * log_probs, axis=-1).mean()

                    # Policy Loss (PPO clipped surrogate or A2C)
                    ratio = jnp.exp(log_prob - traj.log_prob)
                    norm_gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    surr1 = ratio * norm_gae
                    surr2 = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * norm_gae
                    pi_loss = -jnp.minimum(surr1, surr2).mean()

                    # Value Loss
                    vf_loss = 0.5 * jnp.square(value - ret).mean()

                    total_loss = pi_loss + self.vf_coef * vf_loss - self.ent_coef * entropy
                    return total_loss, (pi_loss, vf_loss, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (total_loss, (pi_loss, vf_loss, entropy)), grads = grad_fn(t_state.params)
                t_state = t_state.apply_gradients(grads=grads)
                return t_state, total_loss

            t_state, r = update_state
            r, subkey = jax.random.split(r)
            batch_size = self.num_steps * self.num_envs
            minibatch_size = batch_size // self.num_minibatches

            # Flatten trajectory batch
            flat_traj = jax.tree_util.tree_map(lambda x: x.reshape((batch_size, *x.shape[2:])), traj_batch)
            flat_gae = advantages.reshape((batch_size,))
            flat_ret = returns.reshape((batch_size,))

            # Shuffle and split into minibatches
            permutation = jax.random.permutation(subkey, batch_size)
            batch = (flat_traj, flat_gae, flat_ret)
            shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [self.num_minibatches, minibatch_size] + list(x.shape[1:])),
                shuffled_batch
            )

            t_state, total_loss = jax.lax.scan(_update_minibatch, t_state, minibatches)
            return (t_state, r), total_loss.mean()

        (train_state, rng), loss = jax.lax.scan(
            _update_epoch, (train_state, rng), None, length=self.update_epochs
        )

        mean_reward = traj_batch.reward.sum(axis=0).mean()
        metrics = {
            "loss": loss.mean(),
            "mean_reward": mean_reward
        }

        return (train_state, env_state, last_obs, rng), metrics
