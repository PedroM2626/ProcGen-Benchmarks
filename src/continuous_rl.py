"""
REAL continuous-control trainers in JAX (reusable).

  * GaussianPPOTrainer : on-policy PPO with a diagonal-Gaussian policy (tanh-free,
    clipped actions) + MLP value critic. GAE + clipped surrogate + optax updates.
  * SACTrainer         : off-policy Soft Actor-Critic with twin Q-networks,
    squashed tanh-Gaussian actor, automatic temperature (alpha) tuning, replay
    buffer and Polyak target updates.

Both are vectorized over `num_envs` and expect the interface:
    env.reset(rng) -> (obs, state)
    env.step(rng, state, action) -> (obs, next_state, reward, done)
"""
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from typing import NamedTuple

from src.continuous_modules import ContinuousGaussianActor, SACCritic, SACTanhGaussianActor


class ContinuousValueCritic(nn.Module):
    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.Dense(128)(x))
        h = nn.relu(nn.Dense(128)(h))
        return jnp.squeeze(nn.Dense(1)(h), axis=-1)


def _where_done(done, a, b):
    """Select a where done else b, broadcasting `done` over a's trailing dims."""
    d = done.reshape((-1,) + (1,) * (a.ndim - 1))
    return jnp.where(d, a, b)


# =====================================================================
# GAUSSIAN PPO (continuous)
# =====================================================================
class GaussianPPOTrainer:
    def __init__(self, env, obs_dim, action_dim, num_envs=64, num_steps=64, lr=3e-4,
                 gamma=0.99, gae_lambda=0.95, clip_eps=0.2, ent_coef=0.005, vf_coef=0.5,
                 update_epochs=4, num_minibatches=4):
        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.actor = ContinuousGaussianActor(action_dim=action_dim)
        self.critic = ContinuousValueCritic()
        self.opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr, eps=1e-5))
        self.reset_vmap = jax.jit(jax.vmap(env.reset))
        self.step_vmap = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))

    def create_state(self, rng):
        rng, r1, r2 = jax.random.split(rng, 3)
        dummy = jnp.zeros((1, self.obs_dim))
        params = {'actor': self.actor.init(r1, dummy)['params'],
                  'critic': self.critic.init(r2, dummy)['params']}
        return params, self.opt.init(params)

    def make_train_step(self):
        def train_step(carry, _):
            params, opt_state, env_state, obs, rng = carry

            def _env_step(sc, _):
                params, env_state, obs, rng = sc
                rng, a_rng, s_rng, reset_rng = jax.random.split(rng, 4)
                mu, log_std = self.actor.apply({'params': params['actor']}, obs)
                action, logp = ContinuousGaussianActor.sample_and_log_prob(a_rng, mu, log_std)
                value = self.critic.apply({'params': params['critic']}, obs)
                step_keys = jax.random.split(s_rng, self.num_envs)
                n_obs, n_state, rew, done = self.step_vmap(step_keys, env_state, action)
                reset_keys = jax.random.split(reset_rng, self.num_envs)
                r_obs, r_state = self.reset_vmap(reset_keys)
                n_obs = _where_done(done, r_obs, n_obs)
                n_state = jax.tree_util.tree_map(lambda a, b: _where_done(done, a, b), r_state, n_state)
                trans = (obs, action, logp, value, rew, done.astype(jnp.float32))
                return (params, n_state, n_obs, rng), trans

            (params, env_state, obs, rng), traj = jax.lax.scan(
                _env_step, (params, env_state, obs, rng), None, length=self.num_steps)
            t_obs, t_act, t_logp, t_val, t_rew, t_done = traj
            last_val = self.critic.apply({'params': params['critic']}, obs)

            def _gae(carry, x):
                gae, next_value = carry
                d, v, r = x
                delta = r + self.gamma * next_value * (1.0 - d) - v
                gae = delta + self.gamma * self.gae_lambda * (1.0 - d) * gae
                return (gae, v), gae
            _, adv = jax.lax.scan(_gae, (jnp.zeros_like(last_val), last_val),
                                  (t_done, t_val, t_rew), reverse=True)
            returns = adv + t_val

            B = self.num_steps * self.num_envs
            mb = B // self.num_minibatches
            f_obs = t_obs.reshape((B, self.obs_dim))
            f_act = t_act.reshape((B, self.action_dim))
            f_logp = t_logp.reshape((B,))
            f_adv = adv.reshape((B,))
            f_ret = returns.reshape((B,))

            def _update_epoch(state, _):
                params, opt_state, rng = state

                def _loss(params, ob, ac, olp, g, ret):
                    mu, log_std = self.actor.apply({'params': params['actor']}, ob)
                    std = jnp.exp(log_std)
                    # Gaussian log-prob of the stored action under the current policy
                    lp = -0.5 * jnp.sum(((ac - mu) / (std + 1e-8)) ** 2 + 2 * log_std
                                        + jnp.log(2 * jnp.pi), axis=-1)
                    ent = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1).mean()
                    v = self.critic.apply({'params': params['critic']}, ob)
                    gg = (g - g.mean()) / (g.std() + 1e-8)
                    ratio = jnp.exp(lp - olp)
                    s1 = ratio * gg
                    s2 = jnp.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * gg
                    pi = -jnp.minimum(s1, s2).mean()
                    vf = 0.5 * jnp.square(v - ret).mean()
                    return pi + self.vf_coef * vf - self.ent_coef * ent

                def _minibatch(carry, idx):
                    params, opt_state = carry
                    loss, grads = jax.value_and_grad(_loss)(
                        params, f_obs[idx], f_act[idx], f_logp[idx], f_adv[idx], f_ret[idx])
                    updates, new_opt = self.opt.update(grads, opt_state, params)
                    return (optax.apply_updates(params, updates), new_opt), loss

                rng, perm_rng = jax.random.split(rng)
                idxs = jax.random.permutation(perm_rng, B).reshape((self.num_minibatches, mb))
                (params, opt_state), loss = jax.lax.scan(_minibatch, (params, opt_state), idxs)
                return (params, opt_state, rng), loss.mean()

            (params, opt_state, rng), loss = jax.lax.scan(
                _update_epoch, (params, opt_state, rng), None, length=self.update_epochs)
            metrics = {"loss": loss.mean(), "mean_reward": t_rew.sum(axis=0).mean()}
            return (params, opt_state, env_state, obs, rng), metrics

        return jax.jit(train_step)

    def make_eval_policy(self):
        def select(params, obs, rng):
            mu, _ = self.actor.apply({'params': params['actor']}, obs)
            return jnp.clip(mu, -1.0, 1.0)
        return select


# =====================================================================
# SOFT ACTOR-CRITIC (continuous)
# =====================================================================
class SACBuffer(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray
    idx: jnp.ndarray
    size: jnp.ndarray


class SACTrainer:
    def __init__(self, env, obs_dim, action_dim, num_envs=64, buffer_size=100000,
                 batch_size=256, lr=3e-4, alpha_lr=3e-4, gamma=0.99, tau=0.005,
                 init_alpha=0.2, target_entropy=None):
        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy if target_entropy is not None else -float(action_dim)
        self.actor = SACTanhGaussianActor(action_dim=action_dim)
        self.critic = SACCritic()
        self.opt = optax.adam(lr)
        self.alpha_opt = optax.adam(alpha_lr)
        self.reset_vmap = jax.jit(jax.vmap(env.reset))
        self.step_vmap = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))

    def create_state(self, rng):
        rng, r1, r2 = jax.random.split(rng, 3)
        dummy = jnp.zeros((1, self.obs_dim))
        dummy_a = jnp.zeros((1, self.action_dim))
        actor_params = self.actor.init(r1, dummy)['params']
        critic_params = self.critic.init(r2, dummy, dummy_a)['params']
        params = {'actor': actor_params, 'critic': critic_params}
        target_critic = critic_params
        log_alpha = jnp.log(jnp.array(0.2))
        buffer = SACBuffer(
            obs=jnp.zeros((self.buffer_size, self.obs_dim)),
            action=jnp.zeros((self.buffer_size, self.action_dim)),
            reward=jnp.zeros((self.buffer_size,)),
            next_obs=jnp.zeros((self.buffer_size, self.obs_dim)),
            done=jnp.zeros((self.buffer_size,)),
            idx=jnp.array(0), size=jnp.array(0))
        return params, target_critic, self.opt.init(params), log_alpha, self.alpha_opt.init(log_alpha), buffer

    def make_train_step(self):
        def train_step(carry, _):
            params, target_critic, opt_state, log_alpha, alpha_opt_state, buffer, env_state, obs, rng = carry
            alpha = jnp.exp(log_alpha)
            rng, a_rng, s_rng, reset_rng, samp_rng = jax.random.split(rng, 5)

            # act (exploration)
            mu, log_std = self.actor.apply({'params': params['actor']}, obs)
            action, _ = SACTanhGaussianActor.sample_squashed(a_rng, mu, log_std)
            step_keys = jax.random.split(s_rng, self.num_envs)
            n_obs, n_state, rew, done = self.step_vmap(step_keys, env_state, action)

            # store
            pos = (buffer.idx + jnp.arange(self.num_envs)) % self.buffer_size
            buffer = buffer._replace(
                obs=buffer.obs.at[pos].set(obs), action=buffer.action.at[pos].set(action),
                reward=buffer.reward.at[pos].set(rew), next_obs=buffer.next_obs.at[pos].set(n_obs),
                done=buffer.done.at[pos].set(done.astype(jnp.float32)),
                idx=(buffer.idx + self.num_envs) % self.buffer_size,
                size=jnp.minimum(buffer.size + self.num_envs, self.buffer_size))

            # auto-reset
            reset_keys = jax.random.split(reset_rng, self.num_envs)
            r_obs, r_state = self.reset_vmap(reset_keys)
            n_obs = _where_done(done, r_obs, n_obs)
            n_state = jax.tree_util.tree_map(lambda a, b: _where_done(done, a, b), r_state, n_state)

            # ---- gradient update ----
            idxs = jax.random.randint(samp_rng, (self.batch_size,), 0, jnp.maximum(buffer.size, 1))
            b_obs = buffer.obs[idxs]; b_act = buffer.action[idxs]; b_rew = buffer.reward[idxs]
            b_nobs = buffer.next_obs[idxs]; b_done = buffer.done[idxs]

            rng, targ_rng, act_rng2, alpha_rng = jax.random.split(rng, 4)
            t_mu, t_lstd = self.actor.apply({'params': params['actor']}, b_nobs)
            t_a, t_logp = SACTanhGaussianActor.sample_squashed(targ_rng, t_mu, t_lstd)
            tq1, tq2 = self.critic.apply({'params': target_critic}, b_nobs, t_a)
            y = b_rew + self.gamma * (1.0 - b_done) * (jnp.minimum(tq1, tq2) - alpha * t_logp)

            def _critic_loss(critic_params):
                q1, q2 = self.critic.apply({'params': critic_params}, b_obs, b_act)
                return jnp.mean(jnp.square(q1 - y)) + jnp.mean(jnp.square(q2 - y))

            def _actor_loss(actor_params):
                mu, lstd = self.actor.apply({'params': actor_params}, b_obs)
                a_new, logp_new = SACTanhGaussianActor.sample_squashed(act_rng2, mu, lstd)
                q1, q2 = self.critic.apply({'params': params['critic']}, b_obs, a_new)
                return (alpha * logp_new - jnp.minimum(q1, q2)).mean(), logp_new

            c_loss, c_grad = jax.value_and_grad(_critic_loss)(params['critic'])
            (a_loss, logp_new), a_grad = jax.value_and_grad(_actor_loss, has_aux=True)(params['actor'])
            grads = {'actor': a_grad, 'critic': c_grad}
            updates, opt_state_new = self.opt.update(grads, opt_state, params)
            params_new = optax.apply_updates(params, updates)

            # alpha (temperature) update
            def _alpha_loss(la):
                return -(jnp.exp(la) * (jax.lax.stop_gradient(logp_new) + self.target_entropy)).mean()
            al_loss, al_grad = jax.value_and_grad(_alpha_loss)(log_alpha)
            al_upd, alpha_opt_new = self.alpha_opt.update(al_grad, alpha_opt_state)
            log_alpha_new = optax.apply_updates(log_alpha, al_upd)
            log_alpha_new = jnp.clip(log_alpha_new, jnp.log(1e-4), jnp.log(5.0))

            # Polyak target update
            target_critic_new = jax.tree_util.tree_map(
                lambda t, s: (1 - self.tau) * t + self.tau * s, target_critic, params_new['critic'])

            do_update = buffer.size >= self.batch_size
            params_out = jax.lax.cond(do_update, lambda _: params_new, lambda _: params, None)
            opt_state_out = jax.lax.cond(do_update, lambda _: opt_state_new, lambda _: opt_state, None)
            target_out = jax.lax.cond(do_update, lambda _: target_critic_new, lambda _: target_critic, None)
            log_alpha_out = jax.lax.cond(do_update, lambda _: log_alpha_new, lambda _: log_alpha, None)
            alpha_opt_out = jax.lax.cond(do_update, lambda _: alpha_opt_new, lambda _: alpha_opt_state, None)

            metrics = {"critic_loss": c_loss, "actor_loss": a_loss, "alpha": jnp.exp(log_alpha_out),
                       "mean_reward": rew.mean()}
            return (params_out, target_out, opt_state_out, log_alpha_out, alpha_opt_out,
                    buffer, n_state, n_obs, rng), metrics

        return jax.jit(train_step)

    def make_eval_policy(self):
        def select(params, obs, rng):
            mu, _ = self.actor.apply({'params': params['actor']}, obs)
            return jnp.tanh(mu)
        return select


# =====================================================================
# DISCRETE ACTION WRAPPER + DISCRETE PPO (for a fair discrete-vs-continuous
# comparison on IDENTICAL dynamics/reward)
# =====================================================================
class DiscreteActionWrapper:
    """Wraps a continuous-force env with a quantized 5-action space (stay/L/R/D/U)."""
    def __init__(self, base_env):
        self.base = base_env
        self.max_steps = base_env.max_steps
        self.obs_dim = base_env.obs_dim
        self.num_actions = 5
        self._forces = jnp.array([[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0],
                                  [0.0, -1.0], [0.0, 1.0]])

    def reset(self, rng):
        return self.base.reset(rng)

    def step(self, rng, state, action):
        force = self._forces[action]
        return self.base.step(rng, state, force)


class DiscreteActorCritic(nn.Module):
    action_dim: int = 5

    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.Dense(128)(x))
        h = nn.relu(nn.Dense(128)(h))
        logits = nn.Dense(self.action_dim)(h)
        value = jnp.squeeze(nn.Dense(1)(h), axis=-1)
        return logits, value


class DiscretePPOTrainer:
    def __init__(self, env, obs_dim, action_dim, num_envs=64, num_steps=64, lr=3e-4,
                 gamma=0.99, gae_lambda=0.95, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5,
                 update_epochs=4, num_minibatches=4):
        self.env = env
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.model = DiscreteActorCritic(action_dim=action_dim)
        self.opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr, eps=1e-5))
        self.reset_vmap = jax.jit(jax.vmap(env.reset))
        self.step_vmap = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))

    def create_state(self, rng):
        dummy = jnp.zeros((1, self.obs_dim))
        params = self.model.init(rng, dummy)['params']
        return params, self.opt.init(params)

    def make_train_step(self):
        def train_step(carry, _):
            params, opt_state, env_state, obs, rng = carry

            def _env_step(sc, _):
                params, env_state, obs, rng = sc
                rng, a_rng, s_rng, reset_rng = jax.random.split(rng, 4)
                logits, value = self.model.apply({'params': params}, obs)
                action = jax.random.categorical(a_rng, logits)
                logp = jax.nn.log_softmax(logits)[jnp.arange(self.num_envs), action]
                step_keys = jax.random.split(s_rng, self.num_envs)
                n_obs, n_state, rew, done = self.step_vmap(step_keys, env_state, action)
                reset_keys = jax.random.split(reset_rng, self.num_envs)
                r_obs, r_state = self.reset_vmap(reset_keys)
                n_obs = _where_done(done, r_obs, n_obs)
                n_state = jax.tree_util.tree_map(lambda a, b: _where_done(done, a, b), r_state, n_state)
                trans = (obs, action, logp, value, rew, done.astype(jnp.float32))
                return (params, n_state, n_obs, rng), trans

            (params, env_state, obs, rng), traj = jax.lax.scan(
                _env_step, (params, env_state, obs, rng), None, length=self.num_steps)
            t_obs, t_act, t_logp, t_val, t_rew, t_done = traj
            _, last_val = self.model.apply({'params': params}, obs)

            def _gae(carry, x):
                gae, next_value = carry
                d, v, r = x
                delta = r + self.gamma * next_value * (1.0 - d) - v
                gae = delta + self.gamma * self.gae_lambda * (1.0 - d) * gae
                return (gae, v), gae
            _, adv = jax.lax.scan(_gae, (jnp.zeros_like(last_val), last_val),
                                  (t_done, t_val, t_rew), reverse=True)
            returns = adv + t_val

            B = self.num_steps * self.num_envs
            mb = B // self.num_minibatches
            f_obs = t_obs.reshape((B, self.obs_dim))
            f_act = t_act.reshape((B,))
            f_logp = t_logp.reshape((B,))
            f_adv = adv.reshape((B,))
            f_ret = returns.reshape((B,))

            def _update_epoch(state, _):
                params, opt_state, rng = state

                def _loss(params, ob, ac, olp, g, ret):
                    logits, v = self.model.apply({'params': params}, ob)
                    lp_all = jax.nn.log_softmax(logits)
                    lp = lp_all[jnp.arange(ob.shape[0]), ac]
                    ent = -jnp.sum(jax.nn.softmax(logits) * lp_all, axis=-1).mean()
                    gg = (g - g.mean()) / (g.std() + 1e-8)
                    ratio = jnp.exp(lp - olp)
                    s1 = ratio * gg
                    s2 = jnp.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * gg
                    pi = -jnp.minimum(s1, s2).mean()
                    vf = 0.5 * jnp.square(v - ret).mean()
                    return pi + self.vf_coef * vf - self.ent_coef * ent

                def _minibatch(carry, idx):
                    params, opt_state = carry
                    loss, grads = jax.value_and_grad(_loss)(
                        params, f_obs[idx], f_act[idx], f_logp[idx], f_adv[idx], f_ret[idx])
                    updates, new_opt = self.opt.update(grads, opt_state, params)
                    return (optax.apply_updates(params, updates), new_opt), loss

                rng, perm_rng = jax.random.split(rng)
                idxs = jax.random.permutation(perm_rng, B).reshape((self.num_minibatches, mb))
                (params, opt_state), loss = jax.lax.scan(_minibatch, (params, opt_state), idxs)
                return (params, opt_state, rng), loss.mean()

            (params, opt_state, rng), loss = jax.lax.scan(
                _update_epoch, (params, opt_state, rng), None, length=self.update_epochs)
            metrics = {"loss": loss.mean(), "mean_reward": t_rew.sum(axis=0).mean()}
            return (params, opt_state, env_state, obs, rng), metrics

        return jax.jit(train_step)

    def make_eval_policy(self):
        def select(params, obs, rng):
            logits, _ = self.model.apply({'params': params}, obs)
            return jnp.argmax(logits, axis=-1)
        return select
