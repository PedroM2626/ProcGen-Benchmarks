"""
REAL continuous multi-agent (3D drones) trainers + Brax single-agent wrapper.

  * ContinuousMARLPPOTrainer : IPPO / MAPPO / MA-POCA with a shared diagonal-Gaussian
    policy over continuous 3D thrust, trained with GAE + PPO (optax gradients).
  * BraxWrapper              : adapts a Brax env to the (obs,state)/(obs,state,rew,done)
    interface so the reusable GaussianPPOTrainer / SACTrainer can train on it.
  * make_marl3d_evaluator    : real episodic team-return / 3D-coverage / collision metrics.
"""
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from src.continuous_modules import ContinuousGaussianActor
from src.marl import DecentralizedCritic, CentralizedCritic, MAPOCACritic


def _where_done(done, a, b):
    d = done.reshape((-1,) + (1,) * (a.ndim - 1))
    return jnp.where(d, a, b)


class ContinuousMARLPPOTrainer:
    def __init__(self, algo, env, num_envs=64, num_steps=64, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_eps=0.2, ent_coef=0.005, vf_coef=0.5,
                 update_epochs=4, num_minibatches=4):
        assert algo in ("IPPO", "MAPPO", "MAPOCA")
        self.algo = algo
        self.env = env
        self.N = env.num_agents
        self.A = env.action_dim
        self.d = env.obs_dim
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.actor = ContinuousGaussianActor(action_dim=self.A)
        if algo == "IPPO":
            self.critic = DecentralizedCritic()
        elif algo == "MAPPO":
            self.critic = CentralizedCritic()
        else:
            self.critic = MAPOCACritic()
        self.opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr, eps=1e-5))
        self.reset_vmap = jax.jit(jax.vmap(env.reset))
        self.step_vmap = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))

    def _value(self, params, obs):
        E = obs.shape[0]
        if self.algo == "IPPO":
            v = self.critic.apply({'params': params['critic']}, obs.reshape(-1, self.d)).reshape(E, self.N)
        elif self.algo == "MAPPO":
            gs = obs.reshape(E, self.N * self.d)
            v = jnp.repeat(self.critic.apply({'params': params['critic']}, gs)[:, None], self.N, axis=-1)
        else:
            v_team, _ = self.critic.apply({'params': params['critic']}, obs)
            v = v_team
        return v

    def create_state(self, rng):
        rng, r1, r2 = jax.random.split(rng, 3)
        dummy_obs = jnp.zeros((self.num_envs, self.N, self.d))
        actor_params = self.actor.init(r1, jnp.zeros((1, self.d)))['params']
        if self.algo == "IPPO":
            critic_params = self.critic.init(r2, dummy_obs.reshape(-1, self.d))['params']
        elif self.algo == "MAPPO":
            critic_params = self.critic.init(r2, jnp.zeros((1, self.N * self.d)))['params']
        else:
            critic_params = self.critic.init(r2, dummy_obs)['params']
        params = {'actor': actor_params, 'critic': critic_params}
        return params, self.opt.init(params)

    def _act(self, params, obs, rng):
        E = obs.shape[0]
        mu, log_std = self.actor.apply({'params': params['actor']}, obs.reshape(-1, self.d))
        mu = mu.reshape(E, self.N, self.A)
        log_std = jnp.reshape(log_std, mu.shape)
        action, logp = ContinuousGaussianActor.sample_and_log_prob(rng, mu, log_std)
        return action, logp, mu, log_std

    def make_train_step(self):
        def train_step(carry, _):
            params, opt_state, env_state, obs, rng = carry

            def _env_step(sc, _):
                params, env_state, obs, rng = sc
                rng, a_rng, s_rng, reset_rng = jax.random.split(rng, 4)
                action, logp, _, _ = self._act(params, obs, a_rng)
                value = self._value(params, obs)  # (E,N)
                step_keys = jax.random.split(s_rng, self.num_envs)
                n_obs, n_state, rewards, done = self.step_vmap(step_keys, env_state, action)
                any_done = jnp.any(done)
                reset_keys = jax.random.split(reset_rng, self.num_envs)
                r_obs, r_state = self.reset_vmap(reset_keys)
                n_obs = jax.tree_util.tree_map(lambda a, b: jnp.where(any_done, a, b), r_obs, n_obs)
                n_state = jax.tree_util.tree_map(lambda a, b: jnp.where(any_done, a, b), r_state, n_state)
                done_ag = jnp.repeat(done[:, None].astype(jnp.float32), self.N, axis=-1)
                trans = (obs, action, logp, value, rewards, done_ag)
                return (params, n_state, n_obs, rng), trans

            (params, env_state, obs, rng), traj = jax.lax.scan(
                _env_step, (params, env_state, obs, rng), None, length=self.num_steps)
            t_obs, t_act, t_logp, t_val, t_rew, t_done = traj
            last_value = self._value(params, obs)

            def _gae(carry, x):
                gae, next_value = carry
                v, r, d = x
                delta = r + self.gamma * next_value * (1.0 - d) - v
                gae = delta + self.gamma * self.gae_lambda * (1.0 - d) * gae
                return (gae, v), gae
            _, adv = jax.lax.scan(_gae, (jnp.zeros_like(last_value), last_value),
                                  (t_val, t_rew, t_done), reverse=True)
            returns = adv + t_val

            TE = self.num_steps * self.num_envs
            returns_env = returns.reshape((TE, self.N))
            adv_env = adv.reshape((TE, self.N))
            act_env = t_act.reshape((TE, self.N, self.A))
            logp_env = t_logp.reshape((TE, self.N))
            team_ret_env = returns_env.mean(axis=-1)
            obs_env = t_obs.reshape((TE, self.N, self.d))
            mb_env = TE // self.num_minibatches

            def _update_epoch(state, _):
                params, opt_state, rng = state

                def _loss_fn(params, idx_env):
                    ob = obs_env[idx_env]                       # (mb,N,d)
                    ac = act_env[idx_env]                        # (mb,N,A)
                    mu, log_std = self.actor.apply({'params': params['actor']}, ob.reshape(-1, self.d))
                    mu = mu.reshape(-1, self.N, self.A)
                    log_std = log_std.reshape(mu.shape)
                    std = jnp.exp(log_std)
                    lp = -0.5 * jnp.sum(((ac - mu) / (std + 1e-8)) ** 2 + 2 * log_std
                                        + jnp.log(2 * jnp.pi), axis=-1)   # (mb,N)
                    ent = jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1).mean()
                    g = adv_env[idx_env].reshape(-1)
                    g = ((g - g.mean()) / (g.std() + 1e-8)).reshape(lp.shape)
                    ratio = jnp.exp(lp - logp_env[idx_env])
                    s1 = ratio * g
                    s2 = jnp.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * g
                    pi = -jnp.minimum(s1, s2).mean()
                    if self.algo == "IPPO":
                        v = self.critic.apply({'params': params['critic']}, ob.reshape(-1, self.d)).reshape(-1, self.N)
                        vf = 0.5 * jnp.square(v - returns_env[idx_env]).mean()
                    elif self.algo == "MAPPO":
                        gs = ob.reshape(-1, self.N * self.d)
                        v = self.critic.apply({'params': params['critic']}, gs)
                        vf = 0.5 * jnp.square(v - team_ret_env[idx_env]).mean()
                    else:
                        v_team, v_cf = self.critic.apply({'params': params['critic']}, ob)
                        vf = 0.5 * jnp.square(v_team - returns_env[idx_env]).mean() \
                             + 0.5 * jnp.square(v_cf - returns_env[idx_env]).mean()
                    return pi + self.vf_coef * vf - self.ent_coef * ent

                def _minibatch(carry, _):
                    params, opt_state, rng = carry
                    rng, k1 = jax.random.split(rng)
                    idx_env = jax.random.randint(k1, (mb_env,), 0, TE)
                    loss, grads = jax.value_and_grad(_loss_fn)(params, idx_env)
                    updates, new_opt = self.opt.update(grads, opt_state, params)
                    return (optax.apply_updates(params, updates), new_opt, rng), loss

                (params, opt_state, rng), loss = jax.lax.scan(
                    _minibatch, (params, opt_state, rng), None,
                    length=self.num_minibatches * self.update_epochs)
                return (params, opt_state, rng), loss.mean()

            (params, opt_state, rng), loss = jax.lax.scan(
                _update_epoch, (params, opt_state, rng), None, length=1)
            metrics = {"loss": loss.mean(), "team_reward": t_rew[:, :, 0].sum(axis=0).mean()}
            return (params, opt_state, env_state, obs, rng), metrics

        return jax.jit(train_step)

    def make_selector(self, params):
        def sel(obs, rng):
            E = obs.shape[0]
            mu, _ = self.actor.apply({'params': params['actor']}, obs.reshape(-1, self.d))
            return jnp.clip(mu.reshape(E, self.N, self.A), -1.0, 1.0)
        return sel


def make_marl3d_evaluator(env, select_actions, num_envs=256):
    """select_actions(obs (E,N,d), rng) -> actions (E,N,3). Real episodic metrics."""
    reset_vmap = jax.jit(jax.vmap(env.reset))
    step_vmap = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))
    N = env.num_agents

    def _run(rng):
        keys = jax.random.split(rng, num_envs)
        obs, state = reset_vmap(keys)

        def body(carry, _):
            state, obs, rng, ret, cov, col = carry
            rng, a_rng, s_rng = jax.random.split(rng, 3)
            actions = select_actions(obs, a_rng)
            step_keys = jax.random.split(s_rng, num_envs)
            n_obs, n_state, rewards, done = step_vmap(step_keys, state, actions)
            diff = n_state.landmark_pos[:, :, None, :] - n_state.agent_pos[:, None, :, :]  # (E,L,N,3)
            d = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-6)   # (E,L,N)
            covered = jnp.mean((jnp.min(d, axis=-1) < 0.25).astype(jnp.float32), axis=-1)
            apos = n_state.agent_pos
            adiff = apos[:, :, None, :] - apos[:, None, :, :]
            adist = jnp.sqrt(jnp.sum(adiff ** 2, axis=-1) + 1e-6)
            offdiag = ~jnp.eye(N, dtype=bool)
            col_e = jnp.sum((adist < 0.15) & offdiag[None], axis=(1, 2)) / 2.0
            return (n_state, n_obs, rng, ret + rewards[:, 0], cov + covered, col + col_e), None

        init = (state, obs, rng, jnp.zeros(num_envs), jnp.zeros(num_envs), jnp.zeros(num_envs))
        (_, _, _, ret, cov, col) = jax.lax.scan(body, init, None, length=env.max_steps)[0]
        return ret, cov / env.max_steps, col

    run = jax.jit(_run)

    def eval_fn(rng):
        ret, cov, col = run(rng)
        return (float(jnp.mean(ret)), float(jnp.std(ret)),
                float(jnp.mean(cov)) * 100.0, float(jnp.mean(col)))
    return eval_fn


class BraxWrapper:
    """Adapt a Brax env to reset(rng)->(obs,state); step(rng,state,action)->(obs,state,rew,done)."""
    def __init__(self, brax_env, max_steps=1000):
        self.env = brax_env
        self.obs_dim = int(brax_env.observation_size)
        self.action_dim = int(brax_env.action_size)
        self.max_steps = max_steps

    def reset(self, rng):
        state = self.env.reset(rng)
        return state.obs, state

    def step(self, rng, state, action):
        state = self.env.step(state, action)
        return state.obs, state, state.reward, state.done
