"""
REAL Multi-Agent RL trainers.

The previous `compare_marl.py` hard-coded every reward/coverage number. This module
provides genuine, gradient-trained MARL algorithms on `MultiAgentParticleEnv`:

  * MARLPPOTrainer  -> IPPO, MAPPO, MA-POCA  (CTDE policy-gradient with GAE)
  * MARLQTrainer    -> VDN, QMIX             (monotonic value decomposition + replay buffer)

All updates use `jax.value_and_grad` + `optax`, all metrics come from rolling the
trained policy out in the environment.
"""
import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple

from src.marl import (
    MARLActor, DecentralizedCritic, CentralizedCritic,
    MARLQNetwork, QMIXMixingNetwork, vdn_mix, MAPOCACritic,
)


# =====================================================================
# POLICY-BASED CTDE: IPPO / MAPPO / MA-POCA
# =====================================================================
class MARLPPOTrainer:
    def __init__(self, algo, env, num_envs=64, num_steps=64, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5,
                 update_epochs=4, num_minibatches=4):
        assert algo in ("IPPO", "MAPPO", "MAPOCA")
        self.algo = algo
        self.env = env
        self.N = env.num_agents
        self.A = env.num_actions
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.lr = lr
        self.opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(self.lr, eps=1e-5))

        self.actor = MARLActor(action_dim=self.A)
        if algo == "IPPO":
            self.critic = DecentralizedCritic()
        elif algo == "MAPPO":
            self.critic = CentralizedCritic()
        else:
            self.critic = MAPOCACritic()

        self.reset_vmap = jax.jit(jax.vmap(env.reset))
        self.step_vmap = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))

    # ---------- value / logprob helpers ----------
    def _value(self, params, obs, gstate):
        """Return per-agent value V (..., N)."""
        if self.algo == "IPPO":
            flat = obs.reshape(-1, obs.shape[-1])
            v = self.critic.apply({'params': params['critic']}, flat).reshape(*obs.shape[:-1])
            return v
        if self.algo == "MAPPO":
            v = self.critic.apply({'params': params['critic']}, gstate)  # (...,)
            return jnp.repeat(v[..., None], self.N, axis=-1)
        v_team, _ = self.critic.apply({'params': params['critic']}, obs)
        return v_team

    def create_state(self, rng):
        rng, r1, r2 = jax.random.split(rng, 3)
        dummy_obs = jnp.zeros((self.num_envs, self.N, self.env.obs_dim))
        dummy_gstate = jnp.zeros((self.num_envs, self.env.global_state_dim))
        actor_params = self.actor.init(r1, dummy_obs.reshape(-1, self.env.obs_dim))['params']
        if self.algo == "IPPO":
            critic_params = self.critic.init(r2, dummy_obs.reshape(-1, self.env.obs_dim))['params']
        elif self.algo == "MAPPO":
            critic_params = self.critic.init(r2, dummy_gstate)['params']
        else:
            critic_params = self.critic.init(r2, dummy_obs)['params']
        params = {'actor': actor_params, 'critic': critic_params}
        return params, self.opt.init(params)

    def _act(self, params, obs, rng):
        logits = self.actor.apply({'params': params['actor']}, obs.reshape(-1, self.env.obs_dim))
        logits = logits.reshape(self.num_envs, self.N, self.A)
        action = jax.random.categorical(rng, logits, axis=-1)  # (E,N)
        logp = jax.nn.log_softmax(logits, axis=-1)
        logp = jnp.take_along_axis(logp, action[..., None], axis=-1).squeeze(-1)  # (E,N)
        return action, logp, logits

    def make_train_step(self):
        def train_step(carry, _):
            params, opt_state, env_state, obs, gstate, rng = carry
            rng, a_rng, s_rng = jax.random.split(rng, 3)

            # ---- collect rollout ----
            def _env_step(rollout_carry, _):
                params, env_state, obs, gstate, rng = rollout_carry
                rng, act_rng, step_rng, reset_rng = jax.random.split(rng, 4)
                action, logp, _ = self._act(params, obs, act_rng)
                value = self._value(params, obs, gstate)  # (E,N)
                step_keys = jax.random.split(step_rng, self.num_envs)
                n_obs, n_gstate, n_state, reward, done = self.step_vmap(step_keys, env_state, action)
                # episode boundary is synchronous across envs (lockstep step_count)
                any_done = jnp.any(done)
                reset_keys = jax.random.split(reset_rng, self.num_envs)
                r_obs, r_gstate, r_state = self.reset_vmap(reset_keys)
                n_obs = jax.tree_util.tree_map(lambda a, b: jnp.where(any_done, a, b), r_obs, n_obs)
                n_gstate = jnp.where(any_done, r_gstate, n_gstate)
                n_state = jax.tree_util.tree_map(lambda a, b: jnp.where(any_done, a, b), r_state, n_state)
                trans = (obs, gstate, action, logp, value, jnp.repeat(reward[:, None], self.N, axis=-1),
                         jnp.repeat(done[:, None], self.N, axis=-1).astype(jnp.float32))
                return (params, n_state, n_obs, n_gstate, rng), trans

            (_, env_state, obs, gstate, rng), traj = jax.lax.scan(
                _env_step, (params, env_state, obs, gstate, rng), None, length=self.num_steps)
            t_obs, t_gstate, t_act, t_logp, t_val, t_rew, t_done = traj
            # shapes: (T, E, N, ...) with value/reward/done (T,E,N)

            last_value = self._value(params, obs, gstate)  # (E,N)

            # ---- GAE per agent ----
            def _gae(carry, x):
                gae, next_value = carry
                val, rew, done = x
                delta = rew + self.gamma * next_value * (1.0 - done) - val
                gae = delta + self.gamma * self.gae_lambda * (1.0 - done) * gae
                return (gae, val), gae
            _, adv = jax.lax.scan(_gae, (jnp.zeros_like(last_value), last_value),
                                  (t_val, t_rew, t_done), reverse=True)
            returns = adv + t_val  # (T,E,N)

            # env-level groupings for centralized critics (MAPPO / MA-POCA)
            TE = self.num_steps * self.num_envs
            returns_env = returns.reshape((TE, self.N))            # per-agent returns grouped by (t,e)
            team_ret_env = returns_env.mean(axis=-1)               # (TE,)
            obs_all_env = t_obs.reshape((TE, self.N, self.env.obs_dim))
            gstate_env = t_gstate.reshape((TE, self.env.global_state_dim))

            # ---- PPO update ----
            B = self.num_steps * self.num_envs * self.N
            mb = B // self.num_minibatches
            flat = lambda x: x.reshape((B, *x.shape[3:])) if x.ndim > 3 else x.reshape((B,))
            f_obs = t_obs.reshape((B, self.env.obs_dim))
            f_gstate = t_gstate.reshape((self.num_steps * self.num_envs, self.env.global_state_dim))
            f_act = flat(t_act)
            f_logp = flat(t_logp)
            f_adv = flat(adv)
            f_ret = flat(returns)
            # value targets: per-agent for IPPO/MAPOCA, per-env for MAPPO
            f_val = flat(t_val)

            def _update_epoch(state, _):
                params, opt_state, rng = state
                rng, perm_rng = jax.random.split(rng)

                def _loss_fn(params, idx_agent, idx_env):
                    ob = f_obs[idx_agent]                       # (mb, d)
                    ac = f_act[idx_agent]                        # (mb,)
                    logits = self.actor.apply({'params': params['actor']}, ob)
                    logp_all = jax.nn.log_softmax(logits, axis=-1)
                    logp = jnp.take_along_axis(logp_all, ac[:, None], axis=-1).squeeze(-1)
                    probs = jax.nn.softmax(logits, axis=-1)
                    ent = -jnp.sum(probs * logp_all, axis=-1).mean()
                    g = f_adv[idx_agent]
                    g = (g - g.mean()) / (g.std() + 1e-8)
                    ratio = jnp.exp(logp - f_logp[idx_agent])
                    s1 = ratio * g
                    s2 = jnp.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * g
                    pi_loss = -jnp.minimum(s1, s2).mean()

                    # critic loss
                    if self.algo == "IPPO":
                        v = self.critic.apply({'params': params['critic']}, ob)
                        vf_loss = 0.5 * jnp.square(v - f_ret[idx_agent]).mean()
                    elif self.algo == "MAPPO":
                        gs = gstate_env[idx_env]
                        v = self.critic.apply({'params': params['critic']}, gs)
                        vf_loss = 0.5 * jnp.square(v - team_ret_env[idx_env]).mean()
                    else:
                        # MA-POCA: attention critic over all agents + counterfactual head
                        ob_all = obs_all_env[idx_env]                # (mb, N, d)
                        v_team, v_cf = self.critic.apply({'params': params['critic']}, ob_all)
                        tr = returns_env[idx_env]                    # (mb, N)
                        vf_loss = 0.5 * jnp.square(v_team - tr).mean() \
                                  + 0.5 * jnp.square(v_cf - tr).mean()
                    return pi_loss + self.vf_coef * vf_loss - self.ent_coef * ent, (pi_loss, vf_loss, ent)

                def _minibatch(carry, _):
                    params, opt_state, rng = carry
                    rng, k1, k2 = jax.random.split(rng, 3)
                    idx_agent = jax.random.randint(k1, (mb,), 0, B)
                    idx_env = jax.random.randint(k2, (mb,), 0, self.num_steps * self.num_envs)
                    (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(params, idx_agent, idx_env)
                    updates, opt_state_new = self.opt.update(grads, opt_state, params)
                    params_new = optax.apply_updates(params, updates)
                    return (params_new, opt_state_new, rng), loss

                (params, opt_state, rng), loss = jax.lax.scan(
                    _minibatch, (params, opt_state, rng), None, length=self.num_minibatches * self.update_epochs)
                return (params, opt_state, rng), loss.mean()

            (params, opt_state, rng), loss = jax.lax.scan(
                _update_epoch, (params, opt_state, rng), None, length=1)

            metrics = {"loss": loss.mean(), "team_reward": t_rew[:, :, 0].sum(axis=0).mean()}
            return (params, opt_state, env_state, obs, gstate, rng), metrics

        return jax.jit(train_step)


# =====================================================================
# VALUE DECOMPOSITION: VDN / QMIX
# =====================================================================
class QBuffer(NamedTuple):
    obs: jnp.ndarray        # (C, N, d)
    gstate: jnp.ndarray     # (C, Nd)
    actions: jnp.ndarray    # (C, N)
    rewards: jnp.ndarray    # (C,)
    next_obs: jnp.ndarray   # (C, N, d)
    next_gstate: jnp.ndarray
    dones: jnp.ndarray      # (C,)
    idx: jnp.ndarray
    size: jnp.ndarray


class MARLQTrainer:
    def __init__(self, algo, env, num_envs=64, buffer_size=50000, batch_size=256,
                 lr=5e-4, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay_steps=50000,
                 target_update_every=500, embed_dim=64):
        assert algo in ("VDN", "QMIX")
        self.algo = algo
        self.env = env
        self.N = env.num_agents
        self.A = env.num_actions
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.target_update_every = target_update_every
        self.embed_dim = embed_dim
        self.lr = lr
        self.opt = optax.adam(self.lr)

        self.qnet = MARLQNetwork(action_dim=self.A)
        self.mixer = QMIXMixingNetwork(num_agents=self.N, embed_dim=embed_dim) if algo == "QMIX" else None

        self.reset_vmap = jax.jit(jax.vmap(env.reset))
        self.step_vmap = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))

    def _q(self, params, obs):
        # obs (..., N, d) -> Q (..., N, A)
        flat = obs.reshape(-1, self.env.obs_dim)
        q = self.qnet.apply({'params': params['q']}, flat).reshape(*obs.shape[:-1], self.A)
        return q

    def _mix(self, params, q_chosen, gstate):
        if self.algo == "VDN":
            return vdn_mix(q_chosen)
        return self.mixer.apply({'params': params['mixer']}, q_chosen, gstate)

    def create_state(self, rng):
        rng, r1, r2 = jax.random.split(rng, 3)
        dummy_obs = jnp.zeros((1, self.N, self.env.obs_dim))
        q_params = self.qnet.init(r1, dummy_obs.reshape(-1, self.env.obs_dim))['params']
        params = {'q': q_params}
        if self.algo == "QMIX":
            dummy_gs = jnp.zeros((1, self.env.global_state_dim))
            dummy_q = jnp.zeros((1, self.N))
            params['mixer'] = self.mixer.init(r2, dummy_q, dummy_gs)['params']
        target_params = params
        buffer = QBuffer(
            obs=jnp.zeros((self.buffer_size, self.N, self.env.obs_dim)),
            gstate=jnp.zeros((self.buffer_size, self.env.global_state_dim)),
            actions=jnp.zeros((self.buffer_size, self.N), dtype=jnp.int32),
            rewards=jnp.zeros((self.buffer_size,)),
            next_obs=jnp.zeros((self.buffer_size, self.N, self.env.obs_dim)),
            next_gstate=jnp.zeros((self.buffer_size, self.env.global_state_dim)),
            dones=jnp.zeros((self.buffer_size,)),
            idx=jnp.array(0), size=jnp.array(0))
        return params, target_params, self.opt.init(params), buffer

    def make_train_step(self):
        def train_step(step_idx, carry):
            params, target_params, opt_state, buffer, env_state, obs, gstate, rng = carry
            rng, a_rng, s_rng, reset_rng, samp_rng = jax.random.split(rng, 5)

            eps = jnp.maximum(self.eps_end,
                              self.eps_start - (self.eps_start - self.eps_end) * (step_idx * self.num_envs / self.eps_decay_steps))
            q = self._q(params, obs)  # (E,N,A)
            greedy = jnp.argmax(q, axis=-1)
            rand_a = jax.random.randint(a_rng, (self.num_envs, self.N), 0, self.A)
            do_rand = jax.random.uniform(a_rng, (self.num_envs, self.N)) < eps
            action = jnp.where(do_rand, rand_a, greedy)

            step_keys = jax.random.split(s_rng, self.num_envs)
            n_obs, n_gstate, n_state, reward, done = self.step_vmap(step_keys, env_state, action)

            # store transitions
            positions = (buffer.idx + jnp.arange(self.num_envs)) % self.buffer_size
            buffer = buffer._replace(
                obs=buffer.obs.at[positions].set(obs),
                gstate=buffer.gstate.at[positions].set(gstate),
                actions=buffer.actions.at[positions].set(action),
                rewards=buffer.rewards.at[positions].set(reward),
                next_obs=buffer.next_obs.at[positions].set(n_obs),
                next_gstate=buffer.next_gstate.at[positions].set(n_gstate),
                dones=buffer.dones.at[positions].set(done.astype(jnp.float32)),
                idx=(buffer.idx + self.num_envs) % self.buffer_size,
                size=jnp.minimum(buffer.size + self.num_envs, self.buffer_size))

            # auto-reset (lockstep episodes)
            any_done = jnp.any(done)
            reset_keys = jax.random.split(reset_rng, self.num_envs)
            r_obs, r_gstate, r_state = self.reset_vmap(reset_keys)
            n_obs = jax.tree_util.tree_map(lambda a, b: jnp.where(any_done, a, b), r_obs, n_obs)
            n_gstate = jnp.where(any_done, r_gstate, n_gstate)
            n_state = jax.tree_util.tree_map(lambda a, b: jnp.where(any_done, a, b), r_state, n_state)

            # ---- gradient update ----
            def _loss(params):
                idxs = jax.random.randint(samp_rng, (self.batch_size,), 0, jnp.maximum(buffer.size, 1))
                b_obs = buffer.obs[idxs]
                b_gs = buffer.gstate[idxs]
                b_act = buffer.actions[idxs]
                b_rew = buffer.rewards[idxs]
                b_nobs = buffer.next_obs[idxs]
                b_ngs = buffer.next_gstate[idxs]
                b_done = buffer.dones[idxs]
                q_s = self._q(params, b_obs)  # (B,N,A)
                q_chosen = jnp.take_along_axis(q_s, b_act[..., None], axis=-1).squeeze(-1)  # (B,N)
                q_tot = self._mix(params, q_chosen, b_gs)
                # target
                q_next = self._q(target_params, b_nobs)
                next_greedy = jnp.argmax(q_next, axis=-1)
                q_next_chosen = jnp.take_along_axis(q_next, next_greedy[..., None], axis=-1).squeeze(-1)
                q_tot_next = self._mix(target_params, q_next_chosen, b_ngs)
                target = b_rew + self.gamma * (1.0 - b_done) * q_tot_next
                return jnp.mean(optax.huber_loss(q_tot, target))

            grads = jax.grad(_loss)(params)
            updates, opt_state_new = self.opt.update(grads, opt_state, params)
            params_new = optax.apply_updates(params, updates)
            loss = _loss(params)

            # periodic target update
            do_update = (step_idx % self.target_update_every == 0) & (buffer.size >= self.batch_size)
            target_params = jax.lax.cond(do_update, lambda _: params_new,
                                         lambda _: target_params, None)
            opt_state = jax.lax.cond(buffer.size >= self.batch_size, lambda _: opt_state_new, lambda _: opt_state, None)
            params_out = jax.lax.cond(buffer.size >= self.batch_size, lambda _: params_new, lambda _: params, None)

            metrics = {"loss": loss, "eps": eps, "team_reward": reward.mean()}
            return (params_out, target_params, opt_state, buffer, n_state, n_obs, n_gstate, rng), metrics

        return jax.jit(train_step)


# =====================================================================
# REAL EVALUATION: roll trained policies and measure team return + coverage
# =====================================================================
def make_marl_evaluator(env, select_actions, num_envs=256):
    """select_actions(obs (E,N,d), gstate (E,Nd), rng) -> actions (E,N)."""
    reset_vmap = jax.jit(jax.vmap(env.reset))
    step_vmap = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))

    def _run(rng):
        keys = jax.random.split(rng, num_envs)
        obs, gstate, state = reset_vmap(keys)

        def body(carry, _):
            state, obs, gstate, rng, ret, cov, col = carry
            rng, a_rng, s_rng = jax.random.split(rng, 3)
            actions = select_actions(obs, gstate, a_rng)
            step_keys = jax.random.split(s_rng, num_envs)
            n_obs, n_gstate, n_state, rew, done = step_vmap(step_keys, state, actions)
            # coverage: fraction of landmarks with an agent within 0.25
            diff = n_state.landmark_pos[:, :, None, :] - n_state.agent_pos[:, None, :, :]  # (E,L,N,2)
            d = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-6)  # (E, L, N)
            covered = jnp.mean((jnp.min(d, axis=-1) < 0.25).astype(jnp.float32), axis=-1)  # (E,)
            # collisions: pairs of agents within 0.15 (excluding self)
            apos = n_state.agent_pos  # (E, N, 2)
            N = apos.shape[1]
            adiff = apos[:, :, None, :] - apos[:, None, :, :]  # (E,N,N,2)
            adist = jnp.sqrt(jnp.sum(adiff ** 2, axis=-1) + 1e-6)  # (E,N,N)
            offdiag = ~jnp.eye(N, dtype=bool)
            col_e = jnp.sum((adist < 0.15) & offdiag[None], axis=(1, 2)) / 2.0  # (E,)
            return (n_state, n_obs, n_gstate, rng, ret + rew, cov + covered, col + col_e.mean()), None

        init = (state, obs, gstate, rng, jnp.zeros(num_envs), jnp.zeros(num_envs), 0.0)
        (_, _, _, _, ret, cov, col), _ = jax.lax.scan(body, init, None, length=env.max_steps)
        return ret, cov / env.max_steps, col / env.max_steps

    run = jax.jit(_run)

    def eval_fn(rng):
        ret, cov, col = run(rng)
        return float(jnp.mean(ret)), float(jnp.std(ret)), float(jnp.mean(cov)) * 100.0, float(jnp.mean(col))

    return eval_fn
