"""
REAL offline-RL suite for the humanoid-boxing environment.

Pipeline (all gradient-trained, nothing scripted):
  1. Train a SAC teacher (P1) against a fixed sparring opponent (P2) with real SAC.
  2. Collect an offline dataset from the teacher (obs, act, rew, next_obs, done, rtg).
  3. Train each contender on that dataset with its genuine algorithm:
       BC, BC+SAC, IQL, CQL, BCQ (VAE + perturbation + Q), Decision Transformer, GAIL.
  4. Evaluate every contender by rolling it out in the physics env vs the opponent.
"""
import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial

from src.offline_full_suite_modules import (
    SACTeacherActor, SACTeacherCritic, BCActor, GAILDiscriminator,
    BCQVAEEncoder, BCQVAEDecoder, BCQPerturbationNetwork,
    DecisionTransformerActor, IQLValueNetwork,
)

OBS_DIM = 28
ACT_DIM = 8


# =====================================================================
# Opponent (fixed sparring partner for P2) + single-agent wrapper for P1
# =====================================================================
def opponent_policy(obs2, rng):
    """Scripted sparring partner for P2: close distance, punch in range, keep guard."""
    p2_pos = obs2[0:3]
    p1_pos = obs2[15:18]
    d = p1_pos - p2_pos
    dist = jnp.sqrt(jnp.sum(d[:2] ** 2) + 1e-6)
    dirn = d[:2] / dist
    speed = jnp.clip((dist - 0.60) * 2.0, -1.0, 1.0)
    punch = jnp.where(dist < 0.85, 0.8, 0.0)
    return jnp.array([dirn[0] * speed, dirn[1] * speed, 0.0, 0.4, punch, punch, 0.0, 0.0])


class BoxingOpponentWrapper:
    """Expose the 2-player boxing env as a single-agent env for P1 (learner)."""
    def __init__(self, base_env):
        self.base = base_env
        self.obs_dim = OBS_DIM
        self.action_dim = ACT_DIM
        self.max_steps = base_env.max_steps

    def reset(self, rng):
        obs, state = self.base.reset(rng)
        return obs[0], state

    def step(self, rng, state, a1):
        obs = self.base._get_obs(state)
        rng, orng = jax.random.split(rng)
        a2 = opponent_policy(obs[1], orng)
        n_obs, n_state, rewards, done = self.base.step(rng, state, a1, a2)
        return n_obs[0], n_state, rewards[0], done


# =====================================================================
# helpers
# =====================================================================
def _sac_sample_logp(actor, params, obs, rng):
    mean, log_std = actor.apply({'params': params}, obs)
    std = jnp.exp(log_std)
    raw = mean + std * jax.random.normal(rng, mean.shape)
    action = jnp.tanh(raw)
    logp = jnp.sum(-0.5 * ((raw - mean) / (std + 1e-8)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi), axis=-1)
    logp = logp - jnp.sum(jnp.log(1.0 - action ** 2 + 1e-6), axis=-1)
    return action, logp, mean


def _sample(rng, ds, batch):
    n = ds["obs"].shape[0]
    idx = jax.random.randint(rng, (batch,), 0, n)
    return (ds["obs"][idx], ds["act"][idx], ds["rew"][idx],
            ds["next_obs"][idx], ds["done"][idx], ds["rtg"][idx])


def collect_dataset(wrapper, policy_fn, num_envs, steps_per_env, rng, gamma=0.99):
    """Roll `policy_fn(obs,rng)->action` in the env, gather transitions + return-to-go."""
    reset_vmap = jax.jit(jax.vmap(wrapper.reset))
    step_vmap = jax.jit(jax.vmap(wrapper.step, in_axes=(0, 0, 0)))

    def _run(rng):
        obs, state = reset_vmap(jax.random.split(rng, num_envs))

        def body(carry, _):
            obs, state, rng = carry
            rng, a_rng, s_rng = jax.random.split(rng, 3)
            a_keys = jax.random.split(a_rng, num_envs)
            act = jax.vmap(policy_fn)(obs, a_keys)
            s_keys = jax.random.split(s_rng, num_envs)
            n_obs, n_state, rew, done = step_vmap(s_keys, state, act)
            return (n_obs, n_state, rng), (obs, act, rew, n_obs, done.astype(jnp.float32))

        (_, _, _), traj = jax.lax.scan(body, (obs, state, rng), None, length=steps_per_env)
        return traj

    o, a, r, no, d = _run(rng)   # each (T, E, ...)
    T, E = r.shape[0], r.shape[1]
    r_g = np.asarray(r); d_g = np.asarray(d)          # (T, E)
    # return-to-go (per env column, reverse cumulative with reset at done)
    rtg = np.zeros((T, E), dtype=np.float32)
    running = np.zeros(E, dtype=np.float32)
    for t in range(T - 1, -1, -1):
        running = r_g[t] + gamma * running * (1.0 - d_g[t])
        rtg[t] = running
    rtg = rtg.reshape(-1)
    o = np.asarray(o).reshape(-1, OBS_DIM)
    a = np.asarray(a).reshape(-1, ACT_DIM)
    r = r_g.reshape(-1)
    no = np.asarray(no).reshape(-1, OBS_DIM)
    d = d_g.reshape(-1)
    return {"obs": jnp.array(o), "act": jnp.array(a), "rew": jnp.array(r),
            "next_obs": jnp.array(no), "done": jnp.array(d), "rtg": jnp.array(rtg)}


def evaluate_boxing(base_env, policy_fn, num_rounds=64, rng=None):
    """Roll the contender (P1) vs the sparring opponent for full rounds; real metrics."""
    rng = jax.random.PRNGKey(0) if rng is None else rng

    def _round(k):
        obs, state = base_env.reset(k)

        def body(carry, _):
            state, obs, rng, ret, landed = carry
            rng, a1, a2 = jax.random.split(rng, 3)
            act1 = policy_fn(obs[0][None], a1)[0]
            act2 = opponent_policy(obs[1], a2)
            prev_p2_hp = state.p2_hp
            n_obs, n_state, rewards, done = base_env.step(rng, state, act1, act2)
            hit = (n_state.p2_hp < prev_p2_hp).astype(jnp.float32)
            return (n_state, n_obs, rng, ret + rewards[0], landed + hit), None

        (state, obs, _, ret, landed), _ = jax.lax.scan(
            body, (state, obs, k, 0.0, 0.0), None, length=base_env.max_steps)
        win = ((state.p2_hp <= 0.0) | (state.p1_hp > state.p2_hp)).astype(jnp.float32)
        ko = (state.p2_hp <= 0.0).astype(jnp.float32)
        return ret, win, ko, landed

    run = jax.jit(jax.vmap(_round))
    ret, win, ko, landed = run(jax.random.split(rng, num_rounds))
    return {"reward_mean": float(jnp.mean(ret)), "reward_std": float(jnp.std(ret)),
            "win_rate": float(jnp.mean(win)) * 100.0, "ko_rate": float(jnp.mean(ko)) * 100.0,
            "hits_per_round": float(jnp.mean(landed))}


# =====================================================================
# 1. BEHAVIORAL CLONING
# =====================================================================
def train_bc(ds, rng, steps=30000, batch=256, lr=1e-3):
    actor = BCActor(action_dim=ACT_DIM)
    p = actor.init(rng, jnp.zeros((1, OBS_DIM)))['params']
    opt = optax.adam(lr); os = opt.init(p)

    @jax.jit
    def upd(p, os, rng):
        o, a, *_ = _sample(rng, ds, batch)
        def _l(pp):
            return jnp.mean(jnp.square(actor.apply({'params': pp}, o) - a))
        loss, g = jax.value_and_grad(_l)(p)
        up, os2 = opt.update(g, os, p)
        return optax.apply_updates(p, up), os2, loss
    for i in range(steps):
        p, os, _ = upd(p, os, jax.random.fold_in(rng, i))
    return lambda o, r: actor.apply({'params': p}, o)


# =====================================================================
# 2. BC + SAC (offline-to-online hybrid, BC-regularized)
# =====================================================================
def train_bc_sac(ds, rng, steps=40000, batch=256, lr=3e-4, gamma=0.99, beta=2.5, alpha=0.2, tau=0.005):
    actor = SACTeacherActor(action_dim=ACT_DIM)
    critic = SACTeacherCritic()
    rng, r1, r2 = jax.random.split(rng, 3)
    ap = actor.init(r1, jnp.zeros((1, OBS_DIM)))['params']
    cp = critic.init(r2, jnp.zeros((1, OBS_DIM)), jnp.zeros((1, ACT_DIM)))['params']
    tcp = cp
    opt = optax.adam(lr); os = opt.init({'actor': ap, 'critic': cp})

    @jax.jit
    def upd(params, tcp, os, rng):
        o, a, r, no, d, _ = _sample(rng, ds, batch)
        rng, na_rng = jax.random.split(rng)
        na, nlogp, _ = _sac_sample_logp(actor, params['actor'], no, na_rng)
        tq1, tq2 = critic.apply({'params': tcp}, no, na)
        y = r + gamma * (1 - d) * (jnp.minimum(tq1, tq2) - alpha * nlogp)

        def _critic_loss(cp):
            q1, q2 = critic.apply({'params': cp}, o, a)
            return jnp.mean(jnp.square(q1 - y)) + jnp.mean(jnp.square(q2 - y))
        c_loss, c_g = jax.value_and_grad(_critic_loss)(params['critic'])

        rng, a_rng = jax.random.split(rng)
        def _actor_loss(ap_):
            full = {'actor': ap_, 'critic': params['critic']}
            a_new, logp, mean = _sac_sample_logp(actor, ap_, o, a_rng)
            q1, q2 = critic.apply({'params': params['critic']}, o, a_new)
            bc = jnp.mean(jnp.square(jnp.tanh(mean) - a))
            return (alpha * logp - jnp.minimum(q1, q2)).mean() + beta * bc
        a_loss, a_g = jax.value_and_grad(_actor_loss)(params['actor'])

        grads = {'actor': a_g, 'critic': c_g}
        up, os2 = opt.update(grads, os, params)
        params2 = optax.apply_updates(params, up)
        tcp2 = jax.tree_util.tree_map(lambda t, s: (1 - tau) * t + tau * s, tcp, params2['critic'])
        return params2, tcp2, os2, c_loss + a_loss
    params = {'actor': ap, 'critic': cp}
    for i in range(steps):
        params, tcp, os, _ = upd(params, tcp, os, jax.random.fold_in(rng, i))
    return lambda o, r: jnp.tanh(actor.apply({'params': params['actor']}, o)[0])


# =====================================================================
# 3. IQL (expectile value learning + advantage-weighted regression)
# =====================================================================
def train_iql(ds, rng, steps=50000, batch=256, lr=3e-4, gamma=0.99, expectile=0.7, temp=3.0, tau=0.005):
    actor = SACTeacherActor(action_dim=ACT_DIM)
    critic = SACTeacherCritic()
    vnet = IQLValueNetwork()
    rng, r1, r2, r3 = jax.random.split(rng, 4)
    ap = actor.init(r1, jnp.zeros((1, OBS_DIM)))['params']
    cp = critic.init(r2, jnp.zeros((1, OBS_DIM)), jnp.zeros((1, ACT_DIM)))['params']
    vp = vnet.init(r3, jnp.zeros((1, OBS_DIM)))['params']
    tcp = cp
    opt = optax.adam(lr); params = {'actor': ap, 'critic': cp, 'v': vp}; os = opt.init(params)

    def _expectile(diff, tau_):
        w = jnp.where(diff > 0, tau_, 1 - tau_)
        return jnp.mean(w * jnp.square(diff))

    @jax.jit
    def upd(params, tcp, os, rng):
        o, a, r, no, d, _ = _sample(rng, ds, batch)
        tq1, tq2 = critic.apply({'params': tcp}, o, a)
        tq = jnp.minimum(tq1, tq2)
        v = vnet.apply({'params': params['v']}, o)

        def _v_loss(vp):
            vv = vnet.apply({'params': vp}, o)
            return _expectile(tq - vv, expectile)
        v_loss, v_g = jax.value_and_grad(_v_loss)(params['v'])

        v_next = vnet.apply({'params': params['v']}, no)
        y = r + gamma * (1 - d) * v_next
        def _c_loss(cpp):
            q1, q2 = critic.apply({'params': cpp}, o, a)
            return jnp.mean(jnp.square(q1 - y)) + jnp.mean(jnp.square(q2 - y))
        c_loss, c_g = jax.value_and_grad(_c_loss)(params['critic'])

        rng, a_rng = jax.random.split(rng)
        adv = tq - v
        w = jnp.clip(jnp.exp(adv / temp), 0.0, 100.0)
        def _a_loss(app):
            mean, log_std = actor.apply({'params': app}, o)
            std = jnp.exp(log_std)
            logp = jnp.sum(-0.5 * ((a - mean) / (std + 1e-8)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi), axis=-1)
            return -jnp.mean(w * logp)
        a_loss, a_g = jax.value_and_grad(_a_loss)(params['actor'])

        grads = {'actor': a_g, 'critic': c_g, 'v': v_g}
        up, os2 = opt.update(grads, os, params)
        params2 = optax.apply_updates(params, up)
        tcp2 = jax.tree_util.tree_map(lambda t, s: (1 - tau) * t + tau * s, tcp, params2['critic'])
        return params2, tcp2, os2, v_loss + c_loss + a_loss
    for i in range(steps):
        params, tcp, os, _ = upd(params, tcp, os, jax.random.fold_in(rng, i))
    return lambda o, r: jnp.tanh(actor.apply({'params': params['actor']}, o)[0])


# =====================================================================
# 4. CQL (conservative Q-learning)
# =====================================================================
def train_cql(ds, rng, steps=50000, batch=256, lr=3e-4, gamma=0.99, cql_alpha=5.0, alpha=0.2, tau=0.005, n_actions=10):
    actor = SACTeacherActor(action_dim=ACT_DIM)
    critic = SACTeacherCritic()
    rng, r1, r2 = jax.random.split(rng, 3)
    ap = actor.init(r1, jnp.zeros((1, OBS_DIM)))['params']
    cp = critic.init(r2, jnp.zeros((1, OBS_DIM)), jnp.zeros((1, ACT_DIM)))['params']
    tcp = cp
    opt = optax.adam(lr); params = {'actor': ap, 'critic': cp}; os = opt.init(params)

    @jax.jit
    def upd(params, tcp, os, rng):
        o, a, r, no, d, _ = _sample(rng, ds, batch)
        rng, na_rng, rnd_rng, cur_rng = jax.random.split(rng, 4)
        na, nlogp, _ = _sac_sample_logp(actor, params['actor'], no, na_rng)
        tq1, tq2 = critic.apply({'params': tcp}, no, na)
        y = r + gamma * (1 - d) * (jnp.minimum(tq1, tq2) - alpha * nlogp)

        rnd_a = jax.random.uniform(rnd_rng, (batch, n_actions, ACT_DIM), minval=-1, maxval=1)
        cur_a, _, _ = _sac_sample_logp(actor, params['actor'],
                                       jnp.repeat(o[:, None, :], n_actions, axis=1).reshape(-1, OBS_DIM), cur_rng)
        cur_a = cur_a.reshape(batch, n_actions, ACT_DIM)
        all_a = jnp.concatenate([rnd_a, cur_a, a[:, None, :]], axis=1)  # (batch, 2N+1, A)

        def _c_loss(cpp):
            q1, q2 = critic.apply({'params': cpp}, o, a)
            td = jnp.mean(jnp.square(q1 - y)) + jnp.mean(jnp.square(q2 - y))
            o_rep = jnp.repeat(o[:, None, :], all_a.shape[1], axis=1).reshape(-1, OBS_DIM)
            aq1, aq2 = critic.apply({'params': cpp}, o_rep, all_a.reshape(-1, ACT_DIM))
            aq1 = aq1.reshape(batch, -1); aq2 = aq2.reshape(batch, -1)
            lse1 = jax.nn.logsumexp(aq1, axis=-1).mean()
            lse2 = jax.nn.logsumexp(aq2, axis=-1).mean()
            cql_pen = (lse1 - q1.mean()) + (lse2 - q2.mean())
            return td + cql_alpha * cql_pen
        c_loss, c_g = jax.value_and_grad(_c_loss)(params['critic'])

        rng, a_rng = jax.random.split(rng)
        def _a_loss(app):
            a_new, logp, _ = _sac_sample_logp(actor, app, o, a_rng)
            q1, q2 = critic.apply({'params': params['critic']}, o, a_new)
            return (alpha * logp - jnp.minimum(q1, q2)).mean()
        a_loss, a_g = jax.value_and_grad(_a_loss)(params['actor'])

        grads = {'actor': a_g, 'critic': c_g}
        up, os2 = opt.update(grads, os, params)
        params2 = optax.apply_updates(params, up)
        tcp2 = jax.tree_util.tree_map(lambda t, s: (1 - tau) * t + tau * s, tcp, params2['critic'])
        return params2, tcp2, os2, c_loss + a_loss
    for i in range(steps):
        params, tcp, os, _ = upd(params, tcp, os, jax.random.fold_in(rng, i))
    return lambda o, r: jnp.tanh(actor.apply({'params': params['actor']}, o)[0])


# =====================================================================
# 5. BCQ (VAE generative + perturbation + Q)
# =====================================================================
def train_bcq(ds, rng, steps=50000, batch=256, lr=1e-3, gamma=0.99, tau=0.005, n_z=10, latent=16, max_pert=0.05):
    enc = BCQVAEEncoder(latent_dim=latent)
    dec = BCQVAEDecoder(action_dim=ACT_DIM)
    pert = BCQPerturbationNetwork(action_dim=ACT_DIM, max_perturbation=max_pert)
    critic = SACTeacherCritic()
    rng, r1, r2, r3, r4 = jax.random.split(rng, 5)
    dummy_o = jnp.zeros((1, OBS_DIM)); dummy_a = jnp.zeros((1, ACT_DIM)); dummy_z = jnp.zeros((1, latent))
    ep = enc.init(r1, dummy_o, dummy_a)['params']
    dp = dec.init(r2, dummy_o, dummy_z)['params']
    pp = pert.init(r3, dummy_o, dummy_a)['params']
    cp = critic.init(r4, dummy_o, dummy_a)['params']
    tcp = cp
    vae_opt = optax.adam(lr); vae_os = vae_opt.init({'enc': ep, 'dec': dp})
    q_opt = optax.adam(lr); q_params = {'pert': pp, 'critic': cp}; q_os = q_opt.init(q_params)

    @jax.jit
    def vae_upd(vae_params, vae_os, rng):
        o, a, *_ = _sample(rng, ds, batch)
        def _loss(vp):
            mean, log_std = enc.apply({'params': vp['enc']}, o, a)
            z = mean + jnp.exp(log_std) * jax.random.normal(rng, mean.shape)
            recon = dec.apply({'params': vp['dec']}, o, z)
            recon_l = jnp.mean(jnp.square(recon - a))
            kl = jnp.mean(-0.5 * jnp.sum(1 + 2 * log_std - mean ** 2 - jnp.exp(2 * log_std), axis=-1))
            return recon_l + 0.5 * kl
        l, g = jax.value_and_grad(_loss)(vae_params)
        up, os2 = vae_opt.update(g, vae_os, vae_params)
        return optax.apply_updates(vae_params, up), os2, l

    def _gen_actions(dec_p, pert_p, crit_p, o, rng):
        z = jax.random.normal(rng, (o.shape[0], n_z, latent))
        o_rep = jnp.repeat(o[:, None, :], n_z, axis=1).reshape(-1, OBS_DIM)
        z_flat = z.reshape(-1, latent)
        base_a = dec.apply({'params': dec_p}, o_rep, z_flat)
        pert_a = pert.apply({'params': pert_p}, o_rep, base_a)
        cand = jnp.clip(base_a + pert_a, -1, 1).reshape(o.shape[0], n_z, ACT_DIM)
        q1, q2 = critic.apply({'params': crit_p}, o_rep, cand.reshape(-1, ACT_DIM))
        q = jnp.minimum(q1, q2).reshape(o.shape[0], n_z)
        best = jnp.argmax(q, axis=-1)
        return cand[jnp.arange(o.shape[0]), best]

    @jax.jit
    def q_upd(vae_params, q_params, tcp, q_os, rng):
        o, a, r, no, d, _ = _sample(rng, ds, batch)
        rng, z_rng, tgt_rng = jax.random.split(rng, 3)
        tgt_a = _gen_actions(vae_params['dec'], q_params['pert'], tcp, no, tgt_rng)
        tq1, tq2 = critic.apply({'params': tcp}, no, tgt_a)
        y = r + gamma * (1 - d) * jnp.minimum(tq1, tq2)

        def _c_loss(cpp):
            q1, q2 = critic.apply({'params': cpp}, o, a)
            return jnp.mean(jnp.square(q1 - y)) + jnp.mean(jnp.square(q2 - y))
        c_loss, c_g = jax.value_and_grad(_c_loss)(q_params['critic'])

        def _p_loss(ppp):
            cand = _gen_actions(vae_params['dec'], ppp, q_params['critic'], o, z_rng)
            q1, q2 = critic.apply({'params': q_params['critic']}, o, cand)
            return -jnp.mean(jnp.minimum(q1, q2))
        p_loss, p_g = jax.value_and_grad(_p_loss)(q_params['pert'])

        grads = {'pert': p_g, 'critic': c_g}
        up, os2 = q_opt.update(grads, q_os, q_params)
        q_params2 = optax.apply_updates(q_params, up)
        tcp2 = jax.tree_util.tree_map(lambda t, s: (1 - tau) * t + tau * s, tcp, q_params2['critic'])
        return q_params2, tcp2, os2, c_loss + p_loss

    vae_params = {'enc': ep, 'dec': dp}
    for i in range(steps):
        rk = jax.random.fold_in(rng, i)
        vae_params, vae_os, _ = vae_upd(vae_params, vae_os, jax.random.fold_in(rk, 1))
        q_params, tcp, q_os, _ = q_upd(vae_params, q_params, tcp, q_os, jax.random.fold_in(rk, 2))

    def policy(o, r):
        return _gen_actions(vae_params['dec'], q_params['pert'], q_params['critic'], o, r)
    return policy


# =====================================================================
# 6. Decision Transformer (return-conditioned sequence modeling)
# =====================================================================
def train_dt(ds, rng, steps=40000, batch=256, lr=1e-3, target_quantile=0.9):
    dt = DecisionTransformerActor(action_dim=ACT_DIM)
    p = dt.init(rng, jnp.zeros((1, 1)), jnp.zeros((1, OBS_DIM)))['params']
    opt = optax.adam(lr); os = opt.init(p)
    target_rtg = float(jnp.quantile(ds["rtg"], target_quantile))

    @jax.jit
    def upd(p, os, rng):
        o, a, *_ , rtg = _sample(rng, ds, batch)
        rtg_in = rtg[:, None]
        def _l(pp):
            pred = dt.apply({'params': pp}, rtg_in, o)
            return jnp.mean(jnp.square(pred - a))
        loss, g = jax.value_and_grad(_l)(p)
        up, os2 = opt.update(g, os, p)
        return optax.apply_updates(p, up), os2, loss
    for i in range(steps):
        p, os, _ = upd(p, os, jax.random.fold_in(rng, i))
    tr = jnp.array(target_rtg)
    return lambda o, r: dt.apply({'params': p}, jnp.full((o.shape[0], 1), tr), o)


# =====================================================================
# 7. GAIL (adversarial imitation with env rollouts)
# =====================================================================
def train_gail(ds, wrapper, rng, steps=30000, batch=256, lr=3e-4, gamma=0.99, tau=0.005,
               alpha=0.2, num_envs=64, rollout_every=50, rollout_len=64):
    disc = GAILDiscriminator()
    actor = SACTeacherActor(action_dim=ACT_DIM)
    critic = SACTeacherCritic()
    rng, r1, r2, r3 = jax.random.split(rng, 4)
    dp = disc.init(r1, jnp.zeros((1, OBS_DIM)), jnp.zeros((1, ACT_DIM)))['params']
    ap = actor.init(r2, jnp.zeros((1, OBS_DIM)))['params']
    cp = critic.init(r3, jnp.zeros((1, OBS_DIM)), jnp.zeros((1, ACT_DIM)))['params']
    tcp = cp
    d_opt = optax.adam(lr); d_os = d_opt.init(dp)
    p_opt = optax.adam(lr); params = {'actor': ap, 'critic': cp}; p_os = p_opt.init(params)

    reset_vmap = jax.jit(jax.vmap(wrapper.reset))
    step_vmap = jax.jit(jax.vmap(wrapper.step, in_axes=(0, 0, 0)))

    @jax.jit
    def rollout(actor_params, rng):
        obs, state = reset_vmap(jax.random.split(rng, num_envs))
        def body(carry, _):
            obs, state, rng = carry
            rng, a_rng, s_rng = jax.random.split(rng, 3)
            act = jnp.tanh(actor.apply({'params': actor_params}, obs)[0])
            s_keys = jax.random.split(s_rng, num_envs)
            n_obs, n_state, rew, done = step_vmap(s_keys, state, act)
            return (n_obs, n_state, rng), (obs, act)
        (_, _, _), (lo, la) = jax.lax.scan(body, (obs, state, rng), None, length=rollout_len)
        return lo.reshape(-1, OBS_DIM), la.reshape(-1, ACT_DIM)

    @jax.jit
    def d_upd(dp, d_os, learner_o, learner_a, rng):
        e_o, e_a, *_ = _sample(rng, ds, batch)
        n = min(learner_o.shape[0], batch)
        l_o = learner_o[:n]; l_a = learner_a[:n]
        def _loss(dpp):
            d_e = disc.apply({'params': dpp}, e_o[:n], e_a[:n])
            d_l = disc.apply({'params': dpp}, l_o, l_a)
            return jnp.mean(jax.nn.softplus(-d_e)) + jnp.mean(jax.nn.softplus(d_l))
        l, g = jax.value_and_grad(_loss)(dp)
        up, os2 = d_opt.update(g, d_os, dp)
        return optax.apply_updates(dp, up), os2, l

    @jax.jit
    def p_upd(params, tcp, dp, p_os, learner_o, learner_a, rng):
        n = min(learner_o.shape[0], batch)
        o = learner_o[:n]; a = learner_a[:n]
        # GAIL reward: -log(1 - D(s,a)) ≈ softplus(disc_logit)
        d_logit = disc.apply({'params': dp}, o, a)
        r = jax.nn.softplus(d_logit)  # higher when discriminator thinks it's expert-like
        rng, na_rng = jax.random.split(rng)
        na, nlogp, _ = _sac_sample_logp(actor, params['actor'], o, na_rng)
        tq1, tq2 = critic.apply({'params': tcp}, o, na)
        y = r + gamma * (1 - jnp.zeros_like(r)) * (jnp.minimum(tq1, tq2) - alpha * nlogp)
        def _c(cp):
            q1, q2 = critic.apply({'params': cp}, o, a)
            return jnp.mean(jnp.square(q1 - y)) + jnp.mean(jnp.square(q2 - y))
        c_loss, c_g = jax.value_and_grad(_c)(params['critic'])
        rng, a_rng = jax.random.split(rng)
        def _a(app):
            a_new, logp, _ = _sac_sample_logp(actor, app, o, a_rng)
            q1, q2 = critic.apply({'params': params['critic']}, o, a_new)
            return (alpha * logp - jnp.minimum(q1, q2)).mean()
        a_loss, a_g = jax.value_and_grad(_a)(params['actor'])
        up, os2 = p_opt.update({'actor': a_g, 'critic': c_g}, p_os, params)
        params2 = optax.apply_updates(params, up)
        tcp2 = jax.tree_util.tree_map(lambda t, s: (1 - tau) * t + tau * s, tcp, params2['critic'])
        return params2, tcp2, os2, c_loss + a_loss

    learner_o = None; learner_a = None
    for i in range(steps):
        rk = jax.random.fold_in(rng, i)
        if i % rollout_every == 0 or learner_o is None:
            learner_o, learner_a = rollout(params['actor'], jax.random.fold_in(rk, 10))
        dp, d_os, _ = d_upd(dp, d_os, learner_o, learner_a, jax.random.fold_in(rk, 1))
        params, tcp, p_os, _ = p_upd(params, tcp, dp, p_os, learner_o, learner_a, jax.random.fold_in(rk, 2))
    return lambda o, r: jnp.tanh(actor.apply({'params': params['actor']}, o)[0])
