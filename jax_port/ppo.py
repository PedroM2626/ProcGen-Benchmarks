"""Update PPO jittado (hparams fieis ao estudo, §1.4 + defaults SB3).

Estudo (``compare_suite.py:26``): lr 3e-4, n_steps 256, batch 64,
n_epochs 3, gamma 0.99, gae_lambda 0.95, clip 0.2 (+ defaults SB3:
vf_coef 0.5, ent_coef 0.01, max_grad_norm 0.5, normalize_advantage True).

Ajuste de throughput (documentado): o estudo usa n_envs=1 (DummyVecEnv
unitario) com minibatches de 64 — otimo para comparacao, pessimo para GPU
(centenas de grad-steps minusculos por iteracao). Aqui o paralelismo muda
(N envs x T steps, minibatch >= 1024) e a matematica do objetivo PPO fica
intacta: mesmo surrogate clipado, mesmo value-clip, mesma entropia.
"""

import jax
import jax.numpy as jnp
import optax


def make_optimizer(lr=3e-4, max_grad_norm=0.5):
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """GAE em numpy (T,N) -> advantages, returns. Barato: T~128, N~64."""
    import numpy as np
    T, N = rewards.shape
    adv = np.zeros_like(rewards)
    lastgaelam = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t].astype(np.float32)
        nv = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nv * nonterm - values[t]
        lastgaelam = delta + gamma * lam * nonterm * lastgaelam
        adv[t] = lastgaelam
    return adv, (adv + values).astype(np.float32)


def make_update_fn(model, optimizer, clip_range=0.2, vf_coef=0.5, ent_coef=0.01):
    """Retorna (update_jit, rollout_jit, forward_jit). Estado: (params, opt_state).

    Todo numero que pode morar no device mora no device: o update recebe
    obs uint8 contiguo (preprocess dentro do JIT) e o rollout amostra
    acao+logp dentro do JIT (um dispatch+sync por step de env).
    NOTA medida: fatiar minibatch com indice no device
    (``d_obs[d_mb]``) custa ~457ms/call neste setup; slicing no host +
    H2D contiguo custa ~12ms/call (~40x). O loop de minibatches usa
    portanto slicing numpy + H2D por call (ver ``train.py``).
    """

    def loss_fn(params, obs_u8, act, old_logp, adv, ret):
        obs = obs_u8.astype(jnp.float32) / 255.0
        logits, value = model.apply(params, obs)
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, act[:, None], axis=1).squeeze(1)
        ratio = jnp.exp(logp - old_logp)
        pg1 = ratio * adv
        pg2 = jnp.clip(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
        pg_loss = -jnp.mean(jnp.minimum(pg1, pg2))
        v_clipped = ret + jnp.clip(value - ret, -clip_range, clip_range)
        v_loss = jnp.mean(jnp.maximum((value - ret) ** 2, (v_clipped - ret) ** 2)) / 2.0
        entropy = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * logp_all, axis=1))
        return pg_loss + vf_coef * v_loss - ent_coef * entropy

    @jax.jit
    def update_real(state, obs_u8, act, old_logp, adv, ret):
        params, opt_state = state
        loss, grads = jax.value_and_grad(loss_fn)(params, obs_u8, act, old_logp, adv, ret)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    @jax.jit
    def rollout_step(params, obs_u8, key):
        obs = obs_u8.astype(jnp.float32) / 255.0
        logits, value = model.apply(params, obs)
        key, ks = jax.random.split(key)
        act = jax.random.categorical(ks, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(logits.shape[0]), act]
        return act, logp, value, key

    @jax.jit
    def forward(params, obs):
        return model.apply(params, obs)

    @jax.jit
    def normalize_adv(adv):
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    return update_real, rollout_step, forward, normalize_adv
