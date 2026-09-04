"""SPR (Self-Predictive Representations) — EXTENSAO alem do estudo ProcGen.

Rotulo honesto: o estudo `4f84ed3` NAO tem SPR (SPR/CURL/CPC/ACL eram da
fase Craftax apagada). Aqui o SPR roda como suite extra `spr`, com o
mesmo rigor, mas marcado como extensao — nunca misturado as conclusoes
das seces 1-12 / paridade.

Metodo (Schwarzer et al. 2021, adaptado ao loop PPO):
  encoder ONLINE = backbone da policy (COMPARTILHADO: o gradiente aux
    flui para o backbone) + target encoder = copia EMA (tau=0.99/iter);
  modelo de transicao MLP (512+15)->512->512 prevê z_{t+1} de (z_t, a_t);
  loss = MSE(pred, target) sobre latentes L2-normalizados, com views
    aumentadas (crop proprio, p=1.0) de o_t e o_{t+1};
  otimizador aux Adam 1e-4 (precedente ICM/RND do estudo), spr_coef=1.0
  (somado ao loss PPO na fase SPR, em minibatches de pares).
Requer backbone CNN pixels (nao mlp/vae).
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from jax_port.augment import make_augment

SPR_COEF = 1.0
SPR_LR = 1e-4
SPR_TAU = 0.99


class TransitionMLP(nn.Module):
    n_actions: int = 15

    @nn.compact
    def __call__(self, z_a):
        h = nn.relu(nn.Dense(512)(z_a))
        return nn.Dense(512)(h)


def _norm(x):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def make_spr(backbone_cls, n_actions=15):
    """Retorna dict com init/step/ema sobre o backbone compartilhado."""
    backbone = backbone_cls()
    trans = TransitionMLP(n_actions=n_actions)
    aug = make_augment("crop", p=1.0)
    opt_b = optax.adam(SPR_LR)
    opt_t = optax.adam(SPR_LR)

    def encode(bb_params, x_f, key=None):
        del key
        return backbone.apply({"params": bb_params}, x_f)

    @jax.jit
    def step(bb_p, bb_os, tr_p, tr_os, tgt_p, ot_u8, act, on_u8, key):
        k1, k2, k3 = jax.random.split(key, 3)
        xt = aug(ot_u8.astype(jnp.float32) / 255.0, k1)
        xn = aug(on_u8.astype(jnp.float32) / 255.0, k2)
        a_oh = jax.nn.one_hot(act, n_actions)

        def loss_fn(bb, tr):
            z = encode(bb, xt)
            pred = trans.apply(tr,
                               jnp.concatenate([z, a_oh], axis=1))
            tgt = backbone.apply({"params": tgt_p}, xn)
            return ((_norm(pred) - _norm(tgt)) ** 2).mean()

        (l2, grads) = jax.value_and_grad(
            lambda b, t: loss_fn(b, t), argnums=(0, 1))(bb_p, tr_p)
        upd_b, bb_os = opt_b.update(grads[0], bb_os, bb_p)
        upd_t, tr_os = opt_t.update(grads[1], tr_os, tr_p)
        bb_p = optax.apply_updates(bb_p, upd_b)
        tr_p = optax.apply_updates(tr_p, upd_t)
        return bb_p, bb_os, tr_p, tr_os, l2 * SPR_COEF

    @jax.jit
    def ema(tgt_p, bb_p):
        return jax.tree.map(lambda t, o: SPR_TAU * t + (1.0 - SPR_TAU) * o,
                            tgt_p, bb_p)

    return {"step": step, "ema": ema, "trans": trans, "opt_b": opt_b,
            "opt_t": opt_t, "backbone": backbone}
