"""Aux-losses contrastivas — EXTENSAO alem do estudo (como SPR/GAT).

Rotulo honesto: CURL/CPC/ACL existiram so na fase Craftax apagada; no
ProcGen entram como suite extra `aux`, mesmo rigor, fora da paridade.
Desenho padrao (Srinivas et al. CURL; Oord et al. CPC), adaptado ao loop:
  encoder ONLINE = backbone da policy (compartilhado) + encoder TARGET
    EMA (tau=0.99/iter); views aumentadas (crop proprio, p=1.0);
  CURL: query=z(aug o), key+=tgt(aug' o); bilinear W 512x512; InfoNCE/tau.
  CPC : query=z(aug o_t), pred linear; key+=tgt(o_{t+1}); InfoNCE/tau.
  ACL : query=MLP(z(aug o_t), a_t) [TransitionMLP de spr.py];
        key+=tgt(o_{t+1}); InfoNCE/tau.
  tau InfoNCE=0.1; coef aux=1.0; Adam aux 1e-4 (precedente ICM/RND);
  negativos = resto do minibatch (2048). Requer backbone CNN pixels.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from jax_port.augment import make_augment
from jax_port.spr import TransitionMLP

TAU_NCE = 0.1
AUX_LR = 1e-4
EMA_TAU = 0.99


def _norm(x, eps=1e-6):
    # sqrt(soma+eps): denominador nunca e zero exato (o JVP interno do
    # linalg.norm faz 0/0 em vetor nulo, e maximum() nao blinda).
    n = jnp.sqrt((x ** 2).sum(axis=-1, keepdims=True) + eps)
    return x / n


def _infonce(q, k, tau=TAU_NCE):
    logits = _norm(q) @ _norm(k).T / tau
    labels = jnp.arange(q.shape[0])
    logp = jax.nn.log_softmax(logits)
    return -logp[labels, labels].mean()


def make_contrast(backbone_cls, kind, n_actions=15):
    assert kind in ("curl", "cpc", "acl")
    backbone = backbone_cls()
    aug = make_augment("crop", p=1.0)
    opt_b = optax.adam(AUX_LR)
    opt_h = optax.adam(AUX_LR)
    W = (TransitionMLP(n_actions=n_actions) if kind == "acl"
         else nn.Dense(512, use_bias=False))

    def encode(bb_params, x_f):
        return backbone.apply({"params": bb_params}, x_f)

    @jax.jit
    def step(bb_p, bb_os, h_p, h_os, tgt_p, ot_u8, act, on_u8, key):
        k1, k2 = jax.random.split(key)
        xq = aug(ot_u8.astype(jnp.float32) / 255.0, k1)
        if kind == "curl":
            xk = aug(ot_u8.astype(jnp.float32) / 255.0, k2)
        else:
            xk = on_u8.astype(jnp.float32) / 255.0

        def loss_fn(bb, h):
            z = encode(bb, xq)
            if kind == "acl":
                a_oh = jax.nn.one_hot(act, n_actions)
                q = W.apply(h, jnp.concatenate([z, a_oh], axis=1))
            else:
                q = W.apply(h, z)
            k = backbone.apply({"params": tgt_p}, xk)
            return _infonce(q, k)

        (l2, grads) = jax.value_and_grad(
            lambda b, h: loss_fn(b, h), argnums=(0, 1))(bb_p, h_p)
        upd_b, bb_os = opt_b.update(grads[0], bb_os, bb_p)
        upd_h, h_os = opt_h.update(grads[1], h_os, h_p)
        bb_p = optax.apply_updates(bb_p, upd_b)
        h_p = optax.apply_updates(h_p, upd_h)
        return bb_p, bb_os, h_p, h_os, l2

    @jax.jit
    def ema(tgt_p, bb_p):
        return jax.tree.map(lambda t, o: EMA_TAU * t + (1.0 - EMA_TAU) * o,
                            tgt_p, bb_p)

    def init_head(key):
        if kind == "acl":
            return W.init(key, jnp.zeros((1, 512 + n_actions)))
        return W.init(key, jnp.zeros((1, 512)))

    return {"step": step, "ema": ema, "init_head": init_head,
            "opt_b": opt_b, "opt_h": opt_h, "backbone": backbone}
