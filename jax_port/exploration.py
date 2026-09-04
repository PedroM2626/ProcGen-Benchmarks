"""Exploracao intrinseca fiel a ``compare_maze_heist.py:16`` (ICM/RND/NGU).

Semantica replicada (beta=0.01 nos tres; treino online a cada step):
  ICM: phi = CNN Classic sem FC (1024D); forward (1024+15)->512->1024;
       bonus = MSE(fwd(phi_t,a), phi_t+1); otimizados phi+fwd, Adam 1e-4.
       QUIRK FIEL: o inverse (2048->512->15) e definido mas NUNCA entra
       no loss no estudo — aqui tambem nao (documentado, nao removido).
  RND: target congelada + predictor CNN->512; bonus = MSE; Adam 1e-4.
  NGU: RND + memoria episodica por env (deque 1000): episodic = media
       das 5 menores distancias aos ultimos 100 (1.0 se <10);
       bonus = beta * rnd * episodic.
Ajuste de batching (documentado): o estudo treina 1 obs/step (n_envs=1);
aqui o batch e N envs com o mesmo lr — mesmo objetivo, SGD em lote.
Sem normalizacao de bonus (o estudo nao normaliza) e sem clip.
"""

import collections

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax


class ICMPhi(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(32, (8, 8), strides=(4, 4), padding="VALID")(x))
        x = nn.relu(nn.Conv(64, (4, 4), strides=(2, 2), padding="VALID")(x))
        x = nn.relu(nn.Conv(64, (3, 3), strides=(1, 1), padding="VALID")(x))
        return x.reshape((x.shape[0], -1))  # 1024D


class ICMForward(nn.Module):
    @nn.compact
    def __call__(self, phi_a):
        h = nn.relu(nn.Dense(512)(phi_a))
        return nn.Dense(1024)(h)


class ICMInverse(nn.Module):
    """Definido por fidelidade; o estudo nunca o otimiza (ver loss abaixo)."""

    @nn.compact
    def __call__(self, phi_pair):
        h = nn.relu(nn.Dense(512)(phi_pair))
        return nn.Dense(15)(h)


class RNDNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(32, (8, 8), strides=(4, 4), padding="VALID")(x))
        x = nn.relu(nn.Conv(64, (4, 4), strides=(2, 2), padding="VALID")(x))
        x = x.reshape((x.shape[0], -1))
        return nn.Dense(512)(x)


class Exploration:
    def __init__(self, kind, n_envs, seed=0, beta=0.01, lr=1e-4):
        assert kind in ("icm", "rnd", "ngu")
        self.kind, self.beta, self.n = kind, beta, n_envs
        key = jax.random.PRNGKey(seed)
        dummy = jnp.zeros((1, 64, 64, 3), jnp.float32)
        if kind == "icm":
            key, k1, k2, k3 = jax.random.split(key, 4)
            self.phi = ICMPhi()
            self.fwd = ICMForward()
            self.inv = ICMInverse()  # parametros existem; sem otimizador
            self.p_phi = self.phi.init(k1, dummy)
            self.p_fwd = self.fwd.init(k2, jnp.zeros((1, 1039)))
            self._ = self.inv.init(k3, jnp.zeros((1, 2048)))
            self.opt = optax.adam(lr)
            flat = {"phi": self.p_phi, "fwd": self.p_fwd}
            self.opt_state = self.opt.init(flat)

            def loss_fn(p, xt, a_oh, xtp1):
                phi_t = self.phi.apply(p["phi"], xt)
                tgt = jax.lax.stop_gradient(self.phi.apply(p["phi"], xtp1))
                pred = self.fwd.apply(p["fwd"], jnp.concatenate([phi_t, a_oh], 1))
                per = ((pred - tgt) ** 2).mean(axis=1)
                return per.mean(), (per, tgt)

            @jax.jit
            def train(p_os, xt, a_oh, xtp1):
                p, os_ = p_os
                (loss, (per, tgt)), grads = jax.value_and_grad(
                    loss_fn, has_aux=True)(p, xt, a_oh, xtp1)
                upd, os_ = self.opt.update(grads, os_, p)
                p = optax.apply_updates(p, upd)
                return (p, os_), per, tgt
            self._train = train
        else:
            key, k1, k2 = jax.random.split(key, 3)
            self.tgt = RNDNet()
            self.pred = RNDNet()
            self.p_tgt = self.tgt.init(k1, dummy)
            self.p_pred = self.pred.init(k2, dummy)
            self.opt = optax.adam(lr)
            self.opt_state = self.opt.init(self.p_pred)

            def loss_fn(pp, pt, x):
                t = self.tgt.apply(pt, x)
                p = self.pred.apply(pp, x)
                per = ((p - t) ** 2).mean(axis=1)
                return per.mean(), (per, p)

            @jax.jit
            def train(pp_os, pt, x):
                pp, os_ = pp_os
                (loss, (per, emb)), grads = jax.value_and_grad(
                    loss_fn, has_aux=True)(pp, pt, x)
                upd, os_ = self.opt.update(grads, os_, pp)
                pp = optax.apply_updates(pp, upd)
                return (pp, os_), per, emb
            self._train = train
            self.mem = [collections.deque(maxlen=1000) for _ in range(n_envs)]
        self.prev_phi = None

    def reset(self, obs_u8):
        x = obs_u8.astype(jnp.float32) / 255.0
        if self.kind == "icm":
            self.prev_phi = np.asarray(self.phi.apply(self.p_phi, x))
        else:
            for m in self.mem:
                m.clear()

    def step(self, obs_t_u8, act, rew, obs_tp1_u8, done):
        """Bonus + treino online; devolve recompensas aumentadas (numpy)."""
        xt = obs_t_u8.astype(jnp.float32) / 255.0
        xtp1 = obs_tp1_u8.astype(jnp.float32) / 255.0
        if self.kind == "icm":
            a_oh = jax.nn.one_hot(jnp.asarray(act), 15)
            (p, self.opt_state), per, tgt = self._train(
                ({"phi": self.p_phi, "fwd": self.p_fwd}, self.opt_state),
                xt, a_oh, xtp1)
            self.p_phi, self.p_fwd = p["phi"], p["fwd"]
            self.prev_phi = np.asarray(tgt)
            bonus = np.asarray(per)
        else:
            (self.p_pred, self.opt_state), per, emb = self._train(
                (self.p_pred, self.opt_state), self.p_tgt, xtp1)
            rnd = np.asarray(per)
            if self.kind == "rnd":
                bonus = rnd
            else:
                emb_n = np.asarray(emb)
                ep = np.empty(self.n, np.float32)
                for i in range(self.n):
                    m = self.mem[i]
                    if len(m) > 10:
                        d = np.linalg.norm(
                            np.stack(list(m)[-100:]) - emb_n[i], axis=1)
                        ep[i] = float(np.mean(np.sort(d)[:5]))
                    else:
                        ep[i] = 1.0
                    m.append(emb_n[i].copy())
                bonus = rnd * ep
        return np.asarray(rew, np.float32) + self.beta * bonus.astype(np.float32)
