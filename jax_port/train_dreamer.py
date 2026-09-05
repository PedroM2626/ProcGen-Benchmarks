"""Dreamer completo em Flax: treino no mundo real, comportamento no sonho.

Loop: coleta real com o actor (N envs) -> buffer em anel uint8 ->
updates do world model em sequencias BxL -> imaginacao H=15 a partir
de posteriores -> updates actor/critic em lambda-returns -> repete.
Actor e critic: mesma arquitetura (ActorCriticLatent), pesos
independentes; critic-alvo em EMA (tau=0.02). Eval final + video
imaginado (decode de rollout latente).
Hparams: WM Adam 1e-4, actor/critic 8e-5, KL dyn 0.5/rep 0.1,
lambda 0.95, gamma 0.99, H=15, entropia 3e-4, buffer 100k.
Uso: train_dreamer.py --game coinrun --frames 1000000 --seed 42
"""

import argparse
import json
import os
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from procgen import ProcgenGym3Env

from jax_port.dreamer import (DET, STOCH, ActorCriticLatent, Decoders,
                              Encoder, RSSM)

SEQ_L = 64
SEQ_B = 16
IMAG_H = 15
IMAG_B = 32
LAM = 0.95
GAMMA = 0.99
ENT = 3e-4


def symlog(x):
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))
EMA_TAU = 0.02


def sub(params, *names):
    d = params["params"]
    for n in names:
        d = d[n]
    return {"params": d}


class WorldModel(nn.Module):
    n_actions: int = 15

    def setup(self):
        self.encoder = Encoder()
        self.rssm = RSSM(self.n_actions)
        self.decoder = Decoders()

    def __call__(self, obs_f, act, key):
        B, L = act.shape
        feats = self.encoder(obs_f[:, :-1].reshape(B * L, 64, 64, 3))
        feats = feats.reshape(B, L, -1)
        a0 = jnp.zeros((B,), jnp.int32)
        act_p = jnp.concatenate([a0[:, None], act[:, :-1]], axis=1)
        keys = jax.random.split(key, L)
        # loop Python desenrolado (L estatico): evita lax.scan sobre
        # chamadas a metodos de submodulo (vazamento de tracer no XLA).
        h, z = jnp.zeros((B, DET)), jnp.zeros((B, STOCH))
        posts, prs = [], []
        for t in range(L):
            post = self.rssm.observe({"h": h, "z": z}, act_p[:, t],
                                     feats[:, t], keys[t])
            pr = self.rssm.imagine({"h": h, "z": z}, act_p[:, t], keys[t])
            h, z = post["h"], post["z"]
            posts.append(post)
            prs.append(pr)
        posts = {k: jnp.stack([p[k] for p in posts]).transpose(1, 0, 2)
                 for k in posts[0]}
        prs = {k: jnp.stack([p[k] for p in prs]).transpose(1, 0, 2)
               for k in prs[0]}
        f = jnp.concatenate([posts["h"], posts["z"]], -1)
        o, r, c = self.decoder(f.reshape(B * L, DET + STOCH))
        S = (B, L)
        return posts, prs, o.reshape(S + (64, 64, 3)), r.reshape(S), c.reshape(S)


def kl_gauss(mu_q, lv_q, mu_p, lv_p):
    v_q, v_p = jnp.exp(lv_q), jnp.exp(lv_p)
    return 0.5 * ((v_q + (mu_q - mu_p) ** 2) / v_p - 1 + lv_p - lv_q).sum(-1)


def make_wm_update(wm, opt, reward_mode="raw", w_dyn=0.5, w_rep=0.1):
    @jax.jit
    def update(params, opt_state, obs_f, act, rew, cont, key):
        B, L = act.shape

        def loss_fn(p):
            posts, prs, o, r, c = wm.apply(p, obs_f, act, key)
            mu_q, lv_q = posts["mu"], posts["logvar"]
            mu_p, lv_p = prs["mu"], prs["logvar"]
            kl_rep = kl_gauss(mu_q, lv_q, jax.lax.stop_gradient(mu_p),
                              jax.lax.stop_gradient(lv_p)).mean()
            kl_dyn = kl_gauss(jax.lax.stop_gradient(mu_q),
                              jax.lax.stop_gradient(lv_q), mu_p, lv_p).mean()
            bce = -(obs_f[:, :-1] * jnp.log(o + 1e-7) +
                    (1 - obs_f[:, :-1]) * jnp.log(1 - o + 1e-7)).mean()
            rl = ((r - rew) ** 2).mean()
            if reward_mode == "symlog":
                rl = ((r - symlog(rew)) ** 2).mean()
            cl = -(cont * jnp.log(jax.nn.sigmoid(c) + 1e-7) +
                   (1 - cont) * jnp.log(1 - jax.nn.sigmoid(c) + 1e-7)).mean()
            return bce + rl + cl + w_rep * kl_rep + w_dyn * kl_dyn
        loss, grads = jax.value_and_grad(loss_fn)(params)
        upd, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss

    return update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="coinrun")
    ap.add_argument("--frames", type=int, default=1000000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--reward-mode", default="symlog", choices=["raw", "symlog"])
    ap.add_argument("--ent-coef", type=float, default=3e-4)
    ap.add_argument("--kl-dyn", type=float, default=0.5)
    ap.add_argument("--kl-rep", type=float, default=0.1)
    ap.add_argument("--out", default="jax_port/dreamer_run.json")
    ap.add_argument("--out-dir", default="jax_port/dreams")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir",
                      os.environ.get("JAX_PORT_CACHE", "/tmp/jax_port_cache"))
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    device = jax.devices()[0]
    N = args.num_envs
    print(f"dreamer {args.game} device={device}", flush=True)

    wm = WorldModel()
    actor = ActorCriticLatent()
    key, k0, k1, k2 = jax.random.split(key, 4)
    wparams = wm.init(k0, jnp.zeros((1, SEQ_L + 1, 64, 64, 3), jnp.float32),
                      jnp.zeros((1, SEQ_L), jnp.int32), k0)
    aparams = actor.init(k1, jnp.zeros((1, DET + STOCH)))
    cparams = actor.init(k2, jnp.zeros((1, DET + STOCH)))
    ctgt = cparams
    wopt = optax.chain(optax.clip_by_global_norm(100.0), optax.adam(1e-4))
    aopt = optax.adam(8e-5)
    copt = optax.adam(8e-5)
    wstate = (wparams, wopt.init(wparams))
    astate = (aparams, aopt.init(aparams))
    cstate = (cparams, copt.init(cparams))
    wm_upd = make_wm_update(wm, wopt, reward_mode=args.reward_mode,
                            w_dyn=args.kl_dyn, w_rep=args.kl_rep)

    @jax.jit
    def ac_train(a_s, c_s, ct, rssm_p, dec_p, f0, key, ent_coef):
        keys = jax.random.split(key, IMAG_H)
        # loop Python desenrolado (idem WorldModel): sem lax.scan sobre
        # chamadas Flax.
        h0 = f0[:, :DET]
        z0 = f0[:, DET:]
        s, kk = {"h": h0, "z": z0}, keys[0]
        Rs, Cs, Ls, As, Fs = [], [], [], [], []
        for t in range(IMAG_H):
            f_ = jnp.concatenate([s["h"], s["z"]], -1)
            logits, _ = actor.apply(a_s[0], f_)
            k1, k2 = jax.random.split(keys[t])
            act = jax.random.categorical(k1, logits)
            ns = RSSM().apply(rssm_p, s, act, k2, method="imagine")
            fns = jnp.concatenate([ns["h"], ns["z"]], -1)
            o, r, c_ = Decoders().apply(dec_p, fns)
            del o
            Rs.append(r)
            Cs.append(jax.nn.sigmoid(c_))
            Ls.append(logits)
            As.append(act)
            Fs.append(fns)
            s, kk = ns, k2
        rews = jnp.stack(Rs)
        conts = jnp.stack(Cs)
        logits = jnp.stack(Ls)
        acts = jnp.stack(As)
        fns = jnp.stack(Fs)

        def losses(ap, cp):
            lg, _ = actor.apply(ap, fns)
            _, vv = actor.apply(cp, fns)
            _, tv = actor.apply(ct, fns)
            logp_all = jax.nn.log_softmax(lg)
            logp = jnp.take_along_axis(logp_all, acts[..., None], -1).squeeze(-1)
            ent = -jnp.mean(jnp.sum(jax.nn.softmax(lg) * logp_all, -1))

            def lam_ret(carry, inp):
                r, c_, v, tv_ = inp
                nxt = carry
                cur = r + GAMMA * c_ * ((1 - LAM) * tv_ + LAM * nxt)
                return cur, cur
            _, rets = jax.lax.scan(lam_ret, tv[-1],
                                   (rews, conts, vv, tv), reverse=True)
            al = -(rets * jnp.exp(logp - jax.lax.stop_gradient(logp))).mean() \
                - ent_coef * ent
            cl = ((vv - jax.lax.stop_gradient(rets)) ** 2).mean()
            return al + cl, (al, cl)

        (tot, (al, cl)), (ag, cg) = jax.value_and_grad(
            lambda a, c: losses(a, c), argnums=(0, 1),
            has_aux=True)(a_s[0], c_s[0])
        ua, n_aos = aopt.update(ag, a_s[1], a_s[0])
        uc, n_cos = copt.update(cg, c_s[1], c_s[0])
        a_s = (optax.apply_updates(a_s[0], ua), n_aos)
        c_s = (optax.apply_updates(c_s[0], uc), n_cos)
        return a_s, c_s, tot, al, cl

    # --- buffer em anel uint8 ---
    CAP = 100000
    b_o = np.empty((CAP, 64, 64, 3), np.uint8)
    b_a = np.empty(CAP, np.int32)
    b_r = np.empty(CAP, np.float32)
    b_d = np.empty(CAP, bool)
    bi, bfull = 0, False

    def buf_add(o, a, r, d):
        nonlocal bi, bfull
        n = o.shape[0]
        for k in range(n):
            j = (bi + k) % CAP
            b_o[j], b_a[j], b_r[j], b_d[j] = o[k], a[k], r[k], d[k]
        bi = (bi + n) % CAP
        bfull = bfull or (bi == 0)

    def buf_seq(n_seq, L):
        lim = CAP if bfull else bi
        s0 = rng.integers(0, max(2, lim - L - 1), size=n_seq)
        oo = np.stack([b_o[(s + np.arange(L + 1)) % CAP] for s in s0])
        aa = np.stack([b_a[(s + np.arange(L)) % CAP] for s in s0])
        rr = np.stack([b_r[(s + np.arange(L)) % CAP] for s in s0]).astype(np.float32)
        dd = np.stack([b_d[(s + np.arange(L)) % CAP] for s in s0])
        return oo, aa, rr, dd

    # --- warmup fora da medida ---
    key, kw = jax.random.split(key)
    wparams_w, wopt_w, _ = wm_upd(
        wstate[0], wstate[1],
        jnp.zeros((2, SEQ_L + 1, 64, 64, 3), jnp.float32),
        jnp.zeros((2, SEQ_L), jnp.int32), jnp.zeros((2, SEQ_L)),
        jnp.ones((2, SEQ_L)), kw)
    jax.block_until_ready(jax.tree_util.tree_leaves(wparams_w)[0])
    wstate = (wparams_w, wopt_w)

    env = ProcgenGym3Env(num=N, env_name=args.game, num_levels=200,
                         distribution_mode="easy", rand_seed=args.seed)
    _, d, _ = env.observe()
    obs = d["rgb"] if isinstance(d, dict) else d
    frames, it = 0, 0
    ep_rets, cur = [], np.zeros(N)
    curve, t0 = [], time.perf_counter()
    CYCLE = 2048
    # --- coleta real com o actor (estado recorrente carregado) ---
    carry_h = jnp.zeros((N, DET))
    carry_z = jnp.zeros((N, STOCH))
    carry_a = jnp.zeros((N,), jnp.int32)

    @jax.jit
    def act_step(wp, ap, obs_f, h, z, a_prev, key):
        feat = Encoder().apply({"params": wp["params"]["encoder"]}, obs_f)
        post = RSSM().apply({"params": wp["params"]["rssm"]},
                            {"h": h, "z": z}, a_prev, feat, key,
                            method="observe")
        f_ = jnp.concatenate([post["h"], post["z"]], -1)
        logits, _ = actor.apply(ap, f_)
        return post, logits

    @jax.jit
    def post0(wp, obs_f, key):
        feat = Encoder().apply({"params": wp["params"]["encoder"]}, obs_f)
        n = obs_f.shape[0]
        post = RSSM().apply(
            {"params": wp["params"]["rssm"]},
            {"h": jnp.zeros((n, DET)), "z": jnp.zeros((n, STOCH))},
            jnp.zeros((n,), jnp.int32), feat, key, method="observe")
        return jnp.concatenate([post["h"], post["z"]], -1)

    while frames < args.frames:
        it += 1
        # 1. coleta real com o actor
        for _ in range(CYCLE // N):
            ob = jnp.asarray(obs, device=device).astype(jnp.float32) / 255.0
            key, kf = jax.random.split(key)
            post, logits = act_step(wparams_w, astate[0], ob, carry_h,
                                    carry_z, carry_a, kf)
            jax.block_until_ready((post["h"], logits))
            carry_h, carry_z = np.asarray(post["h"]), np.asarray(post["z"])
            key, ks = jax.random.split(key)
            act = np.asarray(jax.random.categorical(ks, np.asarray(logits)))
            carry_a = act.copy()
            env.act(act)
            rew_d, d, first_d = env.observe()
            obs2 = d["rgb"] if isinstance(d, dict) else d
            buf_add(obs, act, np.asarray(rew_d, np.float32),
                    np.asarray(first_d))
            fin = np.asarray(first_d)
            carry_h = np.where(fin[:, None], 0.0, carry_h)
            carry_z = np.where(fin[:, None], 0.0, carry_z)
            carry_a = np.where(fin, 0, carry_a)
            cur += np.asarray(rew_d, np.float32)
            for i in np.where(fin)[0]:
                ep_rets.append(float(cur[i]))
                cur[i] = 0.0
            obs = obs2
            frames += N
            if frames >= args.frames:
                break
        # 2. updates WM em sequencias
        for _ in range(4):
            oo, aa, rr, dd = buf_seq(SEQ_B, SEQ_L)
            key, kwm = jax.random.split(key)
            wparams_w, wopt_w, _ = wm_upd(
                wparams_w, wopt_w,
                jnp.asarray(oo, device=device).astype(jnp.float32) / 255.0,
                jnp.asarray(aa, device=device),
                jnp.asarray(rr, device=device),
                (1 - jnp.asarray(dd, device=device)).astype(jnp.float32), kwm)
        wstate = (wparams_w, wopt_w)
        # 3. imaginacao + actor/critic (f0 = posterior de obs reais)
        for _ in range(4):
            oo, _, _, _ = buf_seq(IMAG_B, 1)
            key, kf2, kac = jax.random.split(key, 3)
            f0 = post0(wparams_w,
                       jnp.asarray(oo[:, 0], device=device).astype(
                           jnp.float32) / 255.0, kf2)
            jax.block_until_ready(f0)
            rssm_p = {"params": wparams_w["params"]["rssm"]}
            dec_p = {"params": wparams_w["params"]["decoder"]}
            astate, cstate, tot, al, cl = ac_train(
                astate, cstate, ctgt, rssm_p, dec_p, f0, kac, args.ent_coef)
            ctgt = jax.tree.map(
                lambda t, o: (1 - EMA_TAU) * t + EMA_TAU * o, ctgt,
                cstate[0])
        el = time.perf_counter() - t0
        mr = float(np.mean(ep_rets[-20:])) if ep_rets else 0.0
        curve.append({"frames": frames, "ret20": mr})
        print(f"iter={it} frames={frames} sps={frames/el:.0f} ret20={mr:.2f} "
              f"eps={len(ep_rets)}", flush=True)
    dt = time.perf_counter() - t0
    out = {"game": args.game, "seed": args.seed, "algo": "dreamer",
           "reward_mode": args.reward_mode, "ent_coef": args.ent_coef,
           "kl_dyn": args.kl_dyn, "kl_rep": args.kl_rep,
           "frames": frames, "wall_s": round(dt, 1),
           "sps": round(frames / dt, 1),
           "train_episodes": len(ep_rets),
           "train_ret_mean20": float(np.mean(ep_rets[-20:])) if ep_rets else 0.0,
           "curve": curve}
    # video imaginado: actor age no sonho, decoder mostra os frames
    oo, _, _, _ = buf_seq(8, 1)
    key, kf3 = jax.random.split(key)
    f0 = post0(wparams_w, jnp.asarray(oo[:, 0], device=device).astype(
        jnp.float32) / 255.0, kf3)
    jax.block_until_ready(f0)
    h0, z0 = np.asarray(f0[:, :DET]), np.asarray(f0[:, DET:])
    ims, s = [], {"h": jnp.asarray(h0), "z": jnp.asarray(z0)}
    for _ in range(30):
        key, kv = jax.random.split(key)
        f_ = jnp.concatenate([s["h"], s["z"]], -1)
        logits, _ = actor.apply(astate[0], f_)
        key, ks = jax.random.split(key)
        acts_v = np.asarray(jax.random.categorical(ks, logits))
        jax.block_until_ready(acts_v)
        ns = RSSM().apply({"params": wparams_w["params"]["rssm"]}, s,
                          jnp.asarray(acts_v), kv, method="imagine")
        o, _, _ = Decoders().apply(
            {"params": wparams_w["params"]["decoder"]},
            jnp.concatenate([ns["h"], ns["z"]], -1))
        ims.append(np.asarray((o[0] * 255).astype(np.uint8)))
        s = ns
    import imageio.v2 as imageio
    imageio.mimsave(os.path.join(args.out_dir, "dreamer_imagined.gif"), ims,
                    fps=10, loop=0)
    out["imagined_video"] = "dreamer_imagined.gif"
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "curve"},
                     indent=2))


if __name__ == "__main__":
    main()
