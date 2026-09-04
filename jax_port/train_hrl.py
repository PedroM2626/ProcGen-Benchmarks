"""HRL em JAX — paridade com ``compare_hrl.py`` + ``compare_hrl_learned.py`` (§11).

Quatro bracos, mesmo budget em FRAMES primitivos (100k), PPO identico:
  flat        : PPO sobre 15 acoes primitivas (DUR=1)
  skip4       : action-repeat 4 sobre 15 acoes (controle, sem hierarquia)
  hrl         : meta PPO sobre 6 skills FIXAS x 4 frames (SKILLS por jogo)
  hrl_learned : meta PPO (6 skills LATENTES) + low pi(a|obs,z) co-treinada
Skills fixas (``compare_hrl.py:30`` + mapeamento oficial procgen/env.py):
  jumper : [4,1,7,5,2,8]  (wait/left/right/jump/jump_left/jump_right)
  plunder: [4,1,7,9,0,6]  (wait/left/right/shoot/shoot_left/shoot_right)
Macro termina na borda do episodio (estudo ``MacroEnv.step``); autoreset
do gym3 devolve obs ja resetada — irrelevante p/ GAE (done mascara).
Eval definitivo: 100 stoch + 100 det unseen + 15 train (estudo).
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

from jax_port.backbones import BACKBONES
from jax_port.networks import ActorCritic, preprocess
from jax_port.ppo import compute_gae, make_optimizer, make_update_fn

DUR = 4
SKILLS = {
    "jumper": [4, 1, 7, 5, 2, 8],
    "plunder": [4, 1, 7, 9, 0, 6],
}


class MacroGym3:
    """N envs gym3 com macro-steps de ate DUR frames (mapa escolha->prim)."""

    def __init__(self, game, num_envs, num_levels, seed):
        self.env = ProcgenGym3Env(num=num_envs, env_name=game,
                                  num_levels=num_levels,
                                  distribution_mode="easy", rand_seed=seed)
        self.num_envs = num_envs
        _, d, _ = self.env.observe()
        self.obs = d["rgb"] if isinstance(d, dict) else d

    def step_macro(self, prims):
        """prims: (N,) acoes primitivas (ja mapeadas). Retorna
        (obs, rew_sum, done_any, frames_consumidos). Macro quebra por env
        na borda do episodio (estudo); frames contam todo step executado.
        """
        N = self.num_envs
        tot = np.zeros(N, np.float32)
        any_done = np.zeros(N, bool)
        active = np.ones(N, bool)
        rounds = 0
        for _ in range(DUR):
            if not active.any():
                break
            self.env.act(np.where(active, prims, 0))
            rew_d, d, first_d = self.env.observe()
            self.obs = d["rgb"] if isinstance(d, dict) else d
            rew, first = np.asarray(rew_d, np.float32), np.asarray(first_d)
            tot += np.where(active, rew, 0.0)
            any_done |= (active & first)
            active &= ~first
            rounds += 1
        return self.obs, tot, any_done, rounds * N


class LowActorCritic(nn.Module):
    """pi(a|obs,z): backbone pixels + one-hot skill -> Dense256 -> heads."""
    backbone: nn.Module
    n_actions: int = 15
    n_skills: int = 6

    @nn.compact
    def __call__(self, obs_f, z, key=None):
        del key
        feat = self.backbone(obs_f)
        h = jnp.concatenate([feat, jax.nn.one_hot(z, self.n_skills)], -1)
        h = nn.relu(nn.Dense(256)(h))
        return nn.Dense(self.n_actions)(h), nn.Dense(1)(h).squeeze(-1)


def make_low_update(model, optimizer):
    def loss_fn(params, obs_f, z, act, old_logp, adv, ret):
        logits, value = model.apply(params, obs_f, z)
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, act[:, None], 1).squeeze(1)
        ratio = jnp.exp(logp - old_logp)
        pg = -jnp.mean(jnp.minimum(
            ratio * adv, jnp.clip(ratio, 0.8, 1.2) * adv))
        v = jnp.mean(jnp.maximum((value - ret) ** 2,
                                 (ret + jnp.clip(value - ret, -0.2, 0.2)
                                  - ret) ** 2)) / 2.0
        ent = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * logp_all, 1))
        return pg + 0.5 * v - 0.01 * ent

    @jax.jit
    def update(state, obs_f, z, act, old_logp, adv, ret):
        p, os_ = state
        loss, grads = jax.value_and_grad(loss_fn)(p, obs_f, z, act, old_logp,
                                                  adv, ret)
        upd, os_ = optimizer.update(grads, os_, p)
        return (optax.apply_updates(p, upd), os_), loss

    @jax.jit
    def rollout(p, obs_f, z, key):
        key, ks = jax.random.split(key)
        logits, value = model.apply(p, obs_f, z)
        act = jax.random.categorical(ks, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(logits.shape[0]), act]
        return act, logp, value, key

    @jax.jit
    def forward(p, obs_f, z):
        return model.apply(p, obs_f, z)

    return update, rollout, forward


def train(args):
    jax.config.update("jax_compilation_cache_dir",
                      os.environ.get("JAX_PORT_CACHE", "/tmp/jax_port_cache"))
    assert args.game in SKILLS, "HRL do estudo: jumper/plunder"
    assert args.arm in ("flat", "skip4", "hrl", "hrl_learned")
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    device = jax.devices()[0]
    N, T, MB = args.num_envs, args.rollout, args.minibatch
    learned = args.arm == "hrl_learned"
    n_meta = 15 if args.arm in ("flat", "skip4") else 6
    prim_of = (lambda c: c) if args.arm in ("flat", "skip4", "hrl_learned") \
        else (lambda c: np.asarray(SKILLS[args.game])[np.asarray(c)])
    print(f"hrl game={args.game} arm={args.arm} frames={args.frames}")

    menv = MacroGym3(args.game, N, 200, args.seed)
    meta = ActorCritic(backbone=BACKBONES["classic"](), n_actions=n_meta)
    key, k0 = jax.random.split(key)
    mparams = meta.init(k0, jnp.zeros((1, 64, 64, 3), jnp.float32), None)
    mopt = make_optimizer()
    mstate = (mparams, mopt.init(mparams))
    mupd, mroll, mfwd, _ = make_update_fn(meta, mopt)
    key, kw = jax.random.split(key)
    mstate, _ = mupd(mstate, jnp.zeros((MB, 64, 64, 3), jnp.uint8),
                     jnp.zeros((MB,), jnp.int32), jnp.zeros((MB,)),
                     jnp.zeros((MB,)), jnp.zeros((MB,)), kw)
    mroll(mparams, jnp.zeros((N, 64, 64, 3), jnp.uint8), kw)
    if learned:
        low = LowActorCritic(backbone=BACKBONES["classic"]())
        key, k1 = jax.random.split(key)
        lparams = low.init(k1, jnp.zeros((1, 64, 64, 3), jnp.float32),
                           jnp.zeros((1,), jnp.int32))
        lopt = make_optimizer()
        lstate = (lparams, lopt.init(lparams))
        lupd, lroll, lfwd = make_low_update(low, lopt)
        key, kw2 = jax.random.split(key)
        lstate, _ = lupd(lstate, jnp.zeros((MB, 64, 64, 3), jnp.float32),
                         jnp.zeros((MB,), jnp.int32), jnp.zeros((MB,), jnp.int32),
                         jnp.zeros((MB,)), jnp.zeros((MB,)), jnp.zeros((MB,)))
    jax.block_until_ready(jax.tree_util.tree_leaves(mstate[0])[0])

    frames, it = 0, 0
    ep_rets, cur = [], np.zeros(N)
    curve = []
    t0 = time.perf_counter()
    while frames < args.frames:
        it += 1
        b_obs = np.empty((T, N, 64, 64, 3), np.uint8)
        b_act = np.empty((T, N), np.int32)
        b_rew = np.empty((T, N), np.float32)
        b_done = np.empty((T, N), bool)
        b_val = np.empty((T, N), np.float32)
        b_logp = np.empty((T, N), np.float32)
        if learned:
            L = []
        for t in range(T):
            if frames >= args.frames:
                break
            pin = jnp.asarray(menv.obs, device=device)
            act_d, logp_d, val_d, key = mroll(mstate[0], pin, key)
            jax.block_until_ready((act_d, logp_d, val_d))
            ch = np.asarray(act_d)
            b_obs[t], b_act[t] = menv.obs, ch
            b_val[t], b_logp[t] = np.asarray(val_d), np.asarray(logp_d)
            if learned:
                z = ch
                mrew = np.zeros(N, np.float32)
                mdone = np.zeros(N, bool)
                for _ in range(DUR):
                    of = preprocess(jnp.asarray(menv.obs, device=device))
                    a_d, lp_d, _, key = lroll(lstate[0], of,
                                             jnp.asarray(z, device=device), key)
                    jax.block_until_ready((a_d, lp_d))
                    a = np.asarray(a_d)
                    obs_before = np.asarray(menv.obs)
                    menv.env.act(a)
                    rew_d, d, first_d = menv.env.observe()
                    menv.obs = d["rgb"] if isinstance(d, dict) else d
                    r_, f_ = np.asarray(rew_d, np.float32), np.asarray(first_d)
                    L.append((obs_before, z.copy(), a, np.asarray(lp_d),
                              r_, f_))
                    mrew += r_
                    mdone |= f_
                    frames += N
                b_rew[t], b_done[t] = mrew, mdone
            else:
                obs, tot, dn, fr = menv.step_macro(prim_of(ch))
                b_rew[t], b_done[t] = tot, dn
                frames += int(fr)
            cur += b_rew[t]
            for i in np.where(b_done[t])[0]:
                ep_rets.append(float(cur[i]))
                cur[i] = 0.0
        if frames >= args.frames and t < T - 1:
            cut = t + 1
            b_obs, b_act = b_obs[:cut], b_act[:cut]
            b_rew, b_done, b_val, b_logp = (b_rew[:cut], b_done[:cut],
                                            b_val[:cut], b_logp[:cut])
            Tt = cut
        else:
            Tt = T if frames < args.frames else t + 1
        ob = preprocess(jnp.asarray(menv.obs, device=device))
        key, kf = jax.random.split(key)
        _, last_v = mfwd(mstate[0], ob, kf)
        adv, ret = compute_gae(b_rew[:Tt], b_val[:Tt], b_done[:Tt],
                               np.asarray(last_v))
        flat = lambda a: a.reshape(Tt * N, *a.shape[2:])
        F = (flat(b_obs[:Tt]), flat(b_act[:Tt]).astype(np.int32),
             flat(b_logp[:Tt]), flat(adv), flat(ret).astype(np.float32))
        adv_n = F[3]
        adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        idx = rng.permutation(Tt * N)
        for _ep in range(3):
            for s in range(0, Tt * N, MB):
                mb = idx[s:s + MB]
                key, kz = jax.random.split(key)
                mstate, _ = mupd(
                    mstate, jnp.asarray(F[0][mb], device=device),
                    jnp.asarray(F[1][mb], device=device),
                    jnp.asarray(F[2][mb], device=device),
                    jnp.asarray(adv_n[mb], device=device),
                    jnp.asarray(F[4][mb], device=device), kz)
        if learned and L:
            # update low-level sobre transicoes primitivas (obs,z,a,r)
            prim = [(r_[0], r_[1], r_[2], r_[3], r_[4], r_[5])
                    for r_ in L if len(r_) == 6][-N * T * DUR:]
            if prim:
                Po = np.stack([p[0] for p in prim]).reshape(-1, 64, 64, 3)
                Pz = np.concatenate([p[1] for p in prim]).reshape(-1)
                Pa = np.concatenate([p[2] for p in prim]).reshape(-1)
                Plp = np.concatenate([p[3] for p in prim]).reshape(-1)
                Pr = np.concatenate([p[4] for p in prim]).reshape(-1)
                Pd = np.concatenate([p[5] for p in prim]).reshape(-1)
                Pv = np.zeros_like(Pr)
                Pad, Prt = compute_gae(Pr.reshape(-1, 1), Pv.reshape(-1, 1),
                                       Pd.reshape(-1, 1), np.zeros(1))
                Pad = (Pad - Pad.mean()) / (Pad.std() + 1e-8)
                li = rng.permutation(len(Po))
                for s in range(0, len(Po), MB):
                    mb = li[s:s + MB]
                    lstate, _ = lupd(
                        lstate,
                        preprocess(jnp.asarray(Po[mb], device=device)),
                        jnp.asarray(Pz[mb], device=device),
                        jnp.asarray(Pa[mb], device=device),
                        jnp.asarray(Plp[mb], device=device),
                        jnp.asarray(Pad[mb].reshape(-1), device=device),
                        jnp.asarray(Prt[mb].reshape(-1), device=device))
        el = time.perf_counter() - t0
        mr = float(np.mean(ep_rets[-20:])) if ep_rets else 0.0
        curve.append({"frames": frames, "ret20": mr})
        print(f"iter={it} frames={frames} sps={frames/el:.0f} "
              f"ret20={mr:.2f} eps={len(ep_rets)}", flush=True)
    dt = time.perf_counter() - t0
    out = {"game": args.game, "arm": args.arm, "seed": args.seed,
           "frames": frames, "wall_s": round(dt, 1),
           "sps": round(frames / dt, 1),
           "train_episodes": len(ep_rets),
           "train_ret_mean20": float(np.mean(ep_rets[-20:])) if ep_rets else 0.0,
           "curve": curve}
    if args.eval_eps > 0:
        out["eval_unseen"] = evaluate_hrl(
            mstate, mfwd, lstate if learned else None,
            lfwd if learned else None, args, device, 0, args.seed + 1000,
            False, args.eval_eps)
    if args.eval_det_eps > 0:
        out["eval_unseen_det"] = evaluate_hrl(
            mstate, mfwd, lstate if learned else None,
            lfwd if learned else None, args, device, 0, args.seed + 1000,
            True, args.eval_det_eps)
    if args.eval_train_eps > 0:
        out["eval_train"] = evaluate_hrl(
            mstate, mfwd, lstate if learned else None,
            lfwd if learned else None, args, device, 200, args.seed,
            False, args.eval_train_eps)
    if "eval_train" in out and "eval_unseen" in out:
        out["gen_gap"] = round(out["eval_train"]["mean"]
                               - out["eval_unseen"]["mean"], 3)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return out


def evaluate_hrl(mstate, mfwd, lstate, lfwd, args, device, num_levels,
                 seed, deterministic, n_eps):
    """Eval macro: meta argmax/sample; learned usa low argmax/sample."""
    learned = lstate is not None
    table = np.asarray(SKILLS[args.game])
    menv = MacroGym3(args.game, args.eval_envs, num_levels, seed)
    key = jax.random.PRNGKey(seed)
    rets = np.zeros(args.eval_envs)
    all_rets = []
    while len(all_rets) < n_eps:
        ob = preprocess(jnp.asarray(menv.obs, device=device))
        key, kf = jax.random.split(key)
        logits, _ = mfwd(mstate[0], ob, kf)
        jax.block_until_ready(logits)
        if deterministic:
            ch = np.asarray(jnp.argmax(logits, axis=1))
        else:
            key, ks = jax.random.split(key)
            ch = np.asarray(jax.random.categorical(ks, logits))
        if learned:
            tot = np.zeros(args.eval_envs, np.float32)
            dn = np.zeros(args.eval_envs, bool)
            for _ in range(DUR):
                of = preprocess(jnp.asarray(menv.obs, device=device))
                ll, _ = lfwd(lstate[0], of, jnp.asarray(ch, device=device))
                jax.block_until_ready(ll)
                if deterministic:
                    a = np.asarray(jnp.argmax(ll, axis=1))
                else:
                    key, ks = jax.random.split(key)
                    a = np.asarray(jax.random.categorical(ks, ll))
                menv.env.act(a)
                rew_d, d, first_d = menv.env.observe()
                menv.obs = d["rgb"] if isinstance(d, dict) else d
                tot += np.asarray(rew_d, np.float32)
                dn |= np.asarray(first_d)
        elif args.arm == "hrl":
            _, tot, dn, _ = menv.step_macro(table[ch])
        else:
            _, tot, dn, _ = menv.step_macro(ch)
        rets += tot
        for i in np.where(dn)[0]:
            all_rets.append(float(rets[i]))
            rets[i] = 0.0
    sel = all_rets[:n_eps]
    return {"mean": round(float(np.mean(sel)), 3), "eps": len(sel)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="jumper", choices=["jumper", "plunder"])
    ap.add_argument("--arm", default="flat",
                    choices=["flat", "skip4", "hrl", "hrl_learned"])
    ap.add_argument("--frames", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--eval-eps", type=int, default=10)
    ap.add_argument("--eval-det-eps", type=int, default=0)
    ap.add_argument("--eval-train-eps", type=int, default=0)
    ap.add_argument("--eval-envs", type=int, default=8)
    ap.add_argument("--out", default="jax_port/hrl_train.json")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
