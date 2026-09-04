"""Treino PPO em JAX sobre ProcGen real (velocidade do porte, PA2).

Protocolo fiel ao estudo onde importa, ajustado onde o throughput manda:
  fiel: jogo/niveis/sementes (treino 200 easy seed S; eval 0 unseen seed
    S+1000), frame 64x64x3, PPO lr 3e-4 / gamma 0.99 / lambda 0.95 /
    clip 0.2 / epochs 3 / vf 0.5 / ent 0.01 / grad-clip 0.5 / adv-norm.
  ajustado: N envs paralelos (C++ gym3, 1 no estudo) e minibatch >= 1024
    (64 no estudo) — mesmo objetivo PPO, batching GPU-eficiente.
    Micro-ajustes so de batching: vantagem normalizada 1x por epoca em
    numpy (8192 amostras) em vez de 1x por minibatch (64); layout NHWC
    (convencao Flax) em vez de CHW (convencao torch) — rede identica.
    Extratores: ``--extractor`` do zoo ``backbones.py`` (classic/cbam/
    spatial/impala/impoola/resnet18/vit/mlp); ``mlp`` usa obs vetor 256D
    (grayscale 16x16, ``ProcgenVectorWrapper`` do estudo, sem cv2:
    luminancia + mean-pool 4x = INTER_AREA exato).

Uso (venv /root/procgen-jax, via WSL):
    wsl -e env PYTHONPATH=/mnt/c/Users/Acer/Downloads/MLE \
      /root/procgen-jax/bin/python \
      /mnt/c/Users/Acer/Downloads/MLE/jax_port/train.py \
      --game coinrun --timesteps 100000 --seed 42 --num-envs 64
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from procgen import ProcgenGym3Env

from jax_port.augment import make_augment, requantize
from jax_port.backbones import BACKBONES
from jax_port.exploration import Exploration
from jax_port.networks import ActorCritic, preprocess
from jax_port.ppo import compute_gae, make_optimizer, make_update_fn


def to_vector(rgb):
    """(N,64,64,3) uint8 -> (N,256) uint8: luminancia + mean-pool 4x.

    Equivale a ``ProcgenVectorWrapper._to_vector`` (cv2 INTER_AREA com
    fator inteiro exato = media dos blocos 4x4).
    """
    gray = (rgb.astype(np.float32) * np.array([0.2989, 0.5870, 0.1140],
                                              np.float32)).sum(-1)
    small = gray.reshape(rgb.shape[0], 16, 4, 16, 4).mean(axis=(2, 4))
    return np.clip(np.rint(small), 0, 255).astype(np.uint8).reshape(
        rgb.shape[0], 256)


def get_obs(obs_d, mode):
    rgb = obs_d["rgb"] if isinstance(obs_d, dict) else obs_d
    return to_vector(rgb) if mode == "vector" else rgb


def train(args):
    jax.config.update("jax_compilation_cache_dir",
                      os.environ.get("JAX_PORT_CACHE", "/tmp/jax_port_cache"))
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    device = jax.devices()[0]
    mode = args.obs or ("vector" if args.extractor == "mlp" else "pixels")
    assert (args.extractor == "mlp") == (mode == "vector"), \
        "mlp<->vector, demais<->pixels"
    oshape = (256,) if mode == "vector" else (64, 64, 3)
    print(f"jax={jax.__version__} device={device} game={args.game} "
          f"extractor={args.extractor} obs={mode}")

    env = ProcgenGym3Env(num=args.num_envs, env_name=args.game,
                         num_levels=200, distribution_mode=args.distribution,
                         rand_seed=args.seed)
    n_actions = 15
    stoch = args.extractor == "vae"
    a2c = args.algo == "a2c"
    epochs = 1 if a2c else 3
    lam = 1.0 if a2c else 0.95  # defaults SB3: A2C 1.0, PPO 0.95 (estudo)
    model = ActorCritic(backbone=BACKBONES[args.extractor](), n_actions=n_actions,
                        stochastic=stoch)
    key, k0 = jax.random.split(key)
    params = model.init(k0, jnp.zeros((1,) + oshape, jnp.float32),
                        jax.random.PRNGKey(0) if stoch else None)
    opt = make_optimizer(lr=3e-4, kind="rmsprop" if a2c else "adam")
    opt_state = opt.init(params)
    update_fn, rollout_fn, forward_fn, norm_fn = make_update_fn(
        model, opt, clip_range=None if a2c else 0.2,
        ent_coef=0.0 if a2c else 0.01)
    state = (params, opt_state)
    N, T = args.num_envs, args.rollout

    # warmup do JIT fora da medida (update uint8 + rollout uint8)
    key, kw, ku = jax.random.split(key, 3)
    (state, _) = update_fn(
        state, jnp.zeros((args.minibatch,) + oshape, jnp.uint8),
        jnp.zeros((args.minibatch,), jnp.int32),
        jnp.zeros((args.minibatch,), jnp.float32),
        jnp.zeros((args.minibatch,), jnp.float32),
        jnp.zeros((args.minibatch,), jnp.float32), ku)
    _, _, _, key = rollout_fn(
        state[0], jnp.zeros((N,) + oshape, jnp.uint8), kw)
    jax.block_until_ready(jax.tree_util.tree_leaves(state[0])[0])

    _, obs_d, _ = env.observe()
    obs = get_obs(obs_d, mode)  # (N,64,64,3) uint8 NHWC ou (N,256) uint8
    aug = make_augment(args.augment)
    if args.augment != "none" or args.explore != "none":
        assert mode == "pixels", "augment/exploracao sao so-pixels (estudo)"
    exp = (Exploration(args.explore, N, seed=args.seed)
           if args.explore != "none" else None)
    if exp is not None:
        exp.reset(obs)
    n_iters = max(1, (args.timesteps + N * T - 1) // (N * T))
    done_steps = 0
    ep_rets, ep_lens, cur_ret, cur_len = [], [], np.zeros(N), np.zeros(N)
    curve = []
    t0 = time.perf_counter()
    ph = {"rollout": 0.0, "gae": 0.0, "update": 0.0}

    for it in range(n_iters):
        b_obs = np.empty((T, N) + oshape, np.uint8)
        b_act = np.empty((T, N), np.int32)
        b_rew = np.empty((T, N), np.float32)
        b_done = np.empty((T, N), bool)
        b_val = np.empty((T, N), np.float32)
        b_logp = np.empty((T, N), np.float32)
        t_roll0 = time.perf_counter()
        for t in range(T):
            pin = jnp.asarray(obs, device=device)
            if aug is not None:
                key, ka = jax.random.split(key)
                pin = requantize(aug(preprocess(pin), ka))
            act_d, logp_d, val_d, key = rollout_fn(state[0], pin, key)
            jax.block_until_ready((act_d, logp_d, val_d))
            act, logp, value = (np.asarray(act_d), np.asarray(logp_d),
                                np.asarray(val_d))
            b_obs[t], b_act[t] = obs, act
            b_val[t], b_logp[t] = value, logp
            env.act(act)
            rew, obs_d, first = env.observe()
            obs = get_obs(obs_d, mode)
            b_rew[t], b_done[t] = np.asarray(rew, np.float32), np.asarray(first)
            if exp is not None:
                b_rew[t] = exp.step(b_obs[t], act, b_rew[t], obs, b_done[t])
            cur_ret += b_rew[t]
            cur_len += 1
            for i in np.where(b_done[t])[0]:
                ep_rets.append(float(cur_ret[i]))
                ep_lens.append(int(cur_len[i]))
                cur_ret[i], cur_len[i] = 0.0, 0
        ph["rollout"] += time.perf_counter() - t_roll0
        t_gae0 = time.perf_counter()
        ob = preprocess(jnp.asarray(obs, device=device))
        key, kf = jax.random.split(key)
        _, last_v = forward_fn(state[0], ob, kf)
        adv, ret = compute_gae(b_rew, b_val, b_done, np.asarray(last_v),
                               lam=lam)
        ph["gae"] += time.perf_counter() - t_gae0
        t_upd0 = time.perf_counter()
        flat = lambda a: a.reshape(T * N, *a.shape[2:])
        F = (flat(b_obs), flat(b_act).astype(np.int32), flat(b_logp),
             flat(adv), flat(ret).astype(np.float32))
        idx = rng.permutation(T * N)
        # Slicing NO HOST + H2D contiguo por minibatch: gather com indice
        # no device media 457ms/call (medido) vs 12ms/call contiguo (~40x).
        adv_n = F[3]  # (8192,) float32 numpy
        adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        for _ep in range(epochs):
            for s in range(0, T * N, args.minibatch):
                mb = idx[s:s + args.minibatch]
                mo = jnp.asarray(F[0][mb], device=device)
                if aug is not None:
                    key, ka = jax.random.split(key)
                    mo = requantize(aug(preprocess(mo), ka))
                key, kz = jax.random.split(key)
                state, _ = update_fn(
                    state, mo,
                    jnp.asarray(F[1][mb], device=device),
                    jnp.asarray(F[2][mb], device=device),
                    jnp.asarray(adv_n[mb], device=device),
                    jnp.asarray(F[4][mb], device=device), kz)
        ph["update"] += time.perf_counter() - t_upd0
        done_steps += T * N
        if (it + 1) % 5 == 0 or done_steps >= args.timesteps:
            el = time.perf_counter() - t0
            mr = float(np.mean(ep_rets[-20:])) if ep_rets else 0.0
            curve.append({"steps": done_steps, "ret20": mr})
            print(f"iter={it+1} steps={done_steps} sps={done_steps/el:.0f} "
                  f"train_ret20={mr:.2f} eps={len(ep_rets)}", flush=True)
        if done_steps >= args.timesteps:
            break

    dt = time.perf_counter() - t0
    out = {"game": args.game, "seed": args.seed, "algo": args.algo,
           "extractor": args.extractor,
           "obs_mode": mode, "augment": args.augment, "explore": args.explore,
           "timesteps": done_steps,
           "wall_s": round(dt, 1), "sps": round(done_steps / dt, 1),
           "train_episodes": len(ep_rets),
           "train_ret_mean20": float(np.mean(ep_rets[-20:])) if ep_rets else 0.0,
           "curve": curve,
           "hparams": {"algo": args.algo, "lr": 3e-4, "gamma": 0.99,
                       "lambda": lam, "clip": None if a2c else 0.2,
                       "epochs": epochs, "num_envs": N, "rollout": T,
                       "minibatch": args.minibatch},
           "phase_s": {k: round(v, 2) for k, v in ph.items()}}
    if args.eval_eps > 0:
        out["eval_unseen"] = evaluate(state, forward_fn, args, device, mode,
                                      0, args.seed + 1000, False, args.eval_eps)
    if args.eval_det_eps > 0:
        out["eval_unseen_det"] = evaluate(
            state, forward_fn, args, device, mode, 0, args.seed + 1000,
            True, args.eval_det_eps)
    if args.eval_train_eps > 0:
        out["eval_train"] = evaluate(state, forward_fn, args, device, mode,
                                     200, args.seed, False, args.eval_train_eps)
    if "eval_train" in out and "eval_unseen" in out:
        out["gen_gap"] = round(out["eval_train"]["mean"]
                               - out["eval_unseen"]["mean"], 3)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return out


def evaluate(state, forward_fn, args, device, mode, num_levels, seed,
             deterministic, n_eps):
    """Eval generico: unseen (0, seed+1000) ou train (200, seed);
    deterministic=argmax, senao categorical (extrator VAE segue amostrado,
    como no estudo)."""
    ev = ProcgenGym3Env(num=args.eval_envs, env_name=args.game,
                        num_levels=num_levels,
                        distribution_mode=args.distribution, rand_seed=seed)
    _, obs_d, _ = ev.observe()
    obs = get_obs(obs_d, mode)
    key = jax.random.PRNGKey(seed)
    rets = np.zeros(args.eval_envs)
    all_rets = []
    key, ks = jax.random.split(key)
    while len(all_rets) < n_eps:
        ob = preprocess(jnp.asarray(obs, device=device))
        key, kf = jax.random.split(key)
        logits, _ = forward_fn(state[0], ob, kf)
        jax.block_until_ready(logits)
        if deterministic:
            act = np.asarray(jnp.argmax(logits, axis=1))
        else:
            key, ks = jax.random.split(key)
            act = np.asarray(jax.random.categorical(ks, logits))
        ev.act(act)
        rew, obs_d, first = ev.observe()
        obs = get_obs(obs_d, mode)
        rets += np.asarray(rew)
        for i in np.where(np.asarray(first))[0]:
            all_rets.append(float(rets[i]))
            rets[i] = 0.0
    sel = all_rets[:n_eps]
    return {"mean": round(float(np.mean(sel)), 3), "eps": len(sel)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="coinrun")
    ap.add_argument("--distribution", default="easy", choices=["easy", "hard"])
    ap.add_argument("--algo", default="ppo", choices=["ppo", "a2c"])
    ap.add_argument("--extractor", default="classic", choices=sorted(BACKBONES))
    ap.add_argument("--obs", default=None, choices=[None, "pixels", "vector"])
    ap.add_argument("--augment", default="none",
                    choices=["none", "crop", "color", "noise"])
    ap.add_argument("--explore", default="none",
                    choices=["none", "icm", "rnd", "ngu"])
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--eval-eps", type=int, default=10)
    ap.add_argument("--eval-det-eps", type=int, default=0)
    ap.add_argument("--eval-train-eps", type=int, default=0)
    ap.add_argument("--eval-envs", type=int, default=8)
    ap.add_argument("--out", default="jax_port/pa2_train.json")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
