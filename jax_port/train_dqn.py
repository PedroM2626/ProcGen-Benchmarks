"""Loop treino DQN/QR-DQN (estudo §12) — uso analogo a train.py.

Ex.: --algo qrdqn --game starpilot --timesteps 100000 --seed 42
Protocolo: treino 200 easy seed S; eval (argmax/det + eps-greedy/stoch)
em 0 unseen seed S+1000 — ver eval.py p/ o protocolo definitivo.
"""

import argparse
import copy
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from procgen import ProcgenGym3Env

from jax_port.backbones import BACKBONES
from jax_port.dqn import QNet, ReplayBuffer, make_dqn_update
import optax


def evaluate_dqn(params, greedy, args, device, num_levels, seed,
                 deterministic, n_eps, eps=0.05):
    """det=argmax; stoch=eps-greedy(eps) — mesmos niveis do estudo."""
    ev = ProcgenGym3Env(num=args.eval_envs, env_name=args.game,
                        num_levels=num_levels, distribution_mode="easy",
                        rand_seed=seed)
    _, obs_d, _ = ev.observe()
    obs = obs_d["rgb"] if isinstance(obs_d, dict) else obs_d
    rng = np.random.default_rng(seed)
    rets = np.zeros(args.eval_envs)
    all_rets = []
    while len(all_rets) < n_eps:
        if deterministic or rng.random() > eps:
            act = np.asarray(greedy(params, jnp.asarray(obs, device=device)))
        else:
            act = rng.integers(0, 15, size=args.eval_envs).astype(np.int32)
        ev.act(act)
        rew_d, obs_d, first_d = ev.observe()
        obs = obs_d["rgb"] if isinstance(obs_d, dict) else obs_d
        rets += np.asarray(rew_d)
        for i in np.where(np.asarray(first_d))[0]:
            all_rets.append(float(rets[i]))
            rets[i] = 0.0
    sel = all_rets[:n_eps]
    return {"mean": round(float(np.mean(sel)), 3), "eps": len(sel)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="coinrun")
    ap.add_argument("--algo", default="dqn", choices=["dqn", "qrdqn"])
    ap.add_argument("--extractor", default="classic", choices=sorted(BACKBONES))
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--eval-eps", type=int, default=10)
    ap.add_argument("--eval-det-eps", type=int, default=0)
    ap.add_argument("--eval-train-eps", type=int, default=0)
    ap.add_argument("--eval-envs", type=int, default=8)
    ap.add_argument("--out", default="jax_port/dqn_train.json")
    args = ap.parse_args()
    assert args.extractor != "mlp", "DQN do estudo usa CnnPolicy (pixels)"
    assert args.extractor != "vae", "DQN com extrator estocastico fora do estudo"
    train(args)

def train(args):
    jax.config.update("jax_compilation_cache_dir",
                      os.environ.get("JAX_PORT_CACHE", "/tmp/jax_port_cache"))
    rng = np.random.default_rng(args.seed)
    device = jax.devices()[0]
    N, Q = args.num_envs, (200 if args.algo == "qrdqn" else 0)
    print(f"jax={jax.__version__} device={device} {args.algo} {args.game}")

    env = ProcgenGym3Env(num=N, env_name=args.game, num_levels=200,
                         distribution_mode="easy", rand_seed=args.seed)
    net = QNet(backbone=BACKBONES[args.extractor](), n_actions=15, quantiles=Q)
    key = jax.random.PRNGKey(args.seed)
    params = net.init(key, jnp.zeros((1, 64, 64, 3), jnp.float32))
    tgt = copy.deepcopy(params)
    opt = optax.chain(optax.clip_by_global_norm(10.0),
                      optax.adam(args.lr, eps=1e-4))
    opt_state = opt.init(params)
    update, greedy = make_dqn_update(net, opt, quantiles=Q)
    # warmup
    update(params, opt_state, tgt, jnp.zeros((64, 64, 64, 3), jnp.uint8),
           jnp.zeros((64,), jnp.int32), jnp.zeros((64,)),
           jnp.zeros((64, 64, 64, 3), jnp.uint8), jnp.zeros((64,)))
    greedy(params, jnp.zeros((N, 64, 64, 3), jnp.uint8))
    jax.block_until_ready(jax.tree_util.tree_leaves(params)[0])

    buf = ReplayBuffer(100000, (64, 64, 3))
    _, obs_d, _ = env.observe()
    obs = obs_d["rgb"] if isinstance(obs_d, dict) else obs_d
    steps, grads, ep_rets, cur = 0, 0, [], np.zeros(N)
    curve = []
    # SB3: eps 1.0->0.05 linear nos primeiros 25% (exploration_fraction .25)
    eps_end = max(1, args.timesteps // 4)
    t0 = time.perf_counter()
    log_every = max(N, N * 20)
    while steps < args.timesteps:
        eps = max(0.05, 1.0 - 0.95 * steps / eps_end)
        if steps < 5000 or rng.random() < eps:
            act = rng.integers(0, 15, size=N).astype(np.int32)
        else:
            act = np.asarray(greedy(
                params, jnp.asarray(obs, device=device)))
        env.act(act)
        rew_d, obs_d2, first_d = env.observe()
        obs2 = obs_d2["rgb"] if isinstance(obs_d2, dict) else obs_d2
        rew, first = np.asarray(rew_d, np.float32), np.asarray(first_d)
        buf.add(obs, act, rew, obs2, first)
        cur += rew
        for i in np.where(first)[0]:
            ep_rets.append(float(cur[i]))
            cur[i] = 0.0
        obs, steps = obs2, steps + N
        if len(buf) >= 5000:
            # ~1 update por transicao/4 (train_freq 4, gradient_steps 1)
            for _ in range(max(1, N // 4)):
                bo, ba, br, bo2, bd = buf.sample(rng, 64)
                params, opt_state, _ = update(
                    params, opt_state, tgt,
                    jnp.asarray(bo, device=device),
                    jnp.asarray(ba, device=device),
                    jnp.asarray(br, device=device),
                    jnp.asarray(bo2, device=device),
                    jnp.asarray(bd, device=device))
                grads += 1
                if grads % 500 == 0:
                    tgt = copy.deepcopy(params)
        if steps % log_every == 0:
            el = time.perf_counter() - t0
            mr = float(np.mean(ep_rets[-20:])) if ep_rets else 0.0
            curve.append({"steps": steps, "ret20": mr})
            print(f"steps={steps} sps={steps/el:.0f} eps={eps:.2f} "
                  f"ret20={mr:.2f} buf={len(buf)}", flush=True)
    dt = time.perf_counter() - t0
    out = {"game": args.game, "seed": args.seed, "algo": args.algo,
           "extractor": args.extractor, "lr": args.lr, "timesteps": steps,
           "wall_s": round(dt, 1), "sps": round(steps / dt, 1),
           "train_episodes": len(ep_rets),
           "train_ret_mean20": float(np.mean(ep_rets[-20:])) if ep_rets else 0.0,
           "curve": curve}
    if args.eval_eps > 0:
        out["eval_unseen"] = evaluate_dqn(
            params, greedy, args, device, 0, args.seed + 1000, False,
            args.eval_eps)
    if args.eval_det_eps > 0:
        out["eval_unseen_det"] = evaluate_dqn(
            params, greedy, args, device, 0, args.seed + 1000, True,
            args.eval_det_eps)
    if args.eval_train_eps > 0:
        out["eval_train"] = evaluate_dqn(
            params, greedy, args, device, 200, args.seed, False,
            args.eval_train_eps)
    if "eval_train" in out and "eval_unseen" in out:
        out["gen_gap"] = round(out["eval_train"]["mean"]
                               - out["eval_unseen"]["mean"], 3)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
