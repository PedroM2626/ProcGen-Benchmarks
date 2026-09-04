"""Bench PA1: throughput do pipeline ProcGen CPU -> JAX GPU.

Mede, por contagem de envs, (a) FPS so-env (acoes aleatorias, autoreset) e
(b) FPS env+transferencia+preprocessamento JIT (uint8 CHW -> float32 /255
no device cuda:0). Baseline legado do estudo SB3: ~300 FPS (secao 1.4).

Uso (no venv /root/procgen-jax, via WSL):
    /root/procgen-jax/bin/python jax_port/bench_throughput.py \
        --game coinrun --num-envs 1 4 16 --steps 3000 --seed 42
Saida: tabela stdout + JSON em jax_port/pa1_throughput.json.
"""

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from jax_port.vector_env import ProcgenVectorEnv


def preprocess_jit():
    @jax.jit
    def f(batch):
        return batch.astype(jnp.float32) / 255.0
    return f


def bench_env_only(game, num_envs, steps, seed):
    venv = ProcgenVectorEnv(game=game, num_envs=num_envs, seed=seed)
    rng = np.random.default_rng(seed)
    venv.reset()
    t0 = time.perf_counter()
    for _ in range(steps):
        venv.step(venv.sample_actions(rng))
    dt = time.perf_counter() - t0
    return (num_envs * steps) / dt


def bench_env_to_gpu(game, num_envs, steps, seed):
    venv = ProcgenVectorEnv(game=game, num_envs=num_envs, seed=seed)
    rng = np.random.default_rng(seed)
    f = preprocess_jit()
    obs = venv.reset()
    d = jax.devices()[0]
    f(jnp.asarray(obs, device=d)).block_until_ready()  # warmup / compile
    t0 = time.perf_counter()
    for _ in range(steps):
        obs, _, _ = venv.step(venv.sample_actions(rng))
        f(jnp.asarray(obs, device=d)).block_until_ready()
    dt = time.perf_counter() - t0
    return (num_envs * steps) / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="coinrun")
    ap.add_argument("--num-envs", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="jax_port/pa1_throughput.json")
    args = ap.parse_args()

    print(f"jax={jax.__version__} devices={jax.devices()}")
    rows = []
    for n in args.num_envs:
        fps_env = bench_env_only(args.game, n, args.steps, args.seed)
        fps_full = bench_env_to_gpu(args.game, n, args.steps, args.seed)
        rows.append({"game": args.game, "num_envs": n, "steps": args.steps,
                     "fps_env_only": round(fps_env, 1),
                     "fps_env_to_gpu": round(fps_full, 1)})
        print(f"envs={n:3d}  env_only={fps_env:8.1f} FPS  env_to_gpu={fps_full:8.1f} FPS")
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
