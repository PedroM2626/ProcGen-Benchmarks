"""Bench pareado SB3 — protocolo justo (04/09/2026).

Config A (fiel ao estudo, ``compare_suite.py:19-26``): DummyVecEnv n=1 com
Monitor, CnnPolicy default (NatureCNN), lr 3e-4, n_steps 256, batch 64,
epochs 3, gamma 0.99, gae_lambda 0.95, clip 0.2, vf/ent/grad-clip defaults
SB3 (0.5/0.01/0.5), seed fixo, device cuda, SEM tensorboard/eval callbacks.
Config B (throughput): identica, exceto SubprocVecEnv n=64 + batch 1024
(mesmo ajuste de batching do porte JAX; objetivo PPO inalterado).

Medida: wall de ``model.learn()`` APENAS (construcao de envs/modelo, eval
e salvamento ficam de fora). SPS = model.num_timesteps / wall.
Uso (venv do estudo, Windows):
    & "C:/Users/Acer/AppData/Local/Programs/Python/Python310/python.exe" `
      bench_sb3_paired.py --vec dummy --n-envs 1 --batch-size 64 `
      --timesteps 100000 --seed 42 --game coinrun --out sb3_A.json
"""

import argparse
import json
import time

_CFG = {}


def _make_one(rank):
    from stable_baselines3.common.monitor import Monitor
    from procgen_wrapper import make_procgen_env
    return Monitor(make_procgen_env(
        _CFG["game"], num_levels=200, distribution_mode="easy",
        seed=_CFG["seed"] + rank, frame_stack=1, vector=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="coinrun")
    ap.add_argument("--vec", choices=["dummy", "subproc"], default="dummy")
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="sb3_bench.json")
    args = ap.parse_args()
    _CFG.update(game=args.game, seed=args.seed)

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
          f"vec={args.vec} n={args.n_envs} batch={args.batch_size}", flush=True)
    assert torch.cuda.is_available(), "bench justo exige torch CUDA (ver README)"

    if args.vec == "dummy":
        assert args.n_envs == 1
        vec = DummyVecEnv([lambda: _make_one(0)])
    else:
        vec = SubprocVecEnv([lambda r=i: _make_one(r) for i in range(args.n_envs)])
    model = PPO("CnnPolicy", vec, verbose=0, learning_rate=3e-4, n_steps=256,
                batch_size=args.batch_size, n_epochs=3, gamma=0.99,
                gae_lambda=0.95, clip_range=0.2, seed=args.seed,
                device=args.device)
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.timesteps)
    dt = time.perf_counter() - t0
    actual = int(model.num_timesteps)
    out = {"framework": "sb3", "game": args.game, "seed": args.seed,
           "vec": args.vec, "n_envs": args.n_envs,
           "batch_size": args.batch_size, "timesteps": actual,
           "wall_learn_s": round(dt, 1), "sps": round(actual / dt, 1),
           "torch": torch.__version__,
           "cuda": torch.cuda.is_available()}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    vec.close()


if __name__ == "__main__":
    main()
