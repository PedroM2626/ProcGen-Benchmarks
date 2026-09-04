import os
import sys
import time
import json
import argparse
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.55")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from src.env import CraftaxLevelManager
from src.networks import NatureCNN, ImpalaCNN
from src.ppo import PPOTrainer, create_train_state
from src.eval_utils import make_craftax_evaluator


def run_architecture_benchmark(total_steps=4_000_000, num_envs=64, seeds=(0, 1),
                               eval_episodes=64, eval_horizon=1000):
    print("=" * 78, flush=True)
    print("EXPERIMENTO 3 (REAL): ARQUITETURAS CONVOLUCIONAIS EM PIXELS (NatureCNN vs ImpalaCNN)", flush=True)
    print(f"Backend: {jax.default_backend().upper()} | Device: {jax.devices()[0]}", flush=True)
    print(f"total_steps={total_steps:,} | num_envs={num_envs} | seeds={list(seeds)}", flush=True)
    print("=" * 78, flush=True)

    results = {}
    env_manager = CraftaxLevelManager(use_pixels=True, num_train_levels=200, eval_seed_offset=1000)
    obs_sample, _ = env_manager.env.reset(jax.random.PRNGKey(0), env_manager.params)
    input_shape = obs_sample.shape
    print(f"Formato da observação de pixels: {input_shape}", flush=True)

    archs = {
        "NatureCNN": NatureCNN(action_dim=env_manager.num_actions),
        "ImpalaCNN": ImpalaCNN(action_dim=env_manager.num_actions, channel_sequence=(16, 32, 32)),
    }

    for name, model in archs.items():
        results[name] = []
        print(f"\n>>> Treinando {name}...", flush=True)
        for seed in seeds:
            rng = jax.random.PRNGKey(seed)
            rng, init_rng, run_rng = jax.random.split(rng, 3)
            train_state = create_train_state(model, init_rng, input_shape, learning_rate=3e-4)
            trainer = PPOTrainer(model=model, env_manager=env_manager, num_envs=num_envs, num_steps=64)
            obs, env_state, run_rng = env_manager.reset_train(run_rng, num_envs)
            runner_state = (train_state, env_state, obs, run_rng)
            step_fn = jax.jit(trainer.train_step)
            iters = max(1, total_steps // (num_envs * 64))

            t0 = time.time()
            for it in range(iters):
                runner_state, metrics = step_fn(runner_state)
                if it % 100 == 0 or it == iters - 1:
                    print(f"    [{name} s{seed}] it {it}/{iters} loss={float(metrics['loss']):.4f} "
                          f"rew={float(metrics['mean_reward']):.3f}", flush=True)
            elapsed = time.time() - t0
            fps = iters * num_envs * 64 / (elapsed + 1e-8)
            final_params = runner_state[0].params

            select = lambda p, o, r: jnp.argmax(model.apply({'params': p}, o)[0], axis=-1)
            ev = make_craftax_evaluator(env_manager, select, num_envs=eval_episodes, horizon=eval_horizon)
            e_rng = jax.random.PRNGKey(seed + 999)
            e_rng, r1, r2 = jax.random.split(e_rng, 3)
            tr, tr_s = ev(final_params, r1, unseen=False)
            un, un_s = ev(final_params, r2, unseen=True)

            results[name].append({
                "seed": seed, "steps": int(iters * num_envs * 64), "elapsed_sec": round(elapsed, 2),
                "fps": round(fps, 1), "train_score": round(tr, 3), "train_std": round(tr_s, 3),
                "unseen_score": round(un, 3), "unseen_std": round(un_s, 3), "gen_gap": round(tr - un, 3)})
            print(f"  [{name} s{seed}] FPS={fps:,.0f} Train={tr:.2f}±{tr_s:.2f} Unseen={un:.2f}±{un_s:.2f}", flush=True)
            _save(results)

    print(f"\nResultados reais salvos em results/architectures_results.json", flush=True)
    return results


def _save(results):
    out = Path("results/architectures_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4_000_000)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--eval-episodes", type=int, default=64)
    p.add_argument("--eval-horizon", type=int, default=1000)
    a = p.parse_args()
    run_architecture_benchmark(total_steps=a.steps, num_envs=a.num_envs, seeds=tuple(a.seeds),
                               eval_episodes=a.eval_episodes, eval_horizon=a.eval_horizon)
