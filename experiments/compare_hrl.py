import os
import sys
import time
import json
import argparse
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp

from src.env import CraftaxLevelManager
from src.hrl import HRLTrainer
from src.eval_utils import make_craftax_evaluator


def run_hrl_benchmark(total_steps: int = 5_000_000, num_envs: int = 256, seeds=(0, 1, 2),
                      eval_episodes: int = 128, eval_horizon: int = 1000):
    """Train flat / skip4 / hrl / hrl_learned with REAL PPO gradients and evaluate
    genuine episodic returns on train and unseen levels."""
    print("=" * 78, flush=True)
    print("EXPERIMENTO 2 (REAL): HIERARCHICAL RL & ABSTRAÇÃO TEMPORAL", flush=True)
    print(f"Backend: {jax.default_backend().upper()} | Device: {jax.devices()[0]}", flush=True)
    print(f"total_steps={total_steps:,} | num_envs={num_envs} | seeds={list(seeds)}", flush=True)
    print("=" * 78, flush=True)

    env_manager = CraftaxLevelManager(use_pixels=False, num_train_levels=200, eval_seed_offset=1000)
    obs_sample, _ = env_manager.env.reset(jax.random.PRNGKey(0), env_manager.params)
    input_shape = obs_sample.shape

    modes = ["flat", "skip4", "hrl", "hrl_learned"]
    results = {}

    for mode in modes:
        results[mode] = []
        print(f"\n>>> Treinando modo: {mode}", flush=True)
        for seed in seeds:
            rng = jax.random.PRNGKey(seed)
            rng, init_rng, run_rng = jax.random.split(rng, 3)

            trainer = HRLTrainer(mode=mode, env_manager=env_manager, num_envs=num_envs,
                                 num_steps=64, skip_k=4, learning_rate=3e-4)
            train_state = trainer.create_state(init_rng, input_shape)
            obs, env_state, run_rng = env_manager.reset_train(run_rng, num_envs)
            runner_state = (train_state, env_state, obs, run_rng)

            step_fn = jax.jit(trainer.train_step)
            macro = num_envs * trainer.skip_k
            num_iterations = max(1, total_steps // (num_envs * 64 * trainer.skip_k))

            t0 = time.time()
            for it in range(num_iterations):
                runner_state, metrics = step_fn(runner_state)
                if it % 50 == 0 or it == num_iterations - 1:
                    print(f"    [{mode} seed{seed}] it {it}/{num_iterations} "
                          f"loss={float(metrics['loss']):.4f} rew={float(metrics['mean_reward']):.3f}", flush=True)
            elapsed = time.time() - t0
            final_state = runner_state[0]
            real_steps = num_iterations * num_envs * 64 * trainer.skip_k
            fps = real_steps / (elapsed + 1e-8)

            # REAL episodic evaluation
            eval_fn = make_craftax_evaluator(env_manager, trainer.make_eval_policy(deterministic=True),
                                             num_envs=eval_episodes, horizon=eval_horizon)
            e_rng = jax.random.PRNGKey(seed + 999)
            e_rng, r1, r2 = jax.random.split(e_rng, 3)
            train_mean, train_std = eval_fn(final_state.params, r1, unseen=False)
            unseen_mean, unseen_std = eval_fn(final_state.params, r2, unseen=True)

            run_data = {
                "seed": seed, "mode": mode, "steps": int(real_steps),
                "elapsed_sec": round(elapsed, 2), "fps": round(fps, 1),
                "train_score": round(train_mean, 3), "train_std": round(train_std, 3),
                "unseen_score": round(unseen_mean, 3), "unseen_std": round(unseen_std, 3),
                "gen_gap": round(train_mean - unseen_mean, 3),
            }
            results[mode].append(run_data)
            print(f"  [{mode} seed{seed}] steps={real_steps:,} tempo={elapsed:.1f}s FPS={fps:.0f} "
                  f"Train={train_mean:.2f}±{train_std:.2f} Unseen={unseen_mean:.2f}±{unseen_std:.2f}", flush=True)

            # Incremental save (crash-safe)
            out_file = Path("results/hrl_results.json")
            out_file.parent.mkdir(exist_ok=True)
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nResultados reais salvos em: results/hrl_results.json", flush=True)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval-episodes", type=int, default=128)
    p.add_argument("--eval-horizon", type=int, default=1000)
    a = p.parse_args()
    run_hrl_benchmark(total_steps=a.steps, num_envs=a.num_envs, seeds=tuple(a.seeds),
                      eval_episodes=a.eval_episodes, eval_horizon=a.eval_horizon)
