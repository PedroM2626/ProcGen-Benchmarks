import time
import json
import jax
import jax.numpy as jnp
from pathlib import Path

from src.env import CraftaxLevelManager
from src.hrl import HRLTrainer


def run_hrl_benchmark(total_steps: int = 50000, num_envs: int = 64, seeds: list = [42, 123]):
    print("=" * 70)
    print("EXPERIMENTO 2: HIERARCHICAL RL & ABSTRAÇÃO TEMPORAL")
    print("=" * 70)

    results = {}
    env_manager = CraftaxLevelManager(use_pixels=False, num_train_levels=50, eval_seed_offset=1000)
    obs_sample, _ = env_manager.env.reset(jax.random.PRNGKey(0), env_manager.params)
    input_shape = obs_sample.shape

    modes = ["flat", "skip4", "hrl", "hrl_learned"]

    for mode in modes:
        results[mode] = []
        print(f"\n>>> Avaliando modo: {mode}")

        for seed in seeds:
            rng = jax.random.PRNGKey(seed)
            rng, init_rng, run_rng = jax.random.split(rng, 3)

            trainer = HRLTrainer(
                mode=mode,
                env_manager=env_manager,
                num_envs=num_envs,
                skip_k=4,
                learning_rate=3e-4
            )

            train_state = trainer.create_state(init_rng, input_shape)
            obs, env_state, run_rng = env_manager.reset_train(run_rng, num_envs)

            t0 = time.time()
            total_env_steps = 0
            accum_rewards = []

            step_fn = jax.jit(trainer.step_temporal_abstraction)
            num_iterations = total_steps // (num_envs * (4 if mode != 'flat' else 1))

            for it in range(num_iterations):
                obs, env_state, rew, done, act, val, logp, run_rng = step_fn(
                    train_state, env_state, obs, run_rng
                )
                steps_per_iter = num_envs * (4 if mode != 'flat' else 1)
                total_env_steps += steps_per_iter
                accum_rewards.append(float(rew.mean()))

            elapsed = time.time() - t0
            fps = total_env_steps / (elapsed + 1e-8)

            # Evaluation on Train & Unseen
            eval_rng = jax.random.PRNGKey(seed + 999)
            e_obs, e_state, eval_rng = env_manager.reset_train(eval_rng, num_envs)
            _, _, train_rew, _, _, _, _, _ = step_fn(train_state, e_state, e_obs, eval_rng)
            train_score = float(train_rew.mean())

            u_obs, u_state, eval_rng = env_manager.reset_unseen(eval_rng, num_envs)
            _, _, unseen_rew, _, _, _, _, _ = step_fn(train_state, u_state, u_obs, eval_rng)
            unseen_score = float(unseen_rew.mean())

            gen_gap = train_score - unseen_score
            run_data = {
                "seed": seed,
                "mode": mode,
                "steps": total_env_steps,
                "elapsed_sec": round(elapsed, 2),
                "fps": round(fps, 1),
                "train_score": round(train_score, 3),
                "unseen_score": round(unseen_score, 3),
                "gen_gap": round(gen_gap, 3)
            }
            results[mode].append(run_data)
            print(f"  Seed {seed}: Steps={total_env_steps} | Tempo={elapsed:.1f}s | FPS={fps:.0f} | Train={train_score:.2f} | Unseen={unseen_score:.2f} (Gap={gen_gap:+.2f})")

    out_file = Path("results/hrl_results.json")
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados salvos em: {out_file}")
    return results


if __name__ == "__main__":
    run_hrl_benchmark()
