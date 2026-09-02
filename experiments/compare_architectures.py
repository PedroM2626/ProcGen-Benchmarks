import time
import json
import jax
import jax.numpy as jnp
from pathlib import Path

from src.env import CraftaxLevelManager
from src.networks import NatureCNN, ImpalaCNN
from src.ppo import PPOTrainer, create_train_state


def run_architecture_benchmark(total_steps: int = 20000, num_envs: int = 16, seeds: list = [42]):
    print("=" * 70)
    print("EXPERIMENTO 3: COMPARAÇÃO DE ARQUITETURAS CONVOLUCIONAIS EM PIXELS")
    print("=" * 70)

    results = {}
    env_manager = CraftaxLevelManager(use_pixels=True, num_train_levels=20, eval_seed_offset=1000)
    obs_sample, _ = env_manager.env.reset(jax.random.PRNGKey(0), env_manager.params)
    input_shape = obs_sample.shape
    print(f"Formato da Observação de Pixels: {input_shape}")

    archs = {
        "NatureCNN": NatureCNN(action_dim=env_manager.num_actions),
        "ImpalaCNN": ImpalaCNN(action_dim=env_manager.num_actions, channel_sequence=(16, 32, 32))
    }

    for name, model in archs.items():
        results[name] = []
        print(f"\n>>> Avaliando arquitetura: {name}")

        for seed in seeds:
            rng = jax.random.PRNGKey(seed)
            rng, init_rng, run_rng = jax.random.split(rng, 3)

            train_state = create_train_state(model, init_rng, input_shape, learning_rate=3e-4)
            trainer = PPOTrainer(
                model=model,
                env_manager=env_manager,
                num_envs=num_envs,
                num_steps=32,
                learning_rate=3e-4
            )

            obs, env_state, run_rng = env_manager.reset_train(run_rng, num_envs)
            runner_state = (train_state, env_state, obs, run_rng)

            t0 = time.time()
            total_env_steps = 0

            train_step_fn = jax.jit(trainer.train_step)
            num_iterations = total_steps // (num_envs * 32)

            for it in range(num_iterations):
                runner_state, metrics = train_step_fn(runner_state)
                total_env_steps += num_envs * 32

            elapsed = time.time() - t0
            fps = total_env_steps / (elapsed + 1e-8)
            train_state = runner_state[0]

            # Eval
            eval_rng = jax.random.PRNGKey(seed + 999)
            e_obs, e_state, eval_rng = env_manager.reset_train(eval_rng, 16)
            logits, _ = model.apply({'params': train_state.params}, e_obs)
            _, _, r_train, _, _, _ = env_manager.step(eval_rng, e_state, jnp.argmax(logits, axis=-1))
            train_score = float(r_train.mean())

            u_obs, u_state, eval_rng = env_manager.reset_unseen(eval_rng, 16)
            logits_u, _ = model.apply({'params': train_state.params}, u_obs)
            _, _, r_unseen, _, _, _ = env_manager.step(eval_rng, u_state, jnp.argmax(logits_u, axis=-1))
            unseen_score = float(r_unseen.mean())

            gen_gap = train_score - unseen_score
            run_data = {
                "seed": seed,
                "steps": total_env_steps,
                "elapsed_sec": round(elapsed, 2),
                "fps": round(fps, 1),
                "train_score": round(train_score, 3),
                "unseen_score": round(unseen_score, 3),
                "gen_gap": round(gen_gap, 3)
            }
            results[name].append(run_data)
            print(f"  Seed {seed}: Steps={total_env_steps} | Tempo={elapsed:.1f}s | FPS={fps:.0f} | Train={train_score:.2f} | Unseen={unseen_score:.2f} (Gap={gen_gap:+.2f})")

    out_file = Path("results/architectures_results.json")
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados salvos em: {out_file}")
    return results


if __name__ == "__main__":
    run_architecture_benchmark()
