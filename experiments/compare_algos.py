import time
import json
import jax
import jax.numpy as jnp
from pathlib import Path

from src.env import CraftaxLevelManager
from src.networks import SymbolicActorCritic, QNetwork
from src.ppo import PPOTrainer, create_train_state
from src.dqn import DQNTrainer


def run_algo_benchmark(total_steps: int = 50000, num_envs: int = 64, seeds: list = [42, 123]):
    print("=" * 70)
    print("EXPERIMENTO 1: COMPARAÇÃO DE FAMÍLIAS (PPO vs A2C vs DQN)")
    print("=" * 70)
    
    results = {}
    env_manager = CraftaxLevelManager(use_pixels=False, num_train_levels=50, eval_seed_offset=1000)
    obs_sample, _ = env_manager.env.reset(jax.random.PRNGKey(0), env_manager.params)
    input_shape = obs_sample.shape

    algos = ["PPO", "A2C", "DQN"]

    for algo in algos:
        results[algo] = []
        print(f"\n>>> Iniciando benchmark do algoritmo: {algo}")
        
        for seed in seeds:
            rng = jax.random.PRNGKey(seed)
            rng, init_rng, run_rng = jax.random.split(rng, 3)
            
            t0 = time.time()
            total_env_steps = 0

            if algo in ["PPO", "A2C"]:
                is_a2c = (algo == "A2C")
                model = SymbolicActorCritic(action_dim=env_manager.num_actions)
                train_state = create_train_state(model, init_rng, input_shape, learning_rate=3e-4)
                
                trainer = PPOTrainer(
                    model=model,
                    env_manager=env_manager,
                    num_envs=num_envs,
                    num_steps=64,
                    learning_rate=3e-4,
                    is_a2c=is_a2c
                )
                
                obs, env_state, run_rng = env_manager.reset_train(run_rng, num_envs)
                runner_state = (train_state, env_state, obs, run_rng)

                num_iterations = total_steps // (num_envs * 64)
                
                # JIT train step
                train_step_fn = jax.jit(trainer.train_step)

                final_metrics = None
                for it in range(num_iterations):
                    runner_state, metrics = train_step_fn(runner_state)
                    total_env_steps += num_envs * 64
                    final_metrics = metrics

                elapsed = time.time() - t0
                fps = total_env_steps / (elapsed + 1e-8)
                train_state = runner_state[0]

                # EVALUATION (50 eps train, 50 eps unseen)
                eval_rng = jax.random.PRNGKey(seed + 999)
                
                # Train eval
                e_obs, e_state, eval_rng = env_manager.reset_train(eval_rng, 50)
                logits, _ = model.apply({'params': train_state.params}, e_obs)
                act_stoch = jax.random.categorical(eval_rng, logits)
                _, _, r_train, _, _, _ = env_manager.step(eval_rng, e_state, act_stoch)
                train_score = float(r_train.mean())

                # Unseen eval (stochastic)
                u_obs, u_state, eval_rng = env_manager.reset_unseen(eval_rng, 50)
                logits_u, _ = model.apply({'params': train_state.params}, u_obs)
                eval_rng, sub_eval = jax.random.split(eval_rng)
                act_unseen = jax.random.categorical(sub_eval, logits_u)
                _, _, r_unseen, _, _, _ = env_manager.step(eval_rng, u_state, act_unseen)
                unseen_score = float(r_unseen.mean())

                # Unseen eval (deterministic)
                act_det = jnp.argmax(logits_u, axis=-1)
                _, _, r_det, _, _, _ = env_manager.step(eval_rng, u_state, act_det)
                det_score = float(r_det.mean())

            else:  # DQN
                model = QNetwork(action_dim=env_manager.num_actions)
                trainer = DQNTrainer(
                    model=model,
                    env_manager=env_manager,
                    num_envs=num_envs,
                    buffer_size=20000,
                    learning_rate=1e-4
                )
                t_state, tgt_params, buf = trainer.create_state(init_rng, input_shape)
                obs, env_state, run_rng = env_manager.reset_train(run_rng, num_envs)
                runner_state = (t_state, tgt_params, buf, env_state, obs, run_rng)

                num_iterations = total_steps // num_envs
                train_step_fn = jax.jit(trainer.train_step)

                for it in range(num_iterations):
                    runner_state, metrics = train_step_fn(it, runner_state)
                    total_env_steps += num_envs

                elapsed = time.time() - t0
                fps = total_env_steps / (elapsed + 1e-8)
                t_state = runner_state[0]

                # Evaluation
                eval_rng = jax.random.PRNGKey(seed + 999)
                e_obs, e_state, eval_rng = env_manager.reset_train(eval_rng, 50)
                q_train = model.apply({'params': t_state.params}, e_obs)
                _, _, r_train, _, _, _ = env_manager.step(eval_rng, e_state, jnp.argmax(q_train, axis=-1))
                train_score = float(r_train.mean())

                u_obs, u_state, eval_rng = env_manager.reset_unseen(eval_rng, 50)
                q_unseen = model.apply({'params': t_state.params}, u_obs)
                _, _, r_unseen, _, _, _ = env_manager.step(eval_rng, u_state, jnp.argmax(q_unseen, axis=-1))
                unseen_score = float(r_unseen.mean())
                det_score = unseen_score

            gen_gap = train_score - unseen_score
            run_data = {
                "seed": seed,
                "steps": total_env_steps,
                "elapsed_sec": round(elapsed, 2),
                "fps": round(fps, 1),
                "train_score": round(train_score, 3),
                "unseen_score": round(unseen_score, 3),
                "det_score": round(det_score, 3),
                "gen_gap": round(gen_gap, 3)
            }
            results[algo].append(run_data)
            print(f"  Seed {seed}: Steps={total_env_steps} | Tempo={elapsed:.1f}s | FPS={fps:.0f} | Train={train_score:.2f} | Unseen={unseen_score:.2f} (Gap={gen_gap:+.2f})")

    out_file = Path("results/algo_families_results.json")
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados salvos em: {out_file}")
    return results


if __name__ == "__main__":
    run_algo_benchmark()
