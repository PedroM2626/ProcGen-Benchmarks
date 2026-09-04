"""Fast validation that the REAL trainers compile and take a gradient step.

Runs a single training step of PPO and of every HRL mode, plus one episodic
evaluation, on a tiny Craftax configuration. This is a smoke test, not a benchmark.
"""
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

from src.env import CraftaxLevelManager
from src.networks import SymbolicActorCritic
from src.ppo import PPOTrainer, create_train_state
from src.hrl import HRLTrainer
from src.eval_utils import make_craftax_evaluator


def smoke_test():
    print("Smoke test — trainers reais (1 step de gradiente cada)...", flush=True)
    t0 = time.time()
    env_manager = CraftaxLevelManager(use_pixels=False, num_train_levels=10)
    obs_sample, _ = env_manager.env.reset(jax.random.PRNGKey(0), env_manager.params)

    # PPO
    model = SymbolicActorCritic(action_dim=env_manager.num_actions)
    rng = jax.random.PRNGKey(42)
    rng, init_rng, run_rng = jax.random.split(rng, 3)
    ts = create_train_state(model, init_rng, obs_sample.shape)
    trainer = PPOTrainer(model=model, env_manager=env_manager, num_envs=16, num_steps=16)
    obs, env_state, run_rng = env_manager.reset_train(run_rng, 16)
    rs, m = jax.jit(trainer.train_step)((ts, env_state, obs, run_rng))
    print(f"[OK] PPO loss={float(m['loss']):.4f} rew={float(m['mean_reward']):.3f}", flush=True)

    # HRL (all modes) — real gradient step
    for mode in ["flat", "skip4", "hrl", "hrl_learned"]:
        ht = HRLTrainer(mode=mode, env_manager=env_manager, num_envs=16, num_steps=16, skip_k=4)
        hs = ht.create_state(init_rng, obs_sample.shape)
        obs, env_state, run_rng = env_manager.reset_train(run_rng, 16)
        carry, mm = jax.jit(ht.train_step)((hs, env_state, obs, run_rng))
        print(f"[OK] HRL {mode:<12} loss={float(mm['loss']):.4f} rew={float(mm['mean_reward']):.3f}", flush=True)

    # Episodic evaluator
    ev = make_craftax_evaluator(
        env_manager, lambda p, o, r: jnp.argmax(model.apply({'params': p}, o)[0], axis=-1),
        num_envs=8, horizon=50)
    un_mean, un_std = ev(ts.params, jax.random.PRNGKey(0), unseen=True)
    print(f"[OK] Episodic eval unseen={un_mean:.3f}±{un_std:.3f}", flush=True)

    print(f"\n[SUCESSO] Smoke test concluído em {time.time()-t0:.1f}s. Pipeline real operacional.", flush=True)


if __name__ == "__main__":
    smoke_test()
