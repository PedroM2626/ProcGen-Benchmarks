import os
import time

# Configuração de VRAM para GPU de Laptop
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.55"

import jax
import jax.numpy as jnp

from src.env import CraftaxLevelManager
from src.networks import SymbolicActorCritic
from src.ppo import PPOTrainer, create_train_state
from src.hrl import HRLTrainer


def smoke_test():
    print("Iniciando Smoke Test do Craftax + PureJaxRL...")
    t0 = time.time()
    
    # 1. Test Environment and JIT
    env_manager = CraftaxLevelManager(use_pixels=False, num_train_levels=10)
    obs_sample, _ = env_manager.env.reset(jax.random.PRNGKey(0), env_manager.params)
    print(f"[OK] Craftax Environment carregado. Shape da observação: {obs_sample.shape}")

    # 2. Test PPO Step
    model = SymbolicActorCritic(action_dim=env_manager.num_actions)
    rng = jax.random.PRNGKey(42)
    rng, init_rng, run_rng = jax.random.split(rng, 3)
    train_state = create_train_state(model, init_rng, obs_sample.shape)
    
    trainer = PPOTrainer(
        model=model,
        env_manager=env_manager,
        num_envs=16,
        num_steps=16
    )
    obs, env_state, run_rng = env_manager.reset_train(run_rng, 16)
    runner_state = (train_state, env_state, obs, run_rng)

    train_step_fn = jax.jit(trainer.train_step)
    print("Compilando e executando 1 step de PPO...")
    runner_state, metrics = train_step_fn(runner_state)
    print(f"[OK] PPO executado com sucesso! Loss: {metrics['loss']:.4f}, Recompensa média: {metrics['mean_reward']:.4f}")

    # 3. Test HRL Step
    hrl_trainer = HRLTrainer(mode="skip4", env_manager=env_manager, num_envs=16, skip_k=4)
    hrl_state = hrl_trainer.create_state(init_rng, obs_sample.shape)
    step_fn = jax.jit(hrl_trainer.step_temporal_abstraction)
    print("Compilando e executando 1 step de HRL (skip4)...")
    obs, env_state, rew, done, act, val, logp, run_rng = step_fn(hrl_state, env_state, obs, run_rng)
    print(f"[OK] HRL executado com sucesso! Recompensa média: {rew.mean():.4f}")

    elapsed = time.time() - t0
    print(f"\n[SUCESSO] Smoke test concluído em {elapsed:.2f} segundos! Tudo pronto para rodar o benchmark.")


if __name__ == "__main__":
    smoke_test()
