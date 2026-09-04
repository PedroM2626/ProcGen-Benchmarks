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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.continuous_env import ContinuousSingleAgentNavigationEnv
from src.continuous_rl import GaussianPPOTrainer, SACTrainer, DiscretePPOTrainer, DiscreteActionWrapper
from src.marl_env import MultiAgentParticleEnv
from src.marl_trainers import MARLPPOTrainer, make_marl_evaluator
from src.eval_utils import make_continuous_evaluator


def _train_single(kind, total_steps, num_envs, seed, eval_envs):
    base = ContinuousSingleAgentNavigationEnv(max_steps=100)
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, run_rng = jax.random.split(rng, 3)

    if kind == "Discrete_PPO":
        env = DiscreteActionWrapper(base)
        tr = DiscretePPOTrainer(env, obs_dim=base.obs_dim, action_dim=5, num_envs=num_envs, num_steps=64)
        params, opt_state = tr.create_state(init_rng)
        reset_vmap = jax.jit(jax.vmap(env.reset))
        obs, env_state = reset_vmap(jax.random.split(run_rng, num_envs))
        carry = (params, opt_state, env_state, obs, run_rng)
        step = tr.make_train_step()
        iters = max(1, total_steps // (num_envs * 64))
        t0 = time.time()
        for _ in range(iters):
            carry, m = step(carry, None)
        elapsed = time.time() - t0
        sel = tr.make_eval_policy()
        p = carry[0]
        ev = make_continuous_evaluator(env, lambda o, r: sel(p, o, r), num_envs=eval_envs, horizon=base.max_steps * 4)
        ret, neps = ev(jax.random.PRNGKey(seed + 777))
        space = "Discreto (5 forças quantizadas)"
    elif kind == "Continuous_PPO":
        env = base
        tr = GaussianPPOTrainer(env, obs_dim=base.obs_dim, action_dim=2, num_envs=num_envs, num_steps=64)
        params, opt_state = tr.create_state(init_rng)
        reset_vmap = jax.jit(jax.vmap(env.reset))
        obs, env_state = reset_vmap(jax.random.split(run_rng, num_envs))
        carry = (params, opt_state, env_state, obs, run_rng)
        step = tr.make_train_step()
        iters = max(1, total_steps // (num_envs * 64))
        t0 = time.time()
        for _ in range(iters):
            carry, m = step(carry, None)
        elapsed = time.time() - t0
        sel = tr.make_eval_policy()
        p = carry[0]
        ev = make_continuous_evaluator(env, lambda o, r: sel(p, o, r), num_envs=eval_envs, horizon=base.max_steps * 4)
        ret, neps = ev(jax.random.PRNGKey(seed + 777))
        space = "Contínuo (Gaussiana N(mu,sigma) em [-1,1]^2)"
    else:  # SAC
        env = base
        tr = SACTrainer(env, obs_dim=base.obs_dim, action_dim=2, num_envs=num_envs,
                        buffer_size=200000, batch_size=256)
        params, target, opt_state, log_alpha, alpha_opt, buffer = tr.create_state(init_rng)
        reset_vmap = jax.jit(jax.vmap(env.reset))
        obs, env_state = reset_vmap(jax.random.split(run_rng, num_envs))
        carry = (params, target, opt_state, log_alpha, alpha_opt, buffer, env_state, obs, run_rng)
        step = tr.make_train_step()
        iters = max(1, total_steps // num_envs)
        t0 = time.time()
        for _ in range(iters):
            carry, m = step(carry, None)
        elapsed = time.time() - t0
        sel = tr.make_eval_policy()
        p = carry[0]
        ev = make_continuous_evaluator(env, lambda o, r: sel(p, o, r), num_envs=eval_envs, horizon=base.max_steps * 4)
        ret, neps = ev(jax.random.PRNGKey(seed + 777))
        space = "Contínuo (Tanh-squashed MaxEnt)"

    fps = iters * num_envs * (64 if kind != "SAC" else 1) / (elapsed + 1e-8)
    return {"reward": ret, "episodes": neps, "fps": fps, "elapsed": elapsed, "space": space,
            "steps": int(iters * num_envs * (64 if kind != "SAC" else 1))}


def _train_discrete_mappo(total_steps, num_envs, seed, eval_envs):
    env = MultiAgentParticleEnv(num_agents=3, num_landmarks=3, max_steps=50)
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, run_rng = jax.random.split(rng, 3)
    tr = MARLPPOTrainer("MAPPO", env, num_envs=num_envs, num_steps=64)
    params, opt_state = tr.create_state(init_rng)
    reset_vmap = jax.jit(jax.vmap(env.reset))
    obs, gstate, env_state = reset_vmap(jax.random.split(run_rng, num_envs))
    carry = (params, opt_state, env_state, obs, gstate, run_rng)
    step = tr.make_train_step()
    iters = max(1, total_steps // (num_envs * 64))
    t0 = time.time()
    for _ in range(iters):
        carry, m = step(carry, None)
    elapsed = time.time() - t0
    p = carry[0]

    def sel(obs, gstate, rng):
        E, N, d = obs.shape
        logits = tr.actor.apply({'params': p['actor']}, obs.reshape(-1, d)).reshape(E, N, tr.A)
        return jnp.argmax(logits, axis=-1)
    ev = make_marl_evaluator(env, sel, num_envs=eval_envs)
    rew, std, cov, col = ev(jax.random.PRNGKey(seed + 777))
    fps = iters * num_envs * 64 / (elapsed + 1e-8)
    return {"reward": rew, "reward_std": std, "coverage": cov, "collisions": col, "fps": fps,
            "elapsed": elapsed, "steps": int(iters * num_envs * 64)}


def run_discrete_vs_continuous(total_steps=1_000_000, marl_steps=2_000_000, num_envs=128,
                               seeds=(0, 1, 2), eval_envs=256):
    print("=" * 112, flush=True)
    print("   BENCHMARK REAL DISCRETO vs CONTÍNUO: Discrete PPO vs Gaussian PPO vs SAC (+ MAPPO discreto)", flush=True)
    print(f"   Backend: {jax.default_backend().upper()} | Device: {jax.devices()[0]}", flush=True)
    print(f"   single_steps={total_steps:,} | marl_steps={marl_steps:,} | num_envs={num_envs} | seeds={list(seeds)}", flush=True)
    print("=" * 112, flush=True)

    results = {}
    for kind in ["Discrete_PPO", "Continuous_PPO", "SAC"]:
        runs = []
        print(f"\n>>> Treinando {kind}...", flush=True)
        for seed in seeds:
            r = _train_single(kind, total_steps, num_envs, seed, eval_envs)
            runs.append({"seed": seed, **r})
            print(f"  [{kind} s{seed}] Reward/ep={r['reward']:+.2f} ({r['episodes']} eps) FPS={r['fps']:,.0f}", flush=True)
        results[f"Single_{kind}"] = {
            "categoria": "Single-Agent", "espaco": runs[0]["space"],
            "reward": round(float(np.mean([x["reward"] for x in runs])), 3),
            "reward_seed_std": round(float(np.std([x["reward"] for x in runs])), 3),
            "throughput_fps": round(float(np.mean([x["fps"] for x in runs])), 0), "runs": runs,
        }
        _save(results)

    # Discrete MAPPO (multi-agent)
    runs = []
    print("\n>>> Treinando Discrete MAPPO (multi-agente MPE)...", flush=True)
    for seed in seeds:
        r = _train_discrete_mappo(marl_steps, num_envs, seed, eval_envs)
        runs.append({"seed": seed, **r})
        print(f"  [Discrete_MAPPO s{seed}] Reward={r['reward']:+.2f}±{r['reward_std']:.2f} "
              f"Cob={r['coverage']:.1f}% FPS={r['fps']:,.0f}", flush=True)
    results["Multi_Discrete_MAPPO"] = {
        "categoria": "Multi-Agent", "espaco": "Discreto (5 ações por agente)",
        "reward": round(float(np.mean([x["reward"] for x in runs])), 3),
        "reward_seed_std": round(float(np.std([x["reward"] for x in runs])), 3),
        "coverage": round(float(np.mean([x["coverage"] for x in runs])), 1),
        "throughput_fps": round(float(np.mean([x["fps"] for x in runs])), 0), "runs": runs,
    }
    results["_nota_multiagent_continuous"] = (
        "Continuous MAPPO e Continuous MA-POCA multi-agente sao treinados de verdade no benchmark 3D "
        "(experiments/compare_3d_benchmarks.py, ambiente de drones continuos MultiAgent3DCooperativeEnv).")
    _save(results)
    _plot(results)
    print("\n[SUCESSO] Resultados reais em results/discrete_vs_continuous_results.json", flush=True)
    return results


def _plot(results):
    keys = [k for k in ["Single_Discrete_PPO", "Single_Continuous_PPO", "Single_SAC", "Multi_Discrete_MAPPO"]
            if k in results]
    labels = {"Single_Discrete_PPO": "Discrete PPO", "Single_Continuous_PPO": "Gaussian PPO",
              "Single_SAC": "SAC", "Multi_Discrete_MAPPO": "MAPPO (multi)"}
    names = [labels[k] for k in keys]
    rew = [results[k]["reward"] for k in keys]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(names, rew, color=['#ef4444', '#10b981', '#3b82f6', '#f59e0b'][:len(names)], edgecolor='black')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_ylabel('Retorno episódico real', fontweight='bold')
    ax.set_title('Discreto vs Contínuo — Retorno Real Treinado', fontweight='bold')
    for i, v in enumerate(rew):
        ax.text(i, v, f"{v:+.1f}", ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold', fontsize=9)
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/04_discrete_vs_continuous.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


def _save(results):
    out = Path("results/discrete_vs_continuous_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--marl-steps", type=int, default=2_000_000)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval-envs", type=int, default=256)
    a = p.parse_args()
    run_discrete_vs_continuous(total_steps=a.steps, marl_steps=a.marl_steps, num_envs=a.num_envs,
                               seeds=tuple(a.seeds), eval_envs=a.eval_envs)
