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

import brax.envs as brax_envs
from src.marl_3d_env import MultiAgent3DCooperativeEnv
from src.marl3d_trainers import ContinuousMARLPPOTrainer, make_marl3d_evaluator, BraxWrapper
from src.continuous_rl import GaussianPPOTrainer, SACTrainer
from src.eval_utils import make_continuous_evaluator, make_fixed_horizon_evaluator


# ---------------------------------------------------------------------
# 1. SINGLE-AGENT 3D: GOOGLE BRAX (real PPO / SAC training)
# ---------------------------------------------------------------------
def _brax_train(algo, env_name, total_steps, num_envs, seed):
    benv = brax_envs.get_environment(env_name=env_name)
    env = BraxWrapper(benv, max_steps=1000)
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, run_rng = jax.random.split(rng, 3)
    reset_vmap = jax.jit(jax.vmap(env.reset))
    obs, env_state = reset_vmap(jax.random.split(run_rng, num_envs))

    if algo == "PPO":
        tr = GaussianPPOTrainer(env, obs_dim=env.obs_dim, action_dim=env.action_dim,
                                num_envs=num_envs, num_steps=64)
        params, opt_state = tr.create_state(init_rng)
        carry = (params, opt_state, env_state, obs, run_rng)
        step = tr.make_train_step()
        iters = max(1, total_steps // (num_envs * 64))
        t0 = time.time()
        for _ in range(iters):
            carry, m = step(carry, None)
        elapsed = time.time() - t0
        sel = tr.make_eval_policy()
        p = carry[0]
        real_steps = iters * num_envs * 64
    else:  # SAC
        tr = SACTrainer(env, obs_dim=env.obs_dim, action_dim=env.action_dim, num_envs=num_envs,
                        buffer_size=100000, batch_size=256)
        params, target, opt_state, log_alpha, alpha_opt, buffer = tr.create_state(init_rng)
        carry = (params, target, opt_state, log_alpha, alpha_opt, buffer, env_state, obs, run_rng)
        step = tr.make_train_step()
        iters = max(1, total_steps // num_envs)
        t0 = time.time()
        for _ in range(iters):
            carry, m = step(carry, None)
        elapsed = time.time() - t0
        sel = tr.make_eval_policy()
        p = carry[0]
        real_steps = iters * num_envs

    ev = make_fixed_horizon_evaluator(env, lambda o, r: sel(p, o, r), num_envs=64, horizon=1000)
    ret, ret_std = ev(jax.random.PRNGKey(seed + 777))
    return {"reward": ret, "reward_std": ret_std, "fps": real_steps / (elapsed + 1e-8), "steps": int(real_steps)}


def _brax_passive(env_name, num_envs=64):
    benv = brax_envs.get_environment(env_name=env_name)
    env = BraxWrapper(benv, max_steps=1000)
    adim = env.action_dim
    ev = make_fixed_horizon_evaluator(env, lambda o, r: jnp.zeros((o.shape[0], adim)),
                                      num_envs=num_envs, horizon=1000)
    ret, ret_std = ev(jax.random.PRNGKey(0))
    return {"reward": ret, "reward_std": ret_std}


# ---------------------------------------------------------------------
# 2. MULTI-AGENT 3D: continuous drones (real IPPO/MAPPO/MA-POCA training)
# ---------------------------------------------------------------------
def _marl3d_train(algo, total_steps, num_envs, seed, eval_envs):
    env = MultiAgent3DCooperativeEnv(num_agents=3, num_landmarks=3, max_steps=100)
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, run_rng = jax.random.split(rng, 3)
    tr = ContinuousMARLPPOTrainer(algo, env, num_envs=num_envs, num_steps=64)
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
    sel = tr.make_selector(carry[0])
    ev = make_marl3d_evaluator(env, sel, num_envs=eval_envs)
    rew, std, cov, col = ev(jax.random.PRNGKey(seed + 777))
    fps = iters * num_envs * 64 * env.num_agents / (elapsed + 1e-8)
    return {"reward": rew, "reward_std": std, "coverage": cov, "collisions": col,
            "fps": fps, "steps": int(iters * num_envs * 64)}


def run_3d_benchmarks(brax_steps=2_000_000, marl_steps=3_000_000, num_envs=128,
                      seeds=(0, 1), eval_envs=256):
    print("=" * 120, flush=True)
    print("   SUÍTE REAL 3D: GOOGLE BRAX (PPO/SAC treinados) + DRONES 3D (IPPO/MAPPO/MA-POCA treinados)", flush=True)
    print(f"   Backend: {jax.default_backend().upper()} | Device: {jax.devices()[0]}", flush=True)
    print(f"   brax_steps={brax_steps:,} | marl_steps={marl_steps:,} | num_envs={num_envs} | seeds={list(seeds)}", flush=True)
    print("=" * 120, flush=True)

    results = {"single_agent_3d": {}, "multi_agent_3d": {}}
    out = Path("results/3d_benchmarks_results.json")

    for env_name in ["halfcheetah", "ant", "humanoid"]:
        print(f"\n--- Brax {env_name.upper()} ---", flush=True)
        pas = _brax_passive(env_name)
        results["single_agent_3d"][f"{env_name}_Passive"] = {
            "ambiente": f"{env_name.capitalize()} 3D", "algoritmo": "Ação Nula (passiva)",
            "reward_mean": round(pas["reward"], 2), "reward_std": round(pas["reward_std"], 2),
            "throughput_fps": 0.0}
        print(f"  [Passiva] Retorno={pas['reward']:+.2f}±{pas['reward_std']:.2f}", flush=True)
        for algo in ["PPO", "SAC"]:
            runs = []
            for seed in seeds:
                r = _brax_train(algo, env_name, brax_steps, num_envs, seed)
                runs.append(r)
                print(f"  [{env_name} {algo} s{seed}] Retorno={r['reward']:+.2f}±{r['reward_std']:.2f} "
                      f"FPS={r['fps']:,.0f}", flush=True)
            results["single_agent_3d"][f"{env_name}_{algo}"] = {
                "ambiente": f"{env_name.capitalize()} 3D", "algoritmo": f"{env_name.capitalize()} {algo}",
                "reward_mean": round(float(np.mean([x["reward"] for x in runs])), 2),
                "reward_std": round(float(np.mean([x["reward_std"] for x in runs])), 2),
                "throughput_fps": round(float(np.mean([x["fps"] for x in runs])), 0),
                "runs": runs}
            _save(out, results)

    print("\n--- Drones 3D (multi-agente contínuo) ---", flush=True)
    for algo in ["IPPO", "MAPPO", "MAPOCA"]:
        runs = []
        for seed in seeds:
            r = _marl3d_train(algo, marl_steps, num_envs, seed, eval_envs)
            runs.append(r)
            print(f"  [{algo} 3D s{seed}] Retorno={r['reward']:+.2f}±{r['reward_std']:.2f} "
                  f"Cob={r['coverage']:.1f}% Col={r['collisions']:.2f} FPS={r['fps']:,.0f}", flush=True)
        results["multi_agent_3d"][f"{algo}_3D"] = {
            "algoritmo": f"{algo} 3D",
            "reward_mean": round(float(np.mean([x["reward"] for x in runs])), 2),
            "reward_std": round(float(np.mean([x["reward_std"] for x in runs])), 2),
            "cobertura_3d": round(float(np.mean([x["coverage"] for x in runs])), 1),
            "colisoes_medias": round(float(np.mean([x["collisions"] for x in runs])), 2),
            "throughput_fps": round(float(np.mean([x["fps"] for x in runs])), 0),
            "runs": runs}
        _save(out, results)

    _plot(results)
    print("\n[SUCESSO] Resultados reais em results/3d_benchmarks_results.json", flush=True)
    return results


def _save(out, results):
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)


def _plot(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    bk = list(results["single_agent_3d"].keys())
    bn = [results["single_agent_3d"][k]["algoritmo"] for k in bk]
    br = [results["single_agent_3d"][k]["reward_mean"] for k in bk]
    ax1.barh(range(len(bn)), br, color='#0284c7', edgecolor='black')
    ax1.set_yticks(range(len(bn))); ax1.set_yticklabels(bn, fontsize=8)
    ax1.set_title('Brax 3D — Retorno Real Treinado', fontweight='bold')
    ax1.axvline(0, color='black', lw=0.8)
    mk = list(results["multi_agent_3d"].keys())
    mn = [results["multi_agent_3d"][k]["algoritmo"] for k in mk]
    mc = [results["multi_agent_3d"][k]["cobertura_3d"] for k in mk]
    mcol = [results["multi_agent_3d"][k]["colisoes_medias"] for k in mk]
    x = np.arange(len(mn)); w = 0.35
    ax2.bar(x - w/2, mc, w, label='Cobertura 3D (%)', color='#10b981', edgecolor='black')
    ax2.bar(x + w/2, mcol, w, label='Colisões/ep', color='#ef4444', edgecolor='black')
    ax2.set_xticks(x); ax2.set_xticklabels(mn, fontweight='bold')
    ax2.set_title('Drones 3D — Cobertura vs Colisões (treinado)', fontweight='bold')
    ax2.legend()
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/08_3d_benchmarks.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--brax-steps", type=int, default=2_000_000)
    p.add_argument("--marl-steps", type=int, default=3_000_000)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--eval-envs", type=int, default=256)
    a = p.parse_args()
    run_3d_benchmarks(brax_steps=a.brax_steps, marl_steps=a.marl_steps, num_envs=a.num_envs,
                      seeds=tuple(a.seeds), eval_envs=a.eval_envs)
