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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.marl_env import MultiAgentParticleEnv
from src.marl_trainers import MARLPPOTrainer, MARLQTrainer, make_marl_evaluator


def _ppo_selector(trainer, params):
    def sel(obs, gstate, rng):
        E, N, d = obs.shape
        logits = trainer.actor.apply({'params': params['actor']}, obs.reshape(-1, d)).reshape(E, N, trainer.A)
        return jnp.argmax(logits, axis=-1)
    return sel


def _q_selector(trainer, params):
    def sel(obs, gstate, rng):
        E, N, d = obs.shape
        q = trainer.qnet.apply({'params': params['q']}, obs.reshape(-1, d)).reshape(E, N, trainer.A)
        return jnp.argmax(q, axis=-1)
    return sel


def train_eval_policy(algo, env, total_steps, num_envs, seed, eval_envs=256):
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, run_rng = jax.random.split(rng, 3)
    reset_vmap = jax.jit(jax.vmap(env.reset))
    keys = jax.random.split(run_rng, num_envs)
    obs, gstate, env_state = reset_vmap(keys)

    t0 = time.time()
    if algo in ("IPPO", "MAPPO", "MAPOCA"):
        trainer = MARLPPOTrainer(algo, env, num_envs=num_envs, num_steps=64)
        params, opt_state = trainer.create_state(init_rng)
        train_step = trainer.make_train_step()
        carry = (params, opt_state, env_state, obs, gstate, run_rng)
        per_iter = num_envs * 64
        iters = max(1, total_steps // per_iter)
        last = None
        for it in range(iters):
            carry, metrics = train_step(carry, None)
            last = metrics
            if it % 20 == 0 or it == iters - 1:
                print(f"    [{algo} s{seed}] it {it}/{iters} loss={float(metrics['loss']):.4f} "
                      f"team_rew={float(metrics['team_reward']):.3f}", flush=True)
        params = carry[0]
        selector = _ppo_selector(trainer, params)
    else:  # VDN / QMIX
        trainer = MARLQTrainer(algo, env, num_envs=num_envs, eps_decay_steps=total_steps // 2)
        params, target, opt_state, buffer = trainer.create_state(init_rng)
        train_step = trainer.make_train_step()
        carry = (params, target, opt_state, buffer, env_state, obs, gstate, run_rng)
        per_iter = num_envs
        iters = max(1, total_steps // per_iter)
        last = None
        for it in range(iters):
            carry, metrics = train_step(it, carry)
            last = metrics
            if it % 2000 == 0 or it == iters - 1:
                print(f"    [{algo} s{seed}] it {it}/{iters} loss={float(metrics['loss']):.4f} "
                      f"eps={float(metrics['eps']):.3f} team_rew={float(metrics['team_reward']):.3f}", flush=True)
        params = carry[0]
        selector = _q_selector(trainer, params)

    elapsed = time.time() - t0
    real_steps = iters * per_iter
    fps = real_steps / (elapsed + 1e-8)

    eval_fn = make_marl_evaluator(env, selector, num_envs=eval_envs)
    e_rng = jax.random.PRNGKey(seed + 777)
    rew, rew_std, cov, col = eval_fn(e_rng)
    return {"reward": rew, "reward_std": rew_std, "coverage": cov, "collisions": col,
            "fps": fps, "elapsed": elapsed, "steps": int(real_steps)}


def run_marl_benchmark(total_steps=2_000_000, num_envs=64, seeds=(0, 1, 2), eval_envs=256):
    print("=" * 105, flush=True)
    print("   BENCHMARK MARL REAL (TREINADO): IPPO vs MAPPO vs VDN vs QMIX vs MA-POCA", flush=True)
    print(f"   Backend: {jax.default_backend().upper()} | Device: {jax.devices()[0]}", flush=True)
    print(f"   total_steps={total_steps:,} | num_envs={num_envs} | seeds={list(seeds)}", flush=True)
    print("=" * 105, flush=True)

    env = MultiAgentParticleEnv(num_agents=3, num_landmarks=3, max_steps=50)
    algos = ["IPPO", "VDN", "MAPPO", "QMIX", "MAPOCA"]
    labels = {"IPPO": "Descentralizado (Independente)", "VDN": "Fatoração Aditiva Q_tot=ΣQ_i",
              "MAPPO": "CTDE (Crítico Centralizado)", "QMIX": "Fatoração Monotônica (Hiper-redes)",
              "MAPOCA": "CTDE + Auto-Atenção + Contrafactual"}

    results = {}
    for algo in algos:
        runs = []
        print(f"\n>>> Treinando {algo} ({labels[algo]})", flush=True)
        for seed in seeds:
            r = train_eval_policy(algo, env, total_steps, num_envs, seed, eval_envs)
            runs.append({"seed": seed, **r})
            print(f"  [{algo} s{seed}] Reward={r['reward']:+.2f}±{r['reward_std']:.2f} "
                  f"Cobertura={r['coverage']:.1f}% Colisões={r['collisions']:.2f} FPS={r['fps']:,.0f}", flush=True)
        # aggregate
        import numpy as np
        results[algo] = {
            "paradigma": labels[algo],
            "coop_reward": round(float(np.mean([x["reward"] for x in runs])), 3),
            "coop_reward_std": round(float(np.mean([x["reward_std"] for x in runs])), 3),
            "seed_reward_std": round(float(np.std([x["reward"] for x in runs])), 3),
            "cobertura_alvos": round(float(np.mean([x["coverage"] for x in runs])), 1),
            "colisoes": round(float(np.mean([x["collisions"] for x in runs])), 3),
            "throughput_fps": round(float(np.mean([x["fps"] for x in runs])), 0),
            "runs": runs,
        }
        out = Path("results/marl_benchmark_results.json")
        out.parent.mkdir(exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 95, flush=True)
    print(f"{'Algoritmo':<10} | {'Reward Co-op':<18} | {'Cobertura':<10} | {'Colisões':<10} | {'FPS':<12}", flush=True)
    print("-" * 95, flush=True)
    for algo in algos:
        r = results[algo]
        print(f"{algo:<10} | {r['coop_reward']:>+7.2f}±{r['seed_reward_std']:<5.2f}   | "
              f"{r['cobertura_alvos']:>7.1f}% | {r['colisoes']:>8.2f} | {r['throughput_fps']:>10,.0f}", flush=True)
    print("=" * 95, flush=True)
    print("[CONCLUÍDO] Resultados MARL reais salvos em results/marl_benchmark_results.json", flush=True)
    _plot(results)
    return results


def _plot(results):
    order = ["IPPO", "VDN", "MAPPO", "QMIX", "MAPOCA"]
    names = [a for a in order if a in results]
    rew = [results[a]["coop_reward"] for a in names]
    cov = [results[a]["cobertura_alvos"] for a in names]
    import numpy as _np
    x = _np.arange(len(names)); w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(x, rew, w, color='#0284c7', edgecolor='black')
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontweight='bold')
    ax1.set_ylabel('Recompensa cooperativa (episódio)', fontweight='bold')
    ax1.set_title('MARL — Recompensa Real Treinada', fontweight='bold'); ax1.axhline(0, color='k', lw=0.8)
    ax2.bar(x, cov, w, color='#10b981', edgecolor='black')
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontweight='bold')
    ax2.set_ylabel('Cobertura de alvos (%)', fontweight='bold'); ax2.set_ylim(0, 105)
    ax2.set_title('MARL — Cobertura Real Treinada', fontweight='bold')
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/03_marl_benchmarks.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


def _plot_only():
    with open(Path("results/marl_benchmark_results.json")) as f:
        _plot(json.load(f))
    print("[OK] Figura real regenerada em figures/03_marl_benchmarks.png", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval-envs", type=int, default=256)
    p.add_argument("--plot-only", action="store_true")
    a = p.parse_args()
    if a.plot_only:
        _plot_only()
    else:
        run_marl_benchmark(total_steps=a.steps, num_envs=a.num_envs, seeds=tuple(a.seeds), eval_envs=a.eval_envs)
