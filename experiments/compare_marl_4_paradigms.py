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

from src.marl_env import MultiAgentParticleEnv
from src.marl_trainers import MARLPPOTrainer, MARLQTrainer, make_marl_evaluator
from src.marl_paradigms import FogOfWarEnv, JointPPOTrainer


def _ppo_sel(trainer, params):
    def sel(obs, gstate, rng):
        E, N, d = obs.shape
        logits = trainer.actor.apply({'params': params['actor']}, obs.reshape(-1, d)).reshape(E, N, trainer.A)
        return jnp.argmax(logits, axis=-1)
    return sel


def _q_sel(trainer, params):
    def sel(obs, gstate, rng):
        E, N, d = obs.shape
        q = trainer.qnet.apply({'params': params['q']}, obs.reshape(-1, d)).reshape(E, N, trainer.A)
        return jnp.argmax(q, axis=-1)
    return sel


def train_paradigm(paradigm, env, total_steps, num_envs, seed, eval_envs=256):
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, run_rng = jax.random.split(rng, 3)
    reset_vmap = jax.jit(jax.vmap(env.reset))
    obs, gstate, env_state = reset_vmap(jax.random.split(run_rng, num_envs))

    t0 = time.time()
    if paradigm == "CTDE":
        tr = MARLPPOTrainer("MAPPO", env, num_envs=num_envs, num_steps=64)
        params, opt_state = tr.create_state(init_rng)
        step = tr.make_train_step()
        carry = (params, opt_state, env_state, obs, gstate, run_rng)
        per_iter = num_envs * 64
        iters = max(1, total_steps // per_iter)
        for it in range(iters):
            carry, m = step(carry, None)
        sel = _ppo_sel(tr, carry[0])
    elif paradigm == "ValueDecomp":
        tr = MARLQTrainer("QMIX", env, num_envs=num_envs, eps_decay_steps=total_steps // 2)
        params, target, opt_state, buffer = tr.create_state(init_rng)
        step = tr.make_train_step()
        carry = (params, target, opt_state, buffer, env_state, obs, gstate, run_rng)
        per_iter = num_envs
        iters = max(1, total_steps // per_iter)
        for it in range(iters):
            carry, m = step(it, carry)
        sel = _q_sel(tr, carry[0])
    else:  # CTE or COMM
        tr = JointPPOTrainer(paradigm, env, num_envs=num_envs, num_steps=64)
        params, opt_state = tr.create_state(init_rng)
        step = tr.make_train_step()
        carry = (params, opt_state, env_state, obs, gstate, run_rng)
        per_iter = num_envs * 64
        iters = max(1, total_steps // per_iter)
        for it in range(iters):
            carry, m = step(carry, None)
        sel = tr.make_selector(carry[0])

    elapsed = time.time() - t0
    real_steps = iters * per_iter
    fps = real_steps / (elapsed + 1e-8)
    eval_fn = make_marl_evaluator(env, sel, num_envs=eval_envs)
    rew, rew_std, cov, col = eval_fn(jax.random.PRNGKey(seed + 777))
    return {"reward": rew, "reward_std": rew_std, "coverage": cov, "collisions": col,
            "fps": fps, "steps": int(real_steps)}


def run_4_paradigms(total_steps=2_000_000, num_envs=128, seeds=(0, 1, 2), eval_envs=256, fog_radius=0.40):
    print("=" * 120, flush=True)
    print("   BENCHMARK REAL DOS 4 PARADIGMAS MARL (TREINADOS) — VISÃO CLARA vs FOG-OF-WAR", flush=True)
    print(f"   Backend: {jax.default_backend().upper()} | Device: {jax.devices()[0]}", flush=True)
    print(f"   total_steps={total_steps:,} | num_envs={num_envs} | seeds={list(seeds)} | raio fog={fog_radius}", flush=True)
    print("=" * 120, flush=True)

    base = MultiAgentParticleEnv(num_agents=3, num_landmarks=3, max_steps=50)
    paradigms = [
        ("CTDE", "CTDE (MA-POCA / MAPPO)", "Policy-Based CTDE"),
        ("ValueDecomp", "Value Decomposition (QMIX)", "Value-Based Monotonic"),
        ("CTE", "Centralized Joint (CTE)", "Centralized Super-Agent"),
        ("COMM", "Explicit Comm (TarMAC / GAT)", "Graph Attention Messaging"),
    ]
    conditions = [("clear", False), ("fog", True)]

    results_data = []
    for pid, pname, fam in paradigms:
        entry = {"id": pid, "nome": pname, "familia": fam}
        for cond, fog in conditions:
            env = FogOfWarEnv(base, fog=fog, radius=fog_radius)
            runs = []
            print(f"\n>>> Treinando {pname} [{cond.upper()}]...", flush=True)
            for seed in seeds:
                r = train_paradigm(pid, env, total_steps, num_envs, seed, eval_envs)
                runs.append({"seed": seed, **r})
                print(f"  [{pid}/{cond} s{seed}] Reward={r['reward']:+.2f}±{r['reward_std']:.2f} "
                      f"Cob={r['coverage']:.1f}% Col={r['collisions']:.2f} FPS={r['fps']:,.0f}", flush=True)
            entry[f"reward_{cond}"] = round(float(np.mean([x["reward"] for x in runs])), 2)
            entry[f"reward_{cond}_std"] = round(float(np.std([x["reward"] for x in runs])), 2)
            entry[f"cobertura_{cond}"] = round(float(np.mean([x["coverage"] for x in runs])), 1)
            entry[f"colisoes_{cond}"] = round(float(np.mean([x["collisions"] for x in runs])), 2)
            entry[f"fps_{cond}"] = round(float(np.mean([x["fps"] for x in runs])), 0)
            entry[f"runs_{cond}"] = runs
        results_data.append(entry)
        out = Path("results/marl_4_paradigms_results.json")
        out.parent.mkdir(exist_ok=True)
        with open(out, "w") as f:
            json.dump(results_data, f, indent=2)

    # Figure from REAL numbers
    names = [p["nome"].split(" (")[0] for p in results_data]
    x = np.arange(len(names))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    rc = [p["reward_clear"] for p in results_data]
    rf = [p["reward_fog"] for p in results_data]
    ax1.bar(x - w/2, rc, w, label='Visão Clara', color='#10b981', edgecolor='black')
    ax1.bar(x + w/2, rf, w, label=f'Fog-of-War (r={fog_radius}m)', color='#ef4444', edgecolor='black')
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=15, ha='right', fontweight='bold')
    ax1.set_ylabel('Recompensa cooperativa (episódio)', fontweight='bold')
    ax1.set_title('Retorno Real Treinado — 4 Paradigmas MARL', fontweight='bold')
    ax1.legend()
    cc = [p["cobertura_clear"] for p in results_data]
    cf = [p["cobertura_fog"] for p in results_data]
    ax2.bar(x - w/2, cc, w, label='Visão Clara', color='#3b82f6', edgecolor='black')
    ax2.bar(x + w/2, cf, w, label='Fog-of-War', color='#f59e0b', edgecolor='black')
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=15, ha='right', fontweight='bold')
    ax2.set_ylabel('Cobertura de alvos (%)', fontweight='bold'); ax2.set_ylim(0, 115)
    ax2.set_title('Cobertura sob Oclusão (POMDP)', fontweight='bold')
    ax2.legend()
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/07_marl_4_paradigms_benchmark.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("\n[SUCESSO] Resultados reais em results/marl_4_paradigms_results.json "
          "e figures/07_marl_4_paradigms_benchmark.png", flush=True)
    return results_data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval-envs", type=int, default=256)
    p.add_argument("--fog-radius", type=float, default=0.40)
    a = p.parse_args()
    run_4_paradigms(total_steps=a.steps, num_envs=a.num_envs, seeds=tuple(a.seeds),
                    eval_envs=a.eval_envs, fog_radius=a.fog_radius)
