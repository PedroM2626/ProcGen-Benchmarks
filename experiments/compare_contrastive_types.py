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

from src.env import CraftaxLevelManager
from src.combinatorial_engine import FeatureExtractorNatureCNN
from src.aux_ppo import AuxPPOTrainer
from src.eval_utils import make_craftax_evaluator


CONFIGS = [
    ("Baseline_No_Contrastive", "none", "Nenhum (RL puro)"),
    ("Spatial_CURL", "spatial", "Espacial (InfoNCE em crops)"),
    ("Temporal_CPC", "temporal", "Temporal (s_t -> s_t+k)"),
    ("Action_ACL", "action", "Causal (s_t,a_t -> s_t+1)"),
    ("Self_Predictive_SPR", "spr", "Auto-preditivo (BYOL, sem negativos)"),
]


def train_one(aux_type, total_steps, num_envs, seed, eval_episodes, eval_horizon):
    env_manager = CraftaxLevelManager(use_pixels=True, num_train_levels=200, eval_seed_offset=1000)
    obs_sample, _ = env_manager.env.reset(jax.random.PRNGKey(0), env_manager.params)
    input_shape = obs_sample.shape

    rng = jax.random.PRNGKey(seed)
    rng, init_rng, run_rng = jax.random.split(rng, 3)
    trainer = AuxPPOTrainer(extractor_cls=FeatureExtractorNatureCNN, env_manager=env_manager,
                            aux_type=aux_type, num_envs=num_envs, num_steps=64,
                            action_dim=env_manager.num_actions)
    params, opt_state, aux_opt_state = trainer.create_state(init_rng, input_shape)
    obs, env_state, run_rng = env_manager.reset_train(run_rng, num_envs)
    carry = (params, opt_state, aux_opt_state, env_state, obs, run_rng)
    step_fn = trainer.make_train_step()
    iters = max(1, total_steps // (num_envs * 64))

    t0 = time.time()
    last = None
    for it in range(iters):
        carry, metrics = step_fn(carry, None)
        last = metrics
        if it % 50 == 0 or it == iters - 1:
            print(f"    [{aux_type} s{seed}] it {it}/{iters} ppo={float(metrics['ppo_loss']):.4f} "
                  f"aux={float(metrics['aux_loss']):.4f} rew={float(metrics['mean_reward']):.3f}", flush=True)
    elapsed = time.time() - t0
    final_params = carry[0]
    real_steps = iters * num_envs * 64
    fps = real_steps / (elapsed + 1e-8)

    eval_fn = make_craftax_evaluator(env_manager, trainer.make_eval_policy(True),
                                     num_envs=eval_episodes, horizon=eval_horizon)
    e_rng = jax.random.PRNGKey(seed + 999)
    e_rng, r1, r2 = jax.random.split(e_rng, 3)
    tr_mean, tr_std = eval_fn(final_params, r1, unseen=False)
    un_mean, un_std = eval_fn(final_params, r2, unseen=True)
    return {"fps": fps, "elapsed": elapsed, "steps": int(real_steps),
            "train": tr_mean, "train_std": tr_std, "unseen": un_mean, "unseen_std": un_std,
            "final_aux_loss": float(last["aux_loss"]) if last is not None else None}


def run_contrastive_benchmark(total_steps=5_000_000, num_envs=128, seeds=(0, 1, 2),
                              eval_episodes=128, eval_horizon=1000):
    print("=" * 110, flush=True)
    print("   BENCHMARK REAL DAS FAMÍLIAS CONTRASTIVAS (PPO + perda auxiliar treinada)", flush=True)
    print(f"   Backend: {jax.default_backend().upper()} | Device: {jax.devices()[0]}", flush=True)
    print(f"   total_steps={total_steps:,} | num_envs={num_envs} | seeds={list(seeds)}", flush=True)
    print("=" * 110, flush=True)

    results = {}
    for name, aux, desc in CONFIGS:
        runs = []
        print(f"\n>>> Treinando {name} (aux={aux})...", flush=True)
        for seed in seeds:
            r = train_one(aux, total_steps, num_envs, seed, eval_episodes, eval_horizon)
            runs.append({"seed": seed, **r})
            print(f"  [{name} s{seed}] FPS={r['fps']:,.0f} Train={r['train']:.2f}±{r['train_std']:.2f} "
                  f"Unseen={r['unseen']:.2f}±{r['unseen_std']:.2f} aux_loss={r['final_aux_loss']}", flush=True)
        base_unseen = results.get("Baseline_No_Contrastive", {}).get("unseen_score")
        results[name] = {
            "tipo": desc, "aux_type": aux,
            "throughput_fps": round(float(np.mean([x["fps"] for x in runs])), 0),
            "train_score": round(float(np.mean([x["train"] for x in runs])), 3),
            "unseen_score": round(float(np.mean([x["unseen"] for x in runs])), 3),
            "unseen_std": round(float(np.mean([x["unseen_std"] for x in runs])), 3),
            "seed_unseen_std": round(float(np.std([x["unseen"] for x in runs])), 3),
            "final_aux_loss": round(float(np.mean([x["final_aux_loss"] for x in runs
                                                   if x["final_aux_loss"] is not None])), 4)
                              if any(x["final_aux_loss"] is not None for x in runs) else None,
            "runs": runs,
        }
        out = Path("results/contrastive_types_benchmark_results.json")
        out.parent.mkdir(exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    # Figure (real numbers)
    names = [c[0] for c in CONFIGS]
    unseen = [results[n]["unseen_score"] for n in names]
    err = [results[n]["seed_unseen_std"] for n in names]
    short = ["Baseline", "CURL", "CPC", "ACL", "SPR"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(short, unseen, yerr=err, color=['#6b7280', '#ef4444', '#f59e0b', '#10b981', '#3b82f6'],
                  edgecolor='black', capsize=4)
    for b, v in zip(bars, unseen):
        ax.text(b.get_x() + b.get_width()/2., v, f"{v:.2f}", ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel("Retorno episódico real (níveis inéditos)", fontweight='bold')
    ax.set_title("Famílias Contrastivas — Retorno Real Treinado (PPO + aux loss)", fontweight='bold')
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/02_contrastive_learning_families.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("\n[SUCESSO] Resultados reais em results/contrastive_types_benchmark_results.json "
          "e figures/02_contrastive_learning_families.png", flush=True)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval-episodes", type=int, default=128)
    p.add_argument("--eval-horizon", type=int, default=1000)
    a = p.parse_args()
    run_contrastive_benchmark(total_steps=a.steps, num_envs=a.num_envs, seeds=tuple(a.seeds),
                              eval_episodes=a.eval_episodes, eval_horizon=a.eval_horizon)
