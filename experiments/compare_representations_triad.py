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
import flax.linen as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.env import CraftaxLevelManager
from src.combinatorial_engine import FeatureExtractorNatureCNN, UniversalActorCritic
from src.graph_modules import FeatureExtractorGNN
from src.ppo import PPOTrainer, create_train_state
from src.eval_utils import make_craftax_evaluator


class FeatureExtractorMLPVector(nn.Module):
    """Tabular MLP encoder for the symbolic (1345-D) representation."""
    hidden_dim: int = 256
    out_dim: int = 512

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32).reshape((x.shape[0], -1))
        h = nn.relu(nn.Dense(self.hidden_dim)(x))
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        return nn.Dense(self.out_dim)(h)


def _train_one(name, extractor_cls, use_pixels, total_steps, num_envs, seed,
               eval_episodes, eval_horizon):
    env_manager = CraftaxLevelManager(use_pixels=use_pixels, num_train_levels=200, eval_seed_offset=1000)
    obs_sample, _ = env_manager.env.reset(jax.random.PRNGKey(0), env_manager.params)
    input_shape = obs_sample.shape

    model = UniversalActorCritic(extractor_cls=extractor_cls, action_dim=env_manager.num_actions)
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, run_rng = jax.random.split(rng, 3)
    train_state = create_train_state(model, init_rng, input_shape, learning_rate=3e-4)
    trainer = PPOTrainer(model=model, env_manager=env_manager, num_envs=num_envs, num_steps=128)

    obs, env_state, run_rng = env_manager.reset_train(run_rng, num_envs)
    runner_state = (train_state, env_state, obs, run_rng)
    train_step = jax.jit(trainer.train_step)
    iters = max(1, total_steps // (num_envs * 128))

    t0 = time.time()
    for it in range(iters):
        runner_state, metrics = train_step(runner_state)
        if it % 50 == 0 or it == iters - 1:
            print(f"    [{name} s{seed}] it {it}/{iters} loss={float(metrics['loss']):.4f} "
                  f"rew={float(metrics['mean_reward']):.3f}", flush=True)
    elapsed = time.time() - t0
    final_state = runner_state[0]
    real_steps = iters * num_envs * 128
    fps = real_steps / (elapsed + 1e-8)

    def greedy(params, obs, rng):
        logits, _ = model.apply({'params': params}, obs)
        return jnp.argmax(logits, axis=-1)

    eval_fn = make_craftax_evaluator(env_manager, greedy, num_envs=eval_episodes, horizon=eval_horizon)
    e_rng = jax.random.PRNGKey(seed + 999)
    e_rng, r1, r2 = jax.random.split(e_rng, 3)
    tr_mean, tr_std = eval_fn(final_state.params, r1, unseen=False)
    un_mean, un_std = eval_fn(final_state.params, r2, unseen=True)
    return {"fps": fps, "elapsed": elapsed, "steps": int(real_steps),
            "train": tr_mean, "train_std": tr_std, "unseen": un_mean, "unseen_std": un_std}


def run_triad_comparison(total_steps=5_000_000, num_envs=256, seeds=(0, 1, 2),
                         eval_episodes=128, eval_horizon=1000):
    print("=" * 115, flush=True)
    print("   BENCHMARK REAL DA TRÍADE DE REPRESENTAÇÕES: PIXELS vs VETOR vs GRAFO (PPO treinado)", flush=True)
    print(f"   Backend: {jax.default_backend().upper()} | Device: {jax.devices()[0]}", flush=True)
    print(f"   total_steps={total_steps:,} | num_envs={num_envs} | seeds={list(seeds)}", flush=True)
    print("=" * 115, flush=True)

    reps = [
        ("Pixels_NatureCNN", FeatureExtractorNatureCNN, True,
         "Imagem / Grid 2D Convolucional", "Localidade espacial 2D e invariância à translação",
         "Não (depende da coordenada exata do pixel)"),
        ("Vetor_MLP", FeatureExtractorMLPVector, False,
         "Vetor Tabular / Simbólico (1345D)", "Nenhum (conectividade densa cega)",
         "Não (ordem rígida de features nas colunas)"),
        ("Grafo_GNN", FeatureExtractorGNN, False,
         "Grafo Relacional (nós de entidades + arestas + GAT)",
         "Invariância à permutação de entidades + raciocínio relacional",
         "Sim (readout mean+max sobre nós é invariante à permutação de entidades)"),
    ]

    results = {}
    for name, extractor, use_pixels, modality, ind_bias, invariance in reps:
        runs = []
        print(f"\n>>> Treinando representação {name}...", flush=True)
        for seed in seeds:
            r = _train_one(name, extractor, use_pixels, total_steps, num_envs, seed,
                           eval_episodes, eval_horizon)
            runs.append({"seed": seed, **r})
            print(f"  [{name} s{seed}] steps={r['steps']:,} FPS={r['fps']:,.0f} "
                  f"Train={r['train']:.2f}±{r['train_std']:.2f} Unseen={r['unseen']:.2f}±{r['unseen_std']:.2f}",
                  flush=True)
        results[name] = {
            "modalidade": modality,
            "vies_indutivo": ind_bias,
            "throughput_fps": round(float(np.mean([x["fps"] for x in runs])), 0),
            "train_score": round(float(np.mean([x["train"] for x in runs])), 3),
            "train_std": round(float(np.mean([x["train_std"] for x in runs])), 3),
            "unseen_score": round(float(np.mean([x["unseen"] for x in runs])), 3),
            "unseen_std": round(float(np.mean([x["unseen_std"] for x in runs])), 3),
            "seed_unseen_std": round(float(np.std([x["unseen"] for x in runs])), 3),
            "invariancia_permutacao": invariance,
            "runs": runs,
        }
        out = Path("results/representations_triad_results.json")
        out.parent.mkdir(exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    # Figure from REAL numbers
    names = [r[0] for r in reps]
    unseen = [results[n]["unseen_score"] for n in names]
    err = [results[n]["seed_unseen_std"] for n in names]
    fps = [results[n]["throughput_fps"] for n in names]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(names, unseen, yerr=err, color=['#ef4444', '#10b981', '#3b82f6'], edgecolor='black', capsize=4)
    ax1.set_ylabel("Retorno episódico (níveis inéditos)", fontweight='bold')
    ax1.set_title("Tríade de Representações — Retorno Real Treinado (PPO)", fontweight='bold')
    for i, v in enumerate(unseen):
        ax1.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontweight='bold')
    ax2.bar(names, fps, color=['#ef4444', '#10b981', '#3b82f6'], edgecolor='black')
    ax2.set_ylabel("Throughput (FPS)", fontweight='bold')
    ax2.set_title("Throughput de Treino na GPU", fontweight='bold')
    for i, v in enumerate(fps):
        ax2.text(i, v, f"{v:,.0f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/01_representations_triad.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("\n[SUCESSO] Resultados reais salvos em results/representations_triad_results.json "
          "e figures/01_representations_triad.png", flush=True)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval-episodes", type=int, default=128)
    p.add_argument("--eval-horizon", type=int, default=1000)
    a = p.parse_args()
    run_triad_comparison(total_steps=a.steps, num_envs=a.num_envs, seeds=tuple(a.seeds),
                         eval_episodes=a.eval_episodes, eval_horizon=a.eval_horizon)
