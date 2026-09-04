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
from src.recurrent_and_pooling_modules import FeatureExtractorImpoola
from src.aux_ppo import AuxPPOTrainer
from src.recurrent_ppo import RecurrentPPOTrainer
from src.eval_utils import make_craftax_evaluator, make_craftax_recurrent_evaluator


def _env():
    return CraftaxLevelManager(use_pixels=True, num_train_levels=200, eval_seed_offset=1000)


def train_feedforward(extractor_cls, aux_type, total_steps, num_envs, seed, eval_episodes, eval_horizon):
    env_manager = _env()
    obs_sample, _ = env_manager.env.reset(jax.random.PRNGKey(0), env_manager.params)
    input_shape = obs_sample.shape
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, run_rng = jax.random.split(rng, 3)
    trainer = AuxPPOTrainer(extractor_cls=extractor_cls, env_manager=env_manager, aux_type=aux_type,
                            num_envs=num_envs, num_steps=64, action_dim=env_manager.num_actions)
    params, opt_state, aux_opt = trainer.create_state(init_rng, input_shape)
    obs, env_state, run_rng = env_manager.reset_train(run_rng, num_envs)
    carry = (params, opt_state, aux_opt, env_state, obs, run_rng)
    step_fn = trainer.make_train_step()
    iters = max(1, total_steps // (num_envs * 64))
    t0 = time.time()
    last = None
    for it in range(iters):
        carry, m = step_fn(carry, None)
        last = m
    elapsed = time.time() - t0
    fp = carry[0]
    ev = make_craftax_evaluator(env_manager, trainer.make_eval_policy(True),
                                num_envs=eval_episodes, horizon=eval_horizon)
    e_rng = jax.random.PRNGKey(seed + 999)
    e_rng, r1, r2 = jax.random.split(e_rng, 3)
    tr, tr_s = ev(fp, r1, unseen=False)
    un, un_s = ev(fp, r2, unseen=True)
    return {"fps": iters * num_envs * 64 / (elapsed + 1e-8), "train": tr, "train_std": tr_s,
            "unseen": un, "unseen_std": un_s, "gap": tr - un,
            "aux_loss": float(last["aux_loss"]) if last else 0.0, "elapsed": elapsed}


def train_recurrent(total_steps, num_envs, seed, eval_episodes, eval_horizon, latent_dim=256):
    env_manager = _env()
    obs_sample, _ = env_manager.env.reset(jax.random.PRNGKey(0), env_manager.params)
    input_shape = obs_sample.shape
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, run_rng = jax.random.split(rng, 3)
    trainer = RecurrentPPOTrainer(env_manager=env_manager, num_envs=num_envs, num_steps=64,
                                  latent_dim=latent_dim, action_dim=env_manager.num_actions)
    params, opt_state = trainer.create_state(init_rng, input_shape)
    obs, env_state, run_rng = env_manager.reset_train(run_rng, num_envs)
    hidden = jnp.zeros((num_envs, latent_dim))
    carry = (params, opt_state, env_state, obs, hidden, run_rng)
    step_fn = trainer.make_train_step()
    iters = max(1, total_steps // (num_envs * 64))
    t0 = time.time()
    for it in range(iters):
        carry, m = step_fn(carry, None)
    elapsed = time.time() - t0
    fp = carry[0]
    ev = make_craftax_recurrent_evaluator(env_manager, trainer.make_eval_policy(True), latent_dim,
                                          num_envs=eval_episodes, horizon=eval_horizon)
    e_rng = jax.random.PRNGKey(seed + 999)
    e_rng, r1, r2 = jax.random.split(e_rng, 3)
    tr, tr_s = ev(fp, r1, unseen=False)
    un, un_s = ev(fp, r2, unseen=True)
    return {"fps": iters * num_envs * 64 / (elapsed + 1e-8), "train": tr, "train_std": tr_s,
            "unseen": un, "unseen_std": un_s, "gap": tr - un, "aux_loss": None, "elapsed": elapsed}


def run_missing_components_benchmark(total_steps=3_000_000, num_envs=128, seeds=(0, 1, 2),
                                     eval_episodes=128, eval_horizon=1000):
    print("=" * 112, flush=True)
    print("   BENCHMARK REAL: LSTM-ATTN (recorrente) vs IMPOOLA(GAP) vs RND vs NatureCNN", flush=True)
    print(f"   Backend: {jax.default_backend().upper()} | Device: {jax.devices()[0]}", flush=True)
    print(f"   total_steps={total_steps:,} | num_envs={num_envs} | seeds={list(seeds)}", flush=True)
    print("   Nota: Craftax Classic NÃO possui toggle easy/hard como o ProcGen; o stress de", flush=True)
    print("   dificuldade é substituído pelo Generalization Gap REAL (train vs unseen).", flush=True)
    print("=" * 112, flush=True)

    specs = [
        ("NatureCNN_Baseline", "ff", FeatureExtractorNatureCNN, "none", "CNN Nature (baseline visual)"),
        ("Impoola_GAP", "ff", FeatureExtractorImpoola, "none", "Convoluções + Global Average Pooling (64D)"),
        ("RND_Exploration", "ff", FeatureExtractorNatureCNN, "rnd", "RND: bonus intrínseco de novidade (componente do NGU)"),
        ("LSTM_Attention", "rnn", None, None, "CNN + Spatial Attention + GRU recorrente (memória temporal)"),
    ]

    results = {}
    for name, kind, extractor, aux, desc in specs:
        runs = []
        print(f"\n>>> Treinando {name}...", flush=True)
        for seed in seeds:
            if kind == "ff":
                r = train_feedforward(extractor, aux, total_steps, num_envs, seed, eval_episodes, eval_horizon)
            else:
                r = train_recurrent(total_steps, num_envs, seed, eval_episodes, eval_horizon)
            runs.append({"seed": seed, **r})
            print(f"  [{name} s{seed}] FPS={r['fps']:,.0f} Train={r['train']:.2f}±{r['train_std']:.2f} "
                  f"Unseen={r['unseen']:.2f}±{r['unseen_std']:.2f} Gap={r['gap']:+.2f}", flush=True)
        results[name] = {
            "descricao": desc, "kind": kind,
            "throughput_fps": round(float(np.mean([x["fps"] for x in runs])), 0),
            "train_score": round(float(np.mean([x["train"] for x in runs])), 3),
            "unseen_score": round(float(np.mean([x["unseen"] for x in runs])), 3),
            "unseen_std": round(float(np.mean([x["unseen_std"] for x in runs])), 3),
            "seed_unseen_std": round(float(np.std([x["unseen"] for x in runs])), 3),
            "generalization_gap": round(float(np.mean([x["gap"] for x in runs])), 3),
            "runs": runs,
        }
        _save(results)

    # Figure (real numbers): train vs unseen per component
    names = [s[0] for s in specs]
    tr = [results[n]["train_score"] for n in names]
    un = [results[n]["unseen_score"] for n in names]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w/2, tr, w, label='Retorno Train', color='#10b981', edgecolor='black')
    ax.bar(x + w/2, un, w, label='Retorno Unseen (generalização)', color='#3b82f6', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(["NatureCNN", "Impoola", "RND", "LSTM-Attn"], fontweight='bold')
    ax.set_ylabel('Retorno episódico real', fontweight='bold')
    ax.set_title('Componentes ProcGen — Treino Real (PPO/PPO-recorrente) e Gap de Generalização', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/06_procgen_missing_components.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("\n[SUCESSO] Resultados reais em results/procgen_missing_components_results.json", flush=True)
    return results


def _save(results):
    out = Path("results/procgen_missing_components_results.json")
    out.parent.mkdir(exist_ok=True)
    payload = dict(results)
    payload["_nota_easy_hard"] = ("Craftax Classic nao possui modo easy/hard como o ProcGen; "
                                  "a comparacao de dificuldade foi substituida pelo Generalization Gap real.")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval-episodes", type=int, default=128)
    p.add_argument("--eval-horizon", type=int, default=1000)
    a = p.parse_args()
    run_missing_components_benchmark(total_steps=a.steps, num_envs=a.num_envs, seeds=tuple(a.seeds),
                                     eval_episodes=a.eval_episodes, eval_horizon=a.eval_horizon)
