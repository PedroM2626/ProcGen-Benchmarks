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

from src.env import CraftaxLevelManager
from src.combinatorial_engine import FeatureExtractorNatureCNN
from src.advanced_modules import VisionTransformer
from src.aux_ppo import AuxPPOTrainer
from src.eval_utils import make_craftax_evaluator


PARADIGMS = [
    ("Baseline_PPO", "none", "PPO puro (NatureCNN)"),
    ("PPO_ICM", "icm", "PPO + Intrinsic Curiosity (recompensa intrínseca real)"),
    ("PPO_Contrastive", "contrastive", "PPO + CURL/InfoNCE no encoder"),
    ("PPO_WorldModel", "world_model", "PPO + Latent World Model (dinâmica+reward)"),
]


def train_paradigm(aux_type, total_steps, num_envs, seed, eval_episodes, eval_horizon):
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
    tr_mean, _ = eval_fn(final_params, r1, unseen=False)
    un_mean, un_std = eval_fn(final_params, r2, unseen=True)
    return {"fps": fps, "elapsed": elapsed, "steps": int(real_steps), "train": tr_mean,
            "unseen": un_mean, "unseen_std": un_std,
            "final_aux_loss": float(last["aux_loss"]) if last else 0.0}


def run_advanced_benchmark(total_steps=3_000_000, num_envs=128, seeds=(0, 1, 2),
                           eval_episodes=128, eval_horizon=1000):
    print("=" * 100, flush=True)
    print("   BENCHMARK REAL: WORLD MODEL, ICM, CONTRASTIVE & ViT vs CNN (PPO treinado)", flush=True)
    print(f"   Backend: {jax.default_backend().upper()} | Device: {jax.devices()[0]}", flush=True)
    print(f"   total_steps={total_steps:,} | num_envs={num_envs} | seeds={list(seeds)}", flush=True)
    print("=" * 100, flush=True)

    results = {}
    for name, aux, desc in PARADIGMS:
        runs = []
        print(f"\n>>> Treinando {name} (aux={aux})...", flush=True)
        for seed in seeds:
            r = train_paradigm(aux, total_steps, num_envs, seed, eval_episodes, eval_horizon)
            runs.append({"seed": seed, **r})
            print(f"  [{name} s{seed}] FPS={r['fps']:,.0f} Unseen={r['unseen']:.2f}±{r['unseen_std']:.2f} "
                  f"aux_loss={r['final_aux_loss']:.4f}", flush=True)
        results[name] = {
            "descricao": desc, "aux_type": aux,
            "fps": round(float(np.mean([x["fps"] for x in runs])), 1),
            "time_sec": round(float(np.mean([x["elapsed"] for x in runs])), 2),
            "train_score": round(float(np.mean([x["train"] for x in runs])), 3),
            "unseen_score": round(float(np.mean([x["unseen"] for x in runs])), 3),
            "unseen_std": round(float(np.mean([x["unseen_std"] for x in runs])), 3),
            "seed_unseen_std": round(float(np.std([x["unseen"] for x in runs])), 3),
            "final_aux_loss": round(float(np.mean([x["final_aux_loss"] for x in runs])), 4),
            "runs": runs,
        }
        _save(results)

    # ---- Architecture throughput: NatureCNN vs ViT (real forward-pass timing) ----
    print("\n>>> Medindo throughput real: NatureCNN vs Vision Transformer (ViT) em pixels...", flush=True)
    pixel_env = CraftaxLevelManager(use_pixels=True, num_train_levels=10)
    p_obs, _ = pixel_env.env.reset(jax.random.PRNGKey(0), pixel_env.params)
    p_shape = p_obs.shape
    cnn = FeatureExtractorNatureCNN()
    vit = VisionTransformer(action_dim=pixel_env.num_actions, num_heads=4, num_layers=2)
    rng = jax.random.PRNGKey(0)
    rng, r1, r2 = jax.random.split(rng, 3)
    p_batch = jnp.zeros((16, *p_shape))
    cnn_params = cnn.init(r1, p_batch)
    cnn_fn = jax.jit(cnn.apply)
    _ = cnn_fn(cnn_params, p_batch); jax.block_until_ready(_ )
    t0 = time.time()
    for _ in range(50):
        o = cnn_fn(cnn_params, p_batch)
    jax.block_until_ready(o)
    fps_cnn = (50 * 16) / (time.time() - t0)

    vit_params = vit.init(r2, p_batch)
    vit_fn = jax.jit(vit.apply)
    _ = vit_fn(vit_params, p_batch); jax.block_until_ready(_)
    t0 = time.time()
    for _ in range(50):
        o = vit_fn(vit_params, p_batch)
    jax.block_until_ready(o)
    fps_vit = (50 * 16) / (time.time() - t0)

    results["Architectures_Visual"] = {
        "NatureCNN_FPS": round(fps_cnn, 1), "ViT_Transformer_FPS": round(fps_vit, 1),
        "ViT_Relative_Speed": f"{fps_vit / fps_cnn * 100:.1f}% da CNN", "Obs_Shape": list(p_shape),
    }
    print(f"  NatureCNN: {fps_cnn:,.0f} FPS | ViT: {fps_vit:,.0f} FPS ({fps_vit/fps_cnn*100:.1f}% da CNN)", flush=True)
    _save(results)
    print("\n[SUCESSO] Resultados reais em results/advanced_paradigms_results.json", flush=True)
    return results


def _save(results):
    out = Path("results/advanced_paradigms_results.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval-episodes", type=int, default=128)
    p.add_argument("--eval-horizon", type=int, default=1000)
    a = p.parse_args()
    run_advanced_benchmark(total_steps=a.steps, num_envs=a.num_envs, seeds=tuple(a.seeds),
                           eval_episodes=a.eval_episodes, eval_horizon=a.eval_horizon)
