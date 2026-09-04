"""
ProcGen-parity suite — 100% REAL (rewrite).

The previous version hard-coded most axes (QR-DQN/World-Model/Augmentation/Budget numbers)
and used a fabricated multiplier for the HRL deterministic score. This version measures only
genuine quantities with the real trainers:

  Eixo 1 — Budget scaling: PPO treinado em orçamentos crescentes (retorno unseen real + gen-gap).
  Eixo 2 — Exploração: PPO(none) vs ICM vs RND (AuxPPOTrainer), retorno unseen real.
  Eixo 3 — HRL: flat/skip4/hrl/hrl_learned (HRLTrainer real), retorno episódico real.
  Eixo 4 — Throughput de arquiteturas visuais: FPS real de forward-pass (NatureCNN/Attention/CBAM/ViT).

Axes that could not be measured honestly here (QR-DQN, augmentation-for-generalization) are
covered by dedicated experiments or intentionally omitted — never fabricated.
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np

from src.env import CraftaxLevelManager
from src.networks import NatureCNN, SymbolicActorCritic
from src.ppo import PPOTrainer, create_train_state
from src.hrl import HRLTrainer
from src.aux_ppo import AuxPPOTrainer
from src.combinatorial_engine import FeatureExtractorNatureCNN
from src.eval_utils import make_craftax_evaluator


def _ppo_run(env_manager, total_steps, num_envs, seed, eval_episodes, eval_horizon, model=None):
    model = model or SymbolicActorCritic(action_dim=env_manager.num_actions)
    obs_sample, _ = env_manager.env.reset(jax.random.PRNGKey(0), env_manager.params)
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, run_rng = jax.random.split(rng, 3)
    ts = create_train_state(model, init_rng, obs_sample.shape, learning_rate=3e-4)
    tr = PPOTrainer(model=model, env_manager=env_manager, num_envs=num_envs, num_steps=64)
    obs, es, run_rng = env_manager.reset_train(run_rng, num_envs)
    carry = (ts, es, obs, run_rng)
    step = jax.jit(tr.train_step)
    iters = max(1, total_steps // (num_envs * 64))
    for _ in range(iters):
        carry, _ = step(carry)
    ev = make_craftax_evaluator(env_manager, lambda p, o, r: jnp.argmax(model.apply({'params': p}, o)[0], -1),
                                num_envs=eval_episodes, horizon=eval_horizon)
    e = jax.random.PRNGKey(seed + 999); e, r1, r2 = jax.random.split(e, 3)
    tr_m, _ = ev(carry[0].params, r1, False)
    un_m, un_s = ev(carry[0].params, r2, True)
    return tr_m, un_m, un_s


def run_parity_suite(budget_steps=(1_000_000, 4_000_000, 8_000_000), explore_steps=2_000_000,
                     hrl_steps=4_000_000, num_envs=256, seeds=(0, 1), eval_episodes=64, eval_horizon=1000):
    print("=" * 100, flush=True)
    print("   SUÍTE DE PARIDADE PROCGEN — 100% REAL (budget scaling, exploração, HRL, arquiteturas)", flush=True)
    print(f"   Backend: {jax.default_backend().upper()} | Device: {jax.devices()[0]}", flush=True)
    print("=" * 100, flush=True)
    out = Path("results/procgen_parity_master_results.json")
    results = {}

    sym = CraftaxLevelManager(use_pixels=False, num_train_levels=200, eval_seed_offset=1000)

    # EIXO 1 — Budget scaling (real)
    print("\n[EIXO 1] Budget scaling (PPO)...", flush=True)
    budget = {}
    for bs in budget_steps:
        tr_m, un_m, un_s = _ppo_run(sym, bs, num_envs, seeds[0], eval_episodes, eval_horizon)
        budget[f"{bs//1000}k"] = {"train": round(tr_m, 3), "unseen": round(un_m, 3),
                                  "unseen_std": round(un_s, 3), "gen_gap": round(tr_m - un_m, 3)}
        print(f"  {bs:>9,} steps -> unseen={un_m:.3f}±{un_s:.3f} gap={tr_m-un_m:+.3f}", flush=True)
        results["Eixo_1_Budget_Scaling"] = budget; _save(out, results)

    # EIXO 2 — Exploração (real): none vs ICM vs RND
    print("\n[EIXO 2] Exploração (PPO vs ICM vs RND)...", flush=True)
    expl = {}
    for aux in ["none", "icm", "rnd"]:
        runs = []
        for seed in seeds:
            rng = jax.random.PRNGKey(seed); rng, i1, r1 = jax.random.split(rng, 3)
            tr = AuxPPOTrainer(extractor_cls=FeatureExtractorNatureCNN, env_manager=sym, aux_type=aux,
                               num_envs=num_envs, num_steps=64, action_dim=sym.num_actions)
            obs_s, _ = sym.env.reset(jax.random.PRNGKey(0), sym.params)
            p, os_, aos = tr.create_state(i1, obs_s.shape)
            o, es, rr = sym.reset_train(r1, num_envs)
            carry = (p, os_, aos, es, o, rr); step = tr.make_train_step()
            for _ in range(max(1, explore_steps // (num_envs * 64))):
                carry, _ = step(carry, None)
            ev = make_craftax_evaluator(sym, tr.make_eval_policy(True), num_envs=eval_episodes, horizon=eval_horizon)
            e = jax.random.PRNGKey(seed + 999); e, k1, k2 = jax.random.split(e, 3)
            _, us = ev(carry[0], k2, True)
            um, _ = ev(carry[0], k1, False)
            runs.append((um, us))
        expl[aux] = {"train": round(float(np.mean([r[0] for r in runs])), 3),
                     "unseen": round(float(np.mean([r[1] for r in runs])), 3)}
        print(f"  {aux:<6} unseen={expl[aux]['unseen']:.3f}", flush=True)
        results["Eixo_2_Exploracao"] = expl; _save(out, results)

    # EIXO 3 — HRL (real)
    print("\n[EIXO 3] HRL (flat/skip4/hrl/hrl_learned)...", flush=True)
    hrl = {}
    obs_s, _ = sym.env.reset(jax.random.PRNGKey(0), sym.params)
    for mode in ["flat", "skip4", "hrl", "hrl_learned"]:
        runs = []
        for seed in seeds:
            rng = jax.random.PRNGKey(seed); rng, i1, r1 = jax.random.split(rng, 3)
            ht = HRLTrainer(mode=mode, env_manager=sym, num_envs=num_envs, num_steps=64, skip_k=4)
            hs = ht.create_state(i1, obs_s.shape)
            o, es, rr = sym.reset_train(r1, num_envs)
            carry = (hs, es, o, rr); step = jax.jit(ht.train_step)
            for _ in range(max(1, hrl_steps // (num_envs * 64 * ht.skip_k))):
                carry, _ = step(carry)
            ev = make_craftax_evaluator(sym, ht.make_eval_policy(True), num_envs=eval_episodes, horizon=eval_horizon)
            e = jax.random.PRNGKey(seed + 999); e, k1, k2 = jax.random.split(e, 3)
            tm, _ = ev(carry[0].params, k1, False); um, _ = ev(carry[0].params, k2, True)
            runs.append((tm, um))
        hrl[mode] = {"train": round(float(np.mean([r[0] for r in runs])), 3),
                     "unseen": round(float(np.mean([r[1] for r in runs])), 3)}
        print(f"  {mode:<12} train={hrl[mode]['train']:.3f} unseen={hrl[mode]['unseen']:.3f}", flush=True)
        results["Eixo_3_HRL"] = hrl; _save(out, results)

    # EIXO 4 — Throughput real de arquiteturas visuais
    print("\n[EIXO 4] Throughput de arquiteturas (FPS real de forward)...", flush=True)
    from src.procgen_parity_modules import AttentionCNN
    from src.advanced_modules import VisionTransformer
    pix = CraftaxLevelManager(use_pixels=True, num_train_levels=10)
    p_obs, _ = pix.env.reset(jax.random.PRNGKey(0), pix.params)
    pb = jnp.zeros((32, *p_obs.shape))
    arch = {}
    models = {"NatureCNN": NatureCNN(action_dim=pix.num_actions),
              "AttentionCNN": AttentionCNN(action_dim=pix.num_actions, use_cbam=False),
              "CBAM": AttentionCNN(action_dim=pix.num_actions, use_cbam=True),
              "ViT": VisionTransformer(action_dim=pix.num_actions, num_heads=4, num_layers=2)}
    for name, m in models.items():
        rng = jax.random.PRNGKey(0)
        pr = m.init(rng, pb); fn = jax.jit(m.apply)
        _ = fn(pr, pb); jax.block_until_ready(_)
        t0 = time.time()
        for _ in range(50):
            o = fn(pr, pb)
        jax.block_until_ready(o)
        arch[name] = {"fps": round((50 * 32) / (time.time() - t0), 1)}
        print(f"  {name:<12} {arch[name]['fps']:,.0f} FPS", flush=True)
    results["Eixo_4_Arquiteturas"] = arch; _save(out, results)

    print("\n[SUCESSO] Paridade ProcGen real salva em results/procgen_parity_master_results.json", flush=True)
    return results


def _save(out, results):
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--explore-steps", type=int, default=2_000_000)
    p.add_argument("--hrl-steps", type=int, default=4_000_000)
    a = p.parse_args()
    run_parity_suite(num_envs=a.num_envs, seeds=tuple(a.seeds),
                     explore_steps=a.explore_steps, hrl_steps=a.hrl_steps)
