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

from src.boxing_3d_env import HumanoidBoxing3DEnv
from src.continuous_rl import SACTrainer
from src.eval_utils import make_fixed_horizon_evaluator
import src.offline_rl as ORL


def train_teacher(base_env, teacher_steps, num_envs, seed):
    wrapper = ORL.BoxingOpponentWrapper(base_env)
    rng = jax.random.PRNGKey(seed)
    rng, init_rng, run_rng = jax.random.split(rng, 3)
    tr = SACTrainer(wrapper, obs_dim=ORL.OBS_DIM, action_dim=ORL.ACT_DIM, num_envs=num_envs,
                    buffer_size=200000, batch_size=256)
    params, target, opt_state, log_alpha, alpha_opt, buffer = tr.create_state(init_rng)
    reset_vmap = jax.jit(jax.vmap(wrapper.reset))
    obs, env_state = reset_vmap(jax.random.split(run_rng, num_envs))
    carry = (params, target, opt_state, log_alpha, alpha_opt, buffer, env_state, obs, run_rng)
    step = tr.make_train_step()
    iters = max(1, teacher_steps // num_envs)
    t0 = time.time()
    for it in range(iters):
        carry, m = step(carry, None)
        if it % 2000 == 0 or it == iters - 1:
            print(f"    [teacher s{seed}] it {it}/{iters} critic={float(m['critic_loss']):.3f} "
                  f"alpha={float(m['alpha']):.3f} rew={float(m['mean_reward']):.3f}", flush=True)
    elapsed = time.time() - t0
    teacher_params = carry[0]
    sel = tr.make_eval_policy()
    teacher_fn = lambda o, r: sel(teacher_params, o, r)
    return teacher_params, teacher_fn, wrapper, iters * num_envs, elapsed


def main(teacher_steps=1_000_000, offline_steps=40000, dataset_size=200000, num_envs=64,
         num_rounds=64, seed=0):
    print("=" * 118, flush=True)
    print("   GRAND PRIX DE BOXE 3D — 100% TREINADO (SAC teacher real + suíte offline real)", flush=True)
    print(f"   Backend: {jax.default_backend().upper()} | Device: {jax.devices()[0]}", flush=True)
    print(f"   teacher_steps={teacher_steps:,} | dataset={dataset_size:,} | offline_steps={offline_steps:,}", flush=True)
    print("=" * 118, flush=True)

    base_env = HumanoidBoxing3DEnv(max_steps=120)
    results_dir = Path("results"); results_dir.mkdir(exist_ok=True)

    # 1. SAC teacher (real online RL vs sparring opponent)
    print("\n1. Treinando Teacher SAC online (P1) contra sparring (P2)...", flush=True)
    teacher_params, teacher_fn, wrapper, t_steps, t_time = train_teacher(base_env, teacher_steps, num_envs, seed)
    print(f"   [OK] Teacher treinado em {t_time:.1f}s ({t_steps:,} steps).", flush=True)

    # persist teacher weights (real)
    flat = jax.tree_util.tree_map(lambda x: np.asarray(x), teacher_params)
    np.savez(results_dir / "boxing_sac_teacher.npz",
             **{f"{k}_{i}": v for i, (k, v) in enumerate(_flatten(flat))})

    # 2. Dataset offline coletado do teacher
    steps_per_env = max(1, dataset_size // num_envs)
    print(f"\n2. Coletando dataset offline ({num_envs}x{steps_per_env} = {num_envs*steps_per_env:,} transições)...", flush=True)
    ds = ORL.collect_dataset(wrapper, teacher_fn, num_envs, steps_per_env, jax.random.PRNGKey(seed + 1))
    np.savez_compressed(results_dir / "dataset_boxing_expert.npz",
                        obs=np.asarray(ds["obs"]), act=np.asarray(ds["act"]),
                        rew=np.asarray(ds["rew"]), next_obs=np.asarray(ds["next_obs"]),
                        done=np.asarray(ds["done"]), rtg=np.asarray(ds["rtg"]))
    print(f"   [OK] Dataset salvo em results/dataset_boxing_expert.npz", flush=True)

    rng = jax.random.PRNGKey(seed + 2)
    contenders = {}

    # 3. Train each offline contender (real gradient training)
    print("\n3. Treinando a suíte offline completa...", flush=True)
    def _t(name, fn):
        t0 = time.time()
        pol = fn()
        print(f"   [OK] {name} treinado em {time.time()-t0:.1f}s", flush=True)
        contenders[name] = pol

    rng, r = jax.random.split(rng); _t("BC", lambda: ORL.train_bc(ds, r, steps=offline_steps))
    rng, r = jax.random.split(rng); _t("BC_SAC", lambda: ORL.train_bc_sac(ds, r, steps=offline_steps))
    rng, r = jax.random.split(rng); _t("IQL", lambda: ORL.train_iql(ds, r, steps=offline_steps))
    rng, r = jax.random.split(rng); _t("CQL", lambda: ORL.train_cql(ds, r, steps=offline_steps))
    rng, r = jax.random.split(rng); _t("BCQ", lambda: ORL.train_bcq(ds, r, steps=offline_steps))
    rng, r = jax.random.split(rng); _t("DT", lambda: ORL.train_dt(ds, r, steps=offline_steps))
    rng, r = jax.random.split(rng); _t("GAIL", lambda: ORL.train_gail(ds, wrapper, r, steps=offline_steps))

    contenders["Teacher (SAC Online)"] = teacher_fn
    contenders["Random Baseline"] = lambda o, r: jax.random.uniform(r, (o.shape[0], ORL.ACT_DIM), minval=-1.0, maxval=1.0)

    family = {
        "BC": "Imitation Learning Puro", "BC_SAC": "Offline-to-Online Híbrido",
        "IQL": "Offline RL (Expectile)", "CQL": "Offline RL (Pessimista)",
        "BCQ": "Offline Generative VAE", "DT": "Sequence Modeling / Attn",
        "GAIL": "Adversarial Imitation", "Teacher (SAC Online)": "Online Expert (Teacher)",
        "Random Baseline": "Aleatório",
    }

    # 4. Evaluate all contenders in the real physics ring
    print(f"\n4. Avaliando todos os competidores em {num_rounds} rounds reais no ringue 3D...", flush=True)
    leaderboard = []
    for name, pol in contenders.items():
        m = ORL.evaluate_boxing(base_env, pol, num_rounds=num_rounds, rng=jax.random.PRNGKey(123))
        m["nome"] = name; m["familia"] = family.get(name, "-")
        leaderboard.append(m)
        print(f"  [{name:<22}] Score={m['reward_mean']:>+7.2f}±{m['reward_std']:<5.2f} "
              f"Win={m['win_rate']:>5.1f}% KO={m['ko_rate']:>5.1f}% Hits/rd={m['hits_per_round']:.1f}", flush=True)

    leaderboard.sort(key=lambda x: x["reward_mean"], reverse=True)
    for i, r_ in enumerate(leaderboard, 1):
        r_["rank"] = i
        r_["reward_mean"] = round(r_["reward_mean"], 2); r_["reward_std"] = round(r_["reward_std"], 2)
        r_["win_rate"] = round(r_["win_rate"], 1); r_["ko_rate"] = round(r_["ko_rate"], 1)
        r_["hits_per_round"] = round(r_["hits_per_round"], 2)

    with open(results_dir / "boxing_grand_prix_results.json", "w") as f:
        json.dump(leaderboard, f, indent=2)
    with open(results_dir / "boxing_final_results.txt", "w") as f:
        f.write("PLACAR FINAL REAL — TORNEIO DE BOXE 3D (todos treinados por gradiente)\n")
        for r_ in leaderboard:
            f.write(f"#{r_['rank']:<2} {r_['nome']:<22} | {r_['reward_mean']:>+7.2f}±{r_['reward_std']:<5.2f} "
                    f"| Win {r_['win_rate']:>5.1f}% | KO {r_['ko_rate']:>5.1f}% | Hits/rd {r_['hits_per_round']}\n")

    _plot(leaderboard)
    print("\n[SUCESSO] Resultados reais em results/boxing_grand_prix_results.json "
          "e figures/09_humanoid_boxing_offline.png", flush=True)
    return leaderboard


def _flatten(d, prefix=""):
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out.extend(_flatten(v, f"{prefix}{k}_"))
        else:
            out.append((f"{prefix}{k}", v))
    return out


def _plot(lb):
    names = [r["nome"] for r in lb]
    rew = [r["reward_mean"] for r in lb]
    err = [r["reward_std"] for r in lb]
    win = [r["win_rate"] for r in lb]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    y = np.arange(len(names))
    ax1.barh(y[::-1], rew[::-1], xerr=err[::-1], color='#0284c7', edgecolor='black', capsize=3)
    ax1.set_yticks(y); ax1.set_yticklabels(names[::-1], fontsize=8, fontweight='bold')
    ax1.set_xlabel('Recompensa real por round', fontweight='bold')
    ax1.set_title('Boxe 3D — Retorno Real por Round (todos treinados)', fontweight='bold')
    ax1.axvline(0, color='black', lw=0.8)
    ax2.barh(y[::-1], win[::-1], color='#10b981', edgecolor='black')
    ax2.set_yticks(y); ax2.set_yticklabels(names[::-1], fontsize=8, fontweight='bold')
    ax2.set_xlabel('Taxa de vitória (%)', fontweight='bold'); ax2.set_xlim(0, 105)
    ax2.set_title('Taxa de Vitória Real no Ringue', fontweight='bold')
    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/09_humanoid_boxing_offline.png", dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-steps", type=int, default=1_000_000)
    p.add_argument("--offline-steps", type=int, default=40000)
    p.add_argument("--dataset-size", type=int, default=200000)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--num-rounds", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(teacher_steps=a.teacher_steps, offline_steps=a.offline_steps, dataset_size=a.dataset_size,
         num_envs=a.num_envs, num_rounds=a.num_rounds, seed=a.seed)
