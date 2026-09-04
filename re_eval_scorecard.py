"""
Re-avaliação estendida (Roadmap 6.1 itens #1, #2 e #5):
- #1: n_eval_episodes 10 -> 30 unseen (num_levels=0)
- #2: dual-mode deterministic=True + False
- #5: generalization gap = train perf (num_levels=200, seed treino) - unseen perf
Usa os .zip salvos de new_archs (75) e maze_heist (40). Sem retreino.
Salva results/re_eval_results.json incrementalmente após cada modelo.
"""
import os, json, glob, argparse
import numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from procgen_wrapper import make_procgen_env

def eval_model(model, game, seed, n_unseen, n_train, device):
    unseen = DummyVecEnv([lambda: Monitor(make_procgen_env(game, num_levels=0, distribution_mode='easy', seed=seed+1000, vector=False))])
    train = DummyVecEnv([lambda: Monitor(make_procgen_env(game, num_levels=200, distribution_mode='easy', seed=seed, vector=False))])
    m_st, _ = evaluate_policy(model, unseen, n_eval_episodes=n_unseen, deterministic=False)
    m_dt, _ = evaluate_policy(model, unseen, n_eval_episodes=n_unseen, deterministic=True)
    m_tr, _ = evaluate_policy(model, train, n_eval_episodes=n_train, deterministic=False)
    unseen.close(); train.close()
    return float(m_st), float(m_dt), float(m_tr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n_unseen', type=int, default=30)
    parser.add_argument('--n_train', type=int, default=15)
    parser.add_argument('--out', type=str, default='results/re_eval_results.json')
    args = parser.parse_args()
    base = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f: results = json.load(f)  # resume

    zips = sorted(glob.glob(os.path.join(base, 'logs_new_archs', 'new_archs_*', '*.zip'))) + \
           sorted(glob.glob(os.path.join(base, 'logs_maze_heist', 'maze_heist_*', '*.zip')))
    print(f"{len(zips)} modelos, device={args.device}, n_unseen={args.n_unseen} (stoch+det), n_train={args.n_train}")

    for i, z in enumerate(zips):
        # nome: {game}_{config}_seed{seed}.zip
        name = os.path.splitext(os.path.basename(z))[0]
        if name in results: continue
        parts = name.rsplit('_seed', 1)
        seed = int(parts[1]); gk = parts[0]
        game = gk.split('_', 1)[0]
        try:
            model = PPO.load(z, device=args.device)
            m_st, m_dt, m_tr = eval_model(model, game, seed, args.n_unseen, args.n_train, args.device)
            results[name] = {
                'stoch_unseen': round(m_st, 3), 'det_unseen': round(m_dt, 3),
                'stoch_train': round(m_tr, 3), 'gen_gap': round(m_tr - m_st, 3),
                'n_unseen': args.n_unseen, 'n_train': args.n_train,
            }
            print(f"[{i+1}/{len(zips)}] {name}: stoch={m_st:.2f} det={m_dt:.2f} train={m_tr:.2f} gap={m_tr-m_st:+.2f}")
        except Exception as e:
            results[name] = {'error': str(e)}
            print(f"[{i+1}/{len(zips)}] {name}: ERRO {e}")
        with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f"Concluído: {out_path}")

if __name__ == '__main__':
    main()
