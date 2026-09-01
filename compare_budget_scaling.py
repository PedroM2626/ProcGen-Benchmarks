"""
Budget Scaling (Roadmap 6.1 item #6): a vantagem dos vencedores persiste com mais budget?
- Configs: resnet18 (top-2 global) + mlp_vector (top-1 global)
- Jogos: starpilot (casa do mlp_vector) + dodgeball (casa do resnet18)
- Budgets novos: 250k e 500k; o ponto 100k já existe (re-eval 100 eps, seeds 42-44 servem de baseline)
- Hiperparâmetros idênticos a todos os benchmarks; eval definitivo 100 eps stoch+det + 15 train
- Saída: logs_budget/budget_zips + results/budget_results.json (incremental, resume-safe, retry de erros)
"""
import os, json, argparse
import numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from procgen_wrapper import make_procgen_env
from models.combined_extractors import ResNet18Extractor

CONFIGS = [
    ('resnet18', ResNet18Extractor, dict(features_dim=512), False),
    ('mlp_vector', None, {}, True),   # wrapper vetorial 256D, MlpPolicy
]

def eval_model(model, game, seed, vector, n_unseen=100, n_train=15):
    unseen = DummyVecEnv([lambda: Monitor(make_procgen_env(game, num_levels=0, distribution_mode='easy', seed=seed+1000, vector=vector))])
    train = DummyVecEnv([lambda: Monitor(make_procgen_env(game, num_levels=200, distribution_mode='easy', seed=seed, vector=vector))])
    m_st, _ = evaluate_policy(model, unseen, n_eval_episodes=n_unseen, deterministic=False)
    m_dt, _ = evaluate_policy(model, unseen, n_eval_episodes=n_unseen, deterministic=True)
    m_tr, _ = evaluate_policy(model, train, n_eval_episodes=n_train, deterministic=False)
    unseen.close(); train.close()
    return float(m_st), float(m_dt), float(m_tr)

def train_one(game, cls, kwargs, vector, timesteps, seed, log_dir, device):
    vec = DummyVecEnv([lambda: Monitor(make_procgen_env(game, num_levels=200, distribution_mode='easy', seed=seed, vector=vector))])
    if vector:
        policy, pk = "MlpPolicy", {}
    else:
        policy, pk = "CnnPolicy", {"features_extractor_class": cls, "features_extractor_kwargs": kwargs}
    model = PPO(policy, vec, verbose=0, learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=3,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, seed=seed, device=device,
                policy_kwargs=pk, tensorboard_log=log_dir)
    model.learn(total_timesteps=timesteps)
    vec.close()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', nargs='+', default=['starpilot', 'dodgeball'])
    parser.add_argument('--budgets', nargs='+', type=int, default=[250000, 500000])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--log_dir', default='./logs_budget')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--out', default='results/budget_results.json')
    args = parser.parse_args()
    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else ('cpu' if args.device == 'auto' else args.device)

    base = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    zip_dir = os.path.join(base, 'logs_budget', 'budget_zips')
    os.makedirs(zip_dir, exist_ok=True)
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f: results = json.load(f)

    jobs = [(g, key, cls, kw, vec, b, s)
            for b in args.budgets for g in args.games for key, cls, kw, vec in CONFIGS for s in args.seeds]
    print(f"{len(jobs)} jobs ({len(args.budgets)} budgets × {args.games} × {len(CONFIGS)} configs × {args.seeds}), device={device}")
    for i, (game, key, cls, kw, vec, budget, seed) in enumerate(jobs):
        name = f"{game}_{key}_b{budget//1000}k_seed{seed}"
        if name in results and 'error' not in results[name]: continue
        print(f"\n[{i+1}/{len(jobs)}] {name}")
        try:
            model = train_one(game, cls, kw, vec, budget, seed, args.log_dir, device)
            m_st, m_dt, m_tr = eval_model(model, game, seed, vec)
            try: model.save(os.path.join(zip_dir, f"{name}.zip"))
            except Exception as e: print(f"  zip falhou: {e}")
            results[name] = {'stoch_unseen': round(m_st, 3), 'det_unseen': round(m_dt, 3),
                             'stoch_train': round(m_tr, 3), 'gen_gap': round(m_tr - m_st, 3),
                             'n_unseen': 100, 'n_train': 15, 'budget': budget}
            print(f"  stoch={m_st:.2f} det={m_dt:.2f} train={m_tr:.2f} gap={m_tr-m_st:+.2f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = {'error': str(e)}
        with open(out_path, 'w') as f: json.dump(results, f, indent=2)
    print(f"Concluído: {out_path}")

if __name__ == '__main__':
    main()
