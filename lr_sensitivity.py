"""
Teste de sensibilidade de lr (seção 12.2): DQN/QR-DQN a 3e-4 (lr do PPO/A2C) em starpilot,
o jogo com o maior gap value-vs-policy. Responde: a conclusão 'policy > value' depende do lr 1e-4?
10 runs (2 algos × 5 seeds), mesmo protocolo (100 eps stoch+det+15 train), resume-safe.
Baseline lr=1e-4 já está em results/algo_families_results.json (starpilot_dqn/qrdqn).
"""
import os, json, argparse
import numpy as np, torch
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from procgen_wrapper import make_procgen_env
from compare_algo_families import make_algo, eval_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='starpilot')
    parser.add_argument('--algos', nargs='+', default=['dqn', 'qrdqn'])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument('--log_dir', default='./logs_algo_lr')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--out', default='results/lr_sensitivity_results.json')
    args = parser.parse_args()
    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else ('cpu' if args.device == 'auto' else args.device)

    base = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    zip_dir = os.path.join(base, 'logs_algo_lr', 'lr_zips')
    os.makedirs(zip_dir, exist_ok=True)
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f: results = json.load(f)

    lr_tag = f"lr{args.lr:.0e}".replace('-0', '-').replace('+', '')
    jobs = [(a, s) for a in args.algos for s in args.seeds]
    print(f"{len(jobs)} jobs ({args.game} × {args.algos} × {args.seeds}), {lr_tag}, device={device}")
    for i, (algo, seed) in enumerate(jobs):
        name = f"{args.game}_{algo}_{lr_tag}_seed{seed}"
        if name in results and 'error' not in results[name]: continue
        print(f"\n[{i+1}/{len(jobs)}] {name}")
        try:
            vec = DummyVecEnv([lambda: Monitor(make_procgen_env(args.game, num_levels=200, distribution_mode='easy', seed=seed, vector=False))])
            model = make_algo(algo, vec, seed, device, args.log_dir, value_lr=args.lr)
            model.learn(total_timesteps=args.timesteps)
            vec.close()
            m_st, m_dt, m_tr = eval_model(model, args.game, seed)
            try: model.save(os.path.join(zip_dir, f"{name}.zip"))
            except Exception as e: print(f"  zip falhou: {e}")
            results[name] = {'stoch_unseen': round(m_st, 3), 'det_unseen': round(m_dt, 3),
                             'stoch_train': round(m_tr, 3), 'gen_gap': round(m_tr - m_st, 3),
                             'n_unseen': 100, 'n_train': 15, 'lr': args.lr}
            print(f"  stoch={m_st:.2f} det={m_dt:.2f} train={m_tr:.2f} gap={m_tr-m_st:+.2f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = {'error': str(e)}
        with open(out_path, 'w') as f: json.dump(results, f, indent=2)

    # comparativo direto com o baseline 1e-4
    fam = json.load(open(os.path.join(base, 'results/algo_families_results.json'), encoding='utf-8'))
    print('\nCOMPARATIVO starpilot (stoch unseen):')
    for algo in args.algos:
        b = [v['stoch_unseen'] for k, v in fam.items() if k.startswith(f'{args.game}_{algo}_seed') and 'error' not in v]
        n = [v['stoch_unseen'] for k, v in results.items() if k.startswith(f'{args.game}_{algo}_{lr_tag}_seed') and 'error' not in v]
        if b and n:
            print(f"  {algo:6s} lr1e-4: {np.mean(b):.2f}±{np.std(b):.2f}  vs  {lr_tag}: {np.mean(n):.2f}±{np.std(n):.2f}")
    print(f"Concluído: {out_path}")

if __name__ == '__main__':
    main()
