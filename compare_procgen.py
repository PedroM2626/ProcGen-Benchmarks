"""
Comparação Procgen: mesmo jogo com CV vs sem CV + classic vs attention
- Usa Python 3.10 + procgen 0.10.7 + stable-baselines3 PPO
- Rápido: 50k steps ~ 2 min (vs 50 min CarRacing)
"""
import os
import json
import argparse
from datetime import datetime
import numpy as np
import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

from procgen_wrapper import make_procgen_env
from models.sb3_extractors import ClassicCNNExtractor, AttentionCNNExtractor

# Para procgen vetor, usamos MlpPolicy

def train_one(game, num_levels, distribution, use_vector, extractor_class, extractor_kwargs, timesteps, seed, log_dir, device):
    # Env com Monitor para SB3
    def make_env():
        env = make_procgen_env(game, num_levels=num_levels, distribution_mode=distribution, seed=seed, frame_stack=1, vector=use_vector)
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([lambda: Monitor(make_procgen_env(game, num_levels=0, distribution_mode=distribution, seed=seed+1000, frame_stack=1, vector=use_vector))])

    if use_vector:
        policy = "MlpPolicy"
        policy_kwargs = {}
    else:
        policy = "CnnPolicy"
        policy_kwargs = {
            "features_extractor_class": extractor_class,
            "features_extractor_kwargs": extractor_kwargs,
        }

    model = PPO(policy, vec_env, verbose=0,
                learning_rate=3e-4,
                n_steps=256,
                batch_size=64,
                n_epochs=3,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                seed=seed,
                device=device,
                policy_kwargs=policy_kwargs,
                tensorboard_log=log_dir)

    model.learn(total_timesteps=timesteps)

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=False)
    vec_env.close()
    eval_env.close()
    return float(mean_reward), float(std_reward), model

def main():
    parser = argparse.ArgumentParser(description='Procgen comparação CV vs não-CV')
    parser.add_argument('--game', type=str, default='coinrun', help='coinrun, starpilot, bossfight')
    parser.add_argument('--timesteps', type=int, default=50000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--num_levels', type=int, default=200)
    parser.add_argument('--distribution', type=str, default='easy')
    parser.add_argument('--log_dir', type=str, default='./logs_procgen')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Device: {device} torch {torch.__version__}")
    os.makedirs(args.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comp_dir = os.path.join(args.log_dir, f"comparison_{args.game}_{timestamp}")
    os.makedirs(comp_dir, exist_ok=True)

    configs = [
        ('pixels', False, ClassicCNNExtractor, dict(features_dim=512), 'classic_pixels'),
        ('pixels', False, AttentionCNNExtractor, dict(features_dim=512, use_cbam=True), 'attention_cbam_pixels'),
        ('pixels', False, AttentionCNNExtractor, dict(features_dim=512, use_cbam=False), 'attention_spatial_pixels'),
        ('vector', True, None, {}, 'mlp_vector'),  # sem CV
    ]

    results = {}
    for obs_type, use_vector, extractor_class, extractor_kwargs, key in configs:
        results[key] = []
        for seed in args.seeds:
            print(f"\n{'='*60}\n{key} seed {seed} ({obs_type}) game {args.game}\n{'='*60}")
            try:
                mean, std, model = train_one(args.game, args.num_levels, args.distribution, use_vector, extractor_class, extractor_kwargs, args.timesteps, seed, args.log_dir, device)
                print(f"{key} seed {seed}: {mean:.2f} +/- {std:.2f}")
                results[key].append({'seed': seed, 'mean_reward': mean, 'std_reward': std})
                try: model.save(os.path.join(comp_dir, f"{key}_seed{seed}.zip"))
                except: pass
            except Exception as e:
                import traceback
                traceback.print_exc()
                results[key].append({'seed': seed, 'mean_reward': None, 'error': str(e)})

    with open(os.path.join(comp_dir, 'comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    stats = {}
    for k, v in results.items():
        rewards = [x['mean_reward'] for x in v if x['mean_reward'] is not None]
        if rewards:
            stats[k] = {'mean': float(np.mean(rewards)), 'std': float(np.std(rewards)), 'min': float(np.min(rewards)), 'max': float(np.max(rewards)), 'n': len(rewards)}
        else:
            stats[k] = None
    with open(os.path.join(comp_dir, 'statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # Plot
    try:
        import matplotlib.pyplot as plt
        arch_names = []
        means = []
        stds = []
        for k, s in stats.items():
            if s is not None:
                arch_names.append(k.replace('_',' ').title())
                means.append(s['mean'])
                stds.append(s['std'])
        if arch_names:
            plt.figure(figsize=(10,6))
            colors = ['lightblue','lightgreen','lightcoral','lightsalmon']
            bars = plt.bar(arch_names, means, yerr=stds, capsize=5, alpha=0.8, color=colors[:len(arch_names)])
            plt.ylabel('Mean Reward (10 eps)')
            plt.title(f'Procgen {args.game} - {args.timesteps} steps - {args.num_levels} levels')
            plt.xticks(rotation=25, ha='right')
            plt.tight_layout()
            for bar, m, s in zip(bars, means, stds):
                plt.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{m:.1f}±{s:.1f}', ha='center', va='bottom', fontsize=9)
            plt.savefig(os.path.join(comp_dir, 'comparison_plot.png'), dpi=150, bbox_inches='tight')
            print(f"Plot salvo em {comp_dir}")
    except Exception as e:
        print(f"Plot erro: {e}")

    # Txt report
    with open(os.path.join(comp_dir, 'comparison_report.txt'), 'w') as f:
        f.write(f"Procgen {args.game} - {args.timesteps} steps\n")
        f.write("="*60+"\n")
        for k, s in stats.items():
            if s is not None:
                f.write(f"{k}: mean {s['mean']:.2f} std {s['std']:.2f} n {s['n']}\n")
            else:
                f.write(f"{k}: sem dados\n")

    print(f"\nResultados em {comp_dir}")
    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    main()
