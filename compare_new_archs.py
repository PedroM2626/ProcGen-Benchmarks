"""
Apenas 5 novas arquiteturas #4 em 3 jogos 100k 5 seeds — para juntar com benchmark 1 (classic/cbam/spatial/mlp já treinados 5 seeds)
"""
import os, json, argparse
from datetime import datetime
import numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from procgen_wrapper import make_procgen_env
from models.combined_extractors import ImpalaCNNExtractor, ImpoolaCNNExtractor, LSTMAttentionExtractor, ViTExtractor, ResNet18Extractor

def train_one(game, cls, kwargs, timesteps, seed, log_dir, device):
    def make_env(): return Monitor(make_procgen_env(game, num_levels=200, distribution_mode='easy', seed=seed, vector=False))
    vec = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([lambda: Monitor(make_procgen_env(game, num_levels=0, distribution_mode='easy', seed=seed+1000, vector=False))])
    model = PPO("CnnPolicy", vec, verbose=0, learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=3, gamma=0.99, gae_lambda=0.95, clip_range=0.2, seed=seed, device=device, policy_kwargs={"features_extractor_class": cls, "features_extractor_kwargs": kwargs}, tensorboard_log=log_dir)
    model.learn(total_timesteps=timesteps)
    mean,std = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=False)
    vec.close(); eval_env.close()
    return float(mean), float(std), model

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,43,44,45,46])
    parser.add_argument('--games', type=str, nargs='+', default=['bossfight','starpilot','dodgeball'])
    parser.add_argument('--log_dir', type=str, default='./logs_new_archs')
    parser.add_argument('--device', type=str, default='auto')
    args=parser.parse_args()
    device='cuda' if (args.device=='auto' and torch.cuda.is_available()) else args.device if args.device!='auto' else 'cpu'
    print(f"Device: {device} torch {torch.__version__} NEW 5 arquiteturas")
    os.makedirs(args.log_dir, exist_ok=True)
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    comp_dir=os.path.join(args.log_dir, f"new_archs_{'_'.join(args.games)}_{ts}")
    os.makedirs(comp_dir, exist_ok=True)
    configs=[
        ('impala', ImpalaCNNExtractor, dict(features_dim=512)),
        ('impoola', ImpoolaCNNExtractor, dict(features_dim=512)),
        ('lstm_attention', LSTMAttentionExtractor, dict(features_dim=512)),
        ('vit', ViTExtractor, dict(features_dim=512)),
        ('resnet18', ResNet18Extractor, dict(features_dim=512)),
    ]
    results={}
    for game in args.games:
        for key, cls, kw in configs:
            gk=f"{game}_{key}"
            results[gk]=[]
            for seed in args.seeds:
                print(f"\n{'='*60}\n{gk} seed {seed} {game}\n{'='*60}")
                try:
                    mean,std,model = train_one(game, cls, kw, args.timesteps, seed, args.log_dir, device)
                    print(f"{gk} seed {seed}: {mean:.2f} +/- {std:.2f}")
                    results[gk].append({'seed': seed, 'mean_reward': mean, 'std_reward': std})
                    try: model.save(os.path.join(comp_dir, f"{gk}_seed{seed}.zip"))
                    except: pass
                except Exception as e:
                    import traceback; traceback.print_exc()
                    results[gk].append({'seed': seed, 'mean_reward': None, 'error': str(e)})
    with open(os.path.join(comp_dir,'comparison_results.json'),'w') as f: json.dump(results,f,indent=2)
    stats={}
    for k,v in results.items():
        rewards=[x['mean_reward'] for x in v if x['mean_reward'] is not None]
        stats[k]={'mean':float(np.mean(rewards)),'std':float(np.std(rewards)),'min':float(np.min(rewards)),'max':float(np.max(rewards)),'n':len(rewards)} if rewards else None
    with open(os.path.join(comp_dir,'statistics.json'),'w') as f: json.dump(stats,f,indent=2)
    try:
        import matplotlib.pyplot as plt
        for game in args.games:
            keys=[k for k in stats if k.startswith(game+'_')]
            means=[stats[k]['mean'] if stats[k] else 0 for k in keys]
            stds=[stats[k]['std'] if stats[k] else 0 for k in keys]
            plt.figure(figsize=(10,6)); plt.bar(keys, means, yerr=stds, capsize=3, alpha=0.8)
            plt.xticks(rotation=25, ha='right', fontsize=7); plt.ylabel('Mean Reward'); plt.title(f"New Archs {game} - {args.timesteps} steps 5 seeds")
            plt.tight_layout(); plt.savefig(os.path.join(comp_dir, f"new_archs_{game}_plot.png"), dpi=150, bbox_inches='tight'); plt.close()
        print(f"Plots salvos em {comp_dir}")
    except Exception as e: print(f"Plot erro: {e}")
    print(f"\nResultados em {comp_dir}\n"+json.dumps(stats,indent=2))

if __name__=='__main__': main()
