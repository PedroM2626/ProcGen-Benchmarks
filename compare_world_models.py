"""
Comparação World Models em Procgen bossfight: VAE vs AE vs Recon vs Contrastivo
- Python 3.10 + procgen 0.10.7 + SB3 PPO, 100k steps ~60 min
"""
import os, json, argparse
from datetime import datetime
import numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from procgen_wrapper import make_procgen_env
from models.world_model_extractors import VAEExtractor, AEExtractor, ReconExtractor, ContrastiveExtractor

def train_one(game, num_levels, distribution, extractor_class, extractor_kwargs, timesteps, seed, log_dir, device):
    def make_env():
        return Monitor(make_procgen_env(game, num_levels=num_levels, distribution_mode=distribution, seed=seed, vector=False))
    vec_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([lambda: Monitor(make_procgen_env(game, num_levels=0, distribution_mode=distribution, seed=seed+1000, vector=False))])
    policy_kwargs = {"features_extractor_class": extractor_class, "features_extractor_kwargs": extractor_kwargs}
    model = PPO("CnnPolicy", vec_env, verbose=0, learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=3, gamma=0.99, gae_lambda=0.95, clip_range=0.2, seed=seed, device=device, policy_kwargs=policy_kwargs, tensorboard_log=log_dir)
    model.learn(total_timesteps=timesteps)
    mean, std = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=False)
    vec_env.close(); eval_env.close()
    return float(mean), float(std), model

def main():
    parser = argparse.ArgumentParser(description='World Models bossfight')
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,43])
    parser.add_argument('--num_levels', type=int, default=200)
    parser.add_argument('--log_dir', type=str, default='./logs_world_models')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    device = 'cuda' if (args.device=='auto' and torch.cuda.is_available()) else args.device if args.device!='auto' else 'cpu'
    print(f"Device: {device} torch {torch.__version__}")
    os.makedirs(args.log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    comp_dir = os.path.join(args.log_dir, f"comparison_bossfight_{ts}")
    os.makedirs(comp_dir, exist_ok=True)
    game='bossfight'
    configs = [
        ('vae', VAEExtractor, dict(features_dim=512, latent_dim=128)),
        ('ae', AEExtractor, dict(features_dim=512)),
        ('recon', ReconExtractor, dict(features_dim=512)),
        ('contrastive', ContrastiveExtractor, dict(features_dim=512)),
    ]
    results={}
    for key, cls, kwargs in configs:
        results[key]=[]
        for seed in args.seeds:
            print(f"\n{'='*60}\n{key} seed {seed} bossfight\n{'='*60}")
            try:
                mean,std,model = train_one(game, args.num_levels, 'easy', cls, kwargs, args.timesteps, seed, args.log_dir, device)
                print(f"{key} seed {seed}: {mean:.2f} +/- {std:.2f}")
                results[key].append({'seed': seed, 'mean_reward': mean, 'std_reward': std})
                try: model.save(os.path.join(comp_dir, f"{key}_seed{seed}.zip"))
                except: pass
            except Exception as e:
                import traceback; traceback.print_exc()
                results[key].append({'seed': seed, 'mean_reward': None, 'error': str(e)})
    with open(os.path.join(comp_dir, 'comparison_results.json'),'w') as f: json.dump(results,f,indent=2)
    stats={}
    for k,v in results.items():
        rewards=[x['mean_reward'] for x in v if x['mean_reward'] is not None]
        stats[k]={'mean':float(np.mean(rewards)),'std':float(np.std(rewards)),'min':float(np.min(rewards)),'max':float(np.max(rewards)),'n':len(rewards)} if rewards else None
    with open(os.path.join(comp_dir,'statistics.json'),'w') as f: json.dump(stats,f,indent=2)
    try:
        import matplotlib.pyplot as plt
        names, means, stds = [],[],[]
        for k,s in stats.items():
            if s: names.append(k); means.append(s['mean']); stds.append(s['std'])
        if names:
            plt.figure(figsize=(10,6)); colors=['lightblue','lightgreen','lightcoral','gold']
            bars=plt.bar(names, means, yerr=stds, capsize=5, alpha=0.8, color=colors[:len(names)])
            plt.ylabel('Mean Reward (10 eps)'); plt.title(f'World Models bossfight - {args.timesteps} steps')
            plt.tight_layout()
            for bar,m,s in zip(bars,means,stds): plt.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{m:.1f}±{s:.1f}', ha='center', va='bottom', fontsize=9)
            plt.savefig(os.path.join(comp_dir,'comparison_plot.png'), dpi=150, bbox_inches='tight')
            print(f"Plot salvo em {comp_dir}")
    except Exception as e: print(f"Plot erro: {e}")
    with open(os.path.join(comp_dir,'comparison_report.txt'),'w') as f:
        f.write(f"World Models bossfight - {args.timesteps} steps\n"+"="*60+"\n")
        for k,s in stats.items(): f.write(f"{k}: mean {s['mean']:.2f} std {s['std']:.2f} n {s['n']}\n" if s else f"{k}: sem dados\n")
    print(f"\nResultados em {comp_dir}\n"+json.dumps(stats,indent=2))

if __name__=='__main__': main()
