"""
Bossfight HARD extra: 4 WM +4 CNN +3 Augment 5 seeds 100k
- Para medir se precisa hard nos outros jogos
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
from models.sb3_extractors import ClassicCNNExtractor, AttentionCNNExtractor
from compare_augment_contrastive import ContrastiveCrop, ContrastiveColor, ContrastiveNoise

def train_one(game, dist, cls, kwargs, timesteps, seed, log_dir, device, vector=False):
    def make_env(): return Monitor(make_procgen_env(game, num_levels=200, distribution_mode=dist, seed=seed, vector=vector))
    vec = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([lambda: Monitor(make_procgen_env(game, num_levels=0, distribution_mode=dist, seed=seed+1000, vector=vector))])
    policy, pk = ("MlpPolicy", {}) if vector else ("CnnPolicy", {"features_extractor_class": cls, "features_extractor_kwargs": kwargs})
    model = PPO(policy, vec, verbose=0, learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=3, gamma=0.99, gae_lambda=0.95, clip_range=0.2, seed=seed, device=device, policy_kwargs=pk, tensorboard_log=log_dir)
    model.learn(total_timesteps=timesteps)
    mean,std = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=False)
    vec.close(); eval_env.close()
    return float(mean), float(std), model

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,43,44,45,46])
    parser.add_argument('--log_dir', type=str, default='./logs_bossfight_hard')
    parser.add_argument('--device', type=str, default='auto')
    args=parser.parse_args()
    device='cuda' if (args.device=='auto' and torch.cuda.is_available()) else args.device if args.device!='auto' else 'cpu'
    print(f"Device: {device} torch {torch.__version__} HARD")
    os.makedirs(args.log_dir, exist_ok=True)
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    comp_dir=os.path.join(args.log_dir, f"comparison_bossfight_hard_{ts}")
    os.makedirs(comp_dir, exist_ok=True)
    game='bossfight'; dist='hard'
    configs=[
        ('wm_vae', VAEExtractor, dict(features_dim=512, latent_dim=128), False),
        ('wm_ae', AEExtractor, dict(features_dim=512), False),
        ('wm_recon', ReconExtractor, dict(features_dim=512), False),
        ('wm_contrastive', ContrastiveExtractor, dict(features_dim=512), False),
        ('cnn_classic', ClassicCNNExtractor, dict(features_dim=512), False),
        ('cnn_cbam', AttentionCNNExtractor, dict(features_dim=512, use_cbam=True), False),
        ('cnn_spatial', AttentionCNNExtractor, dict(features_dim=512, use_cbam=False), False),
        ('cnn_mlp_vector', None, {}, True),
        ('aug_crop', ContrastiveCrop, dict(features_dim=512), False),
        ('aug_color', ContrastiveColor, dict(features_dim=512), False),
        ('aug_noise', ContrastiveNoise, dict(features_dim=512), False),
    ]
    results={}
    for key, cls, kw, vec in configs:
        results[key]=[]
        for seed in args.seeds:
            print(f"\n{'='*60}\n{key} seed {seed} bossfight HARD\n{'='*60}")
            try:
                mean,std,model = train_one(game, dist, cls, kw, args.timesteps, seed, args.log_dir, device, vector=vec)
                print(f"{key} seed {seed}: {mean:.2f} +/- {std:.2f}")
                results[key].append({'seed': seed, 'mean_reward': mean, 'std_reward': std})
                try: model.save(os.path.join(comp_dir, f"{key}_seed{seed}.zip"))
                except: pass
            except Exception as e:
                import traceback; traceback.print_exc()
                results[key].append({'seed': seed, 'mean_reward': None, 'error': str(e)})
    with open(os.path.join(comp_dir,'comparison_results.json'),'w') as f: json.dump(results,f,indent=2)
    stats={}
    for k,v in results.items():
        rewards=[x['mean_reward'] for x in v if x['mean_reward'] is not None]
        stats[k]={'mean':float(np.mean(rewards)),'std':float(np.std(rewards)),'min':float(np.min(rewards)),'max':float(np.max(rewards)),'n':len(rewards)} if rewards else None
    with open(os.path.join(comp_dir,'statistics.json'),'w') as f: json.dump(stats,f,indent=2)
    try:
        import matplotlib.pyplot as plt
        keys=[k for k in stats if stats[k]]; means=[stats[k]['mean'] for k in keys]; stds=[stats[k]['std'] for k in keys]
        plt.figure(figsize=(14,6)); plt.bar(keys, means, yerr=stds, capsize=4, alpha=0.8)
        plt.xticks(rotation=25, ha='right', fontsize=7); plt.ylabel('Mean Reward (10 eps)'); plt.title(f'Bossfight HARD - {args.timesteps} steps 5 seeds')
        plt.tight_layout(); plt.savefig(os.path.join(comp_dir,'comparison_plot.png'), dpi=150, bbox_inches='tight')
        print(f"Plot salvo em {comp_dir}")
    except Exception as e: print(f"Plot erro: {e}")
    print(f"\nResultados em {comp_dir}\n"+json.dumps(stats,indent=2))

if __name__=='__main__': main()
