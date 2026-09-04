"""
Augment Contrastivo: crop vs color-jitter vs noise em bossfight
- 3 configs ContrastiveExtractor com augment_type diferente, 5 seeds, 100k
"""
import os, json, argparse
from datetime import datetime
import numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from procgen_wrapper import make_procgen_env
from models.world_model_extractors import ContrastiveExtractor

# Patch ContrastiveExtractor para suportar augment_type via monkey-patch simples: recriar classe com param
class ContrastiveCrop(ContrastiveExtractor):
    def __init__(self, obs_space, features_dim=512): super().__init__(obs_space, features_dim); self.augment_type='crop'
    def forward(self, obs):
        if obs.dtype==torch.uint8: obs=obs.float()/255.0
        elif obs.max()>1.5: obs=obs/255.0
        if self.is_hwc and obs.dim()==4 and obs.shape[-1] in [1,3,4]: obs=obs.permute(0,3,1,2)
        if self.training and torch.rand(1).item()<0.5:
            # random crop 56->64 (pad 4)
            obs = torch.nn.functional.pad(obs, (4,4,4,4), mode='replicate')
            h,w = obs.shape[2], obs.shape[3]
            top = torch.randint(0, h-64+1, (1,)).item(); left = torch.randint(0, w-64+1, (1,)).item()
            obs = obs[:,:,top:top+64, left:left+64]
        return self.fc(self.cnn(obs))

class ContrastiveColor(ContrastiveExtractor):
    def __init__(self, obs_space, features_dim=512): super().__init__(obs_space, features_dim); self.augment_type='color'
    def forward(self, obs):
        if obs.dtype==torch.uint8: obs=obs.float()/255.0
        elif obs.max()>1.5: obs=obs/255.0
        if self.is_hwc and obs.dim()==4 and obs.shape[-1] in [1,3,4]: obs=obs.permute(0,3,1,2)
        if self.training and torch.rand(1).item()<0.5:
            # color jitter simples: brightness/contrast
            obs = obs * (0.8 + torch.rand(1,device=obs.device)*0.4)
            obs = torch.clamp(obs,0,1)
        return self.fc(self.cnn(obs))

class ContrastiveNoise(ContrastiveExtractor):
    def __init__(self, obs_space, features_dim=512): super().__init__(obs_space, features_dim); self.augment_type='noise'
    # usa forward original (noise 0.01)

def train_one(game, num_levels, cls, timesteps, seed, log_dir, device):
    def make_env(): return Monitor(make_procgen_env(game, num_levels=num_levels, distribution_mode='easy', seed=seed, vector=False))
    vec = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([lambda: Monitor(make_procgen_env(game, num_levels=0, distribution_mode='easy', seed=seed+1000, vector=False))])
    model = PPO("CnnPolicy", vec, verbose=0, learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=3, gamma=0.99, gae_lambda=0.95, clip_range=0.2, seed=seed, device=device, policy_kwargs={"features_extractor_class": cls, "features_extractor_kwargs": dict(features_dim=512)}, tensorboard_log=log_dir)
    model.learn(total_timesteps=timesteps)
    mean,std = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=False)
    vec.close(); eval_env.close()
    return float(mean), float(std), model

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,43,44,45,46])
    parser.add_argument('--num_levels', type=int, default=200)
    parser.add_argument('--log_dir', type=str, default='./logs_augment')
    parser.add_argument('--device', type=str, default='auto')
    args=parser.parse_args()
    device='cuda' if (args.device=='auto' and torch.cuda.is_available()) else args.device if args.device!='auto' else 'cpu'
    print(f"Device: {device} torch {torch.__version__}")
    os.makedirs(args.log_dir, exist_ok=True)
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    comp_dir=os.path.join(args.log_dir, f"comparison_augment_bossfight_{ts}")
    os.makedirs(comp_dir, exist_ok=True)
    game='bossfight'
    configs=[('contrastive_crop', ContrastiveCrop), ('contrastive_color', ContrastiveColor), ('contrastive_noise', ContrastiveNoise)]
    results={}
    for key, cls in configs:
        results[key]=[]
        for seed in args.seeds:
            print(f"\n{'='*60}\n{key} seed {seed} bossfight\n{'='*60}")
            try:
                mean,std,model = train_one(game, args.num_levels, cls, args.timesteps, seed, args.log_dir, device)
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
        names, means, stds = [],[],[]
        for k,s in stats.items():
            if s: names.append(k); means.append(s['mean']); stds.append(s['std'])
        if names:
            plt.figure(figsize=(10,6)); plt.bar(names, means, yerr=stds, capsize=5, alpha=0.8)
            plt.ylabel('Mean Reward'); plt.title(f'Augment Contrastive bossfight - {args.timesteps} steps')
            plt.xticks(rotation=20, ha='right'); plt.tight_layout()
            for bar,m,s in zip(plt.gca().patches, means, stds): plt.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{m:.1f}±{s:.1f}', ha='center', va='bottom', fontsize=8)
            plt.savefig(os.path.join(comp_dir,'comparison_plot.png'), dpi=150, bbox_inches='tight')
            print(f"Plot salvo em {comp_dir}")
    except Exception as e: print(f"Plot erro: {e}")
    print(f"\nResultados em {comp_dir}\n"+json.dumps(stats,indent=2))

if __name__=='__main__': main()
