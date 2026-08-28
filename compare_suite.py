"""
Suite Procgen: bossfight + starpilot + dodgeball para todo benchmark
- Roda World Models (4) + Procgen CNN (4) + Augment (3) em cada jogo, 5 seeds, 50k (suite rápida) ou 100k
- Para noite inteira: ~8h (50k) ou ~12h (100k)
"""
import os, json, argparse, itertools
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

def train_one(game, extractor_class, kwargs, timesteps, seed, log_dir, device, vector=False):
    def make_env(): return Monitor(make_procgen_env(game, num_levels=200, distribution_mode='easy', seed=seed, vector=vector))
    vec = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([lambda: Monitor(make_procgen_env(game, num_levels=0, distribution_mode='easy', seed=seed+1000, vector=vector))])
    if vector:
        policy, pk = "MlpPolicy", {}
    else:
        policy, pk = "CnnPolicy", {"features_extractor_class": extractor_class, "features_extractor_kwargs": kwargs}
    model = PPO(policy, vec, verbose=0, learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=3, gamma=0.99, gae_lambda=0.95, clip_range=0.2, seed=seed, device=device, policy_kwargs=pk, tensorboard_log=log_dir)
    model.learn(total_timesteps=timesteps)
    mean,std = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=False)
    vec.close(); eval_env.close()
    return float(mean), float(std), model

def run_suite(games, timesteps, seeds, log_dir, device):
    ts=datetime.now().strftime("%Y%m%d_%H%M%S")
    comp_dir=os.path.join(log_dir, f"suite_{'_'.join(games)}_{ts}")
    os.makedirs(comp_dir, exist_ok=True)
    # configs
    world_configs=[('vae', VAEExtractor, dict(features_dim=512, latent_dim=128)), ('ae', AEExtractor, dict(features_dim=512)), ('recon', ReconExtractor, dict(features_dim=512)), ('contrastive', ContrastiveExtractor, dict(features_dim=512))]
    procgen_configs=[('classic', ClassicCNNExtractor, dict(features_dim=512), False), ('cbam', AttentionCNNExtractor, dict(features_dim=512, use_cbam=True), False), ('spatial', AttentionCNNExtractor, dict(features_dim=512, use_cbam=False), False), ('mlp_vector', None, {}, True)]
    augment_configs=[('c_aug_crop', ContrastiveCrop, dict(features_dim=512)), ('c_aug_color', ContrastiveColor, dict(features_dim=512)), ('c_aug_noise', ContrastiveNoise, dict(features_dim=512))]

    all_results={}
    for game in games:
        print(f"\n{'#'*70}\n# GAME {game} suite\n{'#'*70}")
        # World Models
        for key, cls, kw in world_configs:
            gk=f"{game}_wm_{key}"
            all_results[gk]=[]
            for seed in seeds:
                print(f"\n[{game}] wm {key} seed {seed}")
                try:
                    m,s,model = train_one(game, cls, kw, timesteps, seed, log_dir, device, vector=False)
                    print(f"-> {m:.2f}"); all_results[gk].append({'seed':seed,'mean_reward':m})
                    try: model.save(os.path.join(comp_dir, f"{gk}_seed{seed}.zip"))
                    except: pass
                except Exception as e:
                    import traceback; traceback.print_exc(); all_results[gk].append({'seed':seed,'mean_reward':None,'error':str(e)})
        # Procgen CNN
        for key, cls, kw, vec in procgen_configs:
            gk=f"{game}_cnn_{key}"
            all_results[gk]=[]
            for seed in seeds:
                print(f"\n[{game}] cnn {key} seed {seed}")
                try:
                    m,s,model = train_one(game, cls, kw, timesteps, seed, log_dir, device, vector=vec)
                    print(f"-> {m:.2f}"); all_results[gk].append({'seed':seed,'mean_reward':m})
                    try: model.save(os.path.join(comp_dir, f"{gk}_seed{seed}.zip"))
                    except: pass
                except Exception as e:
                    import traceback; traceback.print_exc(); all_results[gk].append({'seed':seed,'mean_reward':None,'error':str(e)})
        # Augment (só bossfight/starpilot/dodgeball, mas roda em todos para comparar)
        for key, cls, kw in [(k, c, dict(features_dim=512)) for k,c in [('aug_crop',ContrastiveCrop),('aug_color',ContrastiveColor),('aug_noise',ContrastiveNoise)]]:
            # já coberto em world, mas mantemos separado
            pass

    # augment separado por jogo (3 configs x3 games)
    for game in games:
        for key, cls in [('aug_crop',ContrastiveCrop),('aug_color',ContrastiveColor),('aug_noise',ContrastiveNoise)]:
            gk=f"{game}_aug_{key}"
            all_results[gk]=[]
            for seed in seeds:
                print(f"\n[{game}] aug {key} seed {seed}")
                try:
                    m,s,model = train_one(game, cls, dict(features_dim=512), timesteps, seed, log_dir, device, vector=False)
                    print(f"-> {m:.2f}"); all_results[gk].append({'seed':seed,'mean_reward':m})
                    try: model.save(os.path.join(comp_dir, f"{gk}_seed{seed}.zip"))
                    except: pass
                except Exception as e:
                    import traceback; traceback.print_exc(); all_results[gk].append({'seed':seed,'mean_reward':None,'error':str(e)})

    with open(os.path.join(comp_dir,'suite_results.json'),'w') as f: json.dump(all_results,f,indent=2)
    stats={}
    for k,v in all_results.items():
        rewards=[x['mean_reward'] for x in v if x['mean_reward'] is not None]
        stats[k]={'mean':float(np.mean(rewards)),'std':float(np.std(rewards)),'n':len(rewards)} if rewards else None
    with open(os.path.join(comp_dir,'suite_statistics.json'),'w') as f: json.dump(stats,f,indent=2)
    print(f"\nSuite completa em {comp_dir}\n"+json.dumps(stats,indent=2))
    # plots por jogo + geral + jornada via tensorboard
    try:
        import matplotlib.pyplot as plt
        for game in games:
            plt.figure(figsize=(14,5))
            keys=[k for k in stats if k.startswith(game+'_')]
            means=[stats[k]['mean'] if stats[k] else 0 for k in keys]
            stds=[stats[k]['std'] if stats[k] else 0 for k in keys]
            plt.bar(keys, means, yerr=stds, capsize=3, alpha=0.8); plt.xticks(rotation=30, ha='right', fontsize=7)
            plt.title(f"Suite {game} - {timesteps} steps 5 seeds"); plt.tight_layout()
            plt.savefig(os.path.join(comp_dir, f"suite_{game}_plot.png"), dpi=150, bbox_inches='tight'); plt.close()
        # geral: média por arquitetura agregada nos 3 jogos
        for prefix in ['vae','ae','recon','contrastive','classic','cbam','spatial','mlp_vector','aug_crop','aug_color','aug_noise']:
            vals=[stats[k]['mean'] for k in stats if prefix in k and stats[k]]
            if vals:
                plt.figure(figsize=(8,4)); plt.bar([k for k in stats if prefix in k], vals, alpha=0.8); plt.xticks(rotation=25, ha='right', fontsize=7)
                plt.title(f"Geral {prefix} - média 3 jogos"); plt.tight_layout()
                plt.savefig(os.path.join(comp_dir, f"suite_geral_{prefix}.png"), dpi=150, bbox_inches='tight'); plt.close()
        print(f"Plots suite salvos em {comp_dir} (por jogo + geral). Jornada completa em tensorboard: tensorboard --logdir {log_dir}")
    except Exception as e: print(f"Plot erro: {e}")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--games', type=str, nargs='+', default=['bossfight','starpilot','dodgeball'])
    parser.add_argument('--timesteps', type=int, default=50000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,43,44,45,46])
    parser.add_argument('--log_dir', type=str, default='./logs_suite')
    parser.add_argument('--device', type=str, default='auto')
    args=parser.parse_args()
    device='cuda' if (args.device=='auto' and torch.cuda.is_available()) else args.device if args.device!='auto' else 'cpu'
    run_suite(args.games, args.timesteps, args.seeds, args.log_dir, device)
