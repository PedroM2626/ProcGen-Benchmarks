"""
Comparação rápida usando SB3 PPO + ambientes rápidos
- cartpole_pixels (CV) vs cartpole_state (vetor sem CV) — mesmo ambiente
- classic CNN vs attention CNN (CBAM True/False)
Roda 50k steps em ~2-5 min vs 50 min CarRacing
"""
import os
import json
import argparse
from datetime import datetime
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from env_setup import create_fast_env
from models.sb3_extractors import ClassicCNNExtractor, AttentionCNNExtractor

def make_vec_env(env_type, seed, frame_stack=4, frame_size=(64,64)):
    def _init():
        env = create_fast_env(env_type, frame_stack=frame_stack, frame_size=frame_size)
        # SB3 espera HWC? Nosso env já é CHW uint8, mas VecTransposeImage espera HWC, então não usamos
        # DummyVecEnv já lida
        return env
    return _init

def train_ppo(env_type, extractor_class, extractor_kwargs, timesteps=50000, seed=42, log_dir='./logs_fast', device='auto'):
    os.makedirs(log_dir, exist_ok=True)
    # VecEnv
    vec_env = DummyVecEnv([make_vec_env(env_type, seed)])
    # Para CNN CHW, SB3 não precisa VecTransposeImage se já é CHW? Mas SB3 espera HWC para CnnPolicy, nosso extractor espera CHW, então não transpõe
    # Avaliacao env separado
    eval_env = DummyVecEnv([make_vec_env(env_type, seed+1000)])
    
    # Policy kwargs
    policy_kwargs = {
        "features_extractor_class": extractor_class,
        "features_extractor_kwargs": extractor_kwargs,
        "net_arch": dict(pi=[256], vf=[256]),
    }
    
    model = PPO("CnnPolicy" if "pixels" in env_type or env_type in ["pong","breakout"] else "MlpPolicy",
                vec_env, verbose=1, 
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                seed=seed,
                device=device,
                policy_kwargs=policy_kwargs if "pixels" in env_type or env_type in ["pong","breakout","carracing"] else {},
                tensorboard_log=log_dir
                )
    # Callback eval a cada 5k
    eval_callback = EvalCallback(eval_env, eval_freq=5000, n_eval_episodes=5, deterministic=True, render=False, verbose=1)
    
    model.learn(total_timesteps=timesteps, callback=eval_callback, tb_log_name=f"{env_type}_{extractor_class.__name__}_{seed}")
    
    # Avaliação final 10 episódios
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"Final eval {env_type} {extractor_class.__name__}: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    vec_env.close()
    eval_env.close()
    
    return mean_reward, std_reward, model

def main():
    parser = argparse.ArgumentParser(description='Comparação rápida CV vs não-CV + Classic vs Attention')
    parser.add_argument('--timesteps', type=int, default=50000, help='Timesteps por experimento (default 50000)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42], help='Seeds')
    parser.add_argument('--env', type=str, default='cartpole_pixels', choices=['cartpole_pixels','cartpole_state','pong','breakout','carracing'], help='Env rápido')
    parser.add_argument('--compare_cv', action='store_true', help='Compara CV (pixels) vs sem CV (state) no mesmo CartPole')
    parser.add_argument('--log_dir', type=str, default='./logs_fast')
    parser.add_argument('--device', type=str, default='auto', help='auto, cpu, cuda')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Device: {device}")
    os.makedirs(args.log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = os.path.join(args.log_dir, f"comparison_fast_{timestamp}")
    os.makedirs(comparison_dir, exist_ok=True)

    results = {}

    if args.compare_cv:
        # Mesmo CartPole, duas observações
        configs = [
            ('cartpole_pixels', ClassicCNNExtractor, dict(features_dim=512), 'classic_pixels'),
            ('cartpole_pixels', AttentionCNNExtractor, dict(features_dim=512, use_cbam=True), 'attention_cbam_pixels'),
            ('cartpole_pixels', AttentionCNNExtractor, dict(features_dim=512, use_cbam=False), 'spatial_pixels'),
            ('cartpole_state', None, {}, 'mlp_state'),  # sem CNN, MLP puro
        ]
    else:
        # Apenas CV, compara arquiteturas
        if args.env in ['cartpole_state']:
            configs = [('cartpole_state', None, {}, 'mlp_state')]
        elif args.env in ['cartpole_pixels','pong','breakout','carracing']:
            configs = [
                (args.env, ClassicCNNExtractor, dict(features_dim=512), 'classic'),
                (args.env, AttentionCNNExtractor, dict(features_dim=512, use_cbam=True), 'attention_cbam_True'),
                (args.env, AttentionCNNExtractor, dict(features_dim=512, use_cbam=False), 'attention_spatial'),
            ]
        else:
            configs = [(args.env, ClassicCNNExtractor, dict(features_dim=512), 'classic')]

    for env_type, extractor_class, extractor_kwargs, key in configs:
        results[key] = []
        for seed in args.seeds:
            print(f"\n{'='*60}\nTreinando {key} seed {seed} env {env_type}\n{'='*60}")
            try:
                # Para cartpole_state, usamos MlpPolicy direto sem extractor custom
                if extractor_class is None:
                    # MLP direto
                    vec_env = DummyVecEnv([make_vec_env(env_type, seed)])
                    eval_env = DummyVecEnv([make_vec_env(env_type, seed+1000)])
                    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=3e-4, n_steps=1024, batch_size=64, seed=seed, device=device, tensorboard_log=args.log_dir)
                    model.learn(total_timesteps=args.timesteps)
                    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
                    vec_env.close(); eval_env.close()
                else:
                    mean_reward, std_reward, _ = train_ppo(env_type, extractor_class, extractor_kwargs, timesteps=args.timesteps, seed=seed, log_dir=args.log_dir, device=device)
                
                results[key].append({'seed': seed, 'mean_reward': float(mean_reward), 'std_reward': float(std_reward)})
                print(f"{key} seed {seed}: {mean_reward:.2f} +/- {std_reward:.2f}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                results[key].append({'seed': seed, 'mean_reward': None, 'error': str(e)})

    # Salvar
    with open(os.path.join(comparison_dir, 'comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Estatísticas
    stats = {}
    for k, v in results.items():
        rewards = [x['mean_reward'] for x in v if x['mean_reward'] is not None]
        if rewards:
            stats[k] = {'mean': float(np.mean(rewards)), 'std': float(np.std(rewards)), 'min': float(np.min(rewards)), 'max': float(np.max(rewards)), 'n': len(rewards)}
        else:
            stats[k] = None
    with open(os.path.join(comparison_dir, 'statistics.json'), 'w') as f:
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
            bars = plt.bar(arch_names, means, yerr=stds, capsize=5, alpha=0.7)
            plt.ylabel('Mean Reward (10 eps)')
            plt.title(f'Comparação Rápida - {args.env} - {args.timesteps} steps')
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            for bar, m, s in zip(bars, means, stds):
                plt.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{m:.1f}±{s:.1f}', ha='center', va='bottom')
            plt.savefig(os.path.join(comparison_dir, 'comparison_plot.png'), dpi=150, bbox_inches='tight')
            print(f"Plot salvo em {comparison_dir}")
            plt.close()
    except Exception as e:
        print(f"Erro plot: {e}")

    print(f"\nResultados salvos em {comparison_dir}")
    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    main()
