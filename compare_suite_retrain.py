"""
Retreino da suite original (compare_suite.py) com o protocolo novo:
- Mesmas configs/hiperparâmetros: 4 WM + 4 CNN + 3 Augment × 3 jogos × 5 seeds (165 modelos)
- PPO idêntico: lr=3e-4, n_steps=256, batch_size=64, n_epochs=3, gamma=0.99, gae_lambda=0.95, clip=0.2
- Treino: num_levels=200, distribution_mode='easy', seed fixo (42-46)
- Eval novo (compatível 1:1 com re_eval_scorecard.py):
  30 eps unseen stoch + 30 eps unseen det (num_levels=0, seed+1000) + 15 eps train (num_levels=200)
- Salva zip por modelo + results/retrain_results.json incremental (resume-safe)
- Zips nomeados {game}_{config}_seed{seed}.zip (compatível com a convenção da re-avaliação)
"""
import os, json, argparse
import numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from procgen_wrapper import make_procgen_env
from models.world_model_extractors import VAEExtractor, AEExtractor, ReconExtractor, ContrastiveExtractor
from models.sb3_extractors import ClassicCNNExtractor, AttentionCNNExtractor
from compare_augment_contrastive import ContrastiveCrop, ContrastiveColor, ContrastiveNoise

def eval_model(model, game, seed, n_unseen, n_train, vector):
    unseen = DummyVecEnv([lambda: Monitor(make_procgen_env(game, num_levels=0, distribution_mode='easy', seed=seed+1000, vector=vector))])
    train = DummyVecEnv([lambda: Monitor(make_procgen_env(game, num_levels=200, distribution_mode='easy', seed=seed, vector=vector))])
    m_st, _ = evaluate_policy(model, unseen, n_eval_episodes=n_unseen, deterministic=False)
    m_dt, _ = evaluate_policy(model, unseen, n_eval_episodes=n_unseen, deterministic=True)
    m_tr, _ = evaluate_policy(model, train, n_eval_episodes=n_train, deterministic=False)
    unseen.close(); train.close()
    return float(m_st), float(m_dt), float(m_tr)

def train_one(game, key, extractor_class, kwargs, vector, timesteps, seed, log_dir, device):
    vec = DummyVecEnv([lambda: Monitor(make_procgen_env(game, num_levels=200, distribution_mode='easy', seed=seed, vector=vector))])
    if vector:
        policy, pk = "MlpPolicy", {}
    else:
        policy, pk = "CnnPolicy", {"features_extractor_class": extractor_class, "features_extractor_kwargs": kwargs}
    model = PPO(policy, vec, verbose=0, learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=3,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, seed=seed, device=device,
                policy_kwargs=pk, tensorboard_log=log_dir)
    model.learn(total_timesteps=timesteps)
    vec.close()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=str, nargs='+', default=['bossfight', 'starpilot', 'dodgeball'])
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    parser.add_argument('--n_unseen', type=int, default=30)
    parser.add_argument('--n_train', type=int, default=15)
    parser.add_argument('--log_dir', type=str, default='./logs_suite_retrain')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--out', type=str, default='results/retrain_results.json')
    args = parser.parse_args()
    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else ('cpu' if args.device == 'auto' else args.device)

    base = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    zip_dir = os.path.join(base, 'logs_suite_retrain', 'suite_retrain_zips')
    os.makedirs(zip_dir, exist_ok=True)
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f: results = json.load(f)  # resume

    # mesmas configs do compare_suite.py (world 4 + cnn 4 + augment 3)
    configs = []
    configs += [('wm_' + k, cls, kw, False) for k, cls, kw in [
        ('vae', VAEExtractor, dict(features_dim=512, latent_dim=128)),
        ('ae', AEExtractor, dict(features_dim=512)),
        ('recon', ReconExtractor, dict(features_dim=512)),
        ('contrastive', ContrastiveExtractor, dict(features_dim=512))]]
    configs += [('cnn_' + k, cls, kw, vec) for k, cls, kw, vec in [
        ('classic', ClassicCNNExtractor, dict(features_dim=512), False),
        ('cbam', AttentionCNNExtractor, dict(features_dim=512, use_cbam=True), False),
        ('spatial', AttentionCNNExtractor, dict(features_dim=512, use_cbam=False), False),
        ('mlp_vector', None, {}, True)]]
    configs += [('aug_' + k, cls, dict(features_dim=512), False) for k, cls in [
        ('crop', ContrastiveCrop), ('color', ContrastiveColor), ('noise', ContrastiveNoise)]]

    jobs = [(g, key, cls, kw, vec, s) for g in args.games for key, cls, kw, vec in configs for s in args.seeds]
    done0 = sum(1 for v in results.values() if 'error' not in v)
    print(f"{len(jobs)} jobs ({len(jobs)-done0} restantes; entries com erro serão re-tentadas), device={device}, {args.timesteps} steps, eval {args.n_unseen}+{args.n_unseen}+{args.n_train}")

    for i, (game, key, cls, kw, vec, seed) in enumerate(jobs):
        name = f"{game}_{key}_seed{seed}"
        if name in results and 'error' not in results[name]: continue
        print(f"\n[{i+1}/{len(jobs)}] {name}")
        try:
            model = train_one(game, key, cls, kw, vec, args.timesteps, seed, args.log_dir, device)
            m_st, m_dt, m_tr = eval_model(model, game, seed, args.n_unseen, args.n_train, vec)
            try: model.save(os.path.join(zip_dir, f"{name}.zip"))
            except Exception as e: print(f"  zip falhou: {e}")
            results[name] = {
                'stoch_unseen': round(m_st, 3), 'det_unseen': round(m_dt, 3),
                'stoch_train': round(m_tr, 3), 'gen_gap': round(m_tr - m_st, 3),
                'n_unseen': args.n_unseen, 'n_train': args.n_train,
            }
            print(f"  stoch={m_st:.2f} det={m_dt:.2f} train={m_tr:.2f} gap={m_tr-m_st:+.2f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = {'error': str(e)}
        with open(out_path, 'w') as f: json.dump(results, f, indent=2)

    # estatísticas finais por config
    stats = {}
    for name, v in results.items():
        if 'error' in v: continue
        gk = name.rsplit('_seed', 1)[0]
        stats.setdefault(gk, []).append(v['stoch_unseen'])
    summary = {k: {'mean': round(float(np.mean(x)), 3), 'std': round(float(np.std(x)), 3), 'n': len(x)} for k, x in stats.items()}
    print(f"\nConcluído: {out_path}\n" + json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
