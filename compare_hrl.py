"""
Benchmark INDEPENDENTE: HRL vs Flat RL em jumper/plunder (não entra no scorecard do estudo principal).

3 braços (mesmo budget de 100k FRAMES primitivos, PPO idêntico, NatureCNN CnnPolicy):
- flat  : PPO sobre ações primitivas, 100k steps
- skip4 : PPO com action-repeat 4 (abstração temporal SEM hierarquia) — controle
- hrl   : PPO meta-controlador sobre biblioteca de skills fixas (framework de opções):
          cada decisão de skill = 4 frames primitivos -> 25k meta-steps = 100k frames

Skills por jogo (sondagem empírica probe_actions.py + mapeamento oficial procgen env.py):
- jumper : WAIT, LEFT, RIGHT, JUMP(UP), JUMP_LEFT, JUMP_RIGHT
- plunder: WAIT, LEFT, RIGHT, SHOOT(D), SHOOT_LEFT, SHOOT_RIGHT

Protocolo de treino/eval idêntico ao estudo (num_levels=200 easy, eval unseen seed+1000,
100 eps stoch + 100 det + 15 train). Saída separada: logs_hrl/ + results/hrl_results.json.
"""
import os, json, argparse
import numpy as np, gymnasium as gymn, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from procgen_wrapper import make_procgen_env

DUR = 4  # frames primitivos por skill / action-repeat
FRAMES = 100_000

# mapeamento oficial (procgen/env.py): 0=LEFT+DOWN 1=LEFT 2=LEFT+UP 3=DOWN 4=() 5=UP
# 6=RIGHT+DOWN 7=RIGHT 8=RIGHT+UP 9=D 10=A 11=W 12=S 13=Q 14=E
SKILLS = {
    'jumper': [4, 1, 7, 5, 2, 8],    # WAIT, LEFT, RIGHT, JUMP, JUMP_LEFT, JUMP_RIGHT
    'plunder': [4, 1, 7, 9, 0, 6],   # WAIT, LEFT, RIGHT, SHOOT(D), SHOOT_LEFT, SHOOT_RIGHT
}
SKILL_NAMES = {
    'jumper': ['wait', 'left', 'right', 'jump', 'jump_left', 'jump_right'],
    'plunder': ['wait', 'left', 'right', 'shoot', 'shoot_left', 'shoot_right'],
}

class MacroEnv(gymn.ActionWrapper):
    """Executa uma skill (ação primitiva mantida por DUR frames) por step do meta-controlador.
    Termina o macro na borda do episódio para não misturar episódios (auto-reset fica no DummyVecEnv)."""
    def __init__(self, env, primitives):
        super().__init__(env)
        self.primitives = primitives
        self.action_space = gymn.spaces.Discrete(len(primitives))

    def action(self, a):  # não usado; step sobrescrito
        return self.primitives[int(a)]

    def step(self, a):
        prim = self.primitives[int(a)]
        tot, term, trunc, obs, info = 0.0, False, False, None, {}
        for _ in range(DUR):
            obs, r, term, trunc, info = self.env.step(prim)
            tot += r
            if term or trunc:
                break
        return obs, tot, term, trunc, info

def make_game(game, num_levels, seed, arm):
    env = make_procgen_env(game, num_levels=num_levels, distribution_mode='easy', seed=seed, vector=False)
    if arm == 'flat':
        return env
    if arm == 'skip4':
        return RepeatEnv(env)
    return MacroEnv(env, SKILLS[game])  # hrl

class RepeatEnv(MacroEnv):
    """action-repeat 4 sobre as 15 ações primitivas (braço skip4)."""
    def __init__(self, env):
        super().__init__(env, list(range(15)))

def train_one(game, arm, seed, log_dir, device):
    vec = DummyVecEnv([lambda: Monitor(make_game(game, 200, seed, arm))])
    model = PPO("CnnPolicy", vec, verbose=0, learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=3,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, seed=seed, device=device,
                tensorboard_log=log_dir)
    ts = FRAMES // DUR if arm != 'flat' else FRAMES  # mesmo budget em frames primitivos
    model.learn(total_timesteps=ts)
    vec.close()
    return model

def eval_model(model, game, arm, seed, n_unseen=100, n_train=15):
    unseen = DummyVecEnv([lambda: Monitor(make_game(game, 0, seed + 1000, arm))])
    train = DummyVecEnv([lambda: Monitor(make_game(game, 200, seed, arm))])
    m_st, _ = evaluate_policy(model, unseen, n_eval_episodes=n_unseen, deterministic=False)
    m_dt, _ = evaluate_policy(model, unseen, n_eval_episodes=n_unseen, deterministic=True)
    m_tr, _ = evaluate_policy(model, train, n_eval_episodes=n_train, deterministic=False)
    unseen.close(); train.close()
    return float(m_st), float(m_dt), float(m_tr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', nargs='+', default=['jumper', 'plunder'])
    parser.add_argument('--arms', nargs='+', default=['flat', 'skip4', 'hrl'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument('--log_dir', default='./logs_hrl')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--out', default='results/hrl_results.json')
    args = parser.parse_args()
    device = 'cuda' if (args.device == 'auto' and torch.cuda.is_available()) else ('cpu' if args.device == 'auto' else args.device)

    base = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(base, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    zip_dir = os.path.join(base, 'logs_hrl', 'hrl_zips')
    os.makedirs(zip_dir, exist_ok=True)
    results = {}
    if os.path.exists(out_path):
        with open(out_path) as f: results = json.load(f)

    jobs = [(g, arm, s) for g in args.games for arm in args.arms for s in args.seeds]
    print(f"{len(jobs)} jobs ({args.games} × {args.arms} × {args.seeds}), device={device}, {FRAMES//1000}k frames/braço")
    for i, (game, arm, seed) in enumerate(jobs):
        name = f"{game}_{arm}_seed{seed}"
        if name in results and 'error' not in results[name]: continue
        print(f"\n[{i+1}/{len(jobs)}] {name}")
        try:
            model = train_one(game, arm, seed, args.log_dir, device)
            m_st, m_dt, m_tr = eval_model(model, game, arm, seed)
            try: model.save(os.path.join(zip_dir, f"{name}.zip"))
            except Exception as e: print(f"  zip falhou: {e}")
            results[name] = {'stoch_unseen': round(m_st, 3), 'det_unseen': round(m_dt, 3),
                             'stoch_train': round(m_tr, 3), 'gen_gap': round(m_tr - m_st, 3),
                             'n_unseen': 100, 'n_train': 15, 'frames': FRAMES}
            print(f"  stoch={m_st:.2f} det={m_dt:.2f} train={m_tr:.2f} gap={m_tr-m_st:+.2f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = {'error': str(e)}
        with open(out_path, 'w') as f: json.dump(results, f, indent=2)

    stats = {}
    for name, v in results.items():
        if 'error' in v: continue
        gk = name.rsplit('_seed', 1)[0]
        stats.setdefault(gk, []).append(v['stoch_unseen'])
    print('\nResumo:'); [print(f"  {k:24s} {np.mean(x):.2f} ± {np.std(x):.2f} (n={len(x)})") for k, x in sorted(stats.items())]
    print(f"Concluído: {out_path}")

if __name__ == '__main__':
    main()
