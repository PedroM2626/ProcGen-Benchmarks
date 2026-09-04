"""Smoke test do compare_hrl: 3 braços × 2 jogos, 1000 steps cada, sem eval longo."""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from compare_hrl import make_game

for game in ['jumper', 'plunder']:
    for arm in ['flat', 'skip4', 'hrl']:
        vec = DummyVecEnv([lambda: Monitor(make_game(game, 16, 42, arm))])
        obs = vec.reset()
        model = PPO("CnnPolicy", vec, verbose=0, n_steps=128, device='cuda', seed=42)
        model.learn(total_timesteps=1000)
        # rollout rápido: verifica step/reset/reward sem exceção
        o = vec.reset()
        total = 0.0
        for _ in range(50):
            a, _ = model.predict(o, deterministic=True)
            o, r, d, _ = vec.step(a)
            total += float(np.sum(r))
        print(f'{game:8s} {arm:6s} obs={o.shape} act_space={vec.action_space.n} rollout_r={total:.2f} OK')
        vec.close()
print('SMOKE OK')
