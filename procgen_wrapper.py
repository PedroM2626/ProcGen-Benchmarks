import gym  # old gym for procgen
import gymnasium as gymn
import numpy as np
import cv2

class ProcgenGymWrapper(gymn.Env):
    """
    Wrapper para Procgen que converte old gym API (obs, done) para gymnasium (obs, info, terminated, truncated)
    e mantém imagem 64x64x3
    """
    def __init__(self, env, frame_stack=1):
        super().__init__()
        self.env = env
        self.frame_stack = frame_stack
        # Procgen 64x64x3 HWC -> convertemos para CHW para CNN
        if frame_stack > 1:
            from collections import deque
            self.frames = deque(maxlen=frame_stack)
            self.observation_space = gymn.spaces.Box(low=0, high=255, shape=(3*frame_stack, 64, 64), dtype=np.uint8)
        else:
            self.observation_space = gymn.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
        # Converter Discrete old gym para gymnasium Discrete
        self.action_space = gymn.spaces.Discrete(env.action_space.n)

    def reset(self, **kwargs):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        chw = np.transpose(obs, (2,0,1))  # 3,64,64 CHW
        if self.frame_stack > 1:
            for _ in range(self.frame_stack):
                self.frames.append(chw)
            stacked = np.concatenate(list(self.frames), axis=0)
            return stacked, {}
        return chw, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = done
        truncated = False
        chw = np.transpose(obs, (2,0,1))
        if self.frame_stack > 1:
            self.frames.append(chw)
            stacked = np.concatenate(list(self.frames), axis=0)
            return stacked, reward, terminated, truncated, info
        return chw, reward, terminated, truncated, info


class ProcgenVectorWrapper(gymn.Env):
    """
    Variação SEM CV do mesmo Procgen: retorna vetor flatten downsampled
    Mesmo jogo coinrun, mas obs é vetor 512D (downsample 16x16 grayscale flatten)
    Permite comparar CNN vs MLP no mesmo ambiente
    """
    def __init__(self, env, downsample=16):
        super().__init__()
        self.env = env
        self.downsample = downsample
        # vetor = downsample*downsample
        self.observation_space = gymn.spaces.Box(low=0, high=255, shape=(downsample*downsample,), dtype=np.uint8)
        self.action_space = gymn.spaces.Discrete(env.action_space.n)

    def reset(self, **kwargs):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        vec = self._to_vector(obs)
        return vec, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        vec = self._to_vector(obs)
        return vec, reward, done, False, info

    def _to_vector(self, obs):
        # obs 64x64x3 -> grayscale -> downsample -> flatten
        if len(obs.shape)==3 and obs.shape[2]==3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        small = cv2.resize(gray, (self.downsample, self.downsample), interpolation=cv2.INTER_AREA)
        return small.flatten().astype(np.uint8)


def make_procgen_env(game='coinrun', num_levels=200, distribution_mode='easy', seed=0, frame_stack=1, vector=False):
    """
    Factory Procgen
    game: coinrun, starpilot, bossfight, dodgeball, etc.
    vector=False -> imagem 64x64x3 (CV), True -> vetor 256D (sem CV)
    """
    env = gym.make(f'procgen:procgen-{game}-v0', num_levels=num_levels, start_level=0, distribution_mode=distribution_mode, rand_seed=seed)
    if vector:
        return ProcgenVectorWrapper(env)
    else:
        return ProcgenGymWrapper(env, frame_stack=frame_stack)
