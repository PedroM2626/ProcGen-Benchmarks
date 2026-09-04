import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from collections import deque

# Mantém CarRacing para compatibilidade
class CarRacingWrapper(gym.Env):
    """
    Wrapper para CarRacing com:
    - Grayscale 64x64
    - Stack de 4 frames
    - Ação contínua normalizada
    """
    
    def __init__(self, frame_stack=4, frame_size=(64, 64), env_name=None):
        # Compatibilidade: tenta v3, fallback para v2 se não existir
        if env_name is None:
            try:
                self.env = gym.make('CarRacing-v3', render_mode='rgb_array')
            except Exception:
                self.env = gym.make('CarRacing-v2', render_mode='rgb_array')
        else:
            self.env = gym.make(env_name, render_mode='rgb_array')
        self.frame_stack = frame_stack
        self.frame_size = frame_size
        self.frames = deque(maxlen=frame_stack)
        
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(frame_stack, *frame_size),
            dtype=np.uint8
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(3,),
            dtype=np.float32
        )
    
    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        processed_obs = self._process_observation(obs)
        for _ in range(self.frame_stack):
            self.frames.append(processed_obs)
        return np.array(self.frames), info
    
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        converted_action = np.array([
            float(action[0]),
            float((action[1] + 1.0) / 2.0),
            float((action[2] + 1.0) / 2.0)
        ], dtype=np.float32)
        converted_action = np.clip(converted_action, -1.0, 1.0)
        obs, reward, terminated, truncated, info = self.env.step(converted_action)
        processed_obs = self._process_observation(obs)
        self.frames.append(processed_obs)
        return np.array(self.frames), reward, terminated, truncated, info
    
    def _process_observation(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, self.frame_size, interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)
    
    def close(self):
        self.env.close()
    
    def render(self):
        return self.env.render()


class CartPolePixelWrapper(gym.Env):
    """
    Mesmo ambiente CartPole-v1 mas com observação visual 64x64 grayscale stack4
    Permite comparar CV vs vetor no MESMO ambiente (solicitação do usuário)
    - Rápido: 200 steps/episódio vs 1000 CarRacing
    - Suporta CNN classic vs attention
    """
    def __init__(self, frame_stack=4, frame_size=(64, 64)):
        self.env = gym.make('CartPole-v1', render_mode='rgb_array')
        self.frame_stack = frame_stack
        self.frame_size = frame_size
        self.frames = deque(maxlen=frame_stack)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(frame_stack, *frame_size),
            dtype=np.uint8
        )
        self.action_space = self.env.action_space  # Discrete(2)

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        # obs é vetor, ignoramos; renderizamos
        img = self.env.render()
        processed = self._process_observation(img)
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        return np.array(self.frames), info

    def step(self, action):
        # action é Discrete
        obs, reward, terminated, truncated, info = self.env.step(action)
        img = self.env.render()
        processed = self._process_observation(img)
        self.frames.append(processed)
        return np.array(self.frames), reward, terminated, truncated, info

    def _process_observation(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, self.frame_size, interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)

    def close(self):
        self.env.close()
    def render(self):
        return self.env.render()


class CartPoleStateWrapper(gym.Env):
    """
    Variação SEM CV do mesmo ambiente CartPole-v1
    Obs vetor 4D, usa MLP — comparação direta CV vs não-CV
    """
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
    def reset(self, seed=None):
        return self.env.reset(seed=seed)
    def step(self, action):
        return self.env.step(action)
    def close(self):
        self.env.close()
    def render(self):
        return self.env.render()


class AtariWrapper(gym.Env):
    """
    Atari rápido para CV: Pong, Breakout
    10x mais rápido que CarRacing, ideal para testar attention
    """
    def __init__(self, game='ALE/Pong-v5', frame_stack=4, frame_size=(84, 84)):
        # frameskip já é 4 no ALE
        self.env = gym.make(game, render_mode='rgb_array', frameskip=1)
        self.frame_stack = frame_stack
        self.frame_size = frame_size
        self.frames = deque(maxlen=frame_stack)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(frame_stack, *frame_size),
            dtype=np.uint8
        )
        self.action_space = self.env.action_space

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        processed = self._process_observation(obs)
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        return np.array(self.frames), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed = self._process_observation(obs)
        self.frames.append(processed)
        return np.array(self.frames), reward, terminated, truncated, info

    def _process_observation(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        resized = cv2.resize(gray, self.frame_size, interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)

    def close(self):
        self.env.close()
    def render(self):
        return self.env.render()


def create_env(frame_stack=4, frame_size=(64, 64), env_name=None):
    """Factory compatível com código antigo — default CarRacing"""
    return CarRacingWrapper(frame_stack=frame_stack, frame_size=frame_size, env_name=env_name)

def create_fast_env(env_type='cartpole_pixels', frame_stack=4, frame_size=(64, 64)):
    """
    Factory para ambientes rápidos:
    - 'cartpole_pixels': CartPole com imagem 64x64 stack4 (CV) — MESMO ambiente que cartpole_state
    - 'cartpole_state': CartPole vetor 4D (sem CV) — comparação direta CV vs vetor
    - 'pong': Atari Pong 84x84 stack4 (CV rápido)
    - 'breakout': Atari Breakout
    - 'carracing': CarRacing original
    """
    if env_type == 'cartpole_pixels':
        return CartPolePixelWrapper(frame_stack=frame_stack, frame_size=frame_size)
    elif env_type == 'cartpole_state':
        return CartPoleStateWrapper()
    elif env_type == 'pong':
        return AtariWrapper(game='ALE/Pong-v5', frame_stack=frame_stack, frame_size=frame_size)
    elif env_type == 'breakout':
        return AtariWrapper(game='ALE/Breakout-v5', frame_stack=frame_stack, frame_size=frame_size)
    elif env_type == 'carracing':
        return CarRacingWrapper(frame_stack=frame_stack, frame_size=frame_size)
    else:
        raise ValueError(f"env_type desconhecido: {env_type}. Use: cartpole_pixels, cartpole_state, pong, breakout, carracing")


if __name__ == "__main__":
    for env_type in ['cartpole_pixels', 'cartpole_state', 'pong']:
        print(f"\n=== Testando {env_type} ===")
        env = create_fast_env(env_type)
        print("Obs:", env.observation_space, "Act:", env.action_space)
        obs, info = env.reset(seed=42)
        print("Obs shape:", obs.shape if hasattr(obs, 'shape') else type(obs), "dtype:", obs.dtype if hasattr(obs, 'dtype') else type(obs))
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i}: reward={reward:.2f} terminated={terminated}")
            if terminated or truncated:
                obs, info = env.reset()
        env.close()
        print(f"{env_type} OK")
