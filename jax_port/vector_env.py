"""Pipeline CPU-env do porte JAX (PA1).

Fronteira fiel ao estudo SB3: mesmos niveis/sementes/pre-processamento de
``procgen_wrapper.py`` (frame_stack=1, HWC 64x64x3 uint8 -> CHW 3x64x64),
porem sem dependencias de torch/sb3/cv2/gymnasium — apenas gym+procgen+numpy
no lado CPU. O lado JAX (GPU) recebe o batch via ``jnp.asarray`` + JIT.

Referencia de fidelidade:
  estudo: ``procgen_wrapper.py:29`` (reset HWC->CHW), ``:41`` (step),
  ``:91`` (factory num_levels/distribution_mode/rand_seed).
"""

import gym
import numpy as np

import procgen  # noqa: F401  (registra procgen:procgen-*-v0)


def make_single_env(game="coinrun", num_levels=200, distribution_mode="easy", rand_seed=0):
    """Um env ProcGen cru (gym API antiga, obs HWC uint8)."""
    return gym.make(
        f"procgen:procgen-{game}-v0",
        num_levels=num_levels,
        start_level=0,
        distribution_mode=distribution_mode,
        rand_seed=rand_seed,
    )


class ProcgenVectorEnv:
    """N envs ProcGen sincronos em CPU, obs empilhada CHW uint8.

    Semantica identica a N x ``ProcgenGymWrapper(frame_stack=1)`` do estudo:
    reset devolve ``(N,3,64,64)`` uint8; step recebe ``(N,)`` int e devolve
    ``obs (N,3,64,64) uint8, rew (N,) float32, done (N,) bool``.
    Autoreset SB3-style no done (o bench mede throughput, nao episodios).
    """

    def __init__(self, game="coinrun", num_envs=4, num_levels=200,
                 distribution_mode="easy", seed=42):
        self.envs = [
            make_single_env(game, num_levels, distribution_mode, seed + i)
            for i in range(num_envs)
        ]
        self.num_envs = num_envs
        self.action_space_n = self.envs[0].action_space.n

    def reset(self):
        batch = [np.transpose(e.reset(), (2, 0, 1)) for e in self.envs]
        return np.stack(batch).astype(np.uint8)

    def step(self, actions):
        obs, rew, done = [], [], []
        for e, a in zip(self.envs, actions):
            o, r, d, _ = e.step(int(a))
            if d:
                o = e.reset()
            obs.append(np.transpose(o, (2, 0, 1)))
            rew.append(r)
            done.append(d)
        return (np.stack(obs).astype(np.uint8),
                np.asarray(rew, dtype=np.float32),
                np.asarray(done, dtype=bool))

    def sample_actions(self, rng):
        return rng.integers(0, self.action_space_n, size=self.num_envs)
