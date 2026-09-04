"""Paridade de env: gym3 batch vs gym unitario, mesma seed/acoes (200 steps).

Se o C++ e o mesmo e os niveis os mesmos, gym3 N=1 deve devolver as
mesmas obs/recompensas que o gym. Requer: gym+procgen (venv do estudo
ou do porte). Uso: python -m jax_port.tests.test_parity
"""

import numpy as np


def test_env_parity(steps=200, seed=42):
    import gym
    from procgen import ProcgenGym3Env
    import procgen  # noqa: F401
    g3 = ProcgenGym3Env(num=1, env_name="coinrun", num_levels=200,
                        distribution_mode="easy", rand_seed=seed)
    ge = gym.make("procgen:procgen-coinrun-v0", num_levels=200, start_level=0,
                  distribution_mode="easy", rand_seed=seed)
    rng = np.random.default_rng(seed)
    _, d3, _ = g3.observe()
    og = ge.reset()
    mo, mr = 0, 0
    for _ in range(steps):
        a = np.array([rng.integers(15)])
        g3.act(a)
        rew3, d3, _ = g3.observe()
        og2, rg, dg, _ = ge.step(int(a[0]))
        if dg:
            og2 = ge.reset()
        mo += not np.array_equal(d3["rgb"][0], og2)
        mr += abs(float(rew3[0]) - float(rg)) > 1e-6
    assert mo == 0 and mr == 0, f"obs_mismatch={mo} rew_mismatch={mr}"
    return {"steps": steps, "obs_mismatch": mo, "rew_mismatch": mr}


if __name__ == "__main__":
    print(test_env_parity())
    print("PARITY_OK")
