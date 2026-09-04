"""Shapes/params do zoo + forward JIT (GPU se houver, CPU se nao).

Uso: python -m jax_port.tests.test_zoo
"""

import jax
import jax.numpy as jnp


def test_zoo():
    from jax_port.backbones import BACKBONES
    from jax_port.networks import ActorCritic
    key = jax.random.PRNGKey(0)
    got = {}
    for name, cls in BACKBONES.items():
        vec = name == "mlp"
        stoch = name == "vae"
        dummy = (jnp.zeros((2, 256), jnp.float32) if vec
                 else jnp.zeros((2, 64, 64, 3), jnp.float32))
        m = ActorCritic(backbone=cls(), n_actions=15, stochastic=stoch)
        key, k, ks = jax.random.split(key, 3)
        params = m.init(k, dummy, ks)
        n = sum(p.size for p in jax.tree_util.tree_leaves(params))
        logits, value = jax.jit(lambda p, x, kk: m.apply(p, x, kk))(
            params, dummy, ks)
        jax.block_until_ready((logits, value))
        assert tuple(logits.shape) == (2, 15), name
        assert tuple(value.shape) == (2,), name
        got[name] = n
    # Fidelidade arquitetural: classic ~600k (VALID); gemeos identicos.
    assert 550_000 < got["classic"] < 650_000, got["classic"]
    assert got["ae"] == got["classic"] and got["recon"] == got["classic"]
    return got


if __name__ == "__main__":
    for k, v in test_zoo().items():
        print(f"{k:14s} {v:,}")
    print("ZOO_OK")
