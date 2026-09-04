"""Augments do estudo, fieis a ``compare_augment_contrastive.py`` + base noise.

Aplicados por forward com p=0.5, em float [0,1] (estudo):
  crop  : pad 4 replicate -> crop 64x64 randomico (1 offset por call)
  color : x * Uniform(0.8, 1.2), clip
  noise : x + N(0, 0.01), clip (forward base do ContrastiveExtractor)
  none  : identidade (caminho sem overhead: train.py nem chama)
"""

import jax
import jax.numpy as jnp


def make_augment(kind, p=0.5):
    if kind in (None, "none"):
        return None

    @jax.jit
    def aug(x, key):
        k1, k2, k3 = jax.random.split(key, 3)
        do = jax.random.uniform(k1) < p
        if kind == "crop":
            y = jnp.pad(x, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="edge")
            top = jax.random.randint(k2, (), 0, 9)
            left = jax.random.randint(k3, (), 0, 9)
            y = jax.lax.dynamic_slice_in_dim(
                jax.lax.dynamic_slice_in_dim(y, top, 64, axis=1),
                left, 64, axis=2)
        elif kind == "color":
            s = 0.8 + 0.4 * jax.random.uniform(k2)
            y = jnp.clip(x * s, 0.0, 1.0)
        elif kind == "noise":
            y = jnp.clip(x + jax.random.normal(k2, x.shape) * 0.01, 0.0, 1.0)
        else:
            raise ValueError(kind)
        return jnp.where(do, y, x)

    return aug


def requantize(x_f):
    """float [0,1] -> uint8 (reuso do caminho uint8, erro max 1/255)."""
    return jnp.clip(jnp.rint(x_f * 255.0), 0, 255).astype(jnp.uint8)
