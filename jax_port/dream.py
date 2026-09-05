"""Dreams VAE/AE em JAX — paridade com visualize_side_by_side.py (modo dream).

Estudo: dream() = fc_dec(z) -> reshape(4,4,64) -> deconv k3s1 ->
deconv k4s2 -> deconv k8s4 -> sigmoid (deconvs espelham o encoder).
Micro-diferenca documentada: aqui padding SAME (saida exata 64x64);
no estudo VALID (saida 60x60, redimensionada no video).
Protocolo: coleta 20k frames (policy aleatoria, bossfight 200 easy)
-> treina VAE (BCE+KL) e AE (BCE) ~2k steps batch 256 -> PNG
lado-a-lado (real/vae/ae) + MP4 rollout com sonhos.
Uso (GPU LIVRE! nao rodar junto da grade):
    .../train.py ... # nao; este script e standalone:
    wsl -e env PYTHONPATH=... /root/procgen-jax/bin/python \
      jax_port/dream.py --game bossfight --seed 42 --out-dir jax_port/dreams
"""

import argparse
import json
import os
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from procgen import ProcgenGym3Env


class Decoder(nn.Module):
    n_channels: int = 3

    @nn.compact
    def __call__(self, z):
        h = nn.relu(nn.Dense(4 * 4 * 64)(z))
        h = h.reshape((-1, 4, 4, 64))
        h = nn.relu(nn.ConvTranspose(64, (3, 3), strides=(1, 1),
                                     padding="SAME")(h))  # 4x4
        h = nn.relu(nn.ConvTranspose(64, (4, 4), strides=(2, 2),
                                     padding="SAME")(h))  # 8x8
        h = nn.relu(nn.ConvTranspose(32, (4, 4), strides=(2, 2),
                                     padding="SAME")(h))  # 16x16
        h = nn.relu(nn.ConvTranspose(32, (4, 4), strides=(2, 2),
                                     padding="SAME")(h))  # 32x32
        h = nn.ConvTranspose(self.n_channels, (4, 4), strides=(2, 2),
                             padding="SAME")(h)  # 64x64
        return nn.sigmoid(h)  # (B,64,64,3)


class VAE(nn.Module):
    latent: int = 128
    stochastic: bool = True

    def setup(self):
        from jax_port.backbones import ClassicCNN
        self.enc = ClassicCNN()
        self.mu = nn.Dense(self.latent)
        self.logvar = nn.Dense(self.latent)
        self.dec = Decoder()

    def __call__(self, x, key):
        f = self.enc(x)
        mu, logvar = self.mu(f), self.logvar(f)
        if self.stochastic:
            z = mu + jnp.exp(0.5 * logvar) * jax.random.normal(key, mu.shape)
        else:
            z = mu  # AE deterministico (estudo)
        return self.dec(z), mu, logvar

    def dream(self, x, key=None):
        del key
        return self.dec(self.mu(self.enc(x)))


def collect(game, seed, n_frames, n_envs=8):
    env = ProcgenGym3Env(num=n_envs, env_name=game, num_levels=200,
                         distribution_mode="easy", rand_seed=seed)
    rng = np.random.default_rng(seed)
    buf, got = [], 0
    _, d, _ = env.observe()
    while got < n_frames:
        a = rng.integers(0, 15, size=n_envs)
        env.act(a)
        _, d, _ = env.observe()
        o = d["rgb"] if isinstance(d, dict) else d
        buf.append(o.copy())
        got += n_envs
    return np.concatenate(buf)[:n_frames]


def train_vae(frames, kind, steps=2000, batch=256, seed=0):
    vae = VAE(stochastic=(kind == "vae"))
    key = jax.random.PRNGKey(seed)
    key, k0 = jax.random.split(key)
    params = vae.init(k0, jnp.zeros((1, 64, 64, 3), jnp.float32),
                      jax.random.PRNGKey(0))
    opt = optax.adam(1e-3)
    os_ = opt.init(params)

    @jax.jit
    def step(p, os_, xb, key):
        key, kz = jax.random.split(key)
        x = xb.astype(jnp.float32) / 255.0

        def loss_fn(pp):
            recon, mu, logvar = vae.apply(pp, x, kz)
            bce = -(x * jnp.log(recon + 1e-7) +
                    (1 - x) * jnp.log(1 - recon + 1e-7)).mean()
            kl = (-0.5 * (1 + logvar - mu ** 2 - jnp.exp(logvar))).mean()
            if kind == "ae":
                return bce, (bce, 0.0)
            return bce + kl, (bce, kl)

        (tot, (bce, kl)), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(p)
        upd, os_ = opt.update(grads, os_, p)
        return optax.apply_updates(p, upd), os_, tot, bce, kl

    rng = np.random.default_rng(seed)
    n = len(frames)
    key, kw = jax.random.split(key)
    _, _, _, _, _ = step(params, os_, jnp.zeros((batch, 64, 64, 3),
                                                jnp.uint8), kw)
    t0 = time.perf_counter()
    for i in range(steps):
        idx = rng.integers(0, n, size=batch)
        key, ks = jax.random.split(key)
        params, os_, tot, bce, kl = step(
            params, os_, jnp.asarray(frames[idx]), ks)
        if (i + 1) % 500 == 0:
            print(f"  {kind} step={i+1} loss={float(tot):.4f} "
                  f"bce={float(bce):.4f} kl={float(kl):.4f}", flush=True)
    dt = time.perf_counter() - t0
    return vae, params, {"bce": float(bce), "kl": float(kl),
                         "wall_s": round(dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="bossfight")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frames", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out-dir", default="jax_port/dreams")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"collect {args.frames} frames {args.game}...", flush=True)
    frames = collect(args.game, args.seed, args.frames)
    print(f"collected {frames.shape}", flush=True)
    res = {}
    for kind in ("vae", "ae"):
        print(f"train {kind}...", flush=True)
        vae, params, info = train_vae(frames, kind, args.steps,
                                      seed=args.seed)
        res[kind] = info
        # PNG lado-a-lado: 8 obs fixas (real / vae-dream / ae-dream depois)
        key = jax.random.PRNGKey(args.seed + 1)
        sample = frames[:8].astype(jnp.float32) / 255.0
        f = vae.apply(params, sample, key, method="dream")
        panel = np.asarray((f * 255).astype(np.uint8))
        np.save(os.path.join(args.out_dir, f"dreams_{kind}.npy"), panel)
        # guarda params do VAE p/ o video combinado abaixo
        if kind == "vae":
            vae_params, vae_mod = params, vae
        else:
            ae_params, ae_mod = params, vae
    # PNG combinado + MP4 rollout
    import imageio.v2 as imageio
    real = frames[:8]
    vd = np.load(os.path.join(args.out_dir, "dreams_vae.npy"))
    ad = np.load(os.path.join(args.out_dir, "dreams_ae.npy"))
    combo = np.concatenate(
        [np.concatenate([r, v, a], axis=1) for r, v, a in zip(real, vd, ad)],
        axis=0)
    imageio.imwrite(os.path.join(args.out_dir, "dreams_panel.png"), combo)
    # rollout video: env + sonhos por step
    env = ProcgenGym3Env(num=1, env_name=args.game, num_levels=0,
                         distribution_mode="easy", rand_seed=args.seed + 1000)
    rng = np.random.default_rng(args.seed)
    _, d, _ = env.observe()
    ims = []
    key = jax.random.PRNGKey(0)
    for _ in range(300):
        a = np.array([rng.integers(15)])
        env.act(a)
        _, d, _ = env.observe()
        o = (d["rgb"] if isinstance(d, dict) else d)[0]
        of = o.astype(np.float32) / 255.0
        key, k1, k2 = jax.random.split(key, 3)
        v = np.asarray((vae_mod.apply(vae_params, of[None], k1,
                                      method="dream")[0] * 255).astype(np.uint8))
        m = np.asarray((ae_mod.apply(ae_params, of[None], k2,
                                     method="dream")[0] * 255).astype(np.uint8))
        ims.append(np.concatenate([o, v, m], axis=1))
    imageio.mimsave(os.path.join(args.out_dir, "bossfight_dreams.gif"), ims,
                    fps=15, loop=0)
    res["video"] = "bossfight_dreams.gif"
    json.dump(res, open(os.path.join(args.out_dir, "dreams.json"), "w"),
              indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
