"""Zoo de backbones Flax — paridade com ``models/`` do estudo (mede-se SPS por arq).

Todos recebem NHWC float32 [0,1] e devolvem 512D (convecao do zoo; o
estudo usa FC 512 em todos os extratores). Referencias:
  classic      <- ``models/sb3_extractors.py:8`` ClassicCNNExtractor
  cbam         <- ``models/cnn_attention.py:82`` Channel+Spatial, reduction 16
  spatial      <- ``models/cnn_attention.py:6`` SpatialAttention + residual
  impala       <- ``models/combined_extractors.py`` ImpalaCNNExtractor
  impoola      <- idem ImpoolaCNNExtractor (GAP + gargalo 64D)
  resnet18     <- idem ResNet18Extractor (padrao, sem afinamento)
  vit          <- idem ViTExtractor (patches 8x8 -> 64, Transformer x4)
  mlp          <- ``models/`` via ``ProcgenVectorWrapper`` (vetor 256D);
                  backbone MLP [64,64] tanh (default SB3 MlpPolicy) + FC512
                  para unificar os heads (documentado em train.py).
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


class ClassicCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        # VALID em todas (SB3 NatureCNN): 64->15->6->4, flat 1024 (~600k).
        x = nn.relu(nn.Conv(32, (8, 8), strides=(4, 4), padding="VALID")(x))
        x = nn.relu(nn.Conv(64, (4, 4), strides=(2, 2), padding="VALID")(x))
        x = nn.relu(nn.Conv(64, (3, 3), strides=(1, 1), padding="VALID")(x))
        x = x.reshape((x.shape[0], -1))
        return nn.relu(nn.Dense(512)(x))


class _ChannelAttention(nn.Module):
    reduction: int = 16

    @nn.compact
    def __call__(self, x):
        c = x.shape[-1]
        m = x.mean(axis=(1, 2))
        h = nn.relu(nn.Dense(c // self.reduction)(m))
        w = nn.sigmoid(nn.Dense(c)(h))[:, None, None, :]
        return x * w


class _SpatialAttention(nn.Module):
    @nn.compact
    def __call__(self, x):
        m = jnp.concatenate([x.mean(axis=-1, keepdims=True),
                             x.max(axis=-1, keepdims=True)], axis=-1)
        w = nn.sigmoid(nn.Conv(1, (7, 7), padding="SAME")(m))
        return x * w + x  # residual: estabiliza o spatial puro (estudo)


class CBAMCnn(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(32, (8, 8), strides=(4, 4), padding="VALID")(x))
        x = _ChannelAttention()(x)
        x = nn.relu(nn.Conv(64, (4, 4), strides=(2, 2), padding="VALID")(x))
        x = _SpatialAttention()(x)
        x = nn.relu(nn.Conv(64, (3, 3), strides=(1, 1), padding="VALID")(x))
        x = x.reshape((x.shape[0], -1))
        return nn.relu(nn.Dense(512)(x))


class SpatialCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(32, (8, 8), strides=(4, 4), padding="VALID")(x))
        x = _SpatialAttention()(x)
        x = nn.relu(nn.Conv(64, (4, 4), strides=(2, 2), padding="VALID")(x))
        x = nn.relu(nn.Conv(64, (3, 3), strides=(1, 1), padding="VALID")(x))
        x = x.reshape((x.shape[0], -1))
        return nn.relu(nn.Dense(512)(x))


class _ImpalaBlock(nn.Module):
    depth: int

    @nn.compact
    def __call__(self, x):
        h = nn.relu(nn.Conv(self.depth, (3, 3), padding="SAME")(x))
        h = nn.relu(nn.Conv(self.depth, (3, 3), padding="SAME")(h))
        if x.shape[-1] != self.depth:
            x = nn.Conv(self.depth, (1, 1), padding="SAME")(x)
        h = x + h
        return nn.max_pool(h, (3, 3), strides=(2, 2), padding="SAME")


class ImpalaCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        for d in (16, 32, 32):
            x = _ImpalaBlock(d)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        return nn.relu(nn.Dense(512)(x))


class ImpoolaCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        for d in (16, 32, 32):
            x = _ImpalaBlock(d)(x)
        x = x.mean(axis=(1, 2))  # GAP
        x = nn.relu(nn.Dense(64)(x))  # gargalo 64D (estudo: colapso aqui)
        return nn.relu(nn.Dense(512)(x))


class _ResBlock(nn.Module):
    depth: int
    stride: int = 1

    @nn.compact
    def __call__(self, x):
        h = nn.Conv(self.depth, (3, 3), strides=(self.stride, self.stride),
                    padding="SAME", use_bias=False)(x)
        h = nn.BatchNorm(use_running_average=True)(h)
        h = nn.relu(h)
        h = nn.Conv(self.depth, (3, 3), padding="SAME", use_bias=False)(h)
        h = nn.BatchNorm(use_running_average=True)(h)
        if x.shape[-1] != self.depth or self.stride != 1:
            x = nn.Conv(self.depth, (1, 1), strides=(self.stride, self.stride),
                        padding="SAME", use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=True)(x)
        return nn.relu(x + h)


class ResNet18(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding="SAME", use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=True)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
        for depth, stride in ((64, 1), (64, 1), (128, 2), (128, 1),
                              (256, 2), (256, 1), (512, 2), (512, 1)):
            x = _ResBlock(depth, stride)(x)
        x = x.mean(axis=(1, 2))
        return nn.relu(nn.Dense(512)(x))


class _TransformerBlock(nn.Module):
    dim: int
    heads: int = 4
    mlp: int = 256

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        h = nn.SelfAttention(num_heads=self.heads)(h)
        x = x + h
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.mlp)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.dim)(h)
        return x + h


class ViTTiny(nn.Module):
    patch: int = 16
    dim: int = 128
    layers: int = 4

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape
        p = self.patch
        x = x.reshape(b, h // p, p, w // p, p, c).transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(b, (h // p) * (w // p), p * p * c)
        x = nn.Dense(self.dim)(x)
        pos = self.param("pos_emb", nn.initializers.normal(0.02),
                         (1, x.shape[1], self.dim))
        x = x + pos
        for _ in range(self.layers):
            x = _TransformerBlock(self.dim)(x)
        x = nn.LayerNorm()(x).mean(axis=1)
        return nn.relu(nn.Dense(512)(x))


class MlpBackbone(nn.Module):
    """Backbone do modo vetor (obs 256D). MLP [64,64] tanh = default SB3
    MlpPolicy; FC512 final unifica os heads com o resto do zoo."""

    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(64)(x))
        x = nn.tanh(nn.Dense(64)(x))
        return nn.relu(nn.Dense(512)(x))


class LSTMAttention(nn.Module):
    """Replica fiel de ``LSTMAttentionExtractor`` (stateless!): CNN + pool
    4x4 -> repete a feature 4x (sequencia fake) -> BiLSTM 256 -> MHA 4
    heads -> media -> FC512. Sem carry entre steps (estudo)."""
    hidden: int = 256
    heads: int = 4

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(32, (8, 8), strides=(4, 4), padding="VALID")(x))
        x = nn.relu(nn.Conv(64, (4, 4), strides=(2, 2), padding="VALID")(x))
        x = nn.relu(nn.Conv(64, (3, 3), strides=(1, 1), padding="VALID")(x))
        # 4x4x64 exatos com VALID -> AdaptiveAvgPool2d(4) e no-op:
        f = x.reshape((x.shape[0], -1))  # 1024D
        seq = jnp.repeat(f[:, None, :], 4, axis=1)  # B x 4 x 1024
        cell_f, cell_b = nn.LSTMCell(self.hidden), nn.LSTMCell(self.hidden)
        b = seq.shape[0]
        cf = cell_f.initialize_carry(jax.random.PRNGKey(0), (b,))
        cb = cell_b.initialize_carry(jax.random.PRNGKey(1), (b,))
        outs_f, outs_b = [], []
        for t in range(4):
            cf, hf = cell_f(cf, seq[:, t])
            outs_f.append(hf)
        for t in reversed(range(4)):
            cb, hb = cell_b(cb, seq[:, t])
            outs_b.append(hb)
        lstm_out = jnp.concatenate(
            [jnp.stack(outs_f, axis=1),
             jnp.stack(outs_b[::-1], axis=1)], axis=-1)  # B x 4 x 512
        attn_out = nn.MultiHeadAttention(num_heads=self.heads)(lstm_out)
        pooled = attn_out.mean(axis=1)
        return nn.relu(nn.Dense(512)(pooled))


class VAEBackbone(nn.Module):
    """Replica de ``VAEExtractor`` (forward estocastico): mu/logvar 128D,
    z = mu + eps*std (reparam), fc_out -> 512. SEM termo KL no loss PPO
    (o estudo tambem nao otimiza KL/dream/proj — so o forward entra)."""
    latent: int = 128

    @nn.compact
    def __call__(self, x, key):
        x = nn.relu(nn.Conv(32, (8, 8), strides=(4, 4), padding="VALID")(x))
        x = nn.relu(nn.Conv(64, (4, 4), strides=(2, 2), padding="VALID")(x))
        x = nn.relu(nn.Conv(64, (3, 3), strides=(1, 1), padding="VALID")(x))
        x = x.reshape((x.shape[0], -1))
        mu, logvar = nn.Dense(self.latent)(x), nn.Dense(self.latent)(x)
        z = mu + jnp.exp(0.5 * logvar) * jax.random.normal(key, mu.shape)
        return nn.relu(nn.Dense(512)(z))


BACKBONES = {
    "classic": ClassicCNN,
    "cbam": CBAMCnn,
    "spatial": SpatialCNN,
    "impala": ImpalaCNN,
    "impoola": ImpoolaCNN,
    "resnet18": ResNet18,
    "vit": ViTTiny,
    "mlp": MlpBackbone,
    "lstm_attention": LSTMAttention,
    "vae": VAEBackbone,
    # Gemeos de treino (achado documentado): AE/Recon forward == classic;
    # contrastive == classic + noise (usar com --augment noise).
    "ae": ClassicCNN,
    "recon": ClassicCNN,
    "contrastive": ClassicCNN,
}
PIXEL_BACKBONES = [k for k in BACKBONES if k != "mlp"]
