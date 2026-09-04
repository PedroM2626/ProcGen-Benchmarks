"""
PPO with a REAL auxiliary representation-learning objective.

Used by the contrastive-family benchmark (CURL / CPC / ACL / SPR), the advanced
paradigms benchmark (ICM curiosity, Latent World Model, Contrastive encoder) and
the visual-architecture benchmark (ViT / Impoola / LSTM-Attention backbones).

Everything here is trained by gradient descent (optax). The auxiliary loss is
computed on on-policy rollout transitions and back-propagated jointly with PPO;
ICM additionally injects a genuine intrinsic (forward-dynamics prediction error)
reward into the training signal. No metric is hard-coded.
"""
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from typing import Any, NamedTuple

from src.contrastive_types import (
    SpatialContrastiveHead, TemporalContrastiveHead,
    ActionConditionalContrastiveHead, SPRPredictorHead, info_nce_similarity,
)
from src.advanced_modules import IntrinsicCuriosityModule, LatentWorldModel, ContrastiveEncoder, info_nce_loss


class AuxActorCritic(nn.Module):
    """Backbone feature extractor + actor/critic heads; also exposes features."""
    extractor_cls: Any
    action_dim: int = 17

    @nn.compact
    def __call__(self, x):
        feats = self.extractor_cls()(x)
        logits = nn.Dense(self.action_dim)(feats)
        value = jnp.squeeze(nn.Dense(1)(feats), axis=-1)
        return logits, value, feats


class RNDNet(nn.Module):
    """Random Network Distillation MLP: predictor (trained) / target (fixed random)."""
    dim: int = 128

    @nn.compact
    def __call__(self, x):
        h = x.astype(jnp.float32).reshape((x.shape[0], -1))
        h = nn.relu(nn.Dense(256)(h))
        h = nn.relu(nn.Dense(256)(h))
        return nn.Dense(self.dim)(h)


def random_shift_augment(rng, x, pad=4):
    """CURL-style random-shift augmentation for image batches (B,H,W,C)."""
    b, h, w, c = x.shape
    xp = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='edge')
    rng1, rng2 = jax.random.split(rng)
    dh = jax.random.randint(rng1, (b,), 0, 2 * pad + 1)
    dw = jax.random.randint(rng2, (b,), 0, 2 * pad + 1)

    def _crop(starts, xx):
        return jax.lax.dynamic_slice(xx, (starts[0], starts[1], 0), (h, w, c))
    return jax.vmap(_crop)(jnp.stack([dh, dw], axis=1), xp)


class AuxPPOTrainer:
    def __init__(self, extractor_cls, env_manager, aux_type="none", num_envs=64, num_steps=64,
                 action_dim=17, lr=3e-4, aux_lr=1e-3, gamma=0.99, gae_lambda=0.95,
                 clip_eps=0.2, ent_coef=0.01, vf_coef=0.5, update_epochs=4, num_minibatches=4,
                 aux_coef=0.1, intrinsic_coef=0.01):
        self.env_manager = env_manager
        self.aux_type = aux_type
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.aux_coef = aux_coef
        self.intrinsic_coef = intrinsic_coef

        self.model = AuxActorCritic(extractor_cls=extractor_cls, action_dim=action_dim)
        self.action_dim = action_dim
        self.opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr, eps=1e-5))
        self.aux_opt = optax.adam(aux_lr)

        # auxiliary head (its own parameter subtree)
        self.rnd_target = None
        if aux_type == "spatial":
            self.aux_head = SpatialContrastiveHead()
        elif aux_type == "temporal":
            self.aux_head = TemporalContrastiveHead()
        elif aux_type == "action":
            self.aux_head = ActionConditionalContrastiveHead(action_dim=action_dim)
        elif aux_type == "spr":
            self.aux_head = SPRPredictorHead()
        elif aux_type == "icm":
            self.aux_head = IntrinsicCuriosityModule(action_dim=action_dim)
        elif aux_type == "world_model":
            self.aux_head = LatentWorldModel(action_dim=action_dim)
        elif aux_type == "contrastive":
            self.aux_head = ContrastiveEncoder()
        elif aux_type == "rnd":
            self.aux_head = RNDNet()
            self.rnd_target = RNDNet()
        else:
            self.aux_head = None

    def create_state(self, rng, input_shape):
        rng, r1, r2 = jax.random.split(rng, 3)
        init_x = jnp.zeros((1, *input_shape))
        ac_params = self.model.init(r1, init_x)['params']
        aux_params = {}
        if self.aux_head is not None:
            feats_dim = 512
            dummy_feat = jnp.zeros((1, feats_dim))
            dummy_flat = jnp.zeros((1, *input_shape))
            dummy_act = jnp.zeros((1,), dtype=jnp.int32)
            if self.aux_type == "spatial":
                aux_params = self.aux_head.init(r2, dummy_feat)['params']
            elif self.aux_type == "temporal":
                aux_params = self.aux_head.init(r2, dummy_feat, dummy_feat)['params']
            elif self.aux_type == "action":
                aux_params = self.aux_head.init(r2, dummy_feat, dummy_act, dummy_feat)['params']
            elif self.aux_type == "spr":
                aux_params = self.aux_head.init(r2, dummy_feat, dummy_feat)['params']
            elif self.aux_type == "icm":
                aux_params = self.aux_head.init(r2, dummy_flat, dummy_flat, dummy_act)['params']
            elif self.aux_type == "world_model":
                aux_params = self.aux_head.init(r2, dummy_flat, dummy_act, dummy_flat)['params']
            elif self.aux_type == "contrastive":
                aux_params = self.aux_head.init(r2, dummy_flat)['params']
            elif self.aux_type == "rnd":
                pred = self.aux_head.init(r2, dummy_flat)['params']
                targ = self.rnd_target.init(jax.random.PRNGKey(1234), dummy_flat)['params']
                aux_params = {'pred': pred, 'target': targ}
        params = {'ac': ac_params, 'aux': aux_params}
        return params, self.opt.init(params), self.aux_opt.init(aux_params)

    # ---------- auxiliary loss on rollout pairs ----------
    def _aux_loss(self, params, obs_seq, act_seq, rng):
        """obs_seq: (T,E,...), act_seq: (T,E). Uses consecutive-frame pairs."""
        if self.aux_type == "none" or self.aux_head is None:
            return jnp.array(0.0)
        curr = obs_seq[:-1].reshape((-1,) + obs_seq.shape[2:])
        nxt = obs_seq[1:].reshape((-1,) + obs_seq.shape[2:])
        acts = act_seq[:-1].reshape(-1)
        _, _, feat_curr = self.model.apply({'params': params['ac']}, curr)
        _, _, feat_nxt = self.model.apply({'params': params['ac']}, nxt)

        if self.aux_type == "spatial":
            rng1, rng2 = jax.random.split(rng)
            v1 = random_shift_augment(rng1, curr)
            v2 = random_shift_augment(rng2, curr)
            _, _, f1 = self.model.apply({'params': params['ac']}, v1)
            _, _, f2 = self.model.apply({'params': params['ac']}, v2)
            z1 = self.aux_head.apply({'params': params['aux']}, f1)
            z2 = self.aux_head.apply({'params': params['aux']}, f2)
            return info_nce_similarity(z1, z2)
        if self.aux_type == "temporal":
            pred, z_fut = self.aux_head.apply({'params': params['aux']}, feat_curr, feat_nxt)
            return info_nce_similarity(pred, z_fut)
        if self.aux_type == "action":
            z_t, z_n = self.aux_head.apply({'params': params['aux']}, feat_curr, acts, feat_nxt)
            return info_nce_similarity(z_t, z_n)
        if self.aux_type == "spr":
            return self.aux_head.apply({'params': params['aux']}, feat_curr, feat_nxt)
        if self.aux_type == "icm":
            _, pred_phi, phi_next, _ = self.aux_head.apply({'params': params['aux']}, curr, nxt, acts)
            inv_logits = self.aux_head.apply({'params': params['aux']}, curr, nxt, acts)[0]
            inv_loss = optax.softmax_cross_entropy_with_integer_labels(inv_logits, acts).mean()
            fwd_loss = jnp.mean(jnp.square(pred_phi - jax.lax.stop_gradient(phi_next)))
            return inv_loss + fwd_loss
        if self.aux_type == "world_model":
            _, _, _, dyn_loss = self.aux_head.apply({'params': params['aux']}, curr, acts, nxt)
            return dyn_loss
        if self.aux_type == "contrastive":
            rng1, rng2 = jax.random.split(rng)
            q = self.aux_head.apply({'params': params['aux']}, random_shift_augment(rng1, curr))
            k = self.aux_head.apply({'params': params['aux']}, random_shift_augment(rng2, curr))
            return info_nce_loss(q, k)
        if self.aux_type == "rnd":
            pred = self.aux_head.apply({'params': params['aux']['pred']}, curr)
            targ = jax.lax.stop_gradient(self.rnd_target.apply({'params': params['aux']['target']}, curr))
            return jnp.mean(jnp.square(pred - targ))
        return jnp.array(0.0)

    def _intrinsic_reward(self, params, obs, next_obs, action):
        if self.aux_type == "icm":
            _, _, _, r_int = self.aux_head.apply({'params': params['aux']}, obs, next_obs, action)
            # Bound the curiosity bonus: an untrained forward model yields huge prediction
            # errors that would otherwise explode the value targets and destabilize PPO.
            return jnp.clip(r_int, 0.0, 5.0)
        if self.aux_type == "rnd":
            pred = self.aux_head.apply({'params': params['aux']['pred']}, next_obs)
            targ = jax.lax.stop_gradient(self.rnd_target.apply({'params': params['aux']['target']}, next_obs))
            r = jnp.mean(jnp.square(pred - targ), axis=-1)
            return jnp.clip(r, 0.0, 5.0)
        return jnp.zeros(obs.shape[0])

    def make_train_step(self):
        def train_step(carry, _):
            params, opt_state, aux_opt_state, env_state, last_obs, rng = carry

            def _env_step(step_state, _):
                params, e_state, obs, r = step_state
                r, a_rng, s_rng = jax.random.split(r, 3)
                logits, value, _ = self.model.apply({'params': params['ac']}, obs)
                action = jax.random.categorical(a_rng, logits)
                log_prob = jax.nn.log_softmax(logits)[jnp.arange(self.num_envs), action]
                next_obs, next_e, reward, done, info, r2 = self.env_manager.step(s_rng, e_state, action)
                r_int = self._intrinsic_reward(params, obs, next_obs, action)
                reward = reward + self.intrinsic_coef * r_int
                trans = (done, action, value, reward, log_prob, obs)
                return (params, next_e, next_obs, r2), trans

            (params, env_state, last_obs, rng), traj = jax.lax.scan(
                _env_step, (params, env_state, last_obs, rng), None, length=self.num_steps)
            done, action, value, reward, log_prob, obs = traj

            _, last_val, _ = self.model.apply({'params': params['ac']}, last_obs)

            def _gae(carry, x):
                gae, next_value = carry
                d, v, r = x
                delta = r + self.gamma * next_value * (1.0 - d) - v
                gae = delta + self.gamma * self.gae_lambda * (1.0 - d) * gae
                return (gae, v), gae
            _, adv = jax.lax.scan(_gae, (jnp.zeros_like(last_val), last_val),
                                  (done, value, reward), reverse=True)
            returns = adv + value

            B = self.num_steps * self.num_envs
            mb = B // self.num_minibatches
            f_obs = obs.reshape((B,) + obs.shape[2:])
            f_act = action.reshape((B,))
            f_logp = log_prob.reshape((B,))
            f_adv = adv.reshape((B,))
            f_ret = returns.reshape((B,))

            def _update_epoch(state, _):
                params, opt_state, rng = state

                def _minibatch(carry, idx):
                    params, opt_state = carry
                    ob, ac, olp, g, ret = (jnp.take(f_obs, idx, 0), jnp.take(f_act, idx, 0),
                                           jnp.take(f_logp, idx, 0), jnp.take(f_adv, idx, 0),
                                           jnp.take(f_ret, idx, 0))

                    def _loss(params):
                        logits, v, _ = self.model.apply({'params': params['ac']}, ob)
                        lp_all = jax.nn.log_softmax(logits)
                        lp = lp_all[jnp.arange(ob.shape[0]), ac]
                        ent = -jnp.sum(jax.nn.softmax(logits) * lp_all, axis=-1).mean()
                        gg = (g - g.mean()) / (g.std() + 1e-8)
                        ratio = jnp.exp(lp - olp)
                        s1 = ratio * gg
                        s2 = jnp.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * gg
                        pi = -jnp.minimum(s1, s2).mean()
                        vf = 0.5 * jnp.square(v - ret).mean()
                        return pi + self.vf_coef * vf - self.ent_coef * ent
                    loss, grads = jax.value_and_grad(_loss)(params)
                    updates, new_opt = self.opt.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    return (params, new_opt), loss

                rng, perm_rng = jax.random.split(rng)
                perm = jax.random.permutation(perm_rng, B)
                idxs = perm.reshape((self.num_minibatches, mb))
                (params, opt_state), loss = jax.lax.scan(_minibatch, (params, opt_state), idxs)
                return (params, opt_state, rng), loss.mean()

            (params, opt_state, rng), ppo_loss = jax.lax.scan(
                _update_epoch, (params, opt_state, rng), None, length=self.update_epochs)

            # ---- auxiliary representation update (joint training) ----
            rng, aux_rng = jax.random.split(rng)
            def _aux_grad(params):
                return self._aux_loss(params, obs, action, aux_rng)
            aux_loss, aux_grads = jax.value_and_grad(_aux_grad)(params)
            aux_grads = {'ac': aux_grads['ac'], 'aux': aux_grads['aux']}
            # scale aux gradient and apply through the main optimizer (joint)
            scaled = jax.tree_util.tree_map(lambda g: self.aux_coef * g, aux_grads)
            updates, opt_state = self.opt.update(scaled, opt_state, params)
            params = optax.apply_updates(params, updates)

            metrics = {"ppo_loss": ppo_loss.mean(), "aux_loss": aux_loss,
                       "mean_reward": reward.sum(axis=0).mean()}
            return (params, opt_state, aux_opt_state, env_state, last_obs, rng), metrics

        return jax.jit(train_step)

    def make_eval_policy(self, deterministic=True):
        def select(params, obs, rng):
            logits, _, _ = self.model.apply({'params': params['ac']}, obs)
            return jnp.argmax(logits, axis=-1) if deterministic else jax.random.categorical(rng, logits)
        return select
