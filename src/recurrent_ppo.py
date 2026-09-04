"""
Recurrent PPO for the LSTM+Attention extractor (real temporal memory).

The GRU hidden state is carried across the rollout (so the policy genuinely uses
temporal context) and is re-computed over stored sequences during the PPO update
(sequence-minibatch recurrent PPO). Hidden state is reset at episode boundaries.
"""
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from src.recurrent_and_pooling_modules import FeatureExtractorLSTMAttention


class RecurrentActorCritic(nn.Module):
    latent_dim: int = 256
    action_dim: int = 17

    @nn.compact
    def __call__(self, x, hidden):
        feat, new_hidden = FeatureExtractorLSTMAttention(latent_dim=self.latent_dim)(x, hidden)
        logits = nn.Dense(self.action_dim)(feat)
        value = jnp.squeeze(nn.Dense(1)(feat), axis=-1)
        return logits, value, new_hidden


class RecurrentPPOTrainer:
    def __init__(self, env_manager, num_envs=64, num_steps=64, latent_dim=256, action_dim=17,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2, ent_coef=0.01,
                 vf_coef=0.5, update_epochs=4, num_minibatches=4):
        self.env_manager = env_manager
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.model = RecurrentActorCritic(latent_dim=latent_dim, action_dim=action_dim)
        self.opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr, eps=1e-5))

    def create_state(self, rng, input_shape):
        init_x = jnp.zeros((1, *input_shape))
        init_h = jnp.zeros((1, self.latent_dim))
        params = self.model.init(rng, init_x, init_h)['params']
        return params, self.opt.init(params)

    def make_train_step(self):
        def train_step(carry, _):
            params, opt_state, env_state, obs, hidden, rng = carry

            def _env_step(sc, _):
                params, e_state, obs, hidden, rng = sc
                rng, a_rng, s_rng = jax.random.split(rng, 3)
                logits, value, new_hidden = self.model.apply({'params': params}, obs, hidden)
                action = jax.random.categorical(a_rng, logits)
                logp = jax.nn.log_softmax(logits)[jnp.arange(self.num_envs), action]
                next_obs, next_e, reward, done, info, rng2 = self.env_manager.step(s_rng, e_state, action)
                new_hidden = jnp.where(done[:, None], jnp.zeros_like(new_hidden), new_hidden)
                trans = (obs, action, logp, value, reward, done.astype(jnp.float32))
                return (params, next_e, next_obs, new_hidden, rng2), trans

            (params, env_state, obs, hidden, rng), traj = jax.lax.scan(
                _env_step, (params, env_state, obs, hidden, rng), None, length=self.num_steps)
            t_obs, t_act, t_logp, t_val, t_rew, t_done = traj

            _, last_val, _ = self.model.apply({'params': params}, obs, hidden)

            def _gae(carry, x):
                gae, next_value = carry
                d, v, r = x
                delta = r + self.gamma * next_value * (1.0 - d) - v
                gae = delta + self.gamma * self.gae_lambda * (1.0 - d) * gae
                return (gae, v), gae
            _, adv = jax.lax.scan(_gae, (jnp.zeros_like(last_val), last_val),
                                  (t_done, t_val, t_rew), reverse=True)
            returns = adv + t_val   # (T,E)

            # done shifted by one step (to reset hidden at the start of a new episode)
            done_prev = jnp.concatenate([jnp.zeros((1, self.num_envs)), t_done[:-1]], axis=0)

            E = self.num_envs
            mb_env = E // self.num_minibatches

            def _seq_forward(params, obs_seq, done_prev_seq):
                """Re-run the GRU over a (T, mb, ...) sequence, resetting hidden at episode starts."""
                def _step(h, x):
                    o, dp = x
                    h = jnp.where(dp[:, None], jnp.zeros_like(h), h)
                    logits, value, new_h = self.model.apply({'params': params}, o, h)
                    return new_h, (logits, value)
                init_h = jnp.zeros((obs_seq.shape[1], self.latent_dim))
                _, (logits_seq, value_seq) = jax.lax.scan(_step, init_h, (obs_seq, done_prev_seq))
                return logits_seq, value_seq

            def _update_epoch(state, _):
                params, opt_state, rng = state

                def _minibatch(carry, env_idx):
                    params, opt_state = carry
                    mb_obs = t_obs[:, env_idx]
                    mb_dp = done_prev[:, env_idx]
                    mb_act = t_act[:, env_idx]
                    mb_logp = t_logp[:, env_idx]
                    mb_adv = adv[:, env_idx]
                    mb_ret = returns[:, env_idx]

                    def _loss(params):
                        logits_seq, value_seq = _seq_forward(params, mb_obs, mb_dp)
                        lp_all = jax.nn.log_softmax(logits_seq, axis=-1)
                        lp = jnp.take_along_axis(lp_all, mb_act[..., None], axis=-1).squeeze(-1)
                        ent = -jnp.sum(jax.nn.softmax(logits_seq, axis=-1) * lp_all, axis=-1).mean()
                        g = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                        ratio = jnp.exp(lp - mb_logp)
                        s1 = ratio * g
                        s2 = jnp.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * g
                        pi = -jnp.minimum(s1, s2).mean()
                        vf = 0.5 * jnp.square(value_seq - mb_ret).mean()
                        return pi + self.vf_coef * vf - self.ent_coef * ent

                    loss, grads = jax.value_and_grad(_loss)(params)
                    updates, new_opt = self.opt.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    return (params, new_opt), loss

                rng, perm_rng = jax.random.split(rng)
                perm = jax.random.permutation(perm_rng, E)
                idxs = perm.reshape((self.num_minibatches, mb_env))
                (params, opt_state), loss = jax.lax.scan(_minibatch, (params, opt_state), idxs)
                return (params, opt_state, rng), loss.mean()

            (params, opt_state, rng), loss = jax.lax.scan(
                _update_epoch, (params, opt_state, rng), None, length=self.update_epochs)

            metrics = {"loss": loss.mean(), "mean_reward": t_rew.sum(axis=0).mean()}
            return (params, opt_state, env_state, obs, hidden, rng), metrics

        return jax.jit(train_step)

    def make_eval_policy(self, deterministic=True):
        """Stateful eval policy: returns select(params, obs, rng, hidden)->(action, new_hidden)."""
        def select(params, obs, rng, hidden):
            logits, _, new_hidden = self.model.apply({'params': params}, obs, hidden)
            action = jnp.argmax(logits, axis=-1) if deterministic else jax.random.categorical(rng, logits)
            return action, new_hidden
        return select
