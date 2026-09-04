"""
REAL trainers for the 4 MARL paradigms benchmark, with genuine Fog-of-War POMDP.

Paradigms:
  1. CTDE Policy-Based   -> reuse MARLPPOTrainer('MAPPO')
  2. Value Decomposition -> reuse MARLQTrainer('QMIX')
  3. Centralized Joint (CTE) -> CentralizedJointActorCritic (one policy over global state)
  4. Explicit Communication (TarMAC/GAT) -> CommActorCritic (messages at execution time)

FogOfWarEnv wraps MultiAgentParticleEnv and masks landmark coordinates that are
outside each agent's local vision radius, turning the task into a real POMDP.
"""
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from typing import Tuple

from src.marl_env import MultiAgentParticleEnv
from src.marl_comm_modules import CommActorCritic


# =====================================================================
# FOG-OF-WAR POMDP WRAPPER
# =====================================================================
class FogOfWarEnv:
    """Masks landmark info beyond `radius` from each agent (partial observability)."""
    def __init__(self, base_env: MultiAgentParticleEnv, fog: bool = True, radius: float = 0.40):
        self.base = base_env
        self.fog = fog
        self.radius = radius
        self.num_agents = base_env.num_agents
        self.num_landmarks = base_env.num_landmarks
        self.num_actions = base_env.num_actions
        self.obs_dim = base_env.obs_dim
        self.global_state_dim = base_env.global_state_dim
        self.max_steps = base_env.max_steps

    def _mask(self, obs, state):
        if not self.fog:
            return obs, obs.flatten()
        N, L = self.num_agents, self.num_landmarks
        diff = state.landmark_pos[None, :, :] - state.agent_pos[:, None, :]  # (N,L,2)
        dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-6)                  # (N,L)
        visible = (dist < self.radius).astype(jnp.float32)                   # (N,L)
        lm = obs[:, 2:2 + 2 * L].reshape(N, L, 2) * visible[..., None]
        obs = obs.at[:, 2:2 + 2 * L].set(lm.reshape(N, 2 * L))
        return obs, obs.flatten()

    def reset(self, rng):
        obs, gstate, state = self.base.reset(rng)
        obs, gstate = self._mask(obs, state)
        return obs, gstate, state

    def step(self, rng, state, actions):
        obs, gstate, n_state, rew, done = self.base.step(rng, state, actions)
        obs, gstate = self._mask(obs, n_state)
        return obs, gstate, n_state, rew, done


# =====================================================================
# CENTRALIZED JOINT ACTOR-CRITIC (CTE)
# =====================================================================
class CentralizedJointActorCritic(nn.Module):
    """Single centralized controller: maps the joint global state to all agents' actions."""
    num_agents: int
    action_dim: int

    @nn.compact
    def __call__(self, gstate):
        h = nn.relu(nn.Dense(256)(gstate))
        h = nn.relu(nn.Dense(256)(h))
        logits = nn.Dense(self.num_agents * self.action_dim)(h).reshape(
            (gstate.shape[0], self.num_agents, self.action_dim))
        value = jnp.squeeze(nn.Dense(1)(h), axis=-1)  # team value (B,)
        return logits, value


# =====================================================================
# JOINT PPO TRAINER for CTE and COMM paradigms
# =====================================================================
class JointPPOTrainer:
    def __init__(self, actor_type, env, num_envs=64, num_steps=64, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_eps=0.2, ent_coef=0.01, vf_coef=0.5,
                 update_epochs=4, num_minibatches=4):
        assert actor_type in ("CTE", "COMM")
        self.actor_type = actor_type
        self.env = env
        self.N = env.num_agents
        self.A = env.num_actions
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.lr = lr
        self.opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(self.lr, eps=1e-5))

        if actor_type == "CTE":
            self.net = CentralizedJointActorCritic(num_agents=self.N, action_dim=self.A)
        else:
            self.net = CommActorCritic(action_dim=self.A)

        self.reset_vmap = jax.jit(jax.vmap(env.reset))
        self.step_vmap = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))

    def _forward(self, params, obs, gstate):
        """Return (logits (...,N,A), value_per_agent (...,N))."""
        if self.actor_type == "CTE":
            logits, value = self.net.apply({'params': params['net']}, gstate)
            return logits, jnp.repeat(value[..., None], self.N, axis=-1)
        logits, values = self.net.apply({'params': params['net']}, obs)
        return logits, values

    def create_state(self, rng):
        rng, r1 = jax.random.split(rng)
        dummy_obs = jnp.zeros((self.num_envs, self.N, self.env.obs_dim))
        dummy_gstate = jnp.zeros((self.num_envs, self.env.global_state_dim))
        if self.actor_type == "CTE":
            net_params = self.net.init(r1, dummy_gstate)['params']
        else:
            net_params = self.net.init(r1, dummy_obs)['params']
        params = {'net': net_params}
        return params, self.opt.init(params)

    def _act(self, params, obs, gstate, rng):
        logits, _ = self._forward(params, obs, gstate)
        action = jax.random.categorical(rng, logits, axis=-1)
        logp_all = jax.nn.log_softmax(logits, axis=-1)
        logp = jnp.take_along_axis(logp_all, action[..., None], axis=-1).squeeze(-1)
        return action, logp

    def make_train_step(self):
        def train_step(carry, _):
            params, opt_state, env_state, obs, gstate, rng = carry

            def _env_step(rollout_carry, _):
                params, env_state, obs, gstate, rng = rollout_carry
                rng, act_rng, step_rng, reset_rng = jax.random.split(rng, 4)
                action, logp = self._act(params, obs, gstate, act_rng)
                _, value = self._forward(params, obs, gstate)  # (E,N)
                step_keys = jax.random.split(step_rng, self.num_envs)
                n_obs, n_gstate, n_state, reward, done = self.step_vmap(step_keys, env_state, action)
                any_done = jnp.any(done)
                reset_keys = jax.random.split(reset_rng, self.num_envs)
                r_obs, r_gstate, r_state = self.reset_vmap(reset_keys)
                n_obs = jax.tree_util.tree_map(lambda a, b: jnp.where(any_done, a, b), r_obs, n_obs)
                n_gstate = jnp.where(any_done, r_gstate, n_gstate)
                n_state = jax.tree_util.tree_map(lambda a, b: jnp.where(any_done, a, b), r_state, n_state)
                rew_ag = jnp.repeat(reward[:, None], self.N, axis=-1)
                done_ag = jnp.repeat(done[:, None], self.N, axis=-1).astype(jnp.float32)
                trans = (obs, gstate, action, logp, value, rew_ag, done_ag)
                return (params, n_state, n_obs, n_gstate, rng), trans

            (_, env_state, obs, gstate, rng), traj = jax.lax.scan(
                _env_step, (params, env_state, obs, gstate, rng), None, length=self.num_steps)
            t_obs, t_gstate, t_act, t_logp, t_val, t_rew, t_done = traj
            _, last_value = self._forward(params, obs, gstate)

            def _gae(carry, x):
                gae, next_value = carry
                val, rew, done = x
                delta = rew + self.gamma * next_value * (1.0 - done) - val
                gae = delta + self.gamma * self.gae_lambda * (1.0 - done) * gae
                return (gae, val), gae
            _, adv = jax.lax.scan(_gae, (jnp.zeros_like(last_value), last_value),
                                  (t_val, t_rew, t_done), reverse=True)
            returns = adv + t_val

            TE = self.num_steps * self.num_envs
            returns_env = returns.reshape((TE, self.N))
            adv_env = adv.reshape((TE, self.N))
            act_env = t_act.reshape((TE, self.N))
            logp_env = t_logp.reshape((TE, self.N))
            team_ret_env = returns_env.mean(axis=-1)
            f_obs = t_obs.reshape((TE, self.N, self.env.obs_dim))
            f_gstate = t_gstate.reshape((TE, self.env.global_state_dim))
            mb_env = TE // self.num_minibatches

            def _update_epoch(state, _):
                params, opt_state, rng = state

                def _loss_fn(params, idx_env):
                    ob_all = f_obs[idx_env]      # (mb_env,N,d)
                    gs = f_gstate[idx_env]       # (mb_env,Nd)
                    logits, value = self._forward(params, ob_all, gs)  # (mb_env,N,A), (mb_env,N)
                    ac = act_env[idx_env]        # (mb_env,N)
                    logp_all = jax.nn.log_softmax(logits, axis=-1)
                    logp = jnp.take_along_axis(logp_all, ac[..., None], axis=-1).squeeze(-1)
                    probs = jax.nn.softmax(logits, axis=-1)
                    ent = -jnp.sum(probs * logp_all, axis=-1).mean()
                    g = adv_env[idx_env].reshape(-1)
                    g = (g - g.mean()) / (g.std() + 1e-8)
                    g = g.reshape(ac.shape)
                    ratio = jnp.exp(logp - logp_env[idx_env])
                    s1 = ratio * g
                    s2 = jnp.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * g
                    pi_loss = -jnp.minimum(s1, s2).mean()
                    if self.actor_type == "CTE":
                        team_v = value[:, 0]
                        vf_loss = 0.5 * jnp.square(team_v - team_ret_env[idx_env]).mean()
                    else:
                        vf_loss = 0.5 * jnp.square(value - returns_env[idx_env]).mean()
                    return pi_loss + self.vf_coef * vf_loss - self.ent_coef * ent

                def _minibatch(carry, _):
                    params, opt_state, rng = carry
                    rng, k1 = jax.random.split(rng)
                    idx_env = jax.random.randint(k1, (mb_env,), 0, TE)
                    loss, grads = jax.value_and_grad(_loss_fn)(params, idx_env)
                    updates, opt_state_new = self.opt.update(grads, opt_state, params)
                    params_new = optax.apply_updates(params, updates)
                    return (params_new, opt_state_new, rng), loss

                (params, opt_state, rng), loss = jax.lax.scan(
                    _minibatch, (params, opt_state, rng), None,
                    length=self.num_minibatches * self.update_epochs)
                return (params, opt_state, rng), loss.mean()

            (params, opt_state, rng), loss = jax.lax.scan(
                _update_epoch, (params, opt_state, rng), None, length=1)
            metrics = {"loss": loss.mean(), "team_reward": t_rew[:, :, 0].sum(axis=0).mean()}
            return (params, opt_state, env_state, obs, gstate, rng), metrics

        return jax.jit(train_step)

    def make_selector(self, params):
        def sel(obs, gstate, rng):
            logits, _ = self._forward(params, obs, gstate)
            return jnp.argmax(logits, axis=-1)
        return sel
