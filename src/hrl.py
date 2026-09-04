import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from typing import NamedTuple, Any

from src.env import CraftaxLevelManager
from src.networks import HierarchicalLearnedActorCritic, SymbolicActorCritic


# Fixed macro-actions mapping for Craftax (17 actions total):
# 0: NOOP, 1: LEFT, 2: RIGHT, 3: UP, 4: DOWN, 5: DO (mine/attack), 6: SLEEP, ...
FIXED_SKILL_TO_ACTION = jnp.array([
    0,  # Skill 0: Wait / NOOP
    1,  # Skill 1: Move Left
    2,  # Skill 2: Move Right
    3,  # Skill 3: Move Up
    4,  # Skill 4: Move Down
    5,  # Skill 5: Mine / Attack (DO action)
], dtype=jnp.int32)


class MacroTransition(NamedTuple):
    """One temporally-abstracted decision (a macro-action spanning skip_k frames)."""
    obs: jnp.ndarray
    action: jnp.ndarray       # macro-action index (primitive, skill, or latent z)
    sub_act: jnp.ndarray      # (num_envs, skip_k) primitive actions executed inside the option
    log_prob: jnp.ndarray     # total log-prob of the macro-action (incl. low-level for hrl_learned)
    value: jnp.ndarray
    reward: jnp.ndarray       # summed environment reward over the skip window
    done: jnp.ndarray         # True if the episode terminated within the window


class HRLTrainer:
    """
    Real-training implementation of the 4 HRL / temporal-abstraction paradigms:
      1. 'flat'        : standard PPO over the 17 primitive actions (skip=1).
      2. 'skip4'       : PPO with action-repeat 4 (temporal abstraction, no hierarchy).
      3. 'hrl'         : PPO meta-controller over 6 fixed macro-skills, each repeated 4 frames.
      4. 'hrl_learned' : jointly-trained 2-level hierarchy. Meta selects latent skill z every
                         4 frames; low-level policy p(a|obs,z) executes primitive actions.

    Unlike the previous version (which only performed forward rollouts and NEVER
    updated weights), this trainer runs a full PPO update (GAE + clipped surrogate
    + minibatch gradient descent via optax) on every `train_step`.
    """
    def __init__(
        self,
        mode: str,  # 'flat', 'skip4', 'hrl', 'hrl_learned'
        env_manager: CraftaxLevelManager,
        num_envs: int = 64,
        num_steps: int = 64,      # number of MACRO-steps collected per update
        skip_k: int = 4,
        update_epochs: int = 4,
        num_minibatches: int = 4,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
    ):
        assert mode in ['flat', 'skip4', 'hrl', 'hrl_learned'], f"Unknown mode: {mode}"
        self.mode = mode
        self.env_manager = env_manager
        self.num_envs = num_envs
        self.num_steps = num_steps
        # 'flat' decides every frame; the others use temporal abstraction over skip_k frames.
        self.skip_k = 1 if mode == 'flat' else skip_k
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        if self.mode == 'hrl_learned':
            self.model = HierarchicalLearnedActorCritic(action_dim=self.env_manager.num_actions, num_skills=6)
            self.num_macro_actions = 6
        elif self.mode == 'hrl':
            self.model = SymbolicActorCritic(action_dim=6)
            self.num_macro_actions = 6
        else:
            self.model = SymbolicActorCritic(action_dim=self.env_manager.num_actions)
            self.num_macro_actions = self.env_manager.num_actions

    def create_state(self, rng, input_shape):
        init_x = jnp.zeros((1, *input_shape))
        if self.mode == 'hrl_learned':
            variables = self.model.init(rng, init_x, jnp.zeros((1,), dtype=jnp.int32))
        else:
            variables = self.model.init(rng, init_x)
        tx = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(self.learning_rate, eps=1e-5)
        )
        return TrainState.create(apply_fn=self.model.apply, params=variables['params'], tx=tx)

    # ------------------------------------------------------------------
    # Policy forward helpers
    # ------------------------------------------------------------------
    def _act(self, params, obs, rng):
        """Sample one macro-action and return (action_index, log_prob, value, extra)."""
        if self.mode == 'hrl_learned':
            meta_logits, value, _ = self.model.apply({'params': params}, obs)
            z = jax.random.categorical(rng, meta_logits)
            logp_meta = jax.nn.log_softmax(meta_logits)[jnp.arange(self.num_envs), z]
            return z, logp_meta, value, meta_logits
        logits, value = self.model.apply({'params': params}, obs)
        action = jax.random.categorical(rng, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(self.num_envs), action]
        return action, logp, value, logits

    def _macro_step(self, params, obs, env_state, macro_action, rng):
        """Execute one macro-action over skip_k frames.

        Returns (next_obs, env_state, reward_sum, done_any, low_logp_sum, sub_act, rng)
        where sub_act has shape (num_envs, skip_k).

        For 'hrl_learned' the low-level policy p(a | obs_decision, z) is open-loop within
        the option window (its logits are computed once from the decision observation).
        This keeps the option's total log-prob exactly recomputable in the PPO update,
        which is required for a correct importance-sampling ratio.
        """
        if self.mode == 'hrl':
            per_frame_primitive = FIXED_SKILL_TO_ACTION[macro_action]
            low_logits = None
        elif self.mode == 'hrl_learned':
            _, _, low_logits = self.model.apply({'params': params}, obs, macro_action)
            per_frame_primitive = None
        else:  # flat / skip4
            per_frame_primitive = macro_action
            low_logits = None

        def _substep(carry, _):
            obs_c, env_state, rew_sum, done_any, low_logp, rng = carry
            rng, s_rng = jax.random.split(rng)
            if self.mode == 'hrl_learned':
                act = jax.random.categorical(s_rng, low_logits)
                low_logp = low_logp + jax.nn.log_softmax(low_logits)[jnp.arange(self.num_envs), act]
            else:
                act = per_frame_primitive
            n_obs, n_env, r, d, _, _ = self.env_manager.step(s_rng, env_state, act)
            return (n_obs, n_env, rew_sum + r, jnp.logical_or(done_any, d), low_logp, rng), act

        init = (obs, env_state, jnp.zeros(self.num_envs), jnp.zeros(self.num_envs, dtype=bool),
                jnp.zeros(self.num_envs), rng)
        (obs, env_state, rew_sum, done_any, low_logp, rng), sub_act = jax.lax.scan(
            _substep, init, None, length=self.skip_k
        )
        # sub_act: (skip_k, num_envs) -> (num_envs, skip_k)
        sub_act = jnp.transpose(sub_act.astype(jnp.int32), (1, 0))
        return obs, env_state, rew_sum, done_any, low_logp, sub_act, rng

    # ------------------------------------------------------------------
    # REAL PPO TRAINING STEP (gradient descent via optax)
    # ------------------------------------------------------------------
    def train_step(self, runner_state):
        train_state, env_state, last_obs, rng = runner_state

        # 1. COLLECT MACRO-ROLLOUT
        def _env_step(carry, _):
            train_state, env_state, obs, rng = carry
            rng, a_rng = jax.random.split(rng)
            macro_action, logp_head, value, _ = self._act(train_state.params, obs, a_rng)
            rng, m_rng = jax.random.split(rng)
            n_obs, n_env, rew_sum, done_any, low_logp, sub_act, rng = self._macro_step(
                train_state.params, obs, env_state, macro_action, m_rng
            )
            total_logp = logp_head + low_logp
            trans = MacroTransition(obs, macro_action, sub_act, total_logp, value, rew_sum,
                                    done_any.astype(jnp.float32))
            return (train_state, n_env, n_obs, rng), trans

        (train_state, env_state, last_obs, rng), traj = jax.lax.scan(
            _env_step, (train_state, env_state, last_obs, rng), None, length=self.num_steps
        )

        # 2. GAE
        _, last_value = self._value_only(train_state.params, last_obs)

        def _gae(carry, transition):
            gae, next_value = carry
            delta = transition.reward + self.gamma * next_value * (1.0 - transition.done) - transition.value
            gae = delta + self.gamma * self.gae_lambda * (1.0 - transition.done) * gae
            return (gae, transition.value), gae

        _, advantages = jax.lax.scan(_gae, (jnp.zeros_like(last_value), last_value), traj, reverse=True)
        returns = advantages + traj.value

        # 3. PPO UPDATE
        def _update_epoch(update_state, _):
            def _update_minibatch(t_state, batch_info):
                tb, gae, ret = batch_info

                def _loss_fn(params):
                    # Recompute log-probs and value for the stored macro-actions.
                    lp, val, ent = self._recompute(params, tb)
                    ratio = jnp.exp(lp - tb.log_prob)
                    norm_gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    surr1 = ratio * norm_gae
                    surr2 = jnp.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * norm_gae
                    pi_loss = -jnp.minimum(surr1, surr2).mean()
                    vf_loss = 0.5 * jnp.square(val - ret).mean()
                    return pi_loss + self.vf_coef * vf_loss - self.ent_coef * ent, (pi_loss, vf_loss, ent)

                (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(t_state.params)
                t_state = t_state.apply_gradients(grads=grads)
                return t_state, loss

            t_state, r = update_state
            r, subkey = jax.random.split(r)
            batch_size = self.num_steps * self.num_envs
            mb_size = batch_size // self.num_minibatches
            flat_traj = jax.tree_util.tree_map(lambda x: x.reshape((batch_size, *x.shape[2:])), traj)
            flat_gae = advantages.reshape((batch_size,))
            flat_ret = returns.reshape((batch_size,))
            perm = jax.random.permutation(subkey, batch_size)
            batch = (flat_traj, flat_gae, flat_ret)
            shuffled = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [self.num_minibatches, mb_size] + list(x.shape[1:])), shuffled
            )
            t_state, loss = jax.lax.scan(_update_minibatch, t_state, minibatches)
            return (t_state, r), loss.mean()

        (train_state, rng), loss = jax.lax.scan(
            _update_epoch, (train_state, rng), None, length=self.update_epochs
        )
        metrics = {"loss": loss.mean(), "mean_reward": traj.reward.sum(axis=0).mean()}
        return (train_state, env_state, last_obs, rng), metrics

    # ------------------------------------------------------------------
    def _value_only(self, params, obs):
        if self.mode == 'hrl_learned':
            _, value, _ = self.model.apply({'params': params}, obs)
        else:
            _, value = self.model.apply({'params': params}, obs)
        return None, value

    def _recompute(self, params, tb):
        """Recompute (log_prob, value, entropy) for stored macro-actions tb."""
        if self.mode == 'hrl_learned':
            meta_logits, value, low_logits = self.model.apply({'params': params}, tb.obs, tb.action)
            logp_meta = jax.nn.log_softmax(meta_logits)[jnp.arange(len(tb.action)), tb.action]
            # tb.sub_act: (mb, skip_k); low_logits: (mb, A). Sum low-level log-probs.
            low_logsoftmax = jax.nn.log_softmax(low_logits)
            idx = jnp.arange(low_logsoftmax.shape[0])[:, None]
            low_lp = low_logsoftmax[idx, tb.sub_act].sum(axis=1)
            lp = logp_meta + low_lp
            p_meta = jax.nn.softmax(meta_logits)
            ent_meta = -jnp.sum(p_meta * jax.nn.log_softmax(meta_logits), axis=-1).mean()
            p_low = jax.nn.softmax(low_logits)
            ent_low = -jnp.sum(p_low * jax.nn.log_softmax(low_logits), axis=-1).mean()
            return lp, value, ent_meta + ent_low
        logits, value = self.model.apply({'params': params}, tb.obs)
        logp = jax.nn.log_softmax(logits)[jnp.arange(len(tb.action)), tb.action]
        probs = jax.nn.softmax(logits)
        ent = -jnp.sum(probs * jax.nn.log_softmax(logits), axis=-1).mean()
        return logp, value, ent

    # ------------------------------------------------------------------
    # Greedy action selection for evaluation (used by src.eval_utils)
    # ------------------------------------------------------------------
    def make_eval_policy(self, deterministic: bool = True):
        """Return select_action(params, obs, rng) -> primitive action, for the evaluator."""
        if self.mode == 'hrl_learned':
            def select(params, obs, rng):
                meta_logits, _, low_logits = self.model.apply({'params': params}, obs)
                z = jnp.argmax(meta_logits, axis=-1) if deterministic else jax.random.categorical(rng, meta_logits)
                _, _, low_logits = self.model.apply({'params': params}, obs, z)
                return jnp.argmax(low_logits, axis=-1) if deterministic else jax.random.categorical(rng, low_logits)
            return select
        if self.mode == 'hrl':
            def select(params, obs, rng):
                logits, _ = self.model.apply({'params': params}, obs)
                skill = jnp.argmax(logits, axis=-1) if deterministic else jax.random.categorical(rng, logits)
                return FIXED_SKILL_TO_ACTION[skill]
            return select

        def select(params, obs, rng):
            logits, _ = self.model.apply({'params': params}, obs)
            return jnp.argmax(logits, axis=-1) if deterministic else jax.random.categorical(rng, logits)
        return select
