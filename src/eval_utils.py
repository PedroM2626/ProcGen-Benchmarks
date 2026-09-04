"""
Real evaluation utilities.

Every benchmark in this repository must report GENUINE metrics measured from a
trained policy rolled out in the environment. This module provides a single,
shared, JIT-compiled episodic evaluator so that no experiment resorts to
single-step reward snapshots or hard-coded numbers.

Protocol (Craftax):
  * Reset a batch of environments on a fixed pool of levels (train or unseen).
  * Roll out the policy for `horizon` steps WITHOUT auto-reset, accumulating
    reward only while the episode is still alive.
  * Report the mean/std of the per-environment episodic return.

This yields the honest "mean episodic return capped at `horizon`" metric that is
standard for procedural-RL generalization studies.
"""
import jax
import jax.numpy as jnp


def make_craftax_evaluator(env_manager, select_action, num_envs: int = 128, horizon: int = 1000):
    """
    Build a vectorized episodic evaluator for a CraftaxLevelManager.

    Args:
        env_manager: CraftaxLevelManager instance (raw `.env` and `.params` used).
        select_action: callable (params, obs, rng) -> action array (num_envs,).
        num_envs: number of parallel evaluation environments (== number of episodes).
        horizon: maximum number of steps per episode.

    Returns:
        eval_fn(params, rng, unseen: bool) -> (mean_return, std_return)
    """
    env = env_manager.env
    params_env = env_manager.params

    def _rollout(params, level_ids):
        keys = jax.vmap(jax.random.PRNGKey)(level_ids)
        obs, states = jax.vmap(env.reset, in_axes=(0, None))(keys, params_env)
        alive = jnp.ones(num_envs, dtype=bool)
        ep_ret = jnp.zeros(num_envs, dtype=jnp.float32)
        rng = jax.random.PRNGKey(0)

        def body(carry, _):
            obs, states, alive, ep_ret, rng = carry
            rng, a_rng, s_rng = jax.random.split(rng, 3)
            action = select_action(params, obs, a_rng)
            step_keys = jax.random.split(s_rng, num_envs)
            next_obs, next_states, rew, done, _info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(step_keys, states, action, params_env)
            # Accumulate reward only while the episode is alive.
            ep_ret = ep_ret + jnp.where(alive, rew, 0.0)
            alive = alive & (~done)
            return (next_obs, next_states, alive, ep_ret, rng), None

        (_, _, _, ep_ret, _) = jax.lax.scan(
            body, (obs, states, alive, ep_ret, rng), None, length=horizon
        )[0]
        return ep_ret

    @jax.jit
    def _eval_train(params, rng):
        sub = jax.random.randint(rng, (num_envs,), 0, env_manager.num_train_levels)
        return _rollout(params, sub)

    @jax.jit
    def _eval_unseen(params, rng):
        lo = env_manager.eval_seed_offset
        sub = jax.random.randint(rng, (num_envs,), lo, lo + 100)
        return _rollout(params, sub)

    def eval_fn(params, rng, unseen: bool):
        rng, sub = jax.random.split(rng)
        rets = _eval_unseen(params, sub) if unseen else _eval_train(params, sub)
        return float(jnp.mean(rets)), float(jnp.std(rets))

    return eval_fn


def make_craftax_recurrent_evaluator(env_manager, select_action, latent_dim: int,
                                     num_envs: int = 128, horizon: int = 1000):
    """Episodic evaluator for recurrent policies that carry a hidden state.

    select_action(params, obs, rng, hidden) -> (action, new_hidden).
    Hidden state is reset at episode boundaries (done).
    """
    env = env_manager.env
    params_env = env_manager.params

    def _rollout(params, level_ids):
        keys = jax.vmap(jax.random.PRNGKey)(level_ids)
        obs, states = jax.vmap(env.reset, in_axes=(0, None))(keys, params_env)
        alive = jnp.ones(num_envs, dtype=bool)
        ep_ret = jnp.zeros(num_envs, dtype=jnp.float32)
        hidden = jnp.zeros((num_envs, latent_dim))
        rng = jax.random.PRNGKey(0)

        def body(carry, _):
            obs, states, alive, ep_ret, hidden, rng = carry
            rng, a_rng, s_rng = jax.random.split(rng, 3)
            action, new_hidden = select_action(params, obs, a_rng, hidden)
            step_keys = jax.random.split(s_rng, num_envs)
            next_obs, next_states, rew, done, _info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(step_keys, states, action, params_env)
            ep_ret = ep_ret + jnp.where(alive, rew, 0.0)
            alive = alive & (~done)
            new_hidden = jnp.where(done[:, None], jnp.zeros_like(new_hidden), new_hidden)
            return (next_obs, next_states, alive, ep_ret, new_hidden, rng), None

        (_, _, _, ep_ret, _, _) = jax.lax.scan(
            body, (obs, states, alive, ep_ret, hidden, rng), None, length=horizon
        )[0]
        return ep_ret

    @jax.jit
    def _eval_train(params, rng):
        sub = jax.random.randint(rng, (num_envs,), 0, env_manager.num_train_levels)
        return _rollout(params, sub)

    @jax.jit
    def _eval_unseen(params, rng):
        lo = env_manager.eval_seed_offset
        sub = jax.random.randint(rng, (num_envs,), lo, lo + 100)
        return _rollout(params, sub)

    def eval_fn(params, rng, unseen: bool):
        rng, sub = jax.random.split(rng)
        rets = _eval_unseen(params, sub) if unseen else _eval_train(params, sub)
        return float(jnp.mean(rets)), float(jnp.std(rets))

    return eval_fn


def make_continuous_evaluator(env, select_action, num_envs: int = 256, horizon: int = None):
    """Episodic evaluator for single-agent continuous/discrete envs with the interface
    env.reset(rng)->(obs,state); env.step(rng,state,action)->(obs,state,reward,done).

    select_action(obs, rng) -> action. Episodes auto-reset on done; reports the mean
    COMPLETED-episode return and the number of completed episodes.
    """
    horizon = horizon or (env.max_steps * 4)
    reset_vmap = jax.jit(jax.vmap(env.reset))
    step_vmap = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))

    def _run(rng):
        keys = jax.random.split(rng, num_envs)
        obs, state = reset_vmap(keys)

        def body(carry, _):
            obs, state, rng, ep_ret, ret_sum, ret_count = carry
            rng, a_rng, s_rng, reset_rng = jax.random.split(rng, 4)
            action = select_action(obs, a_rng)
            step_keys = jax.random.split(s_rng, num_envs)
            n_obs, n_state, rew, done = step_vmap(step_keys, state, action)
            ep_ret = ep_ret + rew
            ret_sum = ret_sum + jnp.where(done, ep_ret, 0.0)
            ret_count = ret_count + done.astype(jnp.float32)
            ep_ret = jnp.where(done, 0.0, ep_ret)
            reset_keys = jax.random.split(reset_rng, num_envs)
            r_obs, r_state = reset_vmap(reset_keys)
            d = done.reshape((-1,) + (1,) * (n_obs.ndim - 1))
            n_obs = jnp.where(d, r_obs, n_obs)
            n_state = jax.tree_util.tree_map(
                lambda a, b: jnp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), a, b), r_state, n_state)
            return (n_obs, n_state, rng, ep_ret, ret_sum, ret_count), None

        init = (obs, state, rng, jnp.zeros(num_envs), jnp.zeros(num_envs), jnp.zeros(num_envs))
        (_, _, _, _, ret_sum, ret_count) = jax.lax.scan(body, init, None, length=horizon)[0]
        mean_ret = ret_sum.sum() / jnp.maximum(ret_count.sum(), 1.0)
        return mean_ret, ret_count.sum()

    run = jax.jit(_run)

    def eval_fn(rng):
        mean_ret, n_eps = run(rng)
        return float(mean_ret), int(n_eps)

    return eval_fn


def make_fixed_horizon_evaluator(env, select_action, num_envs: int = 64, horizon: int = 1000):
    """Fixed-horizon return evaluator (robust for Brax, where `done` may only flag
    termination and not truncation). Rolls `horizon` steps with auto-reset on done and
    returns the mean per-env SUM of rewards over the horizon (== episodic return when
    horizon equals the episode length)."""
    reset_vmap = jax.jit(jax.vmap(env.reset))
    step_vmap = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0)))

    def _run(rng):
        keys = jax.random.split(rng, num_envs)
        obs, state = reset_vmap(keys)

        def body(carry, _):
            obs, state, rng, tot = carry
            rng, a_rng, s_rng, reset_rng = jax.random.split(rng, 4)
            action = select_action(obs, a_rng)
            step_keys = jax.random.split(s_rng, num_envs)
            n_obs, n_state, rew, done = step_vmap(step_keys, state, action)
            tot = tot + rew
            reset_keys = jax.random.split(reset_rng, num_envs)
            r_obs, r_state = reset_vmap(reset_keys)
            d = done.reshape((-1,) + (1,) * (n_obs.ndim - 1))
            n_obs = jnp.where(d, r_obs, n_obs)
            n_state = jax.tree_util.tree_map(
                lambda a, b: jnp.where(done.reshape((-1,) + (1,) * (a.ndim - 1)), a, b), r_state, n_state)
            return (n_obs, n_state, rng, tot), None

        init = (obs, state, rng, jnp.zeros(num_envs))
        (_, _, _, tot) = jax.lax.scan(body, init, None, length=horizon)[0]
        return tot

    run = jax.jit(_run)

    def eval_fn(rng):
        tot = run(rng)
        return float(jnp.mean(tot)), float(jnp.std(tot))

    return eval_fn
