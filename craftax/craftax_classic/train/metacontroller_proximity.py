from functools import partial
from random import randrange
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen import initializers

from craftax.craftax_classic.constants import OBS_DIM
from craftax.craftax_classic.envs.craftax_state import EnvParams, StaticEnvParams
from craftax.craftax_classic.envs.craftax_symbolic_env import (
    CraftaxClassicSymbolicEnvShareStats,
    CraftaxClassicSymbolicEnvShareStatsNoAutoReset,
)
from craftax.craftax_classic.game_logic import are_players_alive
from craftax.craftax_classic.train.logger import TrainLogger
from craftax.craftax_classic.train.nets import LSTM, ZNet


class AuxLossNet(nn.Module):
    output_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_size)(x)
        return x


class CraftaxAgent(nn.Module):
    action_space: int

    def setup(self):
        self.network = nn.Sequential(
            [
                nn.Dense(64, kernel_init=initializers.orthogonal(jnp.sqrt(2))),
                nn.relu,
                nn.Dense(32, kernel_init=initializers.orthogonal(jnp.sqrt(2))),
                nn.relu,
            ]
        )
        self.actor_1 = nn.Dense(64, kernel_init=initializers.orthogonal(2))
        self.actor_out = nn.Dense(
            self.action_space, kernel_init=initializers.orthogonal(0.01)
        )
        self.critic_1 = nn.Dense(64, kernel_init=initializers.orthogonal(2))
        self.critic_out = nn.Dense(1, kernel_init=initializers.orthogonal(1.0))
        self.lstm = LSTM(32)
        self.znet = ZNet()

    def get_states(self, x, lstm_state, done):
        agent_data, inventories = x
        inventories = self.znet(inventories)
        new_inventory_shape = inventories.shape[:-2] + (-1,)
        x = jnp.concatenate(
            [agent_data, inventories.reshape(new_inventory_shape)], axis=-1
        )
        x = self.network(x)
        return self.lstm(x, done, lstm_state)

    def actor(self, x):
        x = self.actor_1(x)
        x = nn.relu(x)
        x = self.actor_out(x)
        return x

    def critic(self, x):
        x = self.critic_1(x)
        x = nn.relu(x)
        x = self.critic_out(x)
        return x

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def __call__(self, x, lstm_state, done):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor(hidden)
        value = self.critic(hidden)
        return logits, value, hidden, lstm_state


class ClassicMetaController:
    def __init__(
        self,
        env_params: EnvParams = EnvParams(),
        static_parameters: StaticEnvParams = StaticEnvParams(),
        num_envs: int = 8,
        num_steps: int = 300,
        fixed_timesteps: bool = False,
        num_iterations: int = 100,
        learning_rate: float = 2.5e-3,
        anneal_lr: bool = True,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        num_minibatches: int = 4,
        update_epochs: int = 4,
        clip_coef: float = 0.2,
        norm_adv: bool = True,
        clip_vloss: bool = True,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        proximity_bonus: float = 0.01,
        aux_coef: float = 0.3,
        max_grad_norm: float = 0.5,
        # target_kl: float | None = None,
    ):
        """
        Params:
        - env_params: non-static parameters
        - static_parameters: static environment parameters
        - device: device to use
        - num_envs: number of environments to run in parallel during rollout
        - num_steps: Number of steps to take for each batch. Note that this can be less than
            the actual number of steps used for training since the agent can be dead for some steps
        - fixed_timesteps: Fix the number of timesteps per episode
        - num_iterations: Number of rollout/training iterations
        - learning_rate: learning rate
        - anneal_lr: whether to anneal learning rate
        - gamma: decay rate
        - gae_lambda: lambda for generalized advantage estimation
        - num_minibatches: number of minibatches to use for each batch
        - update_epochs: number of epochs to perform in update step
        - clip_coef: surrogate clipping coefficient
        - norm_adv: whether to use advantage normalization
        - clip_vloss: Whether to use clipped loss for the value function
        - ent_coef: Entropy coefficient
        - vf_coef: Value function coefficient
        - aux_coef: Auxillary loss coefficient
        - max_grad_norm: The maximum norm for gradient clipping
        - target_kl: target KL divergence threshold
        """
        self.static_params = static_parameters
        self.num_envs = num_envs
        self.fixed_timesteps = fixed_timesteps
        self.env_params = env_params
        self.timestep = self.env_params.max_timesteps
        if fixed_timesteps:
            self.env = CraftaxClassicSymbolicEnvShareStatsNoAutoReset(
                self.static_params
            )
        else:
            self.env = CraftaxClassicSymbolicEnvShareStats(self.static_params)
        self.step_fn = jax.vmap(
            self.env.step, in_axes=(0, 0, 1, None), out_axes=(1, 0, 1, 1, 0)
        )
        self.reset_fn = jax.vmap(self.env.reset, in_axes=(0, None), out_axes=(1, 0))
        self.player_alive_check = jax.vmap(are_players_alive, out_axes=1)
        self.num_steps = num_steps
        # self.rng = jax.random.PRNGKey(randrange(2**31))
        self.rng = jax.random.PRNGKey(56)
        self.observation_space = self.env.observation_space(env_params)
        self.action_space = self.env.action_space(env_params)
        # Learning params
        self.learning_rate = learning_rate
        self.num_envs = num_envs
        self.anneal_lr = anneal_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.clip_coef = clip_coef
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.proximity_bonus = proximity_bonus
        self.aux_coef = aux_coef
        self.max_grad_norm = max_grad_norm
        # self.target_kl = target_kl
        self.num_iterations = num_iterations
        # Save the training configuration as a dict
        self.config = dict(vars(self))
        for unwanted in (
            "env_params",
            "static_params",
            "env",
            "step_fn",
            "reset_fn",
            "player_alive_check",
        ):
            del self.config[unwanted]

        self.agent = CraftaxAgent(self.action_space.n)
        self.aux = AuxLossNet(
            np.prod(self.observation_space.spaces[0].shape)  # pyright: ignore
            + np.prod(self.observation_space.spaces[1].shape)  # pyright: ignore
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(learning_rate=self.learning_rate),
        )
        self.lr_schedule = optax.linear_schedule(
            self.learning_rate, 0.0, self.num_iterations
        )
        self.timestep_schedule = optax.linear_schedule(
            self.env_params.max_timesteps * 0.01,
            self.env_params.max_timesteps,
            round(self.num_iterations * 0.8),
        )

    @partial(jax.jit, static_argnums=(0,))
    def train_some_episodes(
        self, rng, tick, model_params, opt_states, next_lstm_states, next_obs, env_state
    ):
        rng, _rng = jax.random.split(rng)
        agent_params, aux_params = model_params
        next_done = jnp.zeros((self.static_params.num_players, self.num_envs))
        init_lstm_states = next_lstm_states
        if self.anneal_lr:
            # update learning rate
            opt_states[1].hyperparams["learning_rate"] = jnp.full(  # pyright: ignore
                self.static_params.num_players, self.lr_schedule(tick)
            )
        env_params = self.env_params
        if self.fixed_timesteps:
            env_params = env_params.replace(max_timesteps=self.timestep_schedule(tick))  # pyright: ignore

        def rollout_step(carry, step):
            (
                next_obs,
                next_done,
                env_state,
                next_lstm_states,
                rng,
            ) = carry
            init_obs = next_obs
            init_done = next_done
            init_lstm_states = next_lstm_states
            # agents_alive = self.player_alive_check(env_state)

            def eval_agent(rng, agent_idx, agent_param, next_lstm_state):
                rng, _rng = jax.random.split(rng)
                logits, value, _, next_lstm_state = self.agent.apply(  # pyright: ignore
                    agent_param,
                    self._idx(next_obs)[agent_idx],
                    next_lstm_state,
                    next_done[agent_idx],
                )
                # sets the action to NOOP if player is dead or sleeping
                # I think this does more harm than good
                # action = jax.lax.select(
                #     ~agents_alive[agent_idx] | env_state.is_sleeping[:, agent_idx],
                #     jnp.full(self.num_envs, Action.NOOP.value),
                #     jax.random.categorical(_rng, logits),
                # )
                action = jax.random.categorical(_rng, logits)
                return (
                    value.flatten(),  # pyright: ignore
                    action,
                    jax.nn.log_softmax(logits)[jnp.arange(self.num_envs), action],
                    next_lstm_state,
                )

            rng, _rng = jax.random.split(rng)
            value, action, logprob, next_lstm_states = jax.vmap(eval_agent)(
                jax.random.split(_rng, self.static_params.num_players),
                jnp.arange(self.static_params.num_players),
                agent_params,
                next_lstm_states,
            )
            rng, _rng = jax.random.split(rng)
            next_obs, env_state, reward, next_done, _info = self.step_fn(
                jax.random.split(_rng, self.num_envs),
                env_state,
                action.astype(int),
                env_params,
            )

            # check if player is within view
            def within_view(idx):
                offset = jnp.abs(
                    env_state.player_position
                    - jnp.expand_dims(env_state.player_position[:, idx], 1)
                )
                within_x = offset[..., 0] <= OBS_DIM[0] // 2
                within_y = offset[..., 1] <= OBS_DIM[1] // 2
                in_range = jnp.logical_and(within_x, within_y)
                in_range = in_range.at[:, idx].set(False)
                return jnp.any(in_range, axis=1)

            can_see_others = jax.vmap(within_view)(
                jnp.arange(self.static_params.num_players)
            )

            def reset_env(rng):
                """Performs an environment reset"""
                rng, _rng = jax.random.split(rng)
                next_obs, env_state = self.reset_fn(
                    jax.random.split(_rng, self.num_envs), self.env_params
                )
                return next_obs, env_state, jnp.ones_like(next_done), rng

            def do_nothing(rng):
                return next_obs, env_state, next_done, rng

            if self.fixed_timesteps:
                # All timesteps should be synchronized if fixed_timesteps
                # therefore, they reset at the same time
                next_obs, env_state, next_done, rng = jax.lax.cond(
                    env_state.timestep[0] >= self.env_params.max_timesteps,
                    reset_env,
                    do_nothing,
                    rng,
                )

            next_done = next_done.astype(float)
            return (
                next_obs,
                next_done,
                env_state,
                next_lstm_states,
                rng,
            ), (
                init_obs,
                next_obs,
                init_done,
                value,
                action,
                logprob,
                reward,
                can_see_others,
            )

        # ugly code
        (
            (
                next_obs,
                next_done,
                env_state,
                next_lstm_states,
                rng,
            ),
            (
                obs,
                future_obs,
                dones,
                values,
                actions,
                logprobs,
                rewards,
                can_see_others,
            ),
        ) = jax.lax.scan(
            rollout_step,
            (
                next_obs,
                next_done,
                env_state,
                next_lstm_states,
                rng,
            ),
            jnp.arange(self.num_steps),
        )

        # bootstrap value if not done
        def produce_value(agent_param, next_obs, next_lstm_state, next_done):
            return self.agent.apply(
                agent_param,
                next_obs,
                next_lstm_state,
                next_done,
                method=CraftaxAgent.get_value,
            ).flatten()  # pyright: ignore

        next_values = jax.vmap(produce_value)(
            agent_params, next_obs, next_lstm_states, next_done
        ).reshape(self.static_params.num_players, self.num_envs)

        def compute_advantages(carry, transition):
            lastgaelam, next_value, next_done = carry
            done, value, reward = transition
            delta = reward + self.gamma * next_value * (1 - next_done) - value
            lastgaelam = (
                delta + self.gamma * self.gae_lambda * (1 - next_done) * lastgaelam
            )
            return (lastgaelam, value, done), lastgaelam

        _, advantages = jax.lax.scan(
            compute_advantages,
            (jnp.zeros_like(next_done), next_values, next_done),
            (dones, values, rewards),
            reverse=True,
        )
        returns = advantages + values

        # this function calculates the ppo loss function
        def ppo_loss(
            model_params,
            mb_obs,
            mb_future_obs,
            mb_logprobs,
            mb_actions,
            mb_dones,
            mb_advantages,
            mb_returns,
            mb_values,
            mb_proximity,
            init_lstm_state,
        ):
            agent_params, aux_params = model_params
            logits, newvalue, hidden, _ = self.agent.apply(  # pyright: ignore
                agent_params, mb_obs, init_lstm_state, mb_dones
            )
            probs = jax.nn.softmax(logits)
            newlogprobs = jnp.log(probs)
            entropy = -jnp.sum(probs * newlogprobs, axis=-1)
            # basically newlogprobs[action], where action tells us what to take in the last dimension
            newlogprob = jnp.take_along_axis(
                newlogprobs, jnp.expand_dims(mb_actions.astype(int), axis=-1), axis=-1
            ).squeeze(-1)
            logratio = newlogprob - mb_logprobs
            ratio = jnp.exp(logratio)

            # calculate approx kl
            # old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()

            if self.norm_adv:
                if len(mb_advantages) == 1:
                    mb_advantages = 0.0
                else:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * jax.lax.clamp(
                1 - self.clip_coef, ratio, 1 + self.clip_coef
            )
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = jnp.squeeze(newvalue, axis=-1)  # pyright: ignore

            if self.clip_vloss:
                v_loss_unclipped = (newvalue - mb_returns) ** 2
                v_clipped = mb_values + jax.lax.clamp(
                    -self.clip_coef, newvalue - mb_values, self.clip_coef
                )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss_max = jnp.maximum(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

            # Auxillary Loss
            # given the array of predicted accuracy, take the mean
            predicted_future_obs = self.aux.apply(aux_params, hidden)
            reshaped_future_obs = jax.tree_util.tree_map(
                lambda x: x.reshape(x.shape[:2] + (-1,)),
                mb_future_obs,
            )
            concat_future_obs = jnp.concatenate(
                [reshaped_future_obs[0], reshaped_future_obs[1]], axis=2
            )
            aux_error = concat_future_obs - predicted_future_obs
            aux_loss = jnp.mean(jnp.square(aux_error))

            # Proximity bonus
            mean_proximity = mb_proximity.astype(float).mean()

            # Entropy Loss
            entropy_loss = entropy.mean()
            loss = (
                pg_loss
                - self.ent_coef * entropy_loss
                + v_loss * self.vf_coef
                + aux_loss * self.aux_coef
                - mean_proximity * self.proximity_bonus
            )
            return loss, (
                pg_loss,
                v_loss,
                entropy_loss,
                aux_loss,
                mean_proximity,
                jax.lax.stop_gradient(approx_kl),
            )

        grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

        # Optimize the policy and value network

        envsperbatch = self.num_envs // self.num_minibatches
        # environment indices
        envinds = jnp.arange(self.num_envs)

        def process_agent(agent_idx, model_param, opt_state, rng):
            b_obs = self._idx(obs)[:, agent_idx]
            b_future_obs = self._idx(future_obs)[:, agent_idx]
            b_logprobs = logprobs[:, agent_idx]
            b_actions = actions[:, agent_idx]
            b_dones = dones[:, agent_idx]
            b_advantages = advantages[:, agent_idx]
            b_returns = returns[:, agent_idx]
            b_values = values[:, agent_idx]
            b_proximity = can_see_others[:, agent_idx]

            def do_epoch(carry, epoch):
                rng, model_param, optimizer_state = carry
                rng, _rng = jax.random.split(rng)
                shuffled_envinds = jax.random.permutation(_rng, envinds)

                def do_minibatch(carry, start):
                    model_param, optimizer_state = carry

                    mbenvinds = jax.lax.dynamic_slice(
                        shuffled_envinds, (start,), (envsperbatch,)
                    )

                    mb_obs = self._idx(b_obs)[:, mbenvinds]
                    mb_future_obs = self._idx(b_future_obs)[:, mbenvinds]
                    mb_logprobs = b_logprobs[:, mbenvinds]
                    mb_actions = b_actions[:, mbenvinds]
                    mb_dones = b_dones[:, mbenvinds]
                    mb_advantages = b_advantages[:, mbenvinds]
                    mb_returns = b_returns[:, mbenvinds]
                    mb_values = b_values[:, mbenvinds]
                    mb_proximity = b_proximity[:, mbenvinds]
                    init_lstm_state = jax.tree.map(
                        lambda state: state[agent_idx, mbenvinds], init_lstm_states
                    )

                    (
                        (
                            loss,
                            (
                                pg_loss,
                                v_loss,
                                entropy_loss,
                                aux_loss,
                                mean_proximity,
                                approx_kl,
                            ),
                        ),
                        grads,
                    ) = grad_fn(
                        model_param,
                        mb_obs,
                        mb_future_obs,
                        mb_logprobs,
                        mb_actions,
                        mb_dones,
                        mb_advantages,
                        mb_returns,
                        mb_values,
                        mb_proximity,
                        init_lstm_state,
                    )
                    updates, optimizer_state = self.optimizer.update(
                        grads, optimizer_state, model_param
                    )
                    model_param = optax.apply_updates(model_param, updates)
                    return (model_param, optimizer_state), (
                        loss,
                        pg_loss,
                        v_loss,
                        entropy_loss,
                        aux_loss,
                        mean_proximity,
                        approx_kl,
                    )

                (
                    (model_param, optimizer_state),
                    (
                        losses,
                        pg_losses,
                        v_losses,
                        entropy_losses,
                        aux_losses,
                        mean_proximity,
                        approx_kl,
                    ),
                ) = jax.lax.scan(
                    do_minibatch,
                    (model_param, optimizer_state),
                    jnp.arange(0, self.num_envs, envsperbatch),
                )

                return (rng, model_param, optimizer_state), (
                    losses,
                    pg_losses,
                    v_losses,
                    entropy_losses,
                    aux_losses,
                    mean_proximity,
                    approx_kl,
                )

            (
                (_rng, model_param, opt_state),
                (
                    losses,
                    pg_losses,
                    v_losses,
                    entropy_losses,
                    aux_losses,
                    mean_proximity,
                    approx_kl,
                ),
            ) = jax.lax.scan(
                do_epoch, (rng, model_param, opt_state), jnp.arange(self.update_epochs)
            )
            last_epoch_loss = losses[-1].mean()
            last_approx_kl = approx_kl[-1].mean()
            last_pg_loss = pg_losses[-1].mean()
            last_v_loss = v_losses[-1].mean()
            last_entropy_loss = entropy_losses[-1].mean()
            last_aux_loss = aux_losses[-1].mean()
            last_mean_proximity = mean_proximity[-1].mean()
            return (
                model_param,
                opt_state,
                last_epoch_loss,
                last_pg_loss,
                last_v_loss,
                last_entropy_loss,
                last_aux_loss,
                last_mean_proximity,
                last_approx_kl,
            )

        (
            model_params,
            opt_states,
            agent_loss,
            agent_pg_loss,
            agent_v_loss,
            agent_entropy_loss,
            agent_aux_loss,
            agent_mean_proximity,
            last_approx_kl,
        ) = jax.vmap(process_agent)(
            jnp.arange(self.static_params.num_players),
            model_params,
            opt_states,
            jax.random.split(rng, self.static_params.num_players),
        )

        return (
            model_params,
            opt_states,
            next_lstm_states,
            next_obs,
            env_state,
            agent_loss,
            agent_pg_loss,
            agent_v_loss,
            agent_entropy_loss,
            agent_aux_loss,
            agent_mean_proximity,
            rewards.mean(axis=(0, 2)),
            last_approx_kl,
        )

    def train(self, model_params=None):
        if model_params is None:
            dummy_obs = (
                jnp.ones(
                    (self.num_steps, self.num_envs)
                    + self.observation_space.spaces[0].shape  # pyright: ignore
                ),
                jnp.ones(
                    (self.num_steps, self.num_envs)
                    + self.observation_space.spaces[1].shape  # pyright: ignore
                ),
            )
            dummy_lstm_state = (
                jnp.ones((self.num_envs, 32)),
                jnp.ones((self.num_envs, 32)),
            )
            dummy_done = jnp.ones((self.num_steps, self.num_envs))
            rng, _rng = jax.random.split(self.rng)
            agent_params = jax.vmap(self.agent.init, in_axes=(0, None, None, None))(
                jax.random.split(_rng, self.static_params.num_players),
                dummy_obs,
                dummy_lstm_state,
                dummy_done,
            )
            rng, _rng = jax.random.split(rng)
            dummy_hidden = jnp.ones((self.num_steps, self.num_envs, 32))
            aux_params = jax.vmap(self.aux.init, in_axes=(0, None))(
                jax.random.split(_rng, self.static_params.num_players), dummy_hidden
            )
            model_params = (agent_params, aux_params)
        else:
            rng = self.rng
        opt_states = jax.vmap(self.optimizer.init)(model_params)

        # Logger
        log = TrainLogger(self.config, self.env_params, self.static_params)

        # initialize environment
        rng, _rng = jax.random.split(rng)
        next_obs, env_state = self.reset_fn(
            jax.random.split(_rng, self.num_envs), self.env_params
        )
        next_lstm_states = (
            jnp.zeros((self.static_params.num_players, self.num_envs, 32)),
            jnp.zeros((self.static_params.num_players, self.num_envs, 32)),
        )
        for iteration in range(self.num_iterations):
            print("Iteration", iteration)
            rng, _rng = jax.random.split(rng)
            (
                model_params,
                opt_states,
                next_lstm_states,
                next_obs,
                env_state,
                agent_loss,
                agent_pg_loss,
                agent_v_loss,
                agent_entropy_loss,
                agent_aux_loss,
                agent_proximity,
                rewards,
                last_approx_kl,
            ) = self.train_some_episodes(
                _rng,
                iteration,
                model_params,
                opt_states,
                next_lstm_states,
                next_obs,
                env_state,
            )
            log.insert_stat(iteration, "loss", agent_loss)
            log.insert_stat(iteration, "reward", rewards)
            log.insert_stat(iteration, "kl", last_approx_kl)
            log.insert_stat(iteration, "pg_loss", agent_pg_loss)
            log.insert_stat(iteration, "v_loss", agent_v_loss)
            log.insert_stat(iteration, "entropy_loss", agent_entropy_loss)
            log.insert_stat(iteration, "aux_loss", agent_aux_loss)
            log.insert_stat(iteration, "proximity", agent_proximity)
            if iteration % 20 == 0:
                log.insert_model_snapshot(iteration, model_params)
            for agent in range(self.static_params.num_players):
                print("Agent", agent, "loss:", agent_loss[agent])
                print("Agent", agent, "PG loss:", agent_pg_loss[agent])
                print("Agent", agent, "value loss:", agent_v_loss[agent])
                print("Agent", agent, "entropy:", agent_entropy_loss[agent])
                print("Agent", agent, "auxillary loss:", agent_aux_loss[agent])
                print("Agent", agent, "mean proximity:", agent_proximity[agent])
                print("Agent", agent, "reward:", rewards[agent])
                print("Agent", agent, "approx KL:", last_approx_kl[agent], flush=True)
        return model_params, opt_states, log

    def run_one_episode(self, model_params):
        agent_params, _aux_params = model_params
        rng, _rng = jax.random.split(self.rng)
        next_obs, env_state = self.env.reset(_rng, self.env_params)
        next_done = jnp.zeros((self.static_params.num_players, 1), dtype=bool)
        states = []
        actions = []
        logits = []
        rewards = []
        next_lstm_states = (
            jnp.zeros((self.static_params.num_players, 1, 32)),
            jnp.zeros((self.static_params.num_players, 1, 32)),
        )

        def eval_agent(model_param, next_lstm_state, next_obs, next_done, rng):
            logits, _value, _hidden, next_lstm_state = self.agent.apply(  # pyright: ignore
                model_param, next_obs, next_lstm_state, next_done
            )
            action = jax.random.categorical(rng, logits).squeeze()
            return action, logits, next_lstm_state

        eval_fn = jax.jit(jax.vmap(eval_agent))
        while not jnp.all(next_done):
            rng, _rng = jax.random.split(rng)
            states.append(env_state)
            agent_actions, agent_logits, next_lstm_states = eval_fn(
                agent_params,
                next_lstm_states,
                next_obs,
                next_done,
                jax.random.split(_rng, self.static_params.num_players),
            )
            next_obs, env_state, reward, next_done, _info = self.env.step(
                _rng, env_state, agent_actions, self.env_params
            )
            actions.append(agent_actions)
            logits.append(agent_logits)
            rewards.append(reward)
        return states, actions, logits, rewards

    class _Indexer:
        def __init__(self, obj):
            self.obj = obj

        def __getitem__(self, val):
            return jax.tree_util.tree_map(lambda x: x[val], self.obj)

    def _idx(self, obj):
        """
        Helpful utility to index object like a PyTree
        """
        return self._Indexer(obj)


if __name__ == "__main__":
    metacontroller = ClassicMetaController(
        static_parameters=StaticEnvParams(num_players=4),
        num_envs=10,
        num_minibatches=1,
        num_steps=200,
        num_iterations=5,
        update_epochs=5,
        anneal_lr=False,
        learning_rate=2.5e-4,
        max_grad_norm=1.0,
        fixed_timesteps=True,
    )
    params, opt_states, log = metacontroller.train()
    # states, actions, logits, rewards = metacontroller.run_one_episode(params)
    # replay_episode(states, actions, 4, 0)
    # n_steps = 10
    # n_envs = 8
    # dummy_obs = jnp.ones((n_steps, n_envs, 1346))
    # dummy_lstm_state = (jnp.ones((n_envs, 32)), jnp.ones((n_envs, 32)))
    # dummy_done = jnp.ones((n_steps, n_envs))
    # print(
    #     CraftaxAgent(17, True).tabulate(
    #         jax.random.PRNGKey(randrange(2**31)),
    #         dummy_obs,
    #         dummy_lstm_state,
    #         dummy_done,
    #     )
    # )
    # model = CraftaxAgent(17)
    # rng_key = jax.random.PRNGKey(0)
    # params = model.init(rng_key, dummy_obs, dummy_lstm_state, dummy_done)
    # print(params)
    # breakpoint()
