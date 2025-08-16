# Add communication between agents
import craftax
from functools import partial
from random import randrange

import flax.linen as nn
import jax

try:
    from jax.tree_util import tree_map
except Exception:
    # Very old JAX fallback
    from jax import tree_map  # type: ignore


import jax.numpy as jnp
import optax
from flax.linen import initializers

from craftax.craftax_classic.envs.craftax_state import EnvParams, StaticEnvParams
from craftax.craftax_classic.envs.craftax_symbolic_env import (
    CraftaxClassicSymbolicEnv,
    CraftaxClassicSymbolicEnvNoAutoReset,
    CraftaxClassicSymbolicEnvShareStats,
    CraftaxClassicSymbolicEnvShareStatsNoAutoReset,
)
from craftax.craftax_classic.game_logic import are_players_alive
from logger import TrainLogger
from nets import LSTM, ZNet


class CraftaxAgent(nn.Module):
    action_space: int
    observe_others: bool
    msg_vocab_size: int = 16  # NEW: size of the discrete token space
    msg_embed_dim: int = 32  # NEW: size of the message embedding

    def setup(self):
        self.network = nn.Sequential(
            [
                nn.Dense(512, kernel_init=initializers.orthogonal(jnp.sqrt(2))),
                nn.relu,
                nn.Dense(256, kernel_init=initializers.orthogonal(jnp.sqrt(2))),
                nn.relu,
            ]
        )
        self.actor_1 = nn.Dense(128, kernel_init=initializers.orthogonal(2))
        self.actor_out = nn.Dense(
            self.action_space, kernel_init=initializers.orthogonal(0.01)
        )

        # NEW: message “actor” head (categorical over msg_vocab_size)
        self.msg_1 = nn.Dense(128, kernel_init=initializers.orthogonal(2))
        self.msg_out = nn.Dense(
            self.msg_vocab_size, kernel_init=initializers.orthogonal(0.01)
        )

        self.critic_1 = nn.Dense(128, kernel_init=initializers.orthogonal(2))
        self.critic_out = nn.Dense(1, kernel_init=initializers.orthogonal(1.0))

        self.lstm = LSTM(256)
        self.znet = ZNet()

        # NEW: learnable lookup table for received tokens
        self.msg_embed = nn.Embed(
            num_embeddings=self.msg_vocab_size, features=self.msg_embed_dim
        )

    def _embed_received_messages(self, recv_tokens, agent_idx, reduce: str = "sum"):
        """
        recv_tokens: (..., num_envs, num_players) or (num_envs, num_players)
                    We treat the LAST axis as players.
        agent_idx:   scalar index of the current agent (0..P-1)
        Returns:     (..., num_envs, D) or (num_envs, D) — same leading dims as recv_tokens, with players aggregated.
        """
        # Ensure integer indices for the embedding
        tokens = recv_tokens.astype(jnp.int32)  # (..., P)

        # Embed every player's token
        emb_all = self.msg_embed(tokens)  # (..., P, D)

        P = tokens.shape[-1]
        # One-hot for "self", then mask them out without dynamic boolean indexing
        self_hot = jax.nn.one_hot(agent_idx, P, dtype=emb_all.dtype)  # (P,)
        # reshape to broadcast across leading dims and feature dim
        expand = (1,) * (emb_all.ndim - 2) + (P, 1)
        self_hot = self_hot.reshape(expand)  # (..., P, 1)
        mask = 1.0 - self_hot  # (..., P, 1)
        # TODO Change this message aggregation function when number of agents > 2
        emb = (emb_all * mask).sum(axis=-2)  # (..., D)

        if reduce == "mean":
            emb = emb / jnp.maximum(1.0, jnp.array(P - 1, dtype=emb.dtype))

        return emb

    def get_states(self, x, lstm_state, done, recv_tokens=None, agent_idx=None):
        """
        If observe_others: x is (agent_data, inventories)
        Else: x is agent_data
        If recv_tokens is provided, we append aggregated message embedding.
        """
        # jax.debug.print("done shape: {s}", s=done.shape)
        # if recv_tokens is not None:
        #     jax.debug.print("recv_tokens shape: {s}", s=recv_tokens.shape)

        if self.observe_others:
            agent_data, inventories = x
            inventories = self.znet(inventories)
            new_inventory_shape = inventories.shape[:-2] + (-1,)
            x = jnp.concatenate(
                [agent_data, inventories.reshape(new_inventory_shape)], axis=-1
            )
        # Append message embedding if provided
        if recv_tokens is not None and agent_idx is not None:
            msg_emb = self._embed_received_messages(
                recv_tokens, agent_idx
            )  # (num_envs, D)
            # jax.debug.print("msg_emb shape: {s}", s=msg_emb.shape)
            # jax.debug.print("x shape: {s}", s=x.shape)
            x = jnp.concatenate([x, msg_emb], axis=-1)

        x = self.network(x)
        return self.lstm(x, done, lstm_state)

    def actor(self, x):
        x = self.actor_1(x)
        x = nn.relu(x)
        x = self.actor_out(x)
        return x

    def message_actor(self, x):
        x = self.msg_1(x)
        x = nn.relu(x)
        return self.msg_out(x)

    def critic(self, x):
        x = self.critic_1(x)
        x = nn.relu(x)
        x = self.critic_out(x)
        return x

    def get_value(self, x, lstm_state, done, recv_tokens=None, agent_idx=None):
        hidden, _ = self.get_states(x, lstm_state, done, recv_tokens, agent_idx)
        return self.critic(hidden)

    def __call__(self, x, lstm_state, done, recv_tokens=None, agent_idx=None):
        """
        Returns:
          action_logits: (num_envs, action_space)
          message_logits: (num_envs, msg_vocab_size)
          value: (num_envs, 1)
          lstm_state: ...
        """
        hidden, lstm_state = self.get_states(
            x, lstm_state, done, recv_tokens, agent_idx
        )
        action_logits = self.actor(hidden)
        msg_logits = self.message_actor(hidden)
        value = self.critic(hidden)
        return action_logits, msg_logits, value, lstm_state


class ClassicMetaController:
    def __init__(
        self,
        env_params: EnvParams = EnvParams(),
        static_parameters: StaticEnvParams = StaticEnvParams(),
        num_envs: int = 8,
        num_steps: int = 300,
        fixed_timesteps: bool = False,
        observe_others: bool = False,
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
        act_ent_coef: float = 0.01,
        msg_ent_coef: float = 0.002,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        wandb_project=None,
        run_name=None,
        msg_vocab_size: int = 16,
        msg_embed_dim: int = 32,
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
        - observe_others: Players can observe other players' inventories
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
        - max_grad_norm: The maximum norm for gradient clipping
        - target_kl: target KL divergence threshold
        """
        self.static_params = static_parameters
        self.num_envs = num_envs
        self.fixed_timesteps = fixed_timesteps
        self.env_params = env_params
        self.timestep = self.env_params.max_timesteps
        self.observe_others = observe_others
        if observe_others:
            if fixed_timesteps:
                self.env = CraftaxClassicSymbolicEnvShareStatsNoAutoReset(
                    self.static_params
                )
            else:
                self.env = CraftaxClassicSymbolicEnvShareStats(self.static_params)
        else:
            if fixed_timesteps:
                self.env = CraftaxClassicSymbolicEnvNoAutoReset(self.static_params)
            else:
                self.env = CraftaxClassicSymbolicEnv(self.static_params)
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
        self.act_ent_coef = act_ent_coef
        self.msg_ent_coef = msg_ent_coef
        self.vf_coef = vf_coef
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

        self.msg_vocab_size = msg_vocab_size
        self.msg_embed_dim = msg_embed_dim

        self.agent = CraftaxAgent(
            self.action_space.n,
            self.observe_others,
            msg_vocab_size=self.msg_vocab_size,
            msg_embed_dim=self.msg_embed_dim,
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
        self.wandb_project = wandb_project
        self.run_name = run_name

    @partial(jax.jit, static_argnums=(0,))
    def train_some_episodes(
        self, rng, tick, model_params, opt_states, next_lstm_states, next_obs, env_state
    ):
        rng, _rng = jax.random.split(rng)
        next_done = jnp.zeros((self.static_params.num_players, self.num_envs))
        init_lstm_states = next_lstm_states
        if self.anneal_lr:
            # update learning rate
            opt_states[1].hyperparams["learning_rate"] = jnp.full(  # pyright: ignore
                self.static_params.num_players, self.lr_schedule(tick)
            )
        env_params = self.env_params
        if self.fixed_timesteps:
            env_params = env_params.replace(
                max_timesteps=self.timestep_schedule(tick)
            )  # pyright: ignore

        recv_tokens = jnp.zeros(
            (self.num_envs, self.static_params.num_players), dtype=jnp.int32
        )

        def rollout_step(carry, step):
            (next_obs, next_done, env_state, next_lstm_states, rng, recv_tokens) = carry
            init_obs = next_obs
            init_done = next_done
            # agents_alive = self.player_alive_check(env_state)

            def eval_agent(rng, agent_idx, model_param, next_lstm_state):
                rng, _rng = jax.random.split(rng)

                (
                    action_logits,
                    msg_logits,
                    value,
                    next_lstm_state,
                ) = self.agent.apply(  # pyright: ignore
                    model_param,
                    self._idx(next_obs)[agent_idx],
                    next_lstm_state,
                    next_done[agent_idx],
                    recv_tokens=recv_tokens,
                    agent_idx=agent_idx,
                )

                # sample action + message token
                action = jax.random.categorical(_rng, action_logits)
                rng2, _rng2 = jax.random.split(rng)
                message = jax.random.categorical(_rng2, msg_logits)

                # logprobs
                act_logprob = jax.nn.log_softmax(action_logits)[
                    jnp.arange(self.num_envs), action
                ]
                msg_logprob = jax.nn.log_softmax(msg_logits)[
                    jnp.arange(self.num_envs), message
                ]
                joint_logprob = act_logprob + msg_logprob

                # entropies
                act_probs = jax.nn.softmax(action_logits)
                msg_probs = jax.nn.softmax(msg_logits)
                act_entropy = -jnp.sum(
                    act_probs * jax.nn.log_softmax(action_logits), axis=-1
                )
                msg_entropy = -jnp.sum(
                    msg_probs * jax.nn.log_softmax(msg_logits), axis=-1
                )
                joint_entropy = act_entropy + msg_entropy

                return (
                    value.flatten(),
                    action,
                    message,
                    joint_logprob,
                    joint_entropy,
                    next_lstm_state,
                )

            rng, _rng = jax.random.split(rng)
            (
                value,
                action,
                message,
                joint_logprob,
                joint_entropy,
                next_lstm_states,
            ) = jax.vmap(eval_agent)(
                jax.random.split(_rng, self.static_params.num_players),
                jnp.arange(self.static_params.num_players),
                model_params,
                next_lstm_states,
            )

            rng, _rng = jax.random.split(rng)
            next_obs, env_state, reward, next_done, _info = self.step_fn(
                jax.random.split(_rng, self.num_envs),
                env_state,
                action.astype(int),
                env_params,
            )

            recv_tokens_next = message.T.astype(jnp.int32)

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
                recv_tokens_next,
            ), (
                init_obs,
                init_done,
                value,
                action,
                message,
                joint_logprob,
                reward,
                joint_entropy,
            )

        # Rollout loop (use scan instead of for)
        (
            (next_obs, next_done, env_state, next_lstm_states, rng, recv_tokens),
            (obs, dones, values, actions, messages, logprobs, rewards, entropies),
        ) = jax.lax.scan(
            rollout_step,
            (next_obs, next_done, env_state, next_lstm_states, rng, recv_tokens),
            jnp.arange(self.num_steps),
        )

        # bootstrap value if not done
        def produce_value(
            model_param, next_obs, next_lstm_state, next_done, recv_tokens, agent_idx
        ):
            return self.agent.apply(
                model_param,
                next_obs,
                next_lstm_state,
                next_done,
                recv_tokens,
                agent_idx,
                method=CraftaxAgent.get_value,
            ).flatten()  # pyright: ignore

        player_idx = jnp.arange(self.static_params.num_players)

        next_values = jax.vmap(produce_value, in_axes=(0, 0, 0, 0, None, 0))(
            model_params,
            next_obs,
            next_lstm_states,
            next_done,
            recv_tokens,  # shared for all agents
            player_idx,
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
            mb_logprobs,
            mb_actions,
            mb_messages,
            mb_dones,
            mb_advantages,
            mb_returns,
            mb_values,
            init_lstm_state,
            mb_recv_tokens,
            agent_idx,
        ):
            (
                action_logits,
                msg_logits,
                newvalue,
                _,
            ) = self.agent.apply(  # pyright: ignore
                model_params,
                mb_obs,
                init_lstm_state,
                mb_dones,
                recv_tokens=mb_recv_tokens,
                agent_idx=agent_idx,
            )

            # probs/logprobs
            act_logprobs = jax.nn.log_softmax(action_logits)
            msg_logprobs = jax.nn.log_softmax(msg_logits)
            new_act_lp = jnp.take_along_axis(
                act_logprobs, mb_actions[..., None].astype(int), axis=-1
            ).squeeze(-1)
            new_msg_lp = jnp.take_along_axis(
                msg_logprobs, mb_messages[..., None].astype(int), axis=-1
            ).squeeze(-1)
            newlogprob = new_act_lp + new_msg_lp

            # ratio
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

            # total entropy = action entropy + message entropy
            act_probs = jax.nn.softmax(action_logits)
            msg_probs = jax.nn.softmax(msg_logits)
            act_entropy = -jnp.sum(act_probs * act_logprobs, axis=-1)
            msg_entropy = -jnp.sum(msg_probs * msg_logprobs, axis=-1)
            entropy_loss = (
                self.act_ent_coef * act_entropy + self.msg_ent_coef * msg_entropy
            ).mean()

            loss = pg_loss - entropy_loss + v_loss * self.vf_coef
            return loss, (
                pg_loss,
                v_loss,
                entropy_loss,
                jax.lax.stop_gradient(approx_kl),
            )

        grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

        # Optimize the policy and value network

        envsperbatch = self.num_envs // self.num_minibatches
        # environment indices
        envinds = jnp.arange(self.num_envs)

        def process_agent(agent_idx, model_param, opt_state, rng):
            b_obs = self._idx(obs)[:, agent_idx]
            b_logprobs = logprobs[:, agent_idx]
            b_actions = actions[:, agent_idx]
            b_messages = messages[:, agent_idx]
            b_dones = dones[:, agent_idx]
            b_advantages = advantages[:, agent_idx]
            b_returns = returns[:, agent_idx]
            b_values = values[:, agent_idx]

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
                    mb_logprobs = b_logprobs[:, mbenvinds]
                    mb_actions = b_actions[:, mbenvinds]
                    mb_messages = b_messages[:, mbenvinds]
                    mb_dones = b_dones[:, mbenvinds]
                    mb_advantages = b_advantages[:, mbenvinds]
                    mb_returns = b_returns[:, mbenvinds]
                    mb_values = b_values[:, mbenvinds]
                    init_lstm_state = tree_map(
                        lambda state: state[agent_idx, mbenvinds], init_lstm_states
                    )
                    # Messages visible in this minibatch: take the *corresponding* received tokens
                    # We saved only the sent tokens in `messages` above; the received tokens for this step
                    # are simply those sent tokens transposed (per step). For simplicity, reuse `messages`
                    # as "what will be seen next step". Here we align by environment indices.
                    mb_recv_tokens = messages[:, :, mbenvinds]  # (T, P, E_mb)
                    # For PPO, we need one-per-time slice, so align shapes to (T*E_mb, P) by merging dims,
                    # or keep (T, E_mb, P) and rely on broadcasting inside the forward (already works
                    # since we pass the right env slice for each time step). Easiest: roll with the
                    # same time/env slicing we use for mb_obs.
                    (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = grad_fn(
                        model_param,
                        mb_obs,
                        mb_logprobs,
                        mb_actions,
                        mb_messages,
                        mb_dones,
                        mb_advantages,
                        mb_returns,
                        mb_values,
                        init_lstm_state,
                        mb_recv_tokens.transpose(0, 2, 1),  # reshape to (T, E_mb, P)
                        agent_idx,
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
                        approx_kl,
                    )

                (
                    (model_param, optimizer_state),
                    (losses, pg_losses, v_losses, entropy_losses, approx_kl),
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
                    approx_kl,
                )

            (
                (_rng, model_param, opt_state),
                (losses, pg_losses, v_losses, entropy_losses, approx_kl),
            ) = jax.lax.scan(
                do_epoch, (rng, model_param, opt_state), jnp.arange(self.update_epochs)
            )
            last_epoch_loss = losses[-1].mean()
            last_approx_kl = approx_kl[-1].mean()
            last_pg_loss = pg_losses[-1].mean()
            last_v_loss = v_losses[-1].mean()
            last_entropy_loss = entropy_losses[-1].mean()
            return (
                model_param,
                opt_state,
                last_epoch_loss,
                last_pg_loss,
                last_v_loss,
                last_entropy_loss,
                last_approx_kl,
            )

        (
            model_params,
            opt_states,
            agent_loss,
            agent_pg_loss,
            agent_v_loss,
            agent_entropy_loss,
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
            rewards.mean(axis=(0, 2)),
            last_approx_kl,
        )

    def train(self, model_params=None):
        if model_params is None:
            if self.observe_others:
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
            else:
                dummy_obs = jnp.ones(
                    (self.num_steps, self.num_envs)
                    + self.observation_space.shape  # pyright: ignore
                )
            dummy_lstm_state = (
                jnp.ones((self.num_envs, 256)),
                jnp.ones((self.num_envs, 256)),
            )
            dummy_done = jnp.ones((self.num_steps, self.num_envs))
            rng, _rng = jax.random.split(self.rng)
            dummy_recv = jnp.zeros(
                (self.num_steps, self.num_envs, self.static_params.num_players),
                dtype=jnp.int32,
            )
            model_params = jax.vmap(
                self.agent.init, in_axes=(0, None, None, None, None, None)
            )(
                jax.random.split(_rng, self.static_params.num_players),
                dummy_obs,
                dummy_lstm_state,
                dummy_done,
                dummy_recv,  # recv_tokens
                jnp.int32(0),  # agent_idx (any valid index for tracing)
            )
        else:
            rng = self.rng
        opt_states = jax.vmap(self.optimizer.init)(model_params)

        # Logger
        log = TrainLogger(
            self.config,
            self.env_params,
            self.static_params,
            self.wandb_project,
            self.run_name,
        )

        # initialize environment
        rng, _rng = jax.random.split(rng)
        next_obs, env_state = self.reset_fn(
            jax.random.split(_rng, self.num_envs), self.env_params
        )
        next_lstm_states = (
            jnp.zeros((self.static_params.num_players, self.num_envs, 256)),
            jnp.zeros((self.static_params.num_players, self.num_envs, 256)),
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
            if iteration % 20 == 0:
                log.insert_model_snapshot(iteration, model_params)
            for agent in range(self.static_params.num_players):
                print("Agent", agent, "loss:", agent_loss[agent])
                print("Agent", agent, "PG loss:", agent_pg_loss[agent])
                print("Agent", agent, "value loss:", agent_v_loss[agent])
                print("Agent", agent, "entropy:", agent_entropy_loss[agent])
                print("Agent", agent, "reward:", rewards[agent])
                print("Agent", agent, "approx KL:", last_approx_kl[agent], flush=True)
        return model_params, opt_states, log

    def run_one_episode(self, model_params):
        rng, _rng = jax.random.split(self.rng)
        next_obs, env_state = self.env.reset(_rng, self.env_params)
        next_done = jnp.zeros((self.static_params.num_players, 1), dtype=bool)
        states = []
        actions = []
        logits = []
        rewards = []
        next_lstm_states = (
            jnp.zeros((self.static_params.num_players, 1, 256)),
            jnp.zeros((self.static_params.num_players, 1, 256)),
        )

        def eval_agent(model_param, next_lstm_state, next_obs, next_done, rng):
            logits, _value, next_lstm_state = self.agent.apply(  # pyright: ignore
                model_param, next_obs, next_lstm_state, next_done
            )
            action = jax.random.categorical(rng, logits).squeeze()
            return action, logits, next_lstm_state

        eval_fn = jax.jit(jax.vmap(eval_agent))
        while not jnp.all(next_done):
            rng, _rng = jax.random.split(rng)
            states.append(env_state)
            agent_actions, agent_logits, next_lstm_states = eval_fn(
                model_params,
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
        def __init__(self, obj, observe_others: bool):
            self.obj = obj
            self.observe_others = observe_others

        def __getitem__(self, val):
            if self.observe_others:
                return tree_map(lambda x: x[val], self.obj)
            return self.obj[val]

    def _idx(self, obj):
        """
        Helpful utility to index object like a PyTree if observe_others
        is set to true, or treat it like an array otherwise
        """
        return self._Indexer(obj, self.observe_others)


if __name__ == "__main__":
    metacontroller = ClassicMetaController(
        static_parameters=StaticEnvParams(num_players=2),
        num_envs=1024,
        num_minibatches=1,
        num_steps=200,
        num_iterations=1000,
        update_epochs=5,
        anneal_lr=False,
        learning_rate=2.5e-4,
        max_grad_norm=1.0,
        fixed_timesteps=False,
        observe_others=True,
        wandb_project="ma_craftax",
        run_name="comm_ppo",
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
