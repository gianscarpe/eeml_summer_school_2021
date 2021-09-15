#@title Imports  { form-width: "30%" }

from jax.nn import softmax
import acme
from acme import environment_loop
from acme import datasets
from acme import specs
from acme import wrappers
from acme.wrappers import gym_wrapper
from acme.agents.jax import dqn
from acme.adders import reverb as adders
from acme.utils import counting
from acme.utils import loggers
import base64
import collections
from collections import namedtuple
import dm_env
import enum
import functools
import gym
import haiku as hk
import io
import imageio
import itertools
import jax
from jax import tree_util
# from jax.experimental import optix
import optax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import reverb
import rlax
import time
from tools import convolve1D
import warnings
warnings.filterwarnings('ignore')


Transitions = collections.namedtuple('Transitions',
                                     ['s_t', 'a_t', 'r_t', 'd_t', 's_tp1'])
TrainingState = namedtuple('TrainingState', 'params, opt_state, step')


class ReinforceAgent(acme.Actor):

  def __init__(self,
               pi_network,
               observation_spec,
               replay_capacity=100000,
               epsilon=0.1,
               batch_size=1,
               learning_rate=3e-4):

    self._observation_spec = observation_spec
    self.epsilon = epsilon
    self._batch_size = batch_size
    self.replay_capacity = replay_capacity
    self._replay_buffer = ReplayBuffer(self.replay_capacity)
    self.last_loss = 0
    self._done = False
    # Setup Network and loss with Haiku
    self._rng = hk.PRNGSequence(1)
    self._pi_network = hk.without_apply_rng(hk.transform(pi_network))
    
    # Initialize network
    dummy_observation = observation_spec.generate_value()
    initial_params = self._pi_network.init(
        next(self._rng), dummy_observation[None, ...])

    # Setup optimizer
    self._optimizer = optax.adam(learning_rate)
    initial_optimizer_state = self._optimizer.init(initial_params)

    self._state = TrainingState(
        params=initial_params, opt_state=initial_optimizer_state, step=0)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _policy(self, params: hk.Params, rng_key: jnp.ndarray,
              observation: jnp.ndarray, epsilon: float):
    pi_values = self._pi_network.apply(params, observation[None, ...])
    action = jax.random.categorical(rng_key, pi_values)
    return jnp.squeeze(action, axis=0)

  def select_action(self, observation):
    action = self._policy(self._state.params, next(self._rng), observation,
                        self.epsilon)
    action = tree_util.tree_map(lambda x: np.array(x).squeeze(axis=0), action)
    return action


  @functools.partial(jax.jit, static_argnums=(0,))
  def _loss(self, params: hk.Params, transitions: Transitions):

    def _reinforce(action_logits, a, G, d):
      """TD error for a single transition."""

      reinforce_error = action_logits[a]
      return - d * G * reinforce_error

  
  # Compute batched action logits [Batch, actions]
    action_logits = softmax(self._pi_network.apply(params, transitions.s_t))

    nsamples = len(action_logits)
    
    discount_sequence = jnp.power(transitions.d_t, jnp.arange(nsamples))
    filtered_rewards = convolve1D(transitions.r_t[::-1], discount_sequence)
    discounted_returns = filtered_rewards[:nsamples][::-1]
    batch_reinforce_error = jax.vmap(_reinforce)
    reinforce_errors = batch_reinforce_error(action_logits, transitions.a_t, discounted_returns, discount_sequence)
    return jnp.sum(reinforce_errors)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _train_step(self, state: TrainingState, transitions: Transitions):
    # Do one learning step on the batch of transitions
    compute_loss_and_grad = jax.value_and_grad(self._loss)
    loss, dloss_dparams = compute_loss_and_grad(state.params, transitions)
    updates, new_opt_state = self._optimizer.update(dloss_dparams,
                                                    state.opt_state)
    new_params = optax.apply_updates(state.params, updates)

    new_state = TrainingState(
        params=new_params, opt_state=new_opt_state, step=state.step + 1)
    return new_state, loss
  def update(self):
    if self._done:
      # Collect a minibatch of random transitions
      transitions = Transitions(*self._replay_buffer.sample(self._batch_size))
      # Compute loss and update parameters
      self._state, self.last_loss = self._train_step(self._state, transitions)
      self._replay_buffer = ReplayBuffer(self.replay_capacity)

  def observe_first(self, timestep):
    self._done = False
    self._replay_buffer.push(timestep, None)

  def observe(self, action, next_timestep):
    self._done = next_timestep.last()
    self._replay_buffer.push(next_timestep, action)


class ReplayBuffer(object):
  """A simple Python replay buffer."""

  def __init__(self, capacity):
    self._prev = None
    self._action = None
    self._latest = None
    self.buffer = collections.deque(maxlen=capacity)

  def push(self, timestep, action):
    self._prev = self._latest
    self._action = action
    self._latest = timestep

    if action is not None:
      self.buffer.append(
          (self._prev.observation, self._action, self._latest.reward,
           self._latest.discount, self._latest.observation))

  def sample(self, batch_size):
    obs_tm1, a_tm1, r_t, discount_t, obs_t = zip(
        *self.buffer)

    return (jnp.stack(obs_tm1), jnp.asarray(a_tm1), jnp.asarray(r_t),
            jnp.asarray(discount_t), jnp.stack(obs_t))

  def is_ready(self, batch_size):
    return batch_size <= len(self.buffer)
