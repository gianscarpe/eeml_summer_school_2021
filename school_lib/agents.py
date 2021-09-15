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


class MCAgentUpdateAll(acme.Actor):
  def __init__(
      self, number_of_states, number_of_actions, behaviour_policy, 
      num_offline_updates=0, step_size=0.1): 
        self._q = np.zeros((number_of_states, number_of_actions))
        self._behaviour_policy = behaviour_policy
        self._replay_buffer = []
        self._state = None
        self._action = None
        self._next_state = None
        self._replay_buffer = []
        self.returns = {}

  @property
  def q_values(self):
    return self._q

  def select_action(self, observation):
    return self._behaviour_policy(self._q[observation])


  def observe_first(self, timestep):
    """The agent is being notified that environment was reset."""
        self._state = timestep.observation
        self._replay_buffer = [self._state]

  def observe(self, action, next_timestep):
    """The agent is being notified of an environment step."""
        s = self._state
        a = action
        r = next_timestep.reward
        g = next_timestep.discount
        next_s = next_timestep.observation

        # Offline Q-value update
        self._action = a
        self._next_state = next_s

        self._replay_buffer.append((s, a, r, g, next_s))

  def update(self):    
    """Agent should update its parameters."""

        for s, a, r, g, next_s in reverse(self._replay_buffer):
            if (s, a) not in returns:
                self.returns[(s, a)] = []
            self.returns[(s, a)].append(g)
        for (s, a) in returns.keys()
            self._q[s, a] = avg(self.returns[(s,a)])
            
class PolicyEvalAgent(acme.Actor):

  def __init__(
      self, number_of_states, number_of_actions, 
      evaluated_policy, 
      behaviour_policy=random_policy, 
      step_size=0.1):
    self._state = None
    self._number_of_states = number_of_states
    self._number_of_actions = number_of_actions
    self._step_size = step_size
    self._behaviour_policy = behaviour_policy
    self._evaluated_policy = evaluated_policy
    
    self._q = np.zeros((number_of_states, number_of_actions))
    self._action = None
    self._next_state = None

  @property
  def q_values(self):
    return self._q

  def select_action(self, observation):
    return self._behaviour_policy(self._q[observation])
    
  def observe_first(self, timestep):
    self._state = timestep.observation

  def observe(self, action, next_timestep):
    s = self._state
    a = action
    r = next_timestep.reward
    g = next_timestep.discount
    next_s = next_timestep.observation
    
    # Compute TD-Error.
    self._action = a
    self._next_state = next_s
    
    next_a = self._evaluated_policy(self._q[next_s])
    self._td_error = r + g * self._q[next_s, next_a] - self._q[s, a]
    
    
  def update(self):
    # Q-value table update.
    s = self._state
    a = self._action
    self._q[s, a] += self._step_size * self._td_error
    self._state = self._next_state