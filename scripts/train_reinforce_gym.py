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
from world import setup_environment
import models
from tools import run_loop

if __name__ == "__main__":
    epsilon = 1. #@param {type:"number"}
    num_episodes = 3 #@param {type:"number"}

    max_episode_length = 200
    env = gym_wrapper.GymWrapper(gym.make('CartPole-v0'))
    env = wrappers.SinglePrecisionWrapper(env)

    environment, environment_spec = setup_environment(env)


    def pi_network(observation: np.ndarray):
      """Outputs action values given an observation."""
      model = hk.Sequential([
          hk.Flatten(),  # Flattens everything except the batch dimension
          hk.nets.MLP([50, 50, environment_spec.actions.num_values]),          
      ])
      return model(observation)


    # Build the trainable Q-learning agent
    agent = models.reinforce.ReinforceAgent(
        pi_network,
        environment_spec.observations,
        epsilon=epsilon,
        replay_capacity=100000,
        batch_size=10,
        learning_rate=1e-3)

    returns = run_loop(environment=environment, agent=agent, num_episodes=300, 
             logger_time_delta=0.2)

    print("Completed")
    returns_avg = pd.Series(returns).rolling(10, center=True).mean()

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(returns)), returns_avg)
    plt.xlabel('Episodes')
    plt.ylabel('Total reward')
    plt.show()
