
import haiku as hk

import numpy as np

from world import build_gridworld_task, ObservationType, setup_environment
from models.reinforce import ReinforceAgent
from tools import run_loop
import logging

if __name__ == "__main__":
    epsilon = 1. #@param {type:"number"}
    num_episodes = 100 #@param {type:"number"}

    max_episode_length = 500

    # Environment
    grid = build_gridworld_task(
        task='simple',
        observation_type=ObservationType.AGENT_GOAL_POS,
        max_episode_length=max_episode_length)
    environment, environment_spec = setup_environment(grid)


    def pi_network(observation: np.ndarray):
      """Outputs action values given an observation."""
      model = hk.Sequential([
          hk.Flatten(),  # Flattens everything except the batch dimension
          hk.nets.MLP([50, 50, environment_spec.actions.num_values]),          
      ])
      return model(observation)

    # Build the trainable Q-learning agent
    agent = ReinforceAgent(
        pi_network,
        environment_spec.observations,
        epsilon=epsilon,
        replay_capacity=100000,
        batch_size=10,
        learning_rate=1e-3)

    returns = run_loop(
        environment=environment,
        agent=agent,
        num_episodes=num_episodes,
        logger_time_delta=1.,
        log_loss=True)

    print(returns)
    logging.info(returns)
