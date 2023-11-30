from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.algorithm import Algorithm

from TradingEnv import TradingMarket, log_return
import os

import gymnasium as gym
from ray import air, tune

from gymnasium import spaces
import numpy as np

import pandas as pd
import pandas_datareader.data as pdr
import yfinance

yfinance.pdr_override()

data_inicial = "2000-01-01"
data_final = "2020-01-01"
df_train = pdr.get_data_yahoo("BBAS3.SA", data_inicial, data_final)

data_inicial = "2020-01-01"
data_final = None
df_test = pdr.get_data_yahoo("BBAS3.SA", data_inicial, data_final)

config = {
    "env": TradingMarket,
    "env_config": {
        "data":df_train
    },
    "num_workers": 7
}

stop_criteria = {
    "timesteps_total": 200000
}

tuner = tune.Tuner(
    "PPO",
    param_space=config,
    run_config=air.RunConfig(
        stop=stop_criteria,
        verbose=2,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=1, checkpoint_at_end=True
        ),
    ),
)
results = tuner.fit()

checkpoint = results.get_best_result().checkpoint
# Create new Algorithm and restore its state from the last checkpoint.
algo = Algorithm.from_checkpoint(checkpoint)

gym.envs.register(
     id='TradeEnv-v0',
     entry_point=TradingMarket,
     kwargs={
         "data": {"data":df_test},
         "initial_value": 1000,
         "positions": [1, 0],
         "reward_func": log_return, 
         "window":15,
         "dt_fim":"2023-11-29", 
         "epi_len":df_test.shape[0]-16
     }
)

env_test = gym.make('TradeEnv-v0')
obs, info = env_test.reset(seed=42)
rewards = []
done = False
while not done:

    action = algo.compute_single_action(
            observation=obs,
            explore=False,
            policy_id="default_policy",  # <- default value
        )


    obs, rwd, done, *_ = env_test.step(action)

    rewards.append(env_test)

with open('history_test.txt', 'w') as file:
    file.write(f'{str(env_test.history)}')