from ray import tune
from ray.rllib.algorithms.ppo import PPO
from TradingEnv import TradingMarket
import os

import gymnasium as gym

from gymnasium import spaces
import numpy as np

import pandas as pd
import pandas_datareader.data as pdr
import yfinance

yfinance.pdr_override()

data_inicial = "2000-01-01"
data_final = "2020-01-01"
df = pdr.get_data_yahoo("BBAS3.SA", data_inicial, data_final)
    
config = {
    "env": TradingMarket,  # Passando a classe do ambiente
    # Outras configurações do seu agente PPO...
    "env_config": {
        "data":df
    },
    "checkpoint_freq": 1,
    "checkpoint_at_end": True,
    "num_workers": 7
}

stop_criteria = {
    "timesteps_total": 200000  # Defina o número total de timesteps desejado
}
results_dir = os.path.abspath("results")
analysis = tune.run(PPO, config=config, stop=stop_criteria, local_dir=results_dir)
print(type(analysis))
print(analysis)

best_checkpoint = analysis.get_best_checkpoint(analysis.trials[0], metric="episode_reward_mean", mode="max")
print("Melhor checkpoint:", best_checkpoint)