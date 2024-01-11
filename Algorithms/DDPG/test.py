import sys
sys.path.append("C:\\Users\\gutop\\Documents\\BIA\\BIA_Semestre_8\\Residencia\\Semana_6\\code\\ray-rllib\\")

import torch
import time
import random
import numpy as np
from ddpg_agent import Agent
from TradingEnv_continuos import TradingMarket
import gymnasium as gym
import pandas_datareader.data as pdr
import yfinance

yfinance.pdr_override()

num_episodes=1
dt_ini_test = "2018-01-01"
dt_final_test = "2023-12-18"             
df_test = pdr.get_data_yahoo("AAPL", dt_ini_test, dt_final_test)

gym.envs.register(
    id='TradeEnvTest',
    entry_point=TradingMarket,
    kwargs={
        "env_config": {
            "OHCL": df_test,
            "initial_value": 1000,
            "positions": [0, 1], 
            "window": 15,
            "epi_len":df_test.shape[0]-40-1
        }
    }
)

env = gym.make('TradeEnvTest')

action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]
agent = Agent(state_size=state_size, action_size=action_size, random_seed=42)

# Load trained model weights
agent.actor_local.load_state_dict(torch.load('ddpg_actor_20231220-220647.pth'))
agent.critic_local.load_state_dict(torch.load('ddpg_critic_20231220-220647.pth'))

for i_episode in range(1, num_episodes+1):

    state, _ = env.reset()     

    score = 0

    while True:
        action = agent.act(state)             
        next_state, reward, done, *_  = env.step(action)   

        state = next_state
        score += reward
        if done:
            break

    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, score), end="")

with open("log_ddpg.txt", "w") as file:
    file.write(str(env.history))
