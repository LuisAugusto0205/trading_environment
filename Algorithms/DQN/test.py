import sys
sys.path.append("C:\\Users\\gutop\\Documents\\BIA\\BIA_Semestre_8\\Residencia\\Semana_6\\code\\ray-rllib\\")

import torch
import time
import random
import numpy as np
from dqn_agent import Agent
from TradingEnv import TradingMarket
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

action_size = env.action_space.n
state_size = env.observation_space.shape[0]
agent = Agent(
    state_size=state_size, 
    action_size=action_size, 
    dqn_type='DQN',
    replay_memory_size=int(5*1e5),
    batch_size=128,
    gamma=0.99,
    target_tau=1e-3,
    update_rate=4,
    learning_rate=1e-4
)

# Load trained model weights
agent.network.load_state_dict(torch.load('dqnAgent_Trained_Model_20231220-014003_return.pth'))


for i_episode in range(1, num_episodes+1):

    state, _ = env.reset()     

    score = 0

    while True:
        # determine epsilon-greedy action from current sate
        action = agent.act(state)             
        next_state, reward, done, *_  = env.step(action)   

        state = next_state
        score += reward
        if done:
            break

    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, score), end="")

with open("log_dqn_return.txt", "w") as file:
    file.write(str(env.history))
