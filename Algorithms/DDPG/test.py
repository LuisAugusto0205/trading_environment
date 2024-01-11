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
import argparse

yfinance.pdr_override()

num_episodes=1000

parser = argparse.ArgumentParser()
parser.add_argument('--ticket', type=str, default='AAPL',
                    help='Ticket from Yahoo Finance')
parser.add_argument('--test_time', type=str, default='2020-01-01/2024-01-01',
                    help='Time range that agent will be trained')
parser.add_argument('--save_critic', type=str, default='',
                    help='Path to Saved critic model')
parser.add_argument('--save_actor', type=str, default='',
                    help='Path to Saved actor model')

args = parser.parse_args()
path_save_critic = args.save_critic
path_save_actor = args.save_actor
         
ticket=args.ticket
dt_ini_test=args.test_time.split('/')[0]
dt_final_test=args.test_time.split('/')[1]   

df_test = pdr.get_data_yahoo(ticket, dt_ini_test, dt_final_test)


gym.envs.register(
    id='TradeEnvTest',
    entry_point=TradingMarket,
    kwargs={
        "env_config": {
            "OHCL": df_test,
            "initial_value": 1000,
            "positions": [0, 1], 
            "window": 15,
            "epi_len":300
        }
    }
)

env = gym.make('TradeEnvTest')

action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]
agent = Agent(state_size=state_size, action_size=action_size, random_seed=42)

# Load trained model weights
agent.actor_local.load_state_dict(torch.load(f'Algorithms\DDPG\\results\{path_save_actor}.pth'))
agent.critic_local.load_state_dict(torch.load(f'Algorithms\DDPG\\results\{path_save_critic}.pth'))

diffs = []
scores = []
print(num_episodes)
for i_episode in range(1, num_episodes+1):

    state, _ = env.reset()     

    score = 0

    done = False
    while not done:
        # determine epsilon-greedy action from current sate
        action = agent.act(state)          
        next_state, reward, done, *_  = env.step(action)   

        state = next_state
        score += reward
    
    
    init_price = env.history[0][3]
    end_price = env.history[-1][3]
    baseline = (1000/init_price) * end_price
    patrimony = env.history[-1][1][0]

    diff = patrimony - baseline
    diffs.append(diff)
    scores.append(score)
    print('\rEpisode {}\tScore: {:.2f}\tbaseline: {:.2f}\tpatrimony: {:.2f}\tdiff: {:.2f}'.format(i_episode, score, baseline, patrimony, diff), end="")

diffs = np.array(diffs)
scores = np.array(scores)
n_pos_diff = (diffs > 0).sum()
n_neg_diff = num_episodes - n_pos_diff
print(f'\n\npositive diff: {n_pos_diff}\nnegative diff: {n_neg_diff}\n Avg diff: {diffs.mean()}\n Std diff: {diffs.std()}\n Avg rwd: {scores.mean()}\n Std rwd: {scores.std()}')

with open("log_ddpg.txt", "w") as file:
    file.write(str(env.history))
