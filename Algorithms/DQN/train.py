import sys
<<<<<<< HEAD
sys.path.append("C:\\Users\\gutop\\Documents\\BIA\\BIA_Semestre_8\\Residencia\\Semana_6\\code\\ray-rllib\\")

import torch
import time
import random
=======
import os
sys.path.append('\\'.join(os.path.abspath(__file__).split('\\')[:-3]))

import torch
import time
import argparse
>>>>>>> 3a230e5cd2b7bfb0609965529f37fb69a8899198
import numpy as np
from collections import deque
from dqn_agent import Agent
from TradingEnv import TradingMarket
import gymnasium as gym
import pandas_datareader.data as pdr
import yfinance
import utils

yfinance.pdr_override()

num_episodes=2000
epsilon=1.0
epsilon_min=0.01
epsilon_decay=0.995
scores = []
diffs = []
scores_average_window = 20               

parser = argparse.ArgumentParser()
parser.add_argument('--ticket', type=str, default='AAPL',
                    help='Ticket from Yahoo Finance')
parser.add_argument('--train_time', type=str, default='2005-01-01/2020-01-01',
                    help='Time range that agent will be trained')
parser.add_argument('--n_eps', type=int, default=2000,
                    help='Number of episodes')
args = parser.parse_args()

num_episodes=args.n_eps         
ticket=args.ticket
dt_ini_train=args.train_time.split('/')[0]
dt_final_train=args.train_time.split('/')[1]

df_train = pdr.get_data_yahoo(ticket, dt_ini_train, dt_final_train)

gym.envs.register(
    id='TradeEnvTrain',
    entry_point=TradingMarket,
    kwargs={
        "env_config": {
            "OHCL":df_train,
            "initial_value": 1000,
            "positions": [0, 1], 
            "window": 15,
            "epi_len":300
        }
    }
)

env = gym.make('TradeEnvTrain')

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

for i_episode in range(1, num_episodes+1):

    state, _ = env.reset()   
    score = 0
    initial_price = env.historical_price.iloc[env.idx, :]["Close"]
    last_price = env.historical_price.iloc[env.idx + 300, :]["Close"]
    baseline = (1000/initial_price)*last_price

    while True:
        action = agent.act(state, epsilon)             
        next_state, reward, done, *_  = env.step(action)
        agent.step(state, action, reward, next_state, done)

        state = next_state
        score += reward

        if done:
            break
    
    scores.append(score)
    average_score = np.mean(scores[i_episode-min(i_episode,scores_average_window):i_episode+1])

    epsilon = max(epsilon_min, epsilon_decay*epsilon)
    patrimony = env.valorisation(last_price)
    diffs.append(patrimony-baseline)
    average_diff = np.mean(diffs[i_episode-min(i_episode,scores_average_window):i_episode+1])
    
    print('\rEpi {}\tAvg Rwd: {:.2f}\tBaseline: {:.2f}\tpat: {:.2f}\tavg diff: {:.2f}'.format(i_episode, average_score, baseline, patrimony, average_diff), end="")
    # Print average score every scores_average_window episodes
    if i_episode % scores_average_window == 0:
        print('\rEpi {}\tAvg Rwd: {:.2f}\tBaseline: {:.2f}\tpat: {:.2f}\tavg diff: {:.2f}'.format(i_episode, average_score, baseline, patrimony, average_diff))
    

timestr = time.strftime("%Y%m%d-%H%M%S")
nn_filename = "Algorithms/DQN/results/dqnAgent_Trained_Model_" + ticket + "-" + timestr + ".pth"
torch.save(agent.network.state_dict(), nn_filename)

# Save the recorded Scores data
scores_filename = "Algorithms/DQN/results/dqnAgent_scores_" + ticket + "-" + timestr + ".csv"
np.savetxt(scores_filename, scores, delimiter=",")