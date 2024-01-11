import sys
import os
sys.path.append('\\'.join(os.path.abspath(__file__).split('\\')[:-3]))

import time
import torch
import numpy as np
import argparse
from TradingEnv import TradingMarket, SimpleTradingEnv
import gymnasium as gym
import pandas_datareader.data as pdr
import yfinance
import pickle

yfinance.pdr_override()

num_episodes=1000

parser = argparse.ArgumentParser()
parser.add_argument('--ticket', type=str, default='AAPL',
                    help='Ticket from Yahoo Finance')
parser.add_argument('--test_time', type=str, default='2020-01-01/2024-01-01',
                    help='Time range that agent will be trained')
parser.add_argument('--save', type=str, default='',
                    help='Path to Saved model')

args = parser.parse_args()
         
ticket=args.ticket
dt_ini_test=args.test_time.split('/')[0]
dt_final_test=args.test_time.split('/')[1] 
path_save = args.save  

df_test = pdr.get_data_yahoo(ticket, dt_ini_test, dt_final_test)

gym.envs.register(
    id='TradeEnvTest',
    entry_point=SimpleTradingEnv,
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


with open(f'Algorithms/Q-Learning/results/{path_save}.pkl', 'rb') as file:
    Q = pickle.load(file)
print(len(Q))
diffs = []
scores =[]
for i_episode in range(1, num_episodes+1):

    state, _ = env.reset()   

    score = 0

    done = False
    while not done:
        # determine epsilon-greedy action from current sate
        state = ''.join([str(int(x)) for x in state])
        try: 
            action = np.argmax(Q[state]) 
        except:
            Q[state] = [0, 0]
            action = np.argmax(Q[state]) 
        next_state, reward, done, *_  = env.step(action)   

        state = next_state
        score += reward
    
    scores.append(score)
    
    init_price = env.history[0][3]
    end_price = env.history[-1][3]
    baseline = (1000/init_price) * end_price
    patrimony = env.history[-1][1]

    diff = patrimony - baseline
    diffs.append(diff)
    print('\rEpisode {}\tScore: {:.2f}\tbaseline: {:.2f}\tpatrimony: {:.2f}\tdiff: {:.2f}'.format(i_episode, score, baseline, patrimony, diff), end="")

diffs = np.array(diffs)
scores = np.array(scores)
n_pos_diff = (diffs > 0).sum()
n_neg_diff = num_episodes - n_pos_diff
print(f'\n\npositive diff: {n_pos_diff}\nnegative diff: {n_neg_diff}\n Avg diff: {diffs.mean()}\n Std diff: {diffs.std()}\n Avg rwd: {scores.mean()}\n Std rwd: {scores.std()}')

timestr = time.strftime("%Y%m%d-%H%M%S")
with open(f"Algorithms\\Q-Learning\\results\\log_dqn_return-{ticket}-{timestr}.txt", "w") as file:
    file.write(str(env.history))