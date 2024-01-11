import sys
import os
sys.path.append('\\'.join(os.path.abspath(__file__).split('\\')[:-3]))

from sb3_contrib import ARS
import argparse
import time
import numpy as np
from TradingEnv import TradingMarket
import gymnasium as gym
import pandas_datareader.data as pdr
import yfinance

yfinance.pdr_override()

parser = argparse.ArgumentParser()
parser.add_argument('--ticket', type=str, default='AAPL',
                    help='Ticket from Yahoo Finance')
parser.add_argument('--test_time', type=str, default='2020-01-01/2024-01-01',
                    help='Time range that agent will be trained')

parser.add_argument('--save', type=str, default='',
                    help='Path to Saved model')

args = parser.parse_args()
path_save = args.save
num_episodes = 1000

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
agent = ARS.load(f'Algorithms/ARS/results/{path_save}')

diffs = []
scores = []
print(num_episodes)
for i_episode in range(num_episodes):
    state, _ = env.reset()     

    score = 0

    while True:
        action = agent.predict(state)             
        next_state, reward, done, *_  = env.step(action)   

        state = next_state
        score += reward
        if done:
            break


    init_price = env.history[0][3]
    end_price = env.history[-1][3]
    baseline = (1000/init_price) * end_price
    patrimony = env.history[-1][1][0]
    diff = patrimony - baseline
    diffs.append(diff)
    scores.append(score)
    print('\rEp: {}\tAverage Score: {:.2f}\tbaseline: {:.2f}\tpatrimony: {:.2f}\tdiff: {:.2f}'.format(i_episode, score, baseline, patrimony, diff), end="")

diffs = np.array(diffs)
scores = np.array(scores)
n_pos_diff = (diffs > 0).sum()
n_neg_diff = num_episodes - n_pos_diff
print(f'\n\npositive diff: {n_pos_diff}\nnegative diff: {n_neg_diff}\n Avg diff: {diffs.mean()}\n Std diff: {diffs.std()}\n Avg rwd: {scores.mean()}\n Std rwd: {scores.std()}')

timestr = time.strftime("%Y%m%d-%H%M%S")
with open(f"Algorithms\\ARS\\results\\log_ars_return-{ticket.replace('.', '-')}-{timestr}.txt", "w") as file:
    file.write(str(env.history))
