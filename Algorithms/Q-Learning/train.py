import sys
import os
sys.path.append('\\'.join(os.path.abspath(__file__).split('\\')[:-3]))

import numpy as np
import pandas as pd
from TradingEnv import SimpleTradingEnv
import gymnasium as gym

import pandas_datareader.data as pdr
import yfinance

import pickle
import time
import argparse

yfinance.pdr_override()

parser = argparse.ArgumentParser()
parser.add_argument('--ticket', type=str, default='AAPL',
                    help='Ticket from Yahoo Finance')
parser.add_argument('--train_time', type=str, default='2020-01-01/2024-01-01',
                    help='Time range that agent will be trained')
parser.add_argument('--n_eps', type=int, default=6000,
                    help='Time range that agent will be trained')

args = parser.parse_args()
         
ticket=args.ticket
dt_ini_train=args.train_time.split('/')[0]
dt_final_train=args.train_time.split('/')[1]
num_episodes = args.n_eps

df_train = pdr.get_data_yahoo(ticket, dt_ini_train, dt_final_train)
gym.envs.register(
    id='SimpleTradeEnvTrain',
    entry_point=SimpleTradingEnv,
    kwargs={
        "env_config": {
            "OHCL": df_train,
            "initial_value": 1000,
            "positions": [0, 1], 
            "window": 15,
            "epi_len":300
        }
    }
)

env = gym.make('SimpleTradeEnvTrain')

alpha = 0.3
gamma = 0.99
Q={}
scores=[]
diffs = []

def select_action(env, Q, obs, e):
    if np.random.rand(1, 1)[0, 0] > e:
        return np.argmax(Q[obs])
    else:
        return env.action_space.sample()

for ep in range(num_episodes):
    obs, _ = env.reset()
    obs = ''.join([str(int(e)) for e in obs])
    done = False
    score = 0

    initial_price = env.historical_price.iloc[env.idx, :]["Close"]
    last_price = env.historical_price.iloc[env.idx + 300, :]["Close"]
    baseline = (1000/initial_price)*last_price

    if obs not in Q.keys():
        Q[obs] = [0, 0]
    
    while not done:
        action = np.argmax(Q[obs])
        new_obs, rwd, done, *_ = env.step(action)
        new_obs = ''.join([str(int(e)) for e in new_obs])
        score += rwd

        if new_obs not in Q.keys():
            Q[new_obs] = [0, 0]
        
        max_action = np.argmax(Q[new_obs])
        Q[obs][action] += alpha*(rwd + gamma*Q[new_obs][max_action] - Q[obs][action])
        obs = new_obs

    scores.append(score)
    patrimony = env.valorisation(last_price)
    diffs.append(patrimony-baseline)
    avg_score = np.mean(scores[ep-min(20, ep):ep+1])
    avg_diff = np.mean(diffs[ep-min(ep,20):ep+1])

    print('\rEpi {}\tAvg Rwd: {:.2f}\tBaseline: {:.2f}\tpat: {:.2f}\tavg diff: {:.2f}'.format(ep, avg_score, baseline, patrimony, avg_diff), end="")
    if ep % 20 == 0:
        print('\rEpi {}\tAvg Rwd: {:.2f}\tBaseline: {:.2f}\tpat: {:.2f}\tavg diff: {:.2f}'.format(ep, avg_score, baseline, patrimony, avg_diff))

timestr = time.strftime("%Y%m%d-%H%M%S")
with open(f'Algorithms/Q-Learning/results/Q-table-{timestr}.pkl', 'wb') as file:
    pickle.dump(Q, file)

df = pd.DataFrame({"Reward": scores, "Diff baseline": diffs})
df.to_csv(f'Algorithms/Q-Learning/results/Q-table-{timestr}.csv', index=False)