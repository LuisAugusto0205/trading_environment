import sys
import os
sys.path.append('\\'.join(os.path.abspath(__file__).split('\\')[:-3]))

import numpy as np
from TradingEnv import SimpleTradingEnv
import gymnasium as gym

import pandas_datareader.data as pdr
import yfinance

import pickle
import time

yfinance.pdr_override()

df_train = pdr.get_data_yahoo("AAPL", "2005-01-01", "2022-01-01")
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

alpha = 0.1
gamma = 0.99
Q={}
scores=[]
diffs = []

def select_action(env, Q, obs, e):
    if np.random.rand(1, 1)[0, 0] > e:
        return np.argmax(Q[obs])
    else:
        return env.action_space.sample()

e=1
for ep in range(4000):
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
        action = select_action(env, Q, obs, e)
        new_obs, rwd, done, *_ = env.step(action)
        new_obs = ''.join([str(int(e)) for e in new_obs])
        score += rwd

        if new_obs not in Q.keys():
            Q[new_obs] = [0, 0]
        
        max_action = np.argmax(Q[new_obs])
        Q[obs][action] += alpha*(rwd + gamma*Q[new_obs][max_action] - Q[obs][action])
        obs = new_obs
    e = max(0.995*e, 0.1)

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