import sys
import os
sys.path.append('\\'.join(os.path.abspath(__file__).split('\\')[:-3]))

from stable_baselines3 import A2C
import argparse
import time
import numpy as np
from TradingEnv import TradingMarket
import gymnasium as gym
import pandas_datareader.data as pdr
import yfinance

yfinance.pdr_override()

num_episodes=1

parser = argparse.ArgumentParser()
parser.add_argument('--ticket', type=str, default='AAPL',
                    help='Ticket from Yahoo Finance')
parser.add_argument('--test_time', type=str, default='2020-01-01/2024-01-01',
                    help='Time range that agent will be trained')

args = parser.parse_args()
         
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
            "epi_len":df_test.shape[0]-40-1
        }
    }
)

env = gym.make('TradeEnvTest')
agent = A2C.load('Algorithms/A2C/results/TradingEnv-AAPL-20240105-111553')

state, _ = env.reset()     

score = 0

while True:
    action = agent.predict(state)             
    next_state, reward, done, *_  = env.step(action)   

    state = next_state
    score += reward
    if done:
        break

print('\tScore: {:.2f}'.format(score), end="")

timestr = time.strftime("%Y%m%d-%H%M%S")
with open(f"Algorithms\\A2C\\results\\log_a2c_return-{ticket.replace('.', '-')}-{timestr}.txt", "w") as file:
    file.write(str(env.history))
