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


if __name__ == '__main__': 
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

    env_id = "TradeEnvTrain"
    env = gym.make(env_id)

    model = ARS("LinearPolicy", env, verbose=1, device='cuda', tensorboard_log='.\\Algorithms\\ARS\\logs')
    model.learn(total_timesteps=300*num_episodes, log_interval=4, progress_bar=True)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    model.save(f"Algorithms/ARS/results/TradingEnv-{ticket.replace('.', '-')}-{timestr}")