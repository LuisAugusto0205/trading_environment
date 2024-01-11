import sys
sys.path.append("C:\\Users\\gutop\\Documents\\BIA\\BIA_Semestre_8\\Residencia\\Semana_6\\code\\ray-rllib\\")

import torch
import time
import random
import numpy as np
from collections import deque
from ddpg_agent import Agent
from TradingEnv_continuos import TradingMarket
import gymnasium as gym
import pandas_datareader.data as pdr
import yfinance
import utils
import matplotlib.pyplot as plt
import argparse

yfinance.pdr_override()

num_episodes=6000
epsilon=1.0
epsilon_min=0.01
epsilon_decay=0.995
scores = []
scores_average_window = 20               

parser = argparse.ArgumentParser()
parser.add_argument('--ticket', type=str, default='AAPL',
                    help='Ticket from Yahoo Finance')
parser.add_argument('--train_time', type=str, default='2005-01-01/2020-01-01',
                    help='Time range that agent will be trained')
parser.add_argument('--n_eps', type=int, default=6000,
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
            "window": 15,
            "epi_len":300
        }
    }
)

env = gym.make('TradeEnvTrain')
action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]
agent = Agent(state_size=state_size, action_size=action_size, random_seed=42)

def ddpg(n_episodes=2000, max_t=300, print_every=20):
    scores_deque = deque(maxlen=print_every)
    scores = []
    diffs = []
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        agent.reset()
        score = 0

        initial_price = env.historical_price.iloc[env.idx, :]["Close"]
        last_price = env.historical_price.iloc[env.idx + 300, :]["Close"]
        baseline = (1000/initial_price)*last_price

        while True:
            action = agent.act(state)
            next_state, reward, done, *_ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores.append(score)
        average_score = np.mean(scores[i_episode-min(i_episode,scores_average_window):i_episode+1])
        
        try:
            patrimony = env.valorisation(last_price)[0]
        except:
            patrimony = env.valorisation(last_price)
        diffs.append(patrimony-baseline)
        average_diff = np.mean(diffs[i_episode-min(i_episode,scores_average_window):i_episode+1])
        
        print('\rEpi {}\tAvg Rwd: {:.2f}\tBaseline: {:.2f}\tpat: {:.2f}\tavg diff: {:.2f}'.format(i_episode, average_score, baseline, patrimony, average_diff), end="")
        # Print average score every scores_average_window episodes
        if i_episode % scores_average_window == 0:
            print('\rEpi {}\tAvg Rwd: {:.2f}\tBaseline: {:.2f}\tpat: {:.2f}\tavg diff: {:.2f}'.format(i_episode, average_score, baseline, patrimony, average_diff))
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    nn_filename_actor = "Algorithms/DDPG/results/ddpg_actor_" + timestr + ".pth"
    nn_filename_critic = "Algorithms/DDPG/results/ddpg_critic_" + timestr + ".pth"
    torch.save(agent.actor_local.state_dict(), nn_filename_actor)
    torch.save(agent.critic_local.state_dict(), nn_filename_critic)

    scores_filename = "Algorithms/DDPG/results/ddpgAgent_scores_" + timestr + ".csv"
    np.savetxt(scores_filename, scores, delimiter=",")

    scores_filename = "Algorithms/DDPG/results/ddpgAgent_diffs_" + timestr + ".csv"
    np.savetxt(scores_filename, diffs, delimiter=",")

    return scores


scores = ddpg(n_episodes=num_episodes)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()