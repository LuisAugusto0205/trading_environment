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

yfinance.pdr_override()

num_episodes=2000
epsilon=1.0
epsilon_min=0.01
epsilon_decay=0.995
scores = []
scores_average_window = 20               

dt_ini_train = "2012-01-01"
dt_final_train = "2017-12-31"
df_train = pdr.get_data_yahoo("AAPL", dt_ini_train, dt_final_train)

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
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        agent.reset()
        score = 0
        while True:
            action = agent.act(state)
            next_state, reward, done, *_ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    nn_filename_actor = "Algorithms/DDPG/results/ddpg_actor_" + timestr + ".pth"
    nn_filename_critic = "Algorithms/DDPG/results/ddpg_critic_" + timestr + ".pth"
    torch.save(agent.actor_local.state_dict(), nn_filename_actor)
    torch.save(agent.critic_local.state_dict(), nn_filename_critic)

    scores_filename = "Algorithms/DDPG/results/ddpgAgent_scores_" + timestr + ".csv"
    np.savetxt(scores_filename, scores, delimiter=",")

    return scores


scores = ddpg()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()