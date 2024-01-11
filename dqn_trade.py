'''
Jordan Lei, 2020. Some code is based on the following sources:
   https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
'''

import gymnasium as gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import seaborn as sns

import argparse
import os

from TradingEnv import TradingMarket
import pandas_datareader.data as pdr
import yfinance
import utils

yfinance.pdr_override()

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0) #cuda device
parser.add_argument('--verbose', type=int, default=1) #printing preferences
parser.add_argument('--load', type=bool, default = False) #if loading an existing model
parser.add_argument('--save', type=bool, default = False) #if saving an existing model
parser.add_argument('--plot', type=bool, default = True) #if plotting an existing model
parser.add_argument('--model', type=str, default='reinforce_cartpole/model.pt') #model - currently supports resnet and alexnet, with more to come
parser.add_argument('--runtype', type=str, default='train',
                        choices=('train', 'run', 'train_run')) #runtype: train only or train and validate
parser.add_argument('--lr', type=float, default=0.0001)  #learning rate
parser.add_argument('--episodes', type=int, default=500) #number of episodes    
parser.add_argument('--gamma', type=float, default=0.99) #discount factor                                  
args = parser.parse_args()

#virtual display used to satisfy non-screen evironments (e.g. server)
# virtualdisplay = Display(visible=0, size=(1400, 900))
# virtualdisplay.start()

#setup environment

dt_ini_train = "2012-01-01"
dt_final_train = "2017-12-31"
dt_ini_test = "2012-01-01"
dt_final_test = "2019-12-31"
df_train = pdr.get_data_yahoo("AAPL", dt_ini_train, dt_final_train)
df_test = pdr.get_data_yahoo("AAPL", dt_ini_test, dt_final_test)

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

env = gym.make('TradeEnvTrain')

#set the cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"]= "{}".format(args.device)
device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#experience replay
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ExperienceReplay(object): 
  def __init__(self, capacity): 
    self.capacity = capacity
    self.memory = []
    self.position = 0
  
  def push(self, *args): 
    # if memory isn't full, add a new experience
    if len(self.memory) < self.capacity: 
      self.memory.append(None)
    
    self.memory[self.position] = Transition(*args)
    self.position = (self.position + 1) % self.capacity
  
  def sample(self, batch_size): 
    return random.sample(self.memory, batch_size)
  
  def __len__(self): 
    return len(self.memory)


#deep Q network implementation
class DQN(nn.Module): 
  def __init__(self, in_size, out_size): 
    super(DQN, self).__init__()
    self.layer1 = nn.Linear(in_size, 128)
    self.layer2 = nn.Linear(128, 64)
    self.layer3 = nn.Linear(64, out_size)
    self.dropout = nn.Dropout(0.7)
  
  def forward(self, x): 
    # x = Variable(torch.from_numpy(x).float().unsqueeze(0)).to(device) 
    x = F.relu(self.layer1(x))
    x = self.dropout(F.relu(self.layer2(x)))
    x = F.relu(self.layer3(x)) 
    return x

class Runner():
  def __init__(self, dqn, loss, lr = 0.001, eps_start = 0.9, eps_end = 0.1, eps_decay = 200,
               batch_size = 128, target_update = 40, logs = "runs", 
               gamma = 0.995):
    self.writer = SummaryWriter(logs) 
    self.logs = logs
    self.learner = dqn
    self.target = dqn

    self.target.load_state_dict(self.learner.state_dict())
    self.target.eval()

    self.optimizer = optim.Adam(self.learner.parameters(), lr = lr)
    self.loss = loss
    self.memory = ExperienceReplay(100000)
    self.batch_size = batch_size
    self.eps_start = eps_start
    self.eps_end = eps_end
    self.eps_decay = eps_decay
    self.target_update = target_update
    self.gamma = gamma
    self.steps = 0

    self.plots = {"Loss": [], "Reward": [], "Mean Reward": []}
  
  def select_action(self, state, eps_thresh):
    #select an action based on the state
    self.steps = self.steps + 1
    sample = random.random()
    #get a decayed epsilon threshold
    #eps_thresh = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1 * self.steps / self.eps_decay)
    if sample > eps_thresh: 
      with torch.no_grad(): 
        #select the optimal action based on the maximum expected return
        action = torch.argmax(self.learner(state)).view(1, 1)
        return action
    else: 
      return torch.tensor([[random.randrange(env.action_space.n)]], device = device, dtype=torch.long)
    
  def train_inner(self): 
    if len(self.memory) < self.batch_size:
      return 0
    
    sample_transitions = self.memory.sample(self.batch_size)
    batch = Transition(*zip(*sample_transitions))

    #get a list that is True where the next state is not "done" 
    has_next_state = torch.tensor(list(map(lambda s: s is not None, batch.next_state)), device = device, dtype=torch.bool)
    next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    pred_values = self.learner(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(self.batch_size, device = device)
    #get the max along each row using the target network, then detach
    next_state_values[has_next_state] = self.target(next_states).max(1)[0].detach()

    #Q(s, a) = reward(s, a) + Q(s_t+1, a_t+1)* gamma
    target_values = next_state_values * self.gamma + reward_batch

    loss = self.loss(pred_values, target_values.unsqueeze(1))
    
    self.optimizer.zero_grad()
    loss.backward()
    for param in self.learner.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optimizer.step()

    return loss

  def env_step(self, action):
    state, reward, done, *_ = env.step(action)
    return torch.FloatTensor([state]).to(device), torch.FloatTensor([reward]).to(device), done
  
  def train(self, episodes=100, smooth=10): 
    steps = 0 
    smoothed_reward = []
    mean_reward = 0
    eps = self.eps_start
    for episode in range(episodes):
      c_loss = 0
      c_samples = 0
      rewards = 0

      state, *_ = env.reset()
      state = Variable(torch.from_numpy(state).float().unsqueeze(0)).to(device)
      done = False
      i=0
      while not done: 
        i+=1
        action = self.select_action(state, eps_thresh=eps)
        next_state, reward, done = self.env_step(action.item())

        if done:
          next_state = None

        self.memory.push(state, action, next_state, reward)
        state = next_state

        loss = self.train_inner()
        rewards += reward.detach().item()

        if done:
          break
        
        steps += 1
        c_samples += self.batch_size
        c_loss += loss
      eps = max(eps*self.eps_decay, self.eps_end)
      smoothed_reward.append(rewards)
      if len(smoothed_reward) > smooth: 
        smoothed_reward = smoothed_reward[-1*smooth: -1]
      
      self.writer.add_scalar("Loss", c_loss/c_samples, steps) 
      self.writer.add_scalar("Reward", rewards, episode)  
      self.writer.add_scalar("Mean Reward", np.mean(smoothed_reward), episode)

      self.plots["Loss"].append(loss.detach().cpu().numpy() if isinstance(loss, torch.Tensor) else loss)
      self.plots["Reward"].append(rewards)
      self.plots["Mean Reward"].append(np.mean(smoothed_reward))

      if episode % 20 == 0:
        patrimony, rwd, num_act = utils.evaluate(self.learner, gym.make("TradeEnvTest")) 
        with open("log_dqn.txt", "a") as file:
          file.write(f"{patrimony:.2f}\t{rwd:.4f}\t{num_act}\n")
        print("\tEpisode {} \t Final Reward {:.2f} \t Average Reward: {:.2f} \t steps: {}".format(episode, rewards, np.mean(smoothed_reward), steps))
  
      if steps % self.target_update == 0:
        self.target.load_state_dict(self.learner.state_dict())

    env.close()


  def plot(self):
    sns.set()
    sns.set_context("poster")

    plt.figure(figsize=(20, 16))
    plt.plot(np.arange(len(self.plots["Loss"])), self.plots["Loss"])
    plt.title("DQN Gradient Loss")
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    plt.savefig("%s/plot_%s.png"%(self.logs, "loss"))

    plt.figure(figsize=(20, 16))
    plt.plot(np.arange(len(self.plots["Reward"])), self.plots["Reward"], label="Reward")
    plt.plot(np.arange(len(self.plots["Mean Reward"])), self.plots["Mean Reward"], label = "Mean Reward")
    plt.legend()
    plt.title("DQN Gradient Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.savefig("%s/plot_%s.png"%(self.logs, "rewards"))

    # for key in self.plots.keys():
    #     data = self.plots[key]
    #     plt.figure(figsize=(20, 16))
    #     plt.plot(np.arange(len(data)), data)
    #     plt.title("Policy Gradient %s"%key)
    #     plt.xlabel("Episodes")
    #     plt.ylabel(key)
    #     plt.savefig("%s/plot_%s.png"%(self.logs, key))

  def save(self): 
    torch.save(self.learner.state_dict(),'%s/model.pt'%self.logs)

def main(): 
    device_name = "cuda: %s"%(args.device) if torch.cuda.is_available() else "cpu"
    print("[Device]\tDevice selected: ", device_name)

    dqn = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    
    #if we're loading a model
    if args.load: 
        dqn.load_state_dict(torch.load(args.model))

    loss = nn.MSELoss()
    runner = Runner(dqn, loss, lr = args.lr, eps_decay=0.995, gamma = args.gamma, logs = "dqn_trade/%s" %time.time())
    
    if "train" in args.runtype:
        print("[Train]\tTraining Beginning ...")
        runner.train(args.episodes)

        if args.plot:
            print("[Plot]\tPlotting Training Curves ...")
            runner.plot()

    if args.save: 
        print("[Save]\tSaving Model ...")
        runner.save()
    

    print("[End]\tDone. Congratulations!")

if __name__ == '__main__':
    main()