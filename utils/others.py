import torch
import numpy as np

def evaluate(agent, env):

    obs, _ = env.reset(seed=42)
    rewards = []
    done = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    while not done:
        obs = torch.FloatTensor([obs]).to(device)
        with torch.no_grad(): 
            action = torch.argmax(agent(obs)).view(1, 1)
        obs, rwd, done, *_ = env.step(action)

        rewards.append(rwd)

    results_patrimony = np.array([day[1] for day in env.history])
    results_date = np.array([day[0] for day in env.history])
    results_act = np.array([day[2] for day in env.history])
    return results_patrimony[-1], np.mean(rewards), np.sum(results_act)