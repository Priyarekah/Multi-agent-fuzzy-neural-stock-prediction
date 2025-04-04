import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces
from collections import deque
import random

# GAN-Based Market Simulation (WGAN)
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Deep Reinforcement Learning (DDPG) for Hedging
class HedgingEnv(gym.Env):
    def __init__(self, data):
        super(HedgingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32)
    
    def reset(self):
        self.current_step = 0
        return self.data[self.current_step]
    
    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
            reward = 0
        else:
            done = False
            reward = -abs(action - self.data[self.current_step][-1])  # Reward is inverse of hedging error
        return self.data[self.current_step], reward, done, {}

class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
    
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        return self.actor(state).detach().numpy()
    
    def train(self, transitions):
        for state, action, reward, next_state in transitions:
            target = reward + 0.99 * self.critic(torch.cat([torch.tensor(next_state, dtype=torch.float32), self.actor(torch.tensor(next_state, dtype=torch.float32))]))
            loss = (self.critic(torch.cat([torch.tensor(state, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)])) - target.detach()).pow(2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# Portfolio Allocation Strategy (Mean-Variance Optimization)
def optimize_portfolio(returns, risk_aversion=0.1):
    cov_matrix = np.cov(returns, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    weights = np.dot(inv_cov_matrix, np.ones(len(returns[0])))
    weights /= np.sum(weights)
    return weights

# Example Usage
if __name__ == "__main__":
    synthetic_data = np.random.randn(1000, 5)  # Simulated market data
    print (synthetic_data)
    env = HedgingEnv(synthetic_data)
    agent = DDPGAgent(state_dim=5, action_dim=1)
    portfolio_weights = optimize_portfolio(synthetic_data)
    print("Optimized Portfolio Weights:", portfolio_weights)

