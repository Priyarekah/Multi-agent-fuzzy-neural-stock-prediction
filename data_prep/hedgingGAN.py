## 1. Import Required Libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from stable_baselines3 import DDPG, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.data import DataLoader, TensorDataset

## 2. Define GAN-Based Market Simulator
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
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

## 3. Train GAN to Generate Market Data
def train_gan(real_data, epochs=1000, batch_size=32):
    input_dim = real_data.shape[1]
    generator = Generator(input_dim, input_dim)
    discriminator = Discriminator(input_dim)
    
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)
    
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)
    
    for epoch in range(epochs):
        idx = np.random.randint(0, real_data.shape[0], batch_size)
        real_samples = torch.tensor(real_data[idx], dtype=torch.float32)
        
        noise = torch.randn(batch_size, input_dim)
        fake_samples = generator(noise)
        
        # Train Discriminator
        optimizer_d.zero_grad()
        real_loss = criterion(discriminator(real_samples), real_labels)
        fake_loss = criterion(discriminator(fake_samples.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d.step()
        
        # Train Generator
        optimizer_g.zero_grad()
        g_loss = criterion(discriminator(fake_samples), real_labels)
        g_loss.backward()
        optimizer_g.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
    
    return generator

## 4. Define Reinforcement Learning Environment for Hedging
class HedgingEnv(gym.Env):
    def __init__(self, price_data, option_data):
        super(HedgingEnv, self).__init__()
        self.price_data = price_data
        self.option_data = option_data
        self.current_step = 0
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(price_data.shape[1],), dtype=np.float32)
    
    def reset(self):
        self.current_step = 0
        return self.price_data[self.current_step]
    
    def step(self, action):
        self.current_step += 1
        reward = -abs(action * self.option_data[self.current_step])  # Minimize hedging loss
        done = self.current_step >= len(self.price_data) - 1
        return self.price_data[self.current_step], reward, done, {}

## 5. Train DRL Model (A2C/DDPG) for Dynamic Hedging
def train_drl(env, model_type='DDPG', timesteps=10000):
    env = DummyVecEnv([lambda: env])
    if model_type == 'DDPG':
        model = DDPG("MlpPolicy", env, verbose=1)
    else:
        model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    return model

## 6. Evaluate the Model
def evaluate_model(model, env, episodes=10):
    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        print(f"Episode {ep + 1}: Total Reward: {total_reward}")

## 7. Run the Complete Pipeline
if __name__ == "__main__":
    real_market_data = np.random.rand(1000, 5)  # Dummy stock price data
    generator = train_gan(real_market_data)
    
    synthetic_data = generator(torch.randn(1000, 5)).detach().numpy()
    option_data = np.random.rand(1000)  # Dummy option pricing data
    env = HedgingEnv(synthetic_data, option_data)
    model = train_drl(env, model_type='DDPG')
    evaluate_model(model, env)





import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gym

def create_env():
    class HedgingEnv(gym.Env):
        def __init__(self):
            super(HedgingEnv, self).__init__()
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            self.current_price = 100
            self.time_to_maturity = 1.0
            self.volatility = 0.2
            self.dt = 1/252
            self.hedge_position = 0
        
        def step(self, action):
            self.hedge_position = action[0]
            dW = np.random.normal(0, np.sqrt(self.dt))
            self.current_price *= np.exp((self.volatility**2 / 2) * self.dt + self.volatility * dW)
            option_value = max(self.current_price - 100, 0)
            hedging_cost = abs(self.hedge_position) * 0.01  # Transaction cost
            reward = -abs(option_value - self.hedge_position * self.current_price) - hedging_cost
            self.time_to_maturity -= self.dt
            done = self.time_to_maturity <= 0
            return np.array([self.current_price, self.time_to_maturity, self.hedge_position], dtype=np.float32), reward, done, {}
        
        def reset(self):
            self.current_price = 100
            self.time_to_maturity = 1.0
            self.hedge_position = 0
            return np.array([self.current_price, self.time_to_maturity, self.hedge_position], dtype=np.float32)
    
    return HedgingEnv()

env = create_env()

import stable_baselines3 as sb3
from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    if done:
        break