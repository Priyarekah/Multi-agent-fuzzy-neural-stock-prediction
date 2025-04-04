import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium import spaces
import yfinance as yf

# Configuration
CSV_PATH = "/home/priya/Desktop/fyp/alwin/filtered_singapore_stock_metrics.csv"
TICKERS = pd.read_csv(CSV_PATH)["Stock"].tolist()
START_DATE = "2020-01-01"
END_DATE = "2023-01-01"
INITIAL_BALANCE = 100000
COMMISSION = 0.001

# Load stock metrics
print("Loading stock metrics...")
metrics_df = pd.read_csv(CSV_PATH)
metrics_df.set_index('Stock', inplace=True)
print("Stock metrics loaded. Preview:\n", metrics_df.head())

# Data Loading Function
def load_data(tickers):
    print("Downloading price data for tickers:", tickers)
    data = yf.download(tickers, start=START_DATE, end=END_DATE)
    if "Close" in data:
        data = data["Close"]
    else:
        print("Warning: 'Close' column not found. Available columns:", data.columns)
    return data

price_data = load_data(TICKERS)
returns = price_data.pct_change().dropna()
print("Price data and returns computed.")

# Portfolio Optimization Functions
def minimum_variance_portfolio(returns):
    cov_matrix = returns.cov()
    inv_cov = np.linalg.pinv(cov_matrix)
    ones = np.ones(len(inv_cov))
    weights = inv_cov @ ones / (ones.T @ inv_cov @ ones)
    return weights / weights.sum()

def optimized_mvo(returns, l2_penalty=0.1):
    cov_matrix = returns.cov()
    identity_matrix = np.eye(cov_matrix.shape[0])
    regularized_cov = cov_matrix + l2_penalty * identity_matrix
    inv_cov = np.linalg.pinv(regularized_cov)
    ones = np.ones(len(inv_cov))
    weights = inv_cov @ ones / (ones.T @ inv_cov @ ones)
    return weights / weights.sum()

mvo_weights = optimized_mvo(returns)
mvp_weights = minimum_variance_portfolio(returns)

# Custom Reward Function (Sharpe Ratio)
def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return (mean_return - risk_free_rate) / (std_return + 1e-8)  # Avoid division by zero

# Reinforcement Learning-based Portfolio Allocation
class AdvancedPortfolioEnv(gym.Env):
    def __init__(self, price_data, base_alloc):
        super(AdvancedPortfolioEnv, self).__init__()
        self.price_data = price_data
        self.base_alloc = base_alloc
        self.action_space = spaces.Box(low=0, high=1, shape=(len(price_data.columns),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(price_data.columns)*2,), dtype=np.float32)
        self.current_step = 0
        self.balance = INITIAL_BALANCE
        self.holdings = np.zeros(len(price_data.columns))
        self.returns_history = []

    def _next_observation(self):
        price_obs = self.price_data.iloc[self.current_step].values
        observation = np.concatenate((price_obs, self.holdings))
        if np.any(np.isnan(observation)):
            observation = np.nan_to_num(observation, nan=0.0)
        return observation.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = INITIAL_BALANCE
        self.holdings = np.zeros(len(self.price_data.columns))
        self.returns_history = []
        return self._next_observation(), {}

    def step(self, action):
        action = np.nan_to_num(action, nan=0.0)
        action = np.clip(action, 0, 1)
        self.current_step += 1
        done = self.current_step >= len(self.price_data) - 1
        
        # Compute portfolio return
        portfolio_return = np.dot(self.holdings, self.price_data.pct_change().iloc[self.current_step].fillna(0))
        self.returns_history.append(portfolio_return)
        
        # Sharpe Ratio as Reward
        if len(self.returns_history) > 20:
            reward = calculate_sharpe_ratio(self.returns_history)
        else:
            reward = portfolio_return  # Use simple returns initially
        
        self.holdings = action
        next_state = self._next_observation()
        return next_state, reward, done, False, {}

# Training RL Model with Optimized Hyperparameters
env = DummyVecEnv([lambda: Monitor(AdvancedPortfolioEnv(price_data, mvo_weights))])
model = PPO(
    "MlpPolicy", env, verbose=1, device='cpu',
    learning_rate=0.0001,  # Lower for better stability
    clip_range=0.1,        # Adjusted for stable updates
    gamma=0.99,            # Discount factor
    ent_coef=0.005,        # Lower entropy for more deterministic policy
)
eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=1000, deterministic=True, render=False)

# Train for 100,000 timesteps (Increased from 20,000)
model.learn(total_timesteps=100000, callback=eval_callback)
model.save("ppo_portfolio")

# Portfolio Simulation
obs = env.reset()
portfolio_values = [INITIAL_BALANCE]
for i in range(len(price_data) - 1):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    current_prices = price_data.iloc[i]
    current_value = INITIAL_BALANCE + np.sum(action * current_prices)
    portfolio_values.append(current_value)
    if done:
        break

# Performance Analysis
portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
cumulative_returns = (portfolio_returns + 1).cumprod() - 1
plt.figure(figsize=(12, 6))
plt.plot(cumulative_returns)
plt.title("Cumulative Portfolio Returns")
plt.xlabel("Trading Days")
plt.ylabel("Returns")
plt.grid(True)
plt.show()
print(f"Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
print(f"Total Return: {cumulative_returns.iloc[-1]*100:.2f}%")
print(f"Annualized Sharpe Ratio: {np.sqrt(252)*portfolio_returns.mean()/portfolio_returns.std():.2f}")
