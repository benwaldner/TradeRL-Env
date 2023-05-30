import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from env import ForexEnv

import pandas as pd
from util.config import get_config

# RL Algorithms: https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

# Implemented in SB3 Contrib 
# install SB3 Contrib + gymnasium-support
# pip install git+https://github.com/DLR-RM/stable-baselines3
# pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib
from sb3_contrib import ARS, QRDQN, RecurrentPPO, TQC, TRPO, MaskablePPO 

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

if __name__ == "__main__":
    config = get_config()
    bars = pd.read_csv(config['data'], index_col='date', parse_dates=['date'])

    # -------------------------------------------------------------------------------------
    # INIT Env.
    # -------------------------------------------------------------------------------------

    seed = 888 #random seed
    set_random_seed(seed)

    print(f'Preparing Environment for {config["data"]}...')

    train_size = int(len(bars) * config['split_ratio'])
    print(f'Training Period: {bars.index[0]} to {bars.index[train_size]} with {train_size} bars')
    print(f'Testing  Period: {bars.index[train_size+1]} to {bars.index[-1]} with {len(bars) - train_size} bars')
    point = config['point']
    spread = config['spread'] / point

    # Create the Training Environment
    train_env = ForexEnv(
        df=bars[:train_size], window_size=config['window_size'], unit_side='right',
        point = point, bar_limit=config['bar_limit'], spread=spread, trade_fee=config['service_fee'],
        symbol=config["symbol"], normalise=1.0, normalise_path=f'data/{config["symbol"]}_scaler.pkl')

    # Start the Learning Process
    if config['bar_limit'] is not None: learning_step = config['epochs'] * config['bar_limit']
    else: learning_step = config['epochs'] * train_size

    model = RecurrentPPO('MlpLstmPolicy', env=train_env, policy_kwargs={ 'n_lstm_layers': 2 })
    model.learn(total_timesteps=learning_step, log_interval=10, progress_bar=True)
    model.save(f'output/{config["symbol"]}_simple.ckpt')

    # Create the evaluation environment
    # For examples, we combine the evaluation script here
    # This part should be seperate from the training code
    model = RecurrentPPO.load(f'output/{config["symbol"]}_simple.ckpt')

    eval_env = ForexEnv(
        df=bars[train_size:], window_size=config['window_size'], unit_side='right',
        point = point, bar_limit=config['bar_limit'], spread=spread, trade_fee=config['service_fee'],
        symbol=config["symbol"], normalise=True, normalise_path=f'data/{config["symbol"]}_scaler.pkl')

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1, deterministic=True)

    # Get all trades
    trades = eval_env.trades()
    if trades.empty == True:
        print('The Model learned Nothing, no trades at all.')
        exit()

    # Calculate the trade profits
    trades.loc[trades['position'] == -1, 'profit'] = ((trades['openPrice'] - spread) - trades['closePrice']) * point
    trades.loc[trades['position'] == 1, 'profit'] = (trades['closePrice'] - (trades['openPrice'] + spread)) * point
    trades = trades.dropna()
    trades['cashflow'] = trades['profit'].cumsum()

    # Export the trade logs
    trades.to_csv(f'logs/{config["symbol"]}_simple_trades.csv', index=False)

    # Plot the trade cashflow
    trades.index = pd.to_datetime(trades['openTime'])
    trades['cashflow'].plot()
    plt.xlabel('date')
    plt.ylabel('PIPs')
    plt.savefig(f'logs/{config["symbol"]}_simple_benchmark.png')
    plt.close()