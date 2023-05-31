import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from typing import TypedDict
import pandas as pd
from datetime import datetime

# Double Sell means flipping Long / Flat to Sell
# Double Buy means flipping Short / Flat to Buy
class Actions(int, Enum):
    DOUBLE_SELL = 0
    SELL = 1
    HOLD = 2
    BUY = 3
    DOUBLE_BUY = 4


class Positions(int, Enum):
    SHORT = -1.
    FLAT = 0.
    LONG = 1.

def transform(position: Positions, action: int):
    '''
    Overview:
        used by env.tep().
        This func is used to transform the env's position from
        the input (position, action) pair according to the status machine.
    Arguments:
        - position(Positions) : Long, Short or Flat
        - action(int) : Doulbe_Sell, Sell, Hold, Buy, Double_Buy
    Returns:
        - next_position(Positions) : the position after transformation.
    '''
    if action == Actions.SELL:
        if position == Positions.LONG: return Positions.FLAT, False
        if position == Positions.FLAT: return Positions.SHORT, True

    if action == Actions.BUY:
        if position == Positions.SHORT: return Positions.FLAT, False
        if position == Positions.FLAT: return Positions.LONG, True

    if action == Actions.DOUBLE_SELL and (position == Positions.LONG or position == Positions.FLAT):
        return Positions.SHORT, True

    if action == Actions.DOUBLE_BUY and (position == Positions.SHORT or position == Positions.FLAT):
        return Positions.LONG, True

    return position, False

class Trade(TypedDict):
    position: Positions
    openPrice: float
    closePrice: float
    openTime: pd.DatetimeIndex
    closeTime: pd.DatetimeIndex

class TradingEnvV2(gym.Env):

    metadata = {'render_modes': ['human']}

    # eps_length: To normalise the holding time of a position
    def __init__(self, df, window_size, symbol, spread, point, eps_length = 253, render_mode=None, clear_trade=True):
        assert df.ndim == 2

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.df = df
        self.symbol = symbol
        self.window_size = window_size
        self.spread = spread
        self.point = point
        self.prices, self.signal_features, self.dates = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1] + 2,)  # +2 for the position information
        self.eps_length = eps_length

        self.writer = tf.summary.create_file_writer(f'logs/rl/{datetime.now().strftime("%Y_%m_%d_%H_%M")}')
        self.episode = 0
        self.clear_trade = clear_trade

        # spaces
        self.action_space = spaces.Discrete(len(Actions))
        INF = 1e10
        self.observation_space = spaces.Box(low=-INF, high=INF, shape=self.shape, dtype=np.float64)

        # episode
        self._start_tick = self.window_size
        self._end_tick = len(self.prices) - 1
        self._terminated = None
        self._current_tick = None
        self._last_trade_tick = None
        self._trades = None
        self._position = None
        self._position_history = None
        self._position_time_score = None
        self._profit_history = None
        self._pips = None
        self._total_reward = None
        self.history = None

        # training variable
        self._best_score = 0

    def _get_info(self):
        return dict(
            total_reward = self._total_reward,
            position = self._position.value
        )
    
    def terminated(self): return self._terminated

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._terminated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.FLAT
        self._position_history = ((self.window_size + 1) * [Positions.FLAT.value])
        self._position_time_score = ((self.window_size + 1) * [0])
        self._profit_history = [1.]
        self._total_reward = 0.
        self._pips = 0.
        self.episode += 1
        self.history = {}
        self._last_trades = self._trades
        if self.clear_trade: self._trades = []

        info = self._get_info()
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self._terminated = False
        self._current_tick += 1

        # Only terminate after all bars finished
        if self._current_tick >= self._end_tick:
            # Draw the trades to tensorboard
            trades = self.trades()
            if trades.empty == False:
                # Calculate the trade profits
                trades.loc[trades['position'] == -1, 'profit'] = ((trades['openPrice'] - self.spread) - trades['closePrice']) * self.point
                trades.loc[trades['position'] == 1, 'profit'] = (trades['closePrice'] - (trades['openPrice'] + self.spread)) * self.point
                trades = trades.dropna()
                trades['cashflow'] = trades['profit'].cumsum()
                profit = trades['profit'].sum()
                # Draw the Cashflow to the tensorboard
                with self.writer.as_default():
                    for i, trade in trades['cashflow'].items():
                        tf.summary.scalar(f'trades/iteration-{self.episode}', trade, i)

                # Update Model
                if self._model is not None:
                    if profit > self._best_score:
                        print(f'Saving Model... [+{round(profit, 2)} pips]')
                        self._model.save(f'output/RL_{self.symbol}_v2.ckpt')
                        self._best_score = profit

            if self.render_mode == "human": self.render()
            self._terminated = True

        # Calculate step reward
        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        # Compute the next position
        last_position = self._position
        self._position, trade = transform(self._position, action)

        # Update Trade History
        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            trade: Trade = { 'position': last_position.value, 'openPrice': last_trade_price, 'closePrice': current_price,
                             'openTime': self.dates[self._last_trade_tick], 'closeTime': self.dates[self._current_tick] }
            
            # Add Order Pips
            if trade['position'] == -1: self._pips += ((trade['openPrice'] - self.spread) - trade['closePrice']) * self.point
            elif trade['position'] == 1: self._pips += (trade['closePrice'] - (trade['openPrice'] + self.spread)) * self.point

            # Append Trade History
            self._trades.append(trade)

            # Update Last Trade Tick
            self._last_trade_tick = self._current_tick

        # Update New Position History
        self._position_history.append(self._position)
        self._position_time_score.append((self._current_tick - self._last_trade_tick) / self.eps_length)
        self._profit_history.append(float(np.exp(self._total_reward)))

        # Return Observation
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        return observation, step_reward, self._terminated, False, info


    def _get_observation(self):
        # obs shape = (window_size, feature_size)
        obs = self.signal_features[(self._current_tick - self.window_size + 1):self._current_tick+1]

        # Get the Position Holding Information
        positions = np.array(self._position_history[(self._current_tick - self.window_size + 1):self._current_tick+1])
        holding_time = np.array(self._position_time_score[(self._current_tick - self.window_size + 1):self._current_tick+1])

        # Add position info to the observation
        return np.concatenate((obs, positions[:, np.newaxis], holding_time[:, np.newaxis]), axis=1)

    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def render(self) -> None:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(16, 12))
        ax1.set_xlabel('trading days')
        ax1.set_ylabel('profit')
        ax1.plot(self._profit_history)

        ax2.set_xlabel('trading days')
        ax2.set_xlabel('close price')
        window_ticks = np.arange(len(self._position_history) - self.window_size)
        eps_price = self.prices[self._start_tick:self._end_tick + 1]
        ax2.plot(eps_price)

        short_ticks = []
        long_ticks = []
        flat_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.SHORT:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.LONG:
                long_ticks.append(tick)
            else:
                flat_ticks.append(tick)

        ax2.plot(long_ticks, eps_price[long_ticks], 'g^', markersize=3, label="Long")
        ax2.plot(flat_ticks, eps_price[flat_ticks], 'bo', markersize=3, label="Flat")
        ax2.plot(short_ticks, eps_price[short_ticks], 'rv', markersize=3, label="Short")
        ax2.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))

        plt.show()
    
    def trades(self):
        return pd.DataFrame(self._trades)
    
    def pips(self):
        return self._pips
    
    def total_steps(self):
        return self._end_tick - self._start_tick
        
    def close(self):
        plt.close()

    def set_model(self, model): self._model = model

    def save_rendering(self, filepath):
        plt.savefig(filepath)


    def pause_rendering(self):
        plt.show()


    def _process_data(self):
        raise NotImplementedError


    def _calculate_reward(self, action):
        raise NotImplementedError

    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError