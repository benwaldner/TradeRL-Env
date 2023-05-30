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

class Actions(Enum):
    Sell = 0
    Buy = 1


class Positions(Enum):
    Short = 0
    Long = 1

    def opposite(self):
        return Positions.Short if self == Positions.Long else Positions.Long

class Trade(TypedDict):
    position: Positions
    openPrice: float
    closePrice: float
    openTime: pd.DatetimeIndex
    closeTime: pd.DatetimeIndex

class TradingEnv(gym.Env):

    metadata = {'render_modes': ['human']}

    def __init__(self, df, window_size, symbol, spread, point, render_mode=None):
        assert df.ndim == 2

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.df = df
        self.symbol = symbol
        self.window_size = window_size
        self.spread = spread
        self.point = point
        self.prices, self.signal_features, self.dates = self._process_data()
        self.shape = (window_size, self.signal_features.shape[1],)

        self.writer = tf.summary.create_file_writer(f'logs/rl/{datetime.now().strftime("%Y_%m_%d_%H_%M")}')
        self.episode = 0

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
        self._trades = []
        self._position = None
        self._position_history = None
        self._total_reward = None
        self._total_profit = None
        self._first_rendering = None
        self.history = None

        # create cache for faster training
        self._observation_cache = []
        for current_tick in range(self._start_tick, self._end_tick + 1):
            obs = self.signal_features[(current_tick-self.window_size+1):current_tick+1]
            self._observation_cache.append(obs)

    def _get_info(self):
        return dict(
            total_reward = self._total_reward,
            total_profit = self._total_profit,
            position = self._position.value
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._terminated = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = 1.  # unit
        self.episode += 1
        self._trades = []
        self._first_rendering = True
        self.history = {}

        info = self._get_info()
        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        self._terminated = False
        self._current_tick += 1

        # Only terminate after all bars finished
        if self._current_tick >= self._end_tick:
            self._terminated = True

            # Draw the trades to tensorboard
            trades = self.trades()
            if trades.empty == False:
                # Calculate the trade profits
                trades.loc[trades['position'] == 0, 'profit'] = ((trades['openPrice'] - self.spread) - trades['closePrice']) * self.point
                trades.loc[trades['position'] == 1, 'profit'] = (trades['closePrice'] - (trades['openPrice'] + self.spread)) * self.point
                trades = trades.dropna()
                trades['cashflow'] = trades['profit'].cumsum()

                # Draw the Cashflow to the tensorboard
                with self.writer.as_default():
                    for i, trade in trades['cashflow'].items():
                        tf.summary.scalar(f'trades/iteration-{self.episode}', trade, i)

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward
        self._update_profit(action)

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            self._position = self._position.opposite()
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        observation = self._get_observation()
        info = self._get_info()
        self._update_history(info)

        if self.render_mode == "human":
            self._render_frame()

        return observation, step_reward, self._terminated, False, info


    def _get_observation(self):
        return self._observation_cache[self._current_tick-self.window_size]


    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)

    def _render_frame(self):
        self.render()

    def render(self, mode='human'):

        def _plot_position(position, tick):
            color = None
            if position == Positions.Short:
                color = 'red'
            elif position == Positions.Long:
                color = 'green'
            if color:
                plt.scatter(tick, self.prices[tick], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.prices)
            start_position = self._position_history[self._start_tick]
            _plot_position(start_position, self._start_tick)

        _plot_position(self._position, self._current_tick)

        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )

        plt.pause(0.01)


    def render_all(self, title=None):
        window_ticks = np.arange(len(self._position_history))
        plt.plot(self.prices)

        short_ticks = []
        long_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)

        plt.plot(short_ticks, self.prices[short_ticks], 'ro')
        plt.plot(long_ticks, self.prices[long_ticks], 'go')

        if title: plt.title(title)
        plt.suptitle(
            "Total Reward: %.6f" % self._total_reward + ' ~ ' +
            "Total Profit: %.6f" % self._total_profit
        )
    
    def trades(self):
        return pd.DataFrame(self._trades)
        
    def close(self):
        plt.close()


    def save_rendering(self, filepath):
        plt.savefig(filepath)


    def pause_rendering(self):
        plt.show()


    def _process_data(self):
        raise NotImplementedError


    def _calculate_reward(self, action):
        raise NotImplementedError


    def _update_profit(self, action):
        raise NotImplementedError


    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError