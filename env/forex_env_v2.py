import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from .trading_env_v2 import TradingEnvV2, Actions, Positions
from util.ta import extract_features

class ForexEnvV2(TradingEnvV2):

    def __init__(self, df, window_size, symbol, unit_side='left', eps_length=253, spread=0.0003, point=10000, trade_fee=0.0003, bar_limit=None, render_mode=None, normalise=None, normalise_path=None):
        assert unit_side.lower() in ['left', 'right']
        self.normalise = normalise
        self.normalise_path = normalise_path
        self.unit_side = unit_side.lower()
        self.bar_limit = bar_limit
        super().__init__(df=df, window_size=window_size, symbol=symbol, eps_length=eps_length, render_mode=render_mode, spread=spread, point=point)

        self.trade_fee = trade_fee  # unit in %


    def _process_data(self):
        # Technical Indicator Processing

        prices = self.df.loc[:, 'close']
        
        if self.normalise is None: self.df = extract_features(self.df)
        elif isinstance(self.normalise, float): self.df = extract_features(self.df, normalise=self.normalise, normalise_path=self.normalise_path)
        elif self.normalise == True: self.df = extract_features(self.df, normalise=True, normalise_path=self.normalise_path)

        self.df = self.df.sort_index(ascending=True)
        if self.bar_limit is not None: self.df = self.df[-self.bar_limit:]

        prices = prices[prices.index.isin(self.df.index)]
        prices = prices.sort_index(ascending=True)
        prices = prices.to_numpy()
        dates = self.df.index
        # self.df.to_csv('test.csv')

        signal_features = self.df.values
        # print(signal_features.shape)
        # print(signal_features)
        print(f'Forex Environment with {len(prices)} bars and {signal_features.shape} feature shape.')
        return prices, signal_features, dates


    def _calculate_reward(self, action):
        step_reward = 0  # pip

        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        ratio = current_price / last_trade_price
        cost = np.log((1 - self.trade_fee) * (1 - self.trade_fee))

        if action == Actions.BUY and self._position == Positions.SHORT:
            step_reward = np.log(2 - ratio) + cost

        if action == Actions.SELL and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        if action == Actions.DOUBLE_SELL and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        if action == Actions.DOUBLE_BUY and self._position == Positions.SHORT:
            step_reward = np.log(2 - ratio) + cost

        step_reward = float(step_reward)
        return step_reward

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1

                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]

                tmp_profit = profit * (2 - (current_price / last_trade_price)) * (1 - self.trade_fee) * (1 - self.trade_fee)
                profit = max(profit, tmp_profit)
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1

                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]

                tmp_profit = profit * (current_price / last_trade_price) * (1 - self.trade_fee) * (1 - self.trade_fee)
                profit = max(profit, tmp_profit)

            last_trade_tick = current_tick - 1

        return profit