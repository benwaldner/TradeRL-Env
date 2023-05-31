import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from .trading_env import TradingEnv, Actions, Positions, Trade
from util.ta import extract_features

class ForexEnv(TradingEnv):

    def __init__(self, df, window_size, symbol, unit_side='left', point=10000, spread=0.0003, trade_fee=0.00001, bar_limit=None, render_mode=None, normalise=None, normalise_path=None, clear_trade=True):
        assert unit_side.lower() in ['left', 'right']
        self.normalise = normalise
        self.normalise_path = normalise_path
        self.unit_side = unit_side.lower()
        self.bar_limit = bar_limit
        super().__init__(df=df, window_size=window_size, symbol=symbol, render_mode=render_mode, point=point, spread=spread, clear_trade=clear_trade)

        self.trade_fee = trade_fee  # unit


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

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Short:
                step_reward += -price_diff * self.point
            elif self._position == Positions.Long:
                step_reward += price_diff * self.point

        return step_reward


    def _update_profit(self, action):
        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade or self._terminated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            trade: Trade = { 'position': self._position.value, 'openPrice': last_trade_price, 'closePrice': current_price,
                             'openTime': self.dates[self._last_trade_tick], 'closeTime': self.dates[self._current_tick] }
            self._trades.append(trade)

            if self.unit_side == 'left':
                if self._position == Positions.Short:
                    quantity = self._total_profit * (last_trade_price - self.trade_fee)
                    self._total_profit = quantity / current_price

            elif self.unit_side == 'right':
                if self._position == Positions.Long:
                    quantity = self._total_profit / last_trade_price
                    self._total_profit = quantity * (current_price - self.trade_fee)


    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            current_price = self.prices[current_tick - 1]
            last_trade_price = self.prices[last_trade_tick]

            if self.unit_side == 'left':
                if position == Positions.Short:
                    quantity = profit * (last_trade_price - self.trade_fee)
                    profit = quantity / current_price

            elif self.unit_side == 'right':
                if position == Positions.Long:
                    quantity = profit / last_trade_price
                    profit = quantity * (current_price - self.trade_fee)

            last_trade_tick = current_tick - 1

        return profit