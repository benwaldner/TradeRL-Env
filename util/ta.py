import pandas as pd
from ta.volatility import *
from ta.trend import *
from ta.momentum import *
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
import pickle
from ta import add_all_ta_features

def donchain(df: pd.DataFrame, window_size=30):
    '''
    Calculate the simplest Support/Resistance from Donchain.

    Support = Last X Bars Minimum Value.
    Resistance = Last X Bars Maximum Value. 
    '''
    # Compute the rolling minimum and maximum of the low and high prices
    df[f'support_{window_size}'] = df['low'].rolling(window_size, closed='left').min()
    df[f'resistance_{window_size}'] = df['high'].rolling(window_size, closed='left').max()

    # Calculate the closing difference, for better mathematical expression
    df[f'support_{window_size}'] = np.log(df['close']) - np.log(df[f'support_{window_size}'])
    df[f'resistance_{window_size}'] = np.log(df[f'resistance_{window_size}']) - np.log(df['close'])

    # Stationise the result
    # df[f'support_{window_size}'] = np.log(df[f'support_{window_size}']) - np.log(df[f'support_{window_size}'].shift(1))
    # df[f'resistance_{window_size}'] = np.log(df[f'resistance_{window_size}']) - np.log(df[f'resistance_{window_size}'].shift(1))
    return df

def extract_features(df: pd.DataFrame, normalise=None, normalise_path="data/scaler.pkl"):
    print('Computing Donchain...')
    for p in [30, 60, 90, 120, 180]: df = donchain(df, window_size=p)

    print('Computing Bollinger Band...')
    bband = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_hi'] = bband.bollinger_hband()
    df['bb_lo'] = bband.bollinger_lband()

    # Calculate the difference between bollinger band and the price for better math expression
    df['bb_hi_diff'] = np.log(df['close']) - np.log(df['bb_hi'])
    df['bb_lo_diff'] = np.log(df['close']) - np.log(df['bb_lo'])

    # Stationising the bollinger band
    df['bb_hi'] = np.log(df['bb_hi']) - np.log(df['bb_hi'].shift(1))
    df['bb_lo'] = np.log(df['bb_lo']) - np.log(df['bb_lo'].shift(1))

    print('Computing RSI...')
    for p in [14, 30]: df[f'rsi_{p}'] = RSIIndicator(df['close'], window=p).rsi()

    print('Computing ADX...')
    for p in [14, 30, 60]:
        adx_indicator = ADXIndicator(df['high'], df['low'], df['close'], window=p)
        df[f'adx_{p}'] = adx_indicator.adx()
        df[f'+di_{p}'] = adx_indicator.adx_pos()
        df[f'-di_{p}'] = adx_indicator.adx_neg()

    print('Computing CCI...')
    for p in [14, 30, 60]: df[f'cci_{p}'] = CCIIndicator(df['high'], df['low'], df['close'], window=p).cci()

    print('Computing MACD...')
    macd_indicator = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd_indicator.macd()
    df['macd_diff'] = macd_indicator.macd_diff()
    df['macd_signal'] = macd_indicator.macd_signal()

    print('Computing StochRSI...')
    for p in [14, 21]:
        srsi = StochRSIIndicator(df['close'], window=p, smooth1=3, smooth2=3)
        df[f'srsi_{p}'] = srsi.stochrsi()
        df[f'srsi_{p}_k'] = srsi.stochrsi_k()
        df[f'srsi_{p}_d'] = srsi.stochrsi_d()

    print('Computing Moving Average...')
    for ma in [3, 7, 14, 21, 60, 120]:
        df[f'MA_{ma}'] = SMAIndicator(df['close'], window=ma).sma_indicator()
        # Calculate the difference between moving average for better math expression
        df[f'MA_{ma}_diff'] = np.log(df[f'MA_{ma}']) - np.log(df['close'])
        # Stationise Moving Average
        df[f'MA_{ma}'] = np.log(df[f'MA_{ma}']) - np.log(df[f'MA_{ma}'].shift(1))

    # Stationising original data
    df['open'] = np.log(df['open']) - np.log(df['open'].shift(1))
    df['high'] = np.log(df['high']) - np.log(df['high'].shift(1))
    df['low'] = np.log(df['low']) - np.log(df['low'].shift(1))
    df['close'] = np.log(df['close']) - np.log(df['close'].shift(1))

    print(f'Dropping {df.isna().any(axis=1).sum()} from {len(df)} bars...')
    df = df.dropna()

    # Normalising data
    if normalise is not None:
        # Normalise = True -> Transform only
        if (normalise == True) & (isinstance(normalise, bool)):
            scaler: StandardScaler = pickle.load(open(normalise_path, 'rb'))
        else:
            train_size = int(len(df) * normalise)
            print(f'Fitting {train_size}/{len(df)} ({normalise * 100}%) of the data to the Scaler')
            scaler = StandardScaler()
            scaler = scaler.fit(df[:train_size])
            pickle.dump(scaler, open(normalise_path, 'wb'))
        print(f'Transforming all data')
        columns = df.columns
        index = df.index
        df = scaler.transform(df)
        df = pd.DataFrame(df, columns=columns, index=index)

    return df

def extract_features2(df: pd.DataFrame, normalise=None, normalise_path="data/scaler.pkl"):
    df = add_all_ta_features(df, open='open', high='high', low='low', close='close', volume='volume', fillna=True)
    print(f'Dropping {df.isna().any(axis=1).sum()} from {len(df)} bars...')
    df = df.dropna()

    # Normalising data
    if normalise is not None:
        # Normalise = True -> Transform only
        if (normalise == True) & (isinstance(normalise, bool)):
            scaler: StandardScaler = pickle.load(open(normalise_path, 'rb'))
        else:
            train_size = int(len(df) * normalise)
            print(f'Fitting {train_size}/{len(df)} ({normalise * 100}%) of the data to the Scaler')
            scaler = StandardScaler()
            scaler = scaler.fit(df[:train_size])
            pickle.dump(scaler, open(normalise_path, 'wb'))
        print(f'Transforming all data')
        columns = df.columns
        index = df.index
        df = scaler.transform(df)
        df = pd.DataFrame(df, columns=columns, index=index)
    return df