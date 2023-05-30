# Forex Reinforcement Learning Environments

A collection of multiple Forex Reinforcement Learning Environments originated from [AminHP](https://github.com/AminHP).

## Requirements

- Python 3.10
- Anaconda
- Install packages from `requirements.txt`
- MetaTrader5 Terminal for `Collection 3 Env`.
- You need **Windows** to run `Collection 3 Env` as MetaTrader5 only supports Windows OS.

---
## Collection 1 : `env/forex_env.py`

This is a simple Reinforcement Learning Environment which contains 2-Discrete Action: `Buy` or `Sell`.

### Action Cycle

|Previous Action|Next Action|Next Position|
|---------------|-----------|-------------|
|Sell|Sell|Keep The Short Position|
|Sell|Buy|Close Short Position & Open Long Position|
|Buy|Buy|Keep The Long Position|
|Buy|Sell|Close Long Position & Open Short Position|

With the above action cycle, the strategy will be 100% time in the market. There will always be one long/short position.

## Collection 2 : `env/forex_env_v2.py`

This is a better Reinforcement Learning Environment which contains more Discrete Actions: `Double Buy`, `Buy`, `Flat`, `Sell`, `Double Sell`.

### Action Cycle

|Previous Action|Next Action|Next Position|
|---------------|-----------|-------------|
|Flat|Buy|Open Long Position|
|Flat|Sell|Open Short Position|
|Flat|Double Buy|Open Long Position|
|Flat|Double Sell|Open Short Position|
|Flat|Flat|Do nothing|
|Buy|Flat|Keep the Long Position|
|Buy|Double Buy|Keep the Long Position|
|Buy|Sell|Close the Long Position|
|Buy|Double Sell|Close the Long Position & Open the Short Position|
|Sell|Flat|Keep the Short Position|
|Sell|Double Sell|Keep the Sort Position|
|Sell|Buy|Close the Short Position|
|Sell|Double Buy|Close the Short Position & Open the Long Position|

With the above action cycle, the strategy can stay out of the market.

---

## Collection 3 : `env/mt_env.py`

This is the advanced Reinforcement Learning Environment which learns from the holding equities and multiple symbols. The environment directly retrieve data from MetaTrader5 and simulate through `env/simulator/mt5.py`.

The Box Actions contains a range of choices by `max. position per symbol` * `no. of symbols`.

The Environment will sell/buy/hold a position based on a threshold.

With the above actions, the strategy can learn exclusively from existing orders, account equity, account balance.

---

## How to Use

1. Gather your bar data into `./data/{your_bar_data}.csv`, the bar data should contains `date,open,high,low,close`.
2. Edit the `config.yml` to use your bar data.
3. Edit the `util/ta.py` to add your favourite features.
4. Run the Example Script to test it out.
   - Collection 1 : `python examples/simple.py`
   - Collection 2 : `python examples/intermediate.py`
   - Collection 3 : `python examples/advanced.py` 
5. Use `tensorboard --logdir=./logs` to get the real-time reward chart.

---

## References

- [gym-anytrading](https://github.com/AminHP/gym-anytrading)
- [gym-mtsim](https://github.com/AminHP/gym-mtsim)