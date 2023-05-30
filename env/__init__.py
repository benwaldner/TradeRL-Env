from gymnasium.envs.registration import register
from .forex_env import ForexEnv
from .forex_env_v2 import ForexEnvV2
from .mt_env import MtEnv

register(id='forex-v0', entry_point='env:ForexEnv')
register(id='forex-v2', entry_point='env:ForexEnvV2')
