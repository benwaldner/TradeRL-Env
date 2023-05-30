import yaml
from dateutil import tz
import pandas as pd

def get_config(config_path = 'config.yml'):
    with open(config_path, "r") as stream: return yaml.safe_load(stream)

def remove_dst(df: pd.DataFrame, timezone: str, to_tz: str = 'UTC+3'):
    df = df.tz_localize(timezone)
    to_tz = tz.gettz(to_tz)
    df = df.tz_convert(to_tz)
    df = df.tz_localize(None)
    return df