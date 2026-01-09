import numpy as np
import pandas as pd

FEATURE_COLS = ["close","ret","acc","amp","vol_rel"]
TARGET_REG_COLS = ["open_next","high_next","low_next","close_next","amp_next"]

def build_features_targets(df: pd.DataFrame):
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["acc"] = df["ret"].diff()
    df["amp"] = df["high"] - df["low"]
    df["vol_rel"] = df["volume"] / df["volume"].rolling(10).mean()

    df["open_next"]  = df["open"].shift(-1)
    df["high_next"]  = df["high"].shift(-1)
    df["low_next"]   = df["low"].shift(-1)
    df["close_next"] = df["close"].shift(-1)
    df["amp_next"]   = (df["high"].shift(-1) - df["low"].shift(-1))
    df["dir_next"]   = (df["close"].shift(-1) > df["close"]).astype(int)

    df = df.dropna().reset_index(drop=True)
    X = df[FEATURE_COLS].copy()
    Yreg = df[TARGET_REG_COLS].copy()
    Ycls = df["dir_next"].copy()
    return df, X, Yreg, Ycls

def exp_sample_weights(n: int, alpha: float):
    t = np.arange(n)
    return alpha ** (n - 1 - t)
