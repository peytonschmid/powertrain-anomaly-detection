import numpy as np
import pandas as pd
from typing import List

from .config import CONFIG, numeric_cols


def per_mode_standardize(df: pd.DataFrame,
                         cols,
                         per_mode: bool = True,
                         by=("cycle_name", "Gear")) -> pd.DataFrame:
    out = df.copy()
    out[cols] = out[cols].astype(float)

    if (not per_mode) or any(c not in out.columns for c in by):
        mu = out[cols].mean()
        sd = out[cols].std() + 1e-8
        out[cols] = (out[cols] - mu) / sd
        out[cols] = out[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    for keys, g in out.groupby(list(by)):
        mu = g[cols].mean()
        sd = g[cols].std() + 1e-8
        out.loc[g.index, cols] = (g[cols] - mu) / sd

    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def make_windows(df: pd.DataFrame,
                 cols,
                 window_s: int,
                 stride_s: int,
                 hz: int):
    W = window_s * hz
    S = stride_s * hz

    X, y, t0, mode = [], [], [], []
    arr = df[cols].values.astype(np.float32)
    lab = df.get("anomaly", pd.Series(np.zeros(len(df)))).values
    times = df["time_s"].values

    if "cycle_name" in df.columns:
        modes_arr = df["cycle_name"].values
    else:
        modes_arr = np.array(["global"] * len(df))

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    for i in range(0, len(df) - W + 1, S):
        win = arr[i:i+W]
        X.append(win)
        y.append(1 if lab[i:i+W].max() > 0 else 0)
        t0.append(times[i])
        mode.append(pd.Series(modes_arr[i:i+W]).mode().iloc[0])

    return (np.stack(X, axis=0),
            np.array(y),
            np.array(t0),
            np.array(mode, dtype=object))
