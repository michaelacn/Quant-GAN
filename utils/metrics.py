import numpy as np
import pandas as pd


def compute_leverage_effect(log_returns: pd.Series, nlags : int = 250) -> pd.Series:
    volatility = log_returns ** 2
    corr_values = []
    for lag in range(1, nlags + 1):
        corr = np.corrcoef(log_returns.iloc[:-lag], volatility.iloc[lag:])[0, 1]
        corr_values.append(corr)
    return pd.Series(corr_values, index=range(1, nlags + 1))


def score(hist_values, sim_values):
    hist_values = np.asarray(hist_values)
    sim_values = np.asarray(sim_values)
    
    if hist_values.shape != sim_values.shape:
        raise ValueError("Shape different.")
    
    diff = hist_values - sim_values
    return np.sqrt(np.sum(diff**2))