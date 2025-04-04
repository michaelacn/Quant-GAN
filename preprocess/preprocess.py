import numpy as np
import pandas as pd
from scipy.optimize import fmin
from scipy.special import lambertw
from scipy.stats import kurtosis, norm
from statsmodels.tsa.stattools import acf


def compute_leverage_effect(log_returns: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    volatility = log_returns ** 2
    leverage_effect = {}
    
    for asset in log_returns.columns:
        corr_values = []
        for lag in range(1, max_lag + 1):
            asset_returns = log_returns[asset]
            asset_vol = volatility[asset]
            corr = np.corrcoef(asset_returns.iloc[:-lag], asset_vol.iloc[lag:])[0, 1]
            corr_values.append(corr)
        leverage_effect[asset] = corr_values
    
    return pd.DataFrame(leverage_effect, index=range(1, max_lag + 1))


def delta_init(z):
    k = kurtosis(z, fisher=False, bias=False)
    if k < 166. / 62.:
        return 0.01
    return np.clip(1. / 66 * (np.sqrt(66 * k - 162.) - 6.), 0.01, 0.48)

def delta_gmm(z):
    delta = delta_init(z)

    def iter(q):
        u = W_delta(z, np.exp(q))
        if not np.all(np.isfinite(u)):
            return 0.
        k = kurtosis(u, fisher=True, bias=False)**2
        if not np.isfinite(k) or k > 1e10:
            return 1e10
        return k

    res = fmin(iter, np.log(delta), disp=0)
    return np.around(np.exp(res[-1]), 6)

def W_delta(z, delta):
    return np.sign(z) * np.sqrt(np.real(lambertw(delta * z ** 2)) / delta)

def W_params(z, params):
    return params[0] + params[1] * W_delta((z - params[0]) / params[1], params[2])

def inverse(z, params):
    return params[0] + params[1] * (z * np.exp(z * z * (params[2] * 0.5)))

def igmm(z, eps=1e-6, max_iter=100):
    delta = delta_init(z)
    params = [np.median(z), np.std(z) * (1. - 2. * delta) ** 0.75, delta]
    for k in range(max_iter):
        params_old = params
        u = (z - params[0]) / params[1]
        params[2] = delta_gmm(u)
        x = W_params(z, params)
        params[0], params[1] = np.mean(x), np.std(x)

        if np.linalg.norm(np.array(params) - np.array(params_old)) < eps:
            break
        if k == max_iter - 1:
            raise "Solution not found"

    return params