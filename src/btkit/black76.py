import numpy as np
import pandas as pd

from scipy.optimize import root_scalar, RootResults
from scipy.stats import norm


class Black76:
    
    @classmethod
    def option_price(cls, F: float, K: float, T: float, sigma: float, option_type: str, r: float = 0.05):
        d1 = (np.log(F / K) + (sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        discount_factor = np.exp(-r * T)

        cp = 1 if option_type.lower() == "c" else -1
        return discount_factor * cp * (F * norm.cdf(cp * d1) - K * norm.cdf(cp * d2))
    
    
    @classmethod
    def sigma(cls, row: pd.Series):
        def f(sigma: float) -> float:
            return row["close"] - cls.option_price(
                F=row["underlying_close"],
                K=row["strike_price"],
                T=row["T"],
                sigma=sigma,
                option_type=row["instrument_class"]
            )
        result: RootResults = root_scalar(f, x0=0.1, x1=0.5)
        if result.converged:
            return result.root
        else: 
            return np.nan


    @classmethod
    def delta(cls, row: pd.Series):
        r = 0.05 # TODO: make class variable?
        F = row["underlying_close"]
        K = row["strike_price"]
        T = row["T"]
        sigma = row["sigma"]
        d1 = (np.log(F / K) + (sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        disc = np.exp(-r * T)
        if row["instrument_class"].lower() in ('c', 'call'):
            return disc * norm.cdf(d1)
        else:
            # put delta under Black-76 is -e^{-rT} N(-d1)
            return -disc * norm.cdf(-d1)

