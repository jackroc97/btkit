import numpy as np
from math import sqrt, exp
from scipy.optimize import root_scalar
from scipy.stats import norm


class Black76:
    
    
    @classmethod
    def option_price(cls, F, K, T, r, sigma, option_type):
        """Black-76 price for a futures option."""
        if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
            return np.nan
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        disc = exp(-r * T)
        if option_type.lower() in ('c', 'call'):
            return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


    @classmethod
    def delta(cls, F, K, T, r, sigma, option_type):
        """Black-76 delta (futures-style)."""
        if T <= 0 or sigma <= 0 or F <= 0 or K <= 0:
            return np.nan
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt(T))
        disc = exp(-r * T)
        if option_type.lower() in ('c', 'call'):
            return disc * norm.cdf(d1)
        else:
            # put delta under Black-76 is -e^{-rT} N(-d1)
            return -disc * norm.cdf(-d1)


    @classmethod
    def implied_vol(cls, option_price, F, K, T, r, option_type,
                            initial_guess=0.4, tol=1e-6, max_iter=80,
                            sigma_min=1e-6, sigma_max=5.0):
        """
        Solve for implied vol under Black-76 using bounded Newton-Raphson with
        finite-difference vega and a Brent fallback if necessary.
        Returns np.nan if not found.
        """
        # quick invalid checks
        if option_price is None or np.isnan(option_price) or option_price <= 0:
            return np.nan
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (F, K, T)):
            return np.nan
        if F <= 0 or K <= 0 or T <= 0:
            return np.nan

        # Objective: price(sigma) - option_price = 0
        def objective(sigma):
            return cls.option_price(F, K, T, r, sigma, option_type) - option_price

        # Newton-ish loop with finite-diff vega
        sigma = float(initial_guess)
        sigma = max(min(sigma, sigma_max), sigma_min)

        for _ in range(max_iter):
            try:
                price = cls.option_price(F, K, T, r, sigma, option_type)
            except Exception:
                return np.nan

            diff = option_price - price
            if abs(diff) < tol:
                return sigma

            # finite-difference vega
            eps = max(1e-6, sigma * 1e-6)
            price_up = cls.option_price(F, K, T, r, sigma + eps, option_type)
            vega = (price_up - price) / eps

            # if vega is extremely small, break to fallback
            if abs(vega) < 1e-8:
                break

            # Newton update with bounds
            sigma_new = sigma + diff / vega
            sigma = max(min(sigma_new, sigma_max), sigma_min)

        # If Newton didn't converge, try a robust bracket-and-solve (brentq)
        try:
            # Need signs for bracket: objective(sigma_min) and objective(sigma_max) should have opposite signs
            f_lo = objective(sigma_min)
            f_hi = objective(sigma_max)
            if np.sign(f_lo) == np.sign(f_hi):
                # try expanding upper bound if both same sign (rare)
                sigma_hi_try = sigma_max * 2
                f_hi = objective(sigma_hi_try)
                if np.sign(f_lo) == np.sign(f_hi):
                    return np.nan
                else:
                    sigma_max = sigma_hi_try

            result = root_scalar(objective, bracket=[sigma_min, sigma_max], method='brentq', xtol=tol)
            return float(result.root) if result.converged else np.nan
        except Exception:
            return np.nan
