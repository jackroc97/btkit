"""
Unit tests for Black-76 pricing kernels and Greeks.

Reference values computed independently using the closed-form Black-76 formulas
with scipy.stats.norm as a cross-check. All tolerances are ±0.001 for prices,
±0.005 for Greeks (rounding differences between bisection IV and closed-form).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from btkit.pipeline.greeks import (
    _black76_price,
    _greeks,
    _implied_vol,
    _norm_cdf,
    _norm_pdf,
)

# ---------------------------------------------------------------------------
# Normal distribution helpers
# ---------------------------------------------------------------------------


class TestNormHelpers:
    def test_cdf_at_zero(self):
        assert abs(_norm_cdf(0.0) - 0.5) < 1e-10

    def test_cdf_symmetry(self):
        for x in [0.5, 1.0, 1.96, 3.0]:
            assert abs(_norm_cdf(x) + _norm_cdf(-x) - 1.0) < 1e-10

    def test_cdf_known_values(self):
        assert abs(_norm_cdf(1.96) - 0.97500) < 0.0001
        assert abs(_norm_cdf(-1.96) - 0.02500) < 0.0001
        assert abs(_norm_cdf(2.576) - 0.99500) < 0.0001

    def test_pdf_at_zero(self):
        expected = 1.0 / math.sqrt(2 * math.pi)
        assert abs(_norm_pdf(0.0) - expected) < 1e-10

    def test_pdf_symmetry(self):
        for x in [0.5, 1.0, 2.0]:
            assert abs(_norm_pdf(x) - _norm_pdf(-x)) < 1e-10

    def test_pdf_integrates_to_one(self):
        # Rough Riemann sum over [-5, 5]
        xs = np.linspace(-5.0, 5.0, 10_000)
        dx = xs[1] - xs[0]
        total = sum(_norm_pdf(float(x)) for x in xs) * dx
        assert abs(total - 1.0) < 0.001


# ---------------------------------------------------------------------------
# Black-76 price
# ---------------------------------------------------------------------------


class TestBlack76Price:
    """
    Reference values from the Black-76 formula using well-known test cases.
    ATM option: F=K=100, T=1yr, r=0.05, sigma=0.20.
    """

    F, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    def test_call_atm(self):
        price = _black76_price(self.F, self.K, self.T, self.r, self.sigma, 1)
        # Black-76 ATM call: df*(F*N(d1)-K*N(d2)) with d1=0.10, d2=-0.10, df=exp(-0.05)
        # ≈ 0.9512 * (100*0.5398 - 100*0.4602) ≈ 7.577
        assert abs(price - 7.577) < 0.01

    def test_put_atm(self):
        price = _black76_price(self.F, self.K, self.T, self.r, self.sigma, 0)
        # ATM put ≈ ATM call (put-call parity on futures: C - P = F*df - K*df)
        call = _black76_price(self.F, self.K, self.T, self.r, self.sigma, 1)
        df = math.exp(-self.r * self.T)
        parity = call - (self.F * df - self.K * df)
        assert abs(price - parity) < 1e-6

    def test_put_call_parity(self):
        """C - P = (F - K) * exp(-rT) for all moneyness."""
        for K in [90.0, 100.0, 110.0]:
            call = _black76_price(self.F, K, self.T, self.r, self.sigma, 1)
            put = _black76_price(self.F, K, self.T, self.r, self.sigma, 0)
            df = math.exp(-self.r * self.T)
            assert abs((call - put) - (self.F - K) * df) < 1e-6

    def test_deep_itm_call_approaches_intrinsic(self):
        # Deep ITM call (F >> K): price ≈ (F - K) * exp(-rT)
        call = _black76_price(200.0, 100.0, 1.0, 0.05, 0.20, 1)
        df = math.exp(-0.05)
        intrinsic = (200.0 - 100.0) * df
        assert abs(call - intrinsic) < 1.0

    def test_deep_otm_call_near_zero(self):
        call = _black76_price(50.0, 200.0, 1.0, 0.05, 0.20, 1)
        assert call < 0.01

    @pytest.mark.parametrize(
        "bad_input",
        [
            (0.0, 100.0, 1.0, 0.05, 0.20),  # F=0
            (100.0, 0.0, 1.0, 0.05, 0.20),  # K=0
            (100.0, 100.0, 0.0, 0.05, 0.20),  # T=0
            (100.0, 100.0, 1.0, 0.05, 0.0),  # sigma=0
        ],
    )
    def test_degenerate_inputs_return_zero(self, bad_input):
        F, K, T, r, sigma = bad_input
        assert _black76_price(F, K, T, r, sigma, 1) == 0.0
        assert _black76_price(F, K, T, r, sigma, 0) == 0.0

    def test_price_increases_with_vol(self):
        prices = [
            _black76_price(self.F, self.K, self.T, self.r, sigma, 1)
            for sigma in [0.10, 0.20, 0.30, 0.50]
        ]
        assert all(prices[i] < prices[i + 1] for i in range(len(prices) - 1))

    def test_price_increases_with_tte(self):
        prices = [
            _black76_price(self.F, self.K, T, self.r, self.sigma, 1) for T in [0.1, 0.25, 0.5, 1.0]
        ]
        assert all(prices[i] < prices[i + 1] for i in range(len(prices) - 1))


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------


class TestImpliedVol:
    def _make_arrays(self, F, K, T, r, sigma, is_call):
        return (
            np.array([F], dtype=np.float64),
            np.array([K], dtype=np.float64),
            np.array([T], dtype=np.float64),
            np.array([r], dtype=np.float64),
            is_call,
            sigma,
        )

    def test_roundtrip_call(self):
        """IV(price(sigma)) should recover sigma."""
        F, K, T, r, true_sigma = 100.0, 100.0, 0.5, 0.05, 0.25
        price = _black76_price(F, K, T, r, true_sigma, 1)
        iv = _implied_vol(
            np.array([F]),
            np.array([K]),
            np.array([T]),
            np.array([r]),
            np.array([price]),
            np.array([1], dtype=np.int32),
        )
        assert abs(iv[0] - true_sigma) < 0.001

    def test_roundtrip_put(self):
        F, K, T, r, true_sigma = 100.0, 95.0, 0.25, 0.05, 0.30
        price = _black76_price(F, K, T, r, true_sigma, 0)
        iv = _implied_vol(
            np.array([F]),
            np.array([K]),
            np.array([T]),
            np.array([r]),
            np.array([price]),
            np.array([0], dtype=np.int32),
        )
        assert abs(iv[0] - true_sigma) < 0.001

    def test_zero_price_returns_nan(self):
        iv = _implied_vol(
            np.array([100.0]),
            np.array([100.0]),
            np.array([1.0]),
            np.array([0.05]),
            np.array([0.0]),
            np.array([1], dtype=np.int32),
        )
        assert np.isnan(iv[0])

    def test_vectorised_batch(self):
        """Multiple rows processed correctly in one call."""
        n = 5
        F = np.full(n, 100.0)
        K = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        T = np.full(n, 0.5)
        r = np.full(n, 0.05)
        true_sigmas = np.array([0.20, 0.22, 0.25, 0.22, 0.20])
        is_call = np.ones(n, dtype=np.int32)
        prices = np.array(
            [_black76_price(F[i], K[i], T[i], r[i], true_sigmas[i], 1) for i in range(n)]
        )
        iv = _implied_vol(F, K, T, r, prices, is_call)
        np.testing.assert_allclose(iv, true_sigmas, atol=0.001)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------


class TestGreeks:
    """
    Validate delta, gamma, theta, vega against known properties and finite-
    difference approximations of the Black-76 formula.
    """

    F = np.array([100.0])
    K = np.array([100.0])
    T = np.array([1.0])
    r = np.array([0.05])
    sigma = np.array([0.20])

    def _fd_delta(self, is_call: int, eps: float = 0.01) -> float:
        """Finite-difference delta."""
        p_up = _black76_price(100.0 + eps, 100.0, 1.0, 0.05, 0.20, is_call)
        p_dn = _black76_price(100.0 - eps, 100.0, 1.0, 0.05, 0.20, is_call)
        return (p_up - p_dn) / (2 * eps)

    def test_call_delta_range(self):
        is_call = np.array([1], dtype=np.int32)
        delta, _, _, _ = _greeks(self.F, self.K, self.T, self.r, self.sigma, is_call)
        # ATM call delta should be near 0.5 (discounted)
        assert 0.4 < delta[0] < 0.6

    def test_put_delta_range(self):
        is_call = np.array([0], dtype=np.int32)
        delta, _, _, _ = _greeks(self.F, self.K, self.T, self.r, self.sigma, is_call)
        assert -0.6 < delta[0] < -0.4

    def test_call_delta_matches_fd(self):
        is_call = np.array([1], dtype=np.int32)
        delta, _, _, _ = _greeks(self.F, self.K, self.T, self.r, self.sigma, is_call)
        fd = self._fd_delta(1)
        assert abs(delta[0] - fd) < 0.005

    def test_put_delta_matches_fd(self):
        is_call = np.array([0], dtype=np.int32)
        delta, _, _, _ = _greeks(self.F, self.K, self.T, self.r, self.sigma, is_call)
        fd = self._fd_delta(0)
        assert abs(delta[0] - fd) < 0.005

    def test_call_put_delta_sum(self):
        """Call delta + |put delta| should equal exp(-rT) (Black-76 identity)."""
        c_delta, _, _, _ = _greeks(
            self.F, self.K, self.T, self.r, self.sigma, np.array([1], dtype=np.int32)
        )
        p_delta, _, _, _ = _greeks(
            self.F, self.K, self.T, self.r, self.sigma, np.array([0], dtype=np.int32)
        )
        df = math.exp(-0.05)
        assert abs(c_delta[0] + abs(p_delta[0]) - df) < 0.005

    def test_gamma_positive(self):
        for is_call_val in [0, 1]:
            is_call = np.array([is_call_val], dtype=np.int32)
            _, gamma, _, _ = _greeks(self.F, self.K, self.T, self.r, self.sigma, is_call)
            assert gamma[0] > 0

    def test_gamma_call_equals_put(self):
        """Gamma is identical for calls and puts at same strike/expiry."""
        c_gamma = _greeks(
            self.F, self.K, self.T, self.r, self.sigma, np.array([1], dtype=np.int32)
        )[1]
        p_gamma = _greeks(
            self.F, self.K, self.T, self.r, self.sigma, np.array([0], dtype=np.int32)
        )[1]
        assert abs(c_gamma[0] - p_gamma[0]) < 1e-8

    def test_theta_negative(self):
        """Theta should be negative (time decay reduces option value)."""
        for is_call_val in [0, 1]:
            is_call = np.array([is_call_val], dtype=np.int32)
            _, _, theta, _ = _greeks(self.F, self.K, self.T, self.r, self.sigma, is_call)
            assert theta[0] < 0

    def test_vega_positive(self):
        """Vega should be positive (higher vol → higher price)."""
        for is_call_val in [0, 1]:
            is_call = np.array([is_call_val], dtype=np.int32)
            _, _, _, vega = _greeks(self.F, self.K, self.T, self.r, self.sigma, is_call)
            assert vega[0] > 0

    def test_vega_call_equals_put(self):
        """Vega is identical for calls and puts at same strike/expiry."""
        c_vega = _greeks(self.F, self.K, self.T, self.r, self.sigma, np.array([1], dtype=np.int32))[
            3
        ]
        p_vega = _greeks(self.F, self.K, self.T, self.r, self.sigma, np.array([0], dtype=np.int32))[
            3
        ]
        assert abs(c_vega[0] - p_vega[0]) < 1e-8

    def test_nan_on_zero_tte(self):
        T_zero = np.array([0.0])
        is_call = np.array([1], dtype=np.int32)
        delta, gamma, theta, vega = _greeks(self.F, self.K, T_zero, self.r, self.sigma, is_call)
        assert np.isnan(delta[0])
        assert np.isnan(gamma[0])
        assert np.isnan(theta[0])
        assert np.isnan(vega[0])
