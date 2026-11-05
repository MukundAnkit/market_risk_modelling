"""
Unit Test Suite for Market Risk Modelling, Backtesting & Derivatives Framework (`tests/test_risk_engine.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cpp_kernel import bsm_greeks_fast, bsm_price_fast, ewma_volatility_fast, get_kernel
from src.data_loader import (
    align_trading_calendar,
    compute_portfolio_returns,
    compute_returns,
    load_config,
    prepare_and_save_processed_data,
    slice_point_in_time,
)
from src.eda_diagnostics import (
    compute_qq_coordinates,
    compute_rolling_stock_bond_correlation,
    compute_stylized_facts_table,
    run_adf_stationarity_table,
)


def test_cpp17_kernel_loaded_and_put_call_parity():
    """Verify the native C++17 kernel compiles, loads, and satisfies BSM put-call parity."""
    kernel = get_kernel()
    assert kernel is not None, "C++17 shared library should compile and load cleanly"

    S, K, r, q, sigma, T = 500.0, 490.0, 0.045, 0.013, 0.18, 30.0 / 365.0
    call_p = bsm_price_fast(S, K, r, q, sigma, T, is_call=True)
    put_p = bsm_price_fast(S, K, r, q, sigma, T, is_call=False)

    # Put-Call Parity: C - P = S * e^{-qT} - K * e^{-rT}
    lhs = call_p - put_p
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert np.isclose(lhs, rhs, atol=1e-10)

    # Verify analytical Greeks signs
    _, c_delta, c_gamma, c_vega, _ = bsm_greeks_fast(S, K, r, q, sigma, T, is_call=True)
    _, p_delta, p_gamma, p_vega, _ = bsm_greeks_fast(S, K, r, q, sigma, T, is_call=False)
    assert 0.0 < c_delta < 1.0
    assert -1.0 < p_delta < 0.0
    assert c_gamma > 0.0 and np.isclose(c_gamma, p_gamma, atol=1e-12)
    assert c_vega > 0.0 and np.isclose(c_vega, p_vega, atol=1e-12)


def test_data_hygiene_and_zero_lookahead_slicing():
    """Verify return transformations, cross-sectional portfolio aggregation, and F_{t-1} slicing."""
    cfg = load_config()
    bundle = prepare_and_save_processed_data(cfg)
    prices = bundle["prices"]
    arith_rets = bundle["arith_returns"]
    log_rets = bundle["log_returns"]
    port_df = bundle["portfolio"]

    # 1. Multi-period log return additivity
    core_cols = cfg["data"]["core_tickers"]
    ratio_from_log = np.exp(log_rets[core_cols].sum(axis=0))
    exact_ratio = prices[core_cols].iloc[-1] / prices[core_cols].iloc[0]
    assert np.allclose(ratio_from_log, exact_ratio, rtol=1e-10)

    # 2. Cross-sectional portfolio return identity
    w = np.array([cfg["portfolio"]["weights"][c] for c in core_cols])
    expected_port_ret = arith_rets[core_cols].values @ w
    assert np.allclose(port_df["portfolio_arith_return"].values, expected_port_ret, atol=1e-12)

    # 3. Strict point-in-time slicing (zero look-ahead bias)
    target_dt = pd.Timestamp("2021-06-15")
    sliced = slice_point_in_time(arith_rets, target_dt, lookback_days=250)
    assert len(sliced) == 250
    assert sliced.index.max() < target_dt


def test_phase0_stationarity_and_fat_tails():
    """Verify ADF stationarity tests and leptokurtic stylized facts (K > 3)."""
    cfg = load_config()
    bundle = prepare_and_save_processed_data(cfg)
    core_cols = cfg["data"]["core_tickers"]

    adf_df = run_adf_stationarity_table(bundle["prices"][core_cols], bundle["arith_returns"][core_cols])
    price_rows = adf_df[adf_df["Series"] == "Price Level (P_t)"]
    ret_rows = adf_df[adf_df["Series"] == "Daily Return (r_t)"]

    assert not price_rows["Stationary_at_5pct"].any(), "Raw price levels must be non-stationary (unit root)"
    assert ret_rows["Stationary_at_5pct"].all(), "Daily return series must be stationary"

    facts_df = compute_stylized_facts_table(
        bundle["arith_returns"][core_cols],
        portfolio_returns=bundle["portfolio"]["portfolio_arith_return"],
    )
    assert (facts_df["Total_Kurtosis"] > 3.0).all(), "Financial returns must exhibit leptokurtosis (K > 3)"
    assert facts_df.loc["PORTFOLIO", "Tail_Ratio_vs_Normal"] > 2.0

    # Verify Q-Q coordinates and rolling correlation
    theo_q, emp_q = compute_qq_coordinates(bundle["portfolio"]["portfolio_arith_return"])
    assert len(theo_q) == len(emp_q)
    corr_60 = compute_rolling_stock_bond_correlation(bundle["arith_returns"], "SPY", "TLT", window=60)
    assert corr_60.loc["2020-03-01":"2020-04-30"].mean() < -0.20
    assert corr_60.loc["2022-06-01":"2022-12-31"].mean() > 0.20
