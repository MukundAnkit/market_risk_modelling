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
from src.var_estimators import (
    build_phase1_comparison_table,
    cholesky_monte_carlo_var_es,
    compute_pretrade_marginal_var_sizing,
    compute_subadditivity_and_diversification,
    cornish_fisher_var_es,
    historical_simulation_var_es,
    parametric_lognormal_var_es,
    parametric_normal_var_es,
    parametric_student_t_var_es,
)
from src.volatility_models import (
    age_weighted_hs_var_es,
    build_dynamic_var_series,
    compute_ewma_volatility,
    fit_garch11_volatility,
    volatility_weighted_hs_var_es,
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


def test_phase1_closed_form_identities_and_subadditivity():
    """
    Verify closed-form Gaussian VaR/ES identities:
      VaR_{0.99}^{Norm} = 2.32634787 * sigma
      ES_{0.975}^{Norm} = 2.33780279 * sigma
    as well as subadditivity and pre-trade Marginal VaR sizing.
    """
    rng = np.random.default_rng(123)
    n = 5000
    raw = rng.standard_normal(n)
    raw = (raw - np.mean(raw)) / np.std(raw, ddof=1) * 0.01  # exact mu=0, sigma=0.01
    df_syn = pd.DataFrame({"A": raw})
    w_syn = {"A": 1.0}

    est_99 = parametric_normal_var_es(df_syn, w_syn, alpha=0.99, notional_usd=1.0, lookback_days=n, zero_mean=True)
    est_975 = parametric_normal_var_es(df_syn, w_syn, alpha=0.975, notional_usd=1.0, lookback_days=n, zero_mean=True)

    assert np.isclose(est_99.var_pct / 0.01, 2.32634787, atol=1e-5)
    assert np.isclose(est_975.es_pct / 0.01, 2.33780279, atol=1e-5)
    assert est_99.es_usd > est_99.var_usd

    cfg = load_config()
    bundle = prepare_and_save_processed_data(cfg)
    div_df, coherence_proof = compute_subadditivity_and_diversification(
        bundle["arith_returns"], cfg["portfolio"]["weights"], alpha=0.99
    )
    benefit_row = div_df[div_df["Position"] == "Diversification Benefit ($ Saved)"].iloc[0]
    assert benefit_row["Normal_VaR_99_USD"] > 0.0
    assert benefit_row["HS_VaR_99_USD"] > 0.0
    assert coherence_proof["var_violates_subadditivity"] is True
    assert coherence_proof["es_satisfies_subadditivity"] is True

    sizing = compute_pretrade_marginal_var_sizing(
        bundle["arith_returns"],
        base_weights=cfg["portfolio"]["weights"],
        candidate_ticker="TSLA",
        cro_var_99_limit_usd=200_000.0,
    )
    comp_sum = sizing["component_var_table"]["Component_VaR_99_USD"].sum()
    assert np.isclose(comp_sum, sizing["baseline_normal_var_99_usd"], rtol=1e-6)
    assert sizing["max_allowable_tsla_normal_usd"] > 0.0
    assert sizing["max_allowable_tsla_hs_usd"] > 0.0


def test_phase2_ewma_garch_and_hull_white_vwhs():
    """
    Verify RiskMetrics EWMA recursion, GARCH(1,1) stationarity (alpha_1 + beta_1 < 1),
    and that Hull-White Volatility-Weighted HS can exceed the maximum historical loss
    in the lookback window when current volatility spikes.
    """
    cfg = load_config()
    bundle = prepare_and_save_processed_data(cfg)
    port_ret = bundle["portfolio"]["portfolio_arith_return"]

    # 1. EWMA Volatility
    ewma_s = compute_ewma_volatility(port_ret, lambda_decay=0.94)
    assert len(ewma_s) == len(port_ret)
    assert (ewma_s > 0).all()

    # 2. GARCH(1,1) Stationarity
    g_norm = fit_garch11_volatility(port_ret, dist="normal")
    g_t = fit_garch11_volatility(port_ret, dist="t")
    assert 0.50 < g_norm["params"]["persistence"] < 1.0
    assert 0.50 < g_t["params"]["persistence"] < 1.0
    assert g_t["params"]["nu_df"] > 2.5

    # 3. Hull-White VWHS exceeding max window loss when sigma_T > max(sigma_i)
    win_r = np.array([-0.02, -0.015, -0.01, 0.005, 0.01] * 50, dtype=np.float64)
    win_v = np.full_like(win_r, 0.01)
    # Suppose current volatility forecast spikes to 0.025 (2.5x historical window vol)
    hw_var, hw_es = volatility_weighted_hs_var_es(
        win_r, win_v, current_vol_forecast=0.025, alpha=0.99, notional_usd=10_000_000.0
    )
    max_unscaled_loss = np.max(-win_r) * 10_000_000.0  # $200,000
    assert hw_var > max_unscaled_loss, "Hull-White VWHS must exceed max historical window loss during a vol spike"
    assert hw_es >= hw_var
