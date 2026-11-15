"""
Unit Test Suite for Market Risk Modelling, Backtesting & Derivatives Framework (`tests/test_risk_engine.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting import (
    christoffersen_markov_tests,
    classify_basel_traffic_light,
    generate_hit_sequence,
    kupiec_pof_test,
    run_champion_challenger_backtest,
)
from src.cpp_kernel import bsm_greeks_fast, bsm_price_fast, ewma_volatility_fast, get_kernel
from src.data_loader import (
    align_trading_calendar,
    compute_portfolio_returns,
    compute_returns,
    load_config,
    prepare_and_save_processed_data,
    slice_point_in_time,
)
from src.derivatives_stress import (
    bsm_greeks,
    bsm_price,
    build_2d_stress_vulnerability_grid,
    compare_derivative_var_approaches,
    compute_irs_sensitivities,
    compute_stressed_var_comparison,
    evaluate_swap_approximation_breakdown,
    generate_spy_option_chain_snapshot,
    price_interest_rate_swap,
    run_historical_crisis_replay,
    solve_implied_volatility,
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
    egarch_filtered_hs_var_es,
    fit_egarch11_volatility,
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
    """Verify return transformations, cross-sectional portfolio aggregation, FX factor, and F_{t-1} slicing."""
    cfg = load_config()
    bundle = prepare_and_save_processed_data(cfg)
    prices = bundle["prices"]
    arith_rets = bundle["arith_returns"]
    log_rets = bundle["log_returns"]
    port_df = bundle["portfolio"]

    # Verify FXE (EUR/USD currency factor) is present alongside Equity, Rates, and Gold
    assert "FXE" in arith_rets.columns
    fx_port = compute_portfolio_returns(arith_rets, cfg["portfolio"]["multi_asset_fx_weights"])
    assert len(fx_port) == len(port_df)

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


def test_phase2_ewma_garch_egarch_fhs_and_hull_white_vwhs():
    """
    Verify RiskMetrics EWMA recursion, GARCH(1,1) stationarity, Asymmetric EGARCH(1,1)
    Filtered Historical Simulation (gamma_1 < 0 leverage effect), and Hull-White VWHS.
    """
    cfg = load_config()
    bundle = prepare_and_save_processed_data(cfg)
    port_ret = bundle["portfolio"]["portfolio_arith_return"]

    ewma_s = compute_ewma_volatility(port_ret, lambda_decay=0.94)
    assert len(ewma_s) == len(port_ret)
    assert (ewma_s > 0).all()

    g_norm = fit_garch11_volatility(port_ret, dist="normal")
    g_t = fit_garch11_volatility(port_ret, dist="t")
    eg_t = fit_egarch11_volatility(port_ret, dist="t")

    assert 0.50 < g_norm["params"]["persistence"] < 1.0
    assert 0.50 < g_t["params"]["persistence"] < 1.0
    assert 0.50 < eg_t["params"]["persistence"] < 1.0
    assert eg_t["params"]["gamma1_asymmetry"] < 0.0, "EGARCH(1,1) must capture negative leverage asymmetry"

    win_r = np.array([-0.02, -0.015, -0.01, 0.005, 0.01] * 50, dtype=np.float64)
    win_v = np.full_like(win_r, 0.01)
    hw_var, hw_es = volatility_weighted_hs_var_es(
        win_r, win_v, current_vol_forecast=0.025, alpha=0.99, notional_usd=10_000_000.0
    )
    eg_var, eg_es = egarch_filtered_hs_var_es(
        win_r, win_v, current_egarch_vol_forecast=0.025, mu_daily=0.0, alpha=0.99, notional_usd=10_000_000.0
    )
    max_unscaled_loss = np.max(-win_r) * 10_000_000.0  # $200,000
    assert hw_var > max_unscaled_loss
    assert eg_var > max_unscaled_loss
    assert hw_es >= hw_var and eg_es >= eg_var


def test_phase3_kupiec_and_christoffersen_spaced_vs_clustered():
    """
    Verify Basel Traffic Light zones, Kupiec POF test (LR_UC = 0 when x/N == 1 - alpha),
    and Christoffersen Markov Independence test distinguishing spaced vs. clustered exceptions.
    """
    assert classify_basel_traffic_light(3)["zone"] == "Green"
    assert classify_basel_traffic_light(7)["zone"] == "Yellow"
    assert classify_basel_traffic_light(12)["zone"] == "Red"

    n_days = 1000
    spaced_hits = np.zeros(n_days, dtype=np.int32)
    spaced_hits[np.arange(50, 1000, 100)] = 1  # 10 evenly spaced breaches (n11 = 0)

    clustered_hits = np.zeros(n_days, dtype=np.int32)
    clustered_hits[100:105] = 1  # 5 consecutive breaches
    clustered_hits[500:505] = 1  # 5 consecutive breaches (n11 = 8)

    res_spaced = christoffersen_markov_tests(spaced_hits, alpha=0.99)
    res_clustered = christoffersen_markov_tests(clustered_hits, alpha=0.99)

    assert np.isclose(res_spaced["lr_uc"], 0.0, atol=1e-12)
    assert np.isclose(res_clustered["lr_uc"], 0.0, atol=1e-12)
    assert res_spaced["reject_uc_5pct"] is False
    assert res_clustered["reject_uc_5pct"] is False

    assert res_spaced["n11"] == 0
    assert res_spaced["reject_ind_5pct"] is False
    assert res_spaced["reject_cc_5pct"] is False

    assert res_clustered["n11"] == 8
    assert res_clustered["reject_ind_5pct"] is True
    assert res_clustered["reject_cc_5pct"] is True


def test_phase4_bsm_iv_solver_straddle_swap_and_stressed_var():
    """
    Verify BSM Implied Volatility inversion accuracy, Delta-Normal failure on a
    Delta-Neutral Straddle, 5Y Interest-Rate Swap valuation & DV01 convexity breakdown,
    and Basel 2.5 Stressed VaR (sVaR > Normal VaR).
    """
    # 1. IV Inversion round-trip accuracy
    S, K, r, q, true_sigma, T = 500.0, 480.0, 0.045, 0.013, 0.215, 30.0 / 365.0
    for is_call in (True, False):
        mkt_p = bsm_price(S, K, r, q, true_sigma, T, is_call=is_call)
        solved_iv = solve_implied_volatility(mkt_p, S, K, r, q, T, is_call=is_call)
        assert np.isclose(solved_iv, true_sigma, atol=1e-7)

    # 2. Option chain snapshot & skew verification
    chain = generate_spy_option_chain_snapshot(spot_price=500.0)
    assert chain["IV_Inversion_Error"].max() < 1e-6

    # 3. Delta-Normal vs. Delta-Gamma-Vega vs. Full Revaluation MC on Straddle & 5Y IRS
    comp_df, _ = compare_derivative_var_approaches(spot_price=500.0, num_mc_scenarios=10_000)
    short_straddle = comp_df[comp_df["Portfolio_Structure"].str.contains("Short Delta-Neutral Straddle")].iloc[0]
    assert np.isclose(short_straddle["Net_Delta_Shares"], 0.0, atol=1e-6)
    assert np.isclose(short_straddle["Delta_Normal_VaR_99_USD"], 0.0, atol=1e-6)
    assert short_straddle["Delta_Gamma_Vega_VaR_99_USD"] > 30_000.0
    assert short_straddle["Full_Reval_MC_VaR_99_USD"] > 30_000.0

    # 4. 5-Year $5M Pay-Fixed Interest-Rate Swap par valuation & convexity breakdown
    par_npv = price_interest_rate_swap(5_000_000.0, 0.042, 0.042, tenor_years=5.0, pay_fixed=True)
    assert np.isclose(par_npv, 0.0, atol=1e-8), "Par swap must have zero initial NPV"
    swap_bk = evaluate_swap_approximation_breakdown(5_000_000.0)
    minus_300 = swap_bk[swap_bk["Yield_Shock_bps"] == -300.0].iloc[0]
    assert abs(minus_300["Linear_DV01_Error_USD"]) > 40_000.0, "Linear DV01 must break down under -300 bps shock"
    assert abs(minus_300["Quadratic_Error_USD"]) < abs(minus_300["Linear_DV01_Error_USD"]) * 0.10

    # 5. Basel 2.5 Stressed VaR (sVaR) > Normal-Period VaR
    cfg = load_config()
    bundle = prepare_and_save_processed_data(cfg)
    svar_df = compute_stressed_var_comparison(bundle["arith_returns"], bundle["vix"], cfg["portfolio"]["weights"])
    norm_var = svar_df.loc["Normal Period (2017 Low-Vol Calm)", "Historical_Sim_VaR_99_USD"]
    covid_svar = svar_df.loc["Stressed Window A (2020 COVID-19 Crash)", "Historical_Sim_VaR_99_USD"]
    rates_svar = svar_df.loc["Stressed Window B (2022 Rate-Hike Bear)", "Historical_Sim_VaR_99_USD"]
    assert covid_svar > 2.5 * norm_var
    assert rates_svar > 2.0 * norm_var

    # 6. Stress Grid: Protective Put reduces loss in severe crash (-20% spot, +25 vol pts)
    unhedged_grid, hedged_grid = build_2d_stress_vulnerability_grid(spot_price=500.0)
    crash_unhedged = unhedged_grid.loc["Spot -20%", "IV +25 pts"]
    crash_hedged = hedged_grid.loc["Spot -20%", "IV +25 pts"]
    assert crash_hedged > crash_unhedged, "Protective put must cushion portfolio loss during a -20% crash"


def test_phase5_ima_five_pillars_rniv_irc_irrbb_and_frtb():
    """
    Verify Phase 5 IMA Regulatory Capital coverage across all 5 asset classes (Equity, Credit,
    Rates, FX, Inflation), RNIV capital add-ons, 1-Year 99.9% Merton Copula IRC & Economic Capital,
    6-shock Basel IRRBB (Delta EVE & Delta NII), and FRTB Liquidity-Horizon-Adjusted 97.5% ES.
    """
    from src.regulatory_capital_frtb import (
        build_ima_regulatory_capital_bridge,
        compute_five_pillar_var_attribution,
        compute_frtb_liquidity_adjusted_es,
        compute_rniv_capital_addons,
        evaluate_irrbb_supervisory_shocks,
        simulate_irc_and_economic_capital,
    )

    cfg = load_config()
    bundle = prepare_and_save_processed_data(cfg)
    arith_rets = bundle["arith_returns"]
    vix_df = bundle["vix"]

    # 1. Verify all 5 IMA risk factor pillars (Equity, Credit, Rates, FX, Inflation)
    assert {"SPY", "AAPL", "TLT", "FXE", "LQD", "TIP", "GLD"}.issubset(set(arith_rets.columns))
    inst_df, pillar_df = compute_five_pillar_var_attribution(arith_rets, notional_usd=10_000_000.0, alpha=0.99)
    assert set(pillar_df["Risk_Class_Pillar"]) == {"Equity", "Credit", "Rates", "FX", "Inflation"}
    assert np.isclose(pillar_df["Share_of_Total_VaR_pct"].sum(), 100.0, atol=1e-5)
    assert pillar_df["Diversification_Benefit_USD"].sum() > 50_000.0

    # 2. Verify RNIV (Risks Not In VaR) across Vol Skew, Yield Curve Basis, and Credit Basis
    rniv_df, rniv_sum = compute_rniv_capital_addons(arith_rets, vix_df, alpha=0.99)
    assert len(rniv_df) == 3
    assert rniv_sum["total_rniv_capital_charge_usd"] > 200_000.0
    assert rniv_sum["total_stressed_rniv_1d_99_var_usd"] > rniv_sum["total_rniv_1d_99_var_usd"]

    # 3. Verify 1-Year 99.9% IRC (Default + Migration >= Default-Only DRC) & Economic Capital
    issuer_df, irc_ec = simulate_irc_and_economic_capital(num_simulations=20_000, seed=42)
    assert len(issuer_df) == 6
    assert irc_ec["irc_default_plus_migration_999_usd"] >= irc_ec["drc_default_only_999_usd"] > 0.0
    assert irc_ec["irc_es_999_usd"] >= irc_ec["irc_default_plus_migration_999_usd"]
    assert irc_ec["ec_diversification_benefit_usd"] > 100_000.0

    # 4. Verify IRRBB (6 Basel d368 shocks: Delta EVE & Delta NII + ALCO Swap Hedge)
    gap_df, irrbb_df = evaluate_irrbb_supervisory_shocks(tier1_capital_usd=15_000_000.0)
    assert len(gap_df) == 8
    assert len(irrbb_df) == 6
    assert irrbb_df["Unhedged_Delta_EVE_USD"].min() < -0.15 * 15_000_000.0
    assert irrbb_df["Basel_15pct_Outlier_Pass"].all()

    # 5. Verify FRTB Liquidity-Horizon-Adjusted 97.5% ES & Capital Bridge
    frtb_df, frtb_sum = compute_frtb_liquidity_adjusted_es(arith_rets, vix_df, notional_usd=10_000_000.0)
    assert len(frtb_df) == 3
    assert frtb_sum["frtb_liquidity_adjusted_es_current_usd"] > frtb_sum["es_975_10d_unadjusted_current_usd"] > frtb_sum["es_975_1d_current_usd"]
    assert frtb_sum["frtb_liquidity_adjusted_es_stressed_usd"] > frtb_sum["frtb_liquidity_adjusted_es_current_usd"]

    bridge_df = build_ima_regulatory_capital_bridge(arith_rets, vix_df, notional_usd=10_000_000.0)
    assert len(bridge_df) == 4
    assert bridge_df.iloc[-1]["Basel_2_5_IMA_Capital_USD"] > 1_000_000.0
    assert bridge_df.iloc[-1]["FRTB_IMA_Capital_USD"] > 1_000_000.0
