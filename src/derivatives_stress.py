"""
Derivative Portfolio Risk & Scenario Stress Testing (`src/derivatives_stress.py`).

Implements Phase 4 (Run) methodology (accelerated by the C++17 quantitative kernel):
1. Black-Scholes-Merton (BSM) European Call/Put Pricing & Implied Volatility Inversion:
     sigma_IV = f^{-1}(P_mkt, S, K, r, T, q)
   using both C++17 safeguarded Newton-Raphson/bisection and `scipy.optimize.brentq`,
   plus an empirical `SPY` option chain with realistic equity volatility skew across strike prices.
2. Closed-Form Analytical Greeks:
     Delta (dV/dS), Gamma (d^2V/dS^2), Vega (dV/dsigma_IV), and Theta (dV/dt).
3. Comparing Three Derivative VaR & ES Approaches across:
   - Position A: Protective Put Hedge on the $3,000,000 `SPY` allocation (5% OTM Put)
   - Position B: Delta-Neutral `SPY` Straddle (ATM Call + ATM Put, where Delta approx 0)
   Approaches compared:
   - Linear Delta-Normal Approximation: dV approx Delta * dS
     (Proves why Delta-Normal VaR erroneously reports near-zero risk for a delta-neutral straddle!)
   - Quadratic Delta-Gamma-Vega Taylor Approximation:
     dV approx Delta * dS + 0.5 * Gamma * (dS)^2 + Vega * d(sigma_IV) + Theta * dt
   - Full Revaluation Monte Carlo Simulation (M = 20,000 joint scenarios of SPY spot returns
     and implied volatility VIX shocks, repriced via the C++17 BSM kernel).
4. Historical Crisis Replay & 2D Hypothetical Stress Testing:
   - Historical Replay of hedged vs. unhedged portfolio through March 2020 (COVID-19 crash)
     and 2022 (simultaneous stock-bond selloff).
   - 5x5 Hypothetical Spot x Implied Volatility Stress Matrix (-20% to +10% spot vs. -5 to +25 vol pts).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from src.cpp_kernel import bsm_greeks_fast, bsm_price_fast, get_kernel


@dataclass
class OptionContract:
    """Representation of a European option contract on `SPY`."""

    label: str
    spot: float
    strike: float
    rate: float
    div_yield: float
    sigma_iv: float
    time_to_expiry: float
    is_call: bool
    contracts_or_shares: float  # Total underlying share equivalent (e.g., 100 shares per contract)


def bsm_price(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    is_call: bool = True,
) -> float:
    """Exact Black-Scholes-Merton European option price (delegates to C++17 kernel)."""
    return bsm_price_fast(S, K, r, q, sigma, T, is_call=is_call)


def bsm_greeks(
    S: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    is_call: bool = True,
) -> Dict[str, float]:
    """
    Compute closed-form BSM price and analytical Greeks:
      - `price`: Option value per share ($)
      - `delta`: dV/dS
      - `gamma`: d^2V/dS^2
      - `vega`: dV/dsigma (per 1.00 = 100 vol points; divide by 100 for 1 vol point)
      - `theta`: dV/dt (per year; divide by 252 for 1 trading day)
    """
    p, d, g, v, th = bsm_greeks_fast(S, K, r, q, sigma, T, is_call=is_call)
    return {
        "price": p,
        "delta": d,
        "gamma": g,
        "vega": v,
        "vega_per_vol_pt": v / 100.0,
        "theta_annual": th,
        "theta_daily": th / 252.0,
    }


def solve_implied_volatility(
    market_price: float,
    S: float,
    K: float,
    r: float,
    q: float,
    T: float,
    is_call: bool = True,
    tol: float = 1e-8,
) -> float:
    """
    Solve for Black-Scholes-Merton Implied Volatility sigma_IV = f^{-1}(P_mkt, S, K, r, T, q)
    using Brent's root-finding method (`scipy.optimize.brentq`) cross-checked with the C++17 solver.
    """
    lib = get_kernel()
    if lib is not None:
        iv_cpp = float(
            lib.cpp_bsm_implied_vol(
                float(market_price),
                float(S),
                float(K),
                float(r),
                float(q),
                float(T),
                1 if is_call else 0,
                float(tol),
                100,
            )
        )
        if iv_cpp > 0.0 and np.isclose(
            bsm_price(S, K, r, q, iv_cpp, T, is_call=is_call), market_price, atol=1e-6
        ):
            return iv_cpp

    def _obj(sig: float) -> float:
        return bsm_price(S, K, r, q, sig, T, is_call=is_call) - market_price

    return float(optimize.brentq(_obj, 1e-5, 5.0, xtol=tol))


def generate_spy_option_chain_snapshot(
    spot_price: float = 500.0,
    r: float = 0.045,
    q: float = 0.013,
    time_to_expiry_days: int = 30,
    atm_iv: float = 0.165,
) -> pd.DataFrame:
    """
    Generate (or fetch) an institutional `SPY` option chain across strike moneyness (85% to 112%)
    incorporating empirical equity index volatility skew (smirk), bid/ask quotes, mid prices,
    and inverted Implied Volatilities + Greeks.
    """
    T = time_to_expiry_days / 365.0
    moneyness_grid = np.linspace(0.85, 1.12, 19)
    strikes = np.round(spot_price * moneyness_grid, 0)

    rows: List[Dict[str, float | str]] = []
    for K in strikes:
        m = K / spot_price
        # Empirical S&P 500 downside put skew + slight upside smile curvature
        true_iv = atm_iv - 0.38 * (m - 1.0) + 0.85 * ((m - 1.0) ** 2)
        true_iv = float(np.clip(true_iv, 0.08, 0.65))

        for opt_type, is_c in [("PUT", False), ("CALL", True)]:
            theo_mid = bsm_price(spot_price, float(K), r, q, true_iv, T, is_call=is_c)
            half_spread = min(0.01 + 0.006 * theo_mid, 0.5 * theo_mid)
            bid = theo_mid - half_spread
            ask = theo_mid + half_spread
            mkt_mid = 0.5 * (bid + ask)

            solved_iv = solve_implied_volatility(
                mkt_mid, spot_price, float(K), r, q, T, is_call=is_c
            )
            g = bsm_greeks(spot_price, float(K), r, q, solved_iv, T, is_call=is_c)

            rows.append(
                {
                    "Option_Type": opt_type,
                    "Spot": spot_price,
                    "Strike": float(K),
                    "Moneyness": float(K / spot_price),
                    "Bid": bid,
                    "Ask": ask,
                    "Market_Mid": mkt_mid,
                    "True_Surface_IV_pct": true_iv * 100.0,
                    "Inverted_IV_pct": solved_iv * 100.0,
                    "IV_Inversion_Error": abs(solved_iv - true_iv),
                    "Delta": g["delta"],
                    "Gamma": g["gamma"],
                    "Vega_per_1pct_Vol": g["vega_per_vol_pt"],
                    "Theta_Daily": g["theta_daily"],
                }
            )
    return pd.DataFrame(rows)


def compare_derivative_var_approaches(
    spot_price: float = 500.0,
    spy_notional_usd: float = 3_000_000.0,
    r: float = 0.045,
    q: float = 0.013,
    atm_iv: float = 0.165,
    time_to_expiry_days: int = 30,
    daily_spot_vol: float = 0.0110,
    daily_iv_vol: float = 0.0145,
    spot_iv_corr: float = -0.72,
    num_mc_scenarios: int = 20_000,
    alpha: float = 0.99,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Compare 1-Day 99% VaR and Expected Shortfall across three derivative risk engines:
      1. Linear Delta-Normal Approximation: dV approx Delta * dS
      2. Quadratic Delta-Gamma-Vega Taylor Approximation:
         dV approx Delta * dS + 0.5 * Gamma * (dS)^2 + Vega * d(sigma_IV) + Theta * dt
      3. Full Revaluation Monte Carlo Simulation (M = 20,000 joint spot & IV scenarios via C++17 BSM kernel)

    Evaluated on three representative trading book structures:
      - Structure 1: Long ATM Delta-Neutral Straddle (Long Call + Long Put at K=S, notional = 6,000 SPY shares)
      - Structure 2: Short ATM Delta-Neutral Straddle (Short Call + Short Put at K=S, notional = -6,000 SPY shares)
        -> Highlights catastrophic failure of Linear Delta-Normal VaR (reports ~$0 VaR on short gamma/vega!)
      - Structure 3: Hedged SPY Equity Book ($3M SPY Long + 5% OTM Protective Put Hedge)
    """
    T = time_to_expiry_days / 365.0
    dt = 1.0 / 252.0
    T_next = max(T - dt, 1e-6)
    shares_spy = spy_notional_usd / spot_price  # 6,000 shares for $3M at S=$500

    # Simulate M = 20,000 joint correlated scenarios of (dS/S, d_sigma_IV)
    rng = np.random.default_rng(seed)
    cov_2d = np.array(
        [
            [daily_spot_vol ** 2, spot_iv_corr * daily_spot_vol * daily_iv_vol],
            [spot_iv_corr * daily_spot_vol * daily_iv_vol, daily_iv_vol ** 2],
        ],
        dtype=np.float64,
    )
    chol_2d = np.linalg.cholesky(cov_2d)
    half_m = num_mc_scenarios // 2
    z_half = rng.standard_normal(size=(2, half_m))
    z_mat = np.concatenate([z_half, -z_half], axis=1)
    shocks = chol_2d @ z_mat

    ret_spot_sim = shocks[0, :]
    d_iv_sim = shocks[1, :]
    spot_sim = np.ascontiguousarray(spot_price * (1.0 + ret_spot_sim), dtype=np.float64)
    dS_sim = spot_sim - spot_price

    # Construct exact delta-neutral strike K_dn = S * exp((r - q + 0.5 * sigma^2) * T)
    # where N(d1) = 0.5 => Call_Delta + Put_Delta = e^{-qT} * (2*N(d1) - 1) == 0.0
    K_dn = float(spot_price * np.exp((r - q + 0.5 * (atm_iv ** 2)) * T))
    g_call_dn = bsm_greeks(spot_price, K_dn, r, q, atm_iv, T, is_call=True)
    g_put_dn = bsm_greeks(spot_price, K_dn, r, q, atm_iv, T, is_call=False)

    # Protective Put strike (5% OTM: K_put = 0.95 * S)
    K_prot = float(0.95 * spot_price)
    iv_prot = atm_iv + 0.022  # Downside skew on 5% OTM put
    g_prot_put = bsm_greeks(spot_price, K_prot, r, q, iv_prot, T, is_call=False)

    lib = get_kernel()

    def _revalue_leg_mc(
        S_vec: np.ndarray,
        iv_vec: np.ndarray,
        K_val: float,
        base_p: float,
        units: float,
        is_c: bool,
    ) -> np.ndarray:
        out_pnl = np.empty(len(S_vec), dtype=np.float64)
        if lib is not None:
            lib.cpp_bsm_mc_revaluation(
                S_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                iv_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int(len(S_vec)),
                ctypes.c_double(K_val),
                ctypes.c_double(r),
                ctypes.c_double(q),
                ctypes.c_double(T_next),
                ctypes.c_int(1 if is_c else 0),
                ctypes.c_double(base_p),
                ctypes.c_double(units),
                out_pnl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            return out_pnl
        for idx in range(len(S_vec)):
            p_new = bsm_price(float(S_vec[idx]), K_val, r, q, float(iv_vec[idx]), T_next, is_call=is_c)
            out_pnl[idx] = units * (p_new - base_p)
        return out_pnl

    iv_sim_atm = np.ascontiguousarray(np.clip(atm_iv + d_iv_sim, 0.03, 2.0), dtype=np.float64)
    iv_sim_prot = np.ascontiguousarray(np.clip(iv_prot + d_iv_sim, 0.03, 2.0), dtype=np.float64)

    # Define the 3 Option Book Structures
    structures = [
        {
            "name": "Long Delta-Neutral Straddle (6,000 sh)",
            "delta_net": shares_spy * (g_call_dn["delta"] + g_put_dn["delta"]),
            "gamma_net": shares_spy * (g_call_dn["gamma"] + g_put_dn["gamma"]),
            "vega_net": shares_spy * (g_call_dn["vega"] + g_put_dn["vega"]),
            "theta_net": shares_spy * (g_call_dn["theta_annual"] + g_put_dn["theta_annual"]),
            "full_mc_pnl": _revalue_leg_mc(spot_sim, iv_sim_atm, K_dn, g_call_dn["price"], shares_spy, True)
            + _revalue_leg_mc(spot_sim, iv_sim_atm, K_dn, g_put_dn["price"], shares_spy, False),
        },
        {
            "name": "Short Delta-Neutral Straddle (-6,000 sh)",
            "delta_net": -shares_spy * (g_call_dn["delta"] + g_put_dn["delta"]),
            "gamma_net": -shares_spy * (g_call_dn["gamma"] + g_put_dn["gamma"]),
            "vega_net": -shares_spy * (g_call_dn["vega"] + g_put_dn["vega"]),
            "theta_net": -shares_spy * (g_call_dn["theta_annual"] + g_put_dn["theta_annual"]),
            "full_mc_pnl": _revalue_leg_mc(spot_sim, iv_sim_atm, K_dn, g_call_dn["price"], -shares_spy, True)
            + _revalue_leg_mc(spot_sim, iv_sim_atm, K_dn, g_put_dn["price"], -shares_spy, False),
        },
        {
            "name": "Hedged $3M SPY + 5% OTM Protective Put",
            "delta_net": shares_spy * (1.0 + g_prot_put["delta"]),
            "gamma_net": shares_spy * g_prot_put["gamma"],
            "vega_net": shares_spy * g_prot_put["vega"],
            "theta_net": shares_spy * g_prot_put["theta_annual"],
            "full_mc_pnl": shares_spy * dS_sim
            + _revalue_leg_mc(spot_sim, iv_sim_prot, K_prot, g_prot_put["price"], shares_spy, False),
        },
    ]

    z_alpha = float(stats.norm.ppf(alpha))
    phi_z = float(stats.norm.pdf(z_alpha))
    records: List[Dict[str, object]] = []
    pnl_distributions: Dict[str, np.ndarray] = {}

    for st in structures:
        d_net = float(st["delta_net"])
        g_net = float(st["gamma_net"])
        v_net = float(st["vega_net"])
        th_net = float(st["theta_net"])
        mc_pnl = st["full_mc_pnl"]

        # 1. Linear Delta-Normal P&L & Analytic VaR/ES
        dn_pnl = d_net * dS_sim
        sd_dn = abs(d_net) * spot_price * daily_spot_vol
        dn_var = z_alpha * sd_dn
        dn_es = (phi_z / (1.0 - alpha)) * sd_dn

        # 2. Quadratic Delta-Gamma-Vega Taylor Expansion P&L
        dgv_pnl = (
            d_net * dS_sim
            + 0.5 * g_net * (dS_sim ** 2)
            + v_net * d_iv_sim
            + th_net * dt
        )
        dgv_losses = -dgv_pnl
        dgv_var = float(np.quantile(dgv_losses, alpha))
        dgv_es = float(np.mean(dgv_losses[dgv_losses >= dgv_var]))

        # 3. Full Revaluation Monte Carlo P&L
        mc_losses = -mc_pnl
        mc_var = float(np.quantile(mc_losses, alpha))
        mc_es = float(np.mean(mc_losses[mc_losses >= mc_var]))

        pnl_distributions[f"{st['name']} - Delta-Normal"] = dn_pnl
        pnl_distributions[f"{st['name']} - Delta-Gamma-Vega"] = dgv_pnl
        pnl_distributions[f"{st['name']} - Full Revaluation MC"] = mc_pnl

        records.append(
            {
                "Portfolio_Structure": st["name"],
                "Net_Delta_Shares": d_net,
                "Net_Gamma_Dollar": 0.5 * g_net * ((0.01 * spot_price) ** 2),
                "Net_Vega_per_1pct_Vol": v_net / 100.0,
                "Delta_Normal_VaR_99_USD": dn_var,
                "Delta_Gamma_Vega_VaR_99_USD": dgv_var,
                "Full_Reval_MC_VaR_99_USD": mc_var,
                "Delta_Normal_ES_99_USD": dn_es,
                "Delta_Gamma_Vega_ES_99_USD": dgv_es,
                "Full_Reval_MC_ES_99_USD": mc_es,
                "Delta_Normal_Understatement_USD": mc_var - dn_var,
            }
        )

    return pd.DataFrame(records), pnl_distributions


def run_historical_crisis_replay(
    arith_returns: pd.DataFrame,
    vix_df: pd.DataFrame,
    weights: Dict[str, float],
    notional_usd: float = 10_000_000.0,
    r: float = 0.045,
    q: float = 0.013,
) -> pd.DataFrame:
    """
    Replay the exact daily factor shocks of:
      1. 2020 COVID-19 Crash ("2020-02-19" to "2020-03-31")
      2. 2022 Inflation & Rate-Hike Selloff ("2022-01-03" to "2022-06-30")
    comparing the Unhedged $10M Linear Portfolio against the Hedged Portfolio
    (where the $3M SPY equity sleeve is protected by a rolling 30-day 5% OTM Put).
    """
    windows = [
        ("2020 COVID-19 Crash (Feb–Mar 2020)", "2020-02-19", "2020-03-31"),
        ("2022 Stock-Bond Selloff (H1 2022)", "2022-01-03", "2022-06-30"),
    ]
    core_tickers = list(weights.keys())
    w_vec = np.array([weights[t] for t in core_tickers], dtype=np.float64)
    spy_alloc_usd = weights["SPY"] * notional_usd

    summary_rows: List[Dict[str, object]] = []
    for label, s_dt, e_dt in windows:
        sub_r = arith_returns.loc[s_dt:e_dt, core_tickers].copy()
        sub_vix = vix_df.loc[sub_r.index, "CLOSE"] / 100.0
        d_vix = sub_vix.diff().fillna(0.0).values

        unhedged_daily_pnl = (sub_r.values @ w_vec) * notional_usd
        spy_r = sub_r["SPY"].values

        # Revalue 30-day 5% OTM protective put daily using exact BSM with realized SPY return and VIX change
        S_ref = 500.0
        K_put = 0.95 * S_ref
        T_init = 30.0 / 365.0
        dt = 1.0 / 252.0
        n_shares = spy_alloc_usd / S_ref

        put_daily_pnl = np.zeros(len(sub_r), dtype=np.float64)
        for i in range(len(sub_r)):
            iv_prev = float(sub_vix.iloc[i - 1]) if i > 0 else float(sub_vix.iloc[0])
            iv_curr = float(sub_vix.iloc[i])
            p_prev = bsm_price(S_ref, K_put, r, q, iv_prev, T_init, is_call=False)
            S_curr = S_ref * (1.0 + float(spy_r[i]))
            p_curr = bsm_price(S_curr, K_put, r, q, iv_curr, T_init - dt, is_call=False)
            put_daily_pnl[i] = n_shares * (p_curr - p_prev)

        hedged_daily_pnl = unhedged_daily_pnl + put_daily_pnl

        unhedged_cum_pnl = float(np.sum(unhedged_daily_pnl))
        hedged_cum_pnl = float(np.sum(hedged_daily_pnl))
        worst_1d_unhedged = float(np.min(unhedged_daily_pnl))
        worst_1d_hedged = float(np.min(hedged_daily_pnl))

        summary_rows.append(
            {
                "Crisis_Scenario": label,
                "Trading_Days": len(sub_r),
                "Unhedged_Cumulative_PnL_USD": unhedged_cum_pnl,
                "Hedged_Cumulative_PnL_USD": hedged_cum_pnl,
                "Put_Hedge_Gain_USD": hedged_cum_pnl - unhedged_cum_pnl,
                "Worst_1Day_Unhedged_PnL_USD": worst_1d_unhedged,
                "Worst_1Day_Hedged_PnL_USD": worst_1d_hedged,
                "Tail_Loss_Reduction_pct": (1.0 - abs(worst_1d_hedged) / abs(worst_1d_unhedged)) * 100.0,
            }
        )

    return pd.DataFrame(summary_rows).set_index("Crisis_Scenario")


def build_2d_stress_vulnerability_grid(
    spot_price: float = 500.0,
    spy_notional_usd: float = 3_000_000.0,
    other_assets_notional_usd: float = 7_000_000.0,
    r: float = 0.045,
    q: float = 0.013,
    base_iv: float = 0.165,
    time_to_expiry_days: int = 30,
    spot_shocks_pct: Optional[List[float]] = None,
    vol_shocks_pts: Optional[List[float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a 5x5 2D Hypothetical Stress Vulnerability Grid combining underlying equity spot
    shocks (-20% to +10%) with implied volatility shocks (-5 to +25 vol points = -0.05 to +0.25):
      - Returns (`unhedged_grid_df`, `hedged_grid_df`) showing portfolio P&L ($ USD) across the 5x5 matrix.
    """
    s_shocks = spot_shocks_pct or [-0.20, -0.10, -0.05, 0.00, 0.10]
    v_shocks = vol_shocks_pts or [-0.05, 0.00, 0.05, 0.15, 0.25]

    T = time_to_expiry_days / 365.0
    shares_spy = spy_notional_usd / spot_price
    K_prot = 0.95 * spot_price
    base_put_iv = base_iv + 0.022
    base_put_price = bsm_price(spot_price, K_prot, r, q, base_put_iv, T, is_call=False)

    # Non-SPY sleeve beta to equity crash (AAPL 20% has beta ~1.15, TLT 30% + GLD 20% provide partial offset)
    # Net non-SPY portfolio sensitivity to SPY shock:
    non_spy_beta_pnl_per_pct = 2_000_000.0 * 1.15 - 3_000_000.0 * 0.25 + 2_000_000.0 * 0.05  # +$1.65M effective equity beta

    row_labels = [f"Spot {int(s * 100):+d}%" for s in s_shocks]
    col_labels = [f"IV {int(v * 100):+d} pts" for v in v_shocks]

    unhedged_mat = np.zeros((len(s_shocks), len(v_shocks)), dtype=np.float64)
    hedged_mat = np.zeros((len(s_shocks), len(v_shocks)), dtype=np.float64)

    for i, ds in enumerate(s_shocks):
        S_stressed = spot_price * (1.0 + ds)
        linear_book_pnl = spy_notional_usd * ds + non_spy_beta_pnl_per_pct * ds

        for j, dv in enumerate(v_shocks):
            iv_stressed = max(0.04, base_put_iv + dv)
            stressed_put_price = bsm_price(S_stressed, K_prot, r, q, iv_stressed, T, is_call=False)
            put_hedge_pnl = shares_spy * (stressed_put_price - base_put_price)

            unhedged_mat[i, j] = linear_book_pnl
            hedged_mat[i, j] = linear_book_pnl + put_hedge_pnl

    unhedged_df = pd.DataFrame(unhedged_mat, index=row_labels, columns=col_labels)
    hedged_df = pd.DataFrame(hedged_mat, index=row_labels, columns=col_labels)
    return unhedged_df, hedged_df
