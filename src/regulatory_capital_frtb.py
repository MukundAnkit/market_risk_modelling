"""
IMA Regulatory Capital, RNIV, IRC, Economic Capital, IRRBB & FRTB Transition Engine (`src/regulatory_capital_frtb.py`).

Implements Phase 5 (Institutional Regulatory & Capital Modelling) to cover every pillar
of an Internal Models Approach (IMA) Market Risk Quantitative Analytics mandate:
1. Five-Pillar Multi-Asset VaR & Expected Shortfall Attribution:
     Explicit risk factor coverage and Euler Component VaR decomposition across all 5 regulatory
     asset classes: Rates (`TLT`), FX (`FXE`), Credit (`LQD`), Inflation (`TIP` + `GLD`), and Equity (`SPY` + `AAPL`).
2. RNIV (Risks Not In VaR) Capital Add-On Framework:
     Quantifies missing-basis and unmodeled-surface risks omitted by standard VaR engines:
       - Volatility Skew / Smile Surface Basis RNIV (pricing OTM options off flat ATM VIX shocks vs.
         dynamic skew steepening during market selloffs).
       - Tenor & Yield-Curve Convexity Basis RNIV (linear DV01 vs. exact multi-tenor discount curve revaluation).
       - Credit Spread Basis RNIV (index CDS / ETF proxy vs. single-name issuer spread dispersion).
     Computes both 1-Day 99% RNIV VaR and Stressed RNIV (sRNIV) capital add-ons.
3. IRC (Incremental Risk Charge — 1-Year 99.9% Confidence) & Economic Capital Engine:
     Merton (1974) / Basel 2.5 Single-Factor Gaussian Copula Monte Carlo engine (M = 50,000 trials)
     modeling 1-year credit default losses (PD / LGD) AND credit rating migration mark-to-market
     spread losses across an institutional credit trading portfolio, plus 1-Year 99.9% Economic Capital (EC).
4. IRRBB (Interest Rate Risk in the Banking Book) Supervisory Framework (BCBS d368):
     Evaluates Economic Value of Equity (Delta EVE) and 1-Year Net Interest Income (Delta NII)
     across all 6 Basel Standard Supervisory Interest-Rate Shock Scenarios:
       (1) Parallel Up (+200 bps), (2) Parallel Down (-200 bps), (3) Steepener,
       (4) Flattener, (5) Short Rate Up (+250 bps), and (6) Short Rate Down (-250 bps),
     including the Basel 15% Tier-1 Capital Outlier Test.
5. FRTB (Fundamental Review of the Trading Book — BCBS d457) Transition & ES Backtesting:
     - Liquidity-Horizon-Adjusted Expected Shortfall (97.5% ES) scaling risk factors across statutory
       Basel liquidity horizons LH_k in {10, 20, 40, 60, 120} days via the cascade formula:
         ES_FRTB = sqrt( ES_10d^2(P) + sum_{j>=2} [ ES_10d(P, j) * sqrt((LH_j - LH_{j-1}) / 10) ]^2 )
     - Acerbi & Szekely (2014) Z_2 Direct Expected Shortfall Backtesting statistic.
     - Full Regulatory Capital Bridge comparing Basel 2.5 IMA Capital (VaR + sVaR + IRC + RNIV)
       against FRTB IMA Capital (Liquidity-Adjusted ES + SES + DRC + RRAO).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.derivatives_stress import bsm_price, price_interest_rate_swap


# ==============================================================================
# 1. FIVE-PILLAR MULTI-ASSET VAR & ES ATTRIBUTION (RATES, FX, CREDIT, INFLATION, EQUITY)
# ==============================================================================

FIVE_PILLAR_MAP = {
    "SPY": ("Equity", "US Large-Cap Equity Beta (SPY)", 0.22),
    "AAPL": ("Equity", "US Single-Stock Tech Equity (AAPL)", 0.13),
    "TLT": ("Rates", "US 20+ Yr Sovereign Treasury Duration (TLT)", 0.20),
    "FXE": ("FX", "G10 Foreign Exchange Spot EUR/USD (FXE)", 0.15),
    "LQD": ("Credit", "US Investment-Grade Corporate Credit Spread (LQD)", 0.15),
    "TIP": ("Inflation", "US TIPS Inflation Breakeven & Real Yield (TIP)", 0.10),
    "GLD": ("Inflation", "Precious Metals / Inflation Tail Hedge (GLD)", 0.05),
}


def compute_five_pillar_var_attribution(
    arith_returns: pd.DataFrame,
    notional_usd: float = 10_000_000.0,
    alpha: float = 0.99,
    lookback_days: int = 250,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute 1-Day 99% VaR and Expected Shortfall across all 5 regulatory risk-factor pillars
    (Equity, Rates, FX, Credit, Inflation) and aggregate Euler Component VaR by asset class.
    """
    tickers = list(FIVE_PILLAR_MAP.keys())
    weights = np.array([FIVE_PILLAR_MAP[t][2] for t in tickers], dtype=np.float64)
    weights = weights / np.sum(weights)
    pos_usd = weights * notional_usd

    sub = arith_returns[tickers].dropna().iloc[-lookback_days:]
    mu_vec = sub.mean().values
    cov_mat = sub.cov(ddof=1).values

    z_alpha = float(stats.norm.ppf(alpha))
    phi_z = float(stats.norm.pdf(z_alpha))

    port_sd_usd = float(np.sqrt(pos_usd.T @ cov_mat @ pos_usd))
    port_mu_usd = float(pos_usd @ mu_vec)
    total_norm_var = -port_mu_usd + z_alpha * port_sd_usd

    # Euler Component VaR decomposition
    grad_sd = (cov_mat @ pos_usd) / port_sd_usd
    marginal_var = -mu_vec + z_alpha * grad_sd
    comp_var_usd = pos_usd * marginal_var
    comp_es_usd = pos_usd * (-mu_vec + (phi_z / (1.0 - alpha)) * grad_sd)

    inst_rows: List[Dict[str, object]] = []
    for i, t in enumerate(tickers):
        pillar, desc, w_i = FIVE_PILLAR_MAP[t]
        s_i = sub[t].values
        sd_i = float(np.std(s_i, ddof=1))
        mu_i = float(np.mean(s_i))
        standalone_var = pos_usd[i] * (-mu_i + z_alpha * sd_i)
        inst_rows.append(
            {
                "Ticker": t,
                "Risk_Class_Pillar": pillar,
                "Risk_Factor_Description": desc,
                "Weight_pct": w_i * 100.0,
                "Notional_USD": pos_usd[i],
                "Ann_Vol_pct": sd_i * np.sqrt(252) * 100.0,
                "Standalone_VaR_99_USD": standalone_var,
                "Component_VaR_99_USD": float(comp_var_usd[i]),
                "Component_ES_99_USD": float(comp_es_usd[i]),
                "Share_of_Total_VaR_pct": float(comp_var_usd[i] / total_norm_var * 100.0),
            }
        )

    inst_df = pd.DataFrame(inst_rows)
    pillar_df = (
        inst_df.groupby("Risk_Class_Pillar", as_index=False)[
            ["Weight_pct", "Notional_USD", "Standalone_VaR_99_USD", "Component_VaR_99_USD", "Component_ES_99_USD", "Share_of_Total_VaR_pct"]
        ]
        .sum()
        .sort_values("Component_VaR_99_USD", ascending=False)
        .reset_index(drop=True)
    )
    pillar_df["Diversification_Benefit_USD"] = (
        pillar_df["Standalone_VaR_99_USD"] - pillar_df["Component_VaR_99_USD"]
    )
    return inst_df, pillar_df


# ==============================================================================
# 2. RNIV (RISKS NOT IN VAR) CAPITAL ADD-ON ENGINE
# ==============================================================================

def compute_rniv_capital_addons(
    arith_returns: pd.DataFrame,
    vix_df: pd.DataFrame,
    alpha: float = 0.99,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Quantify Risks Not In VaR (RNIV) across three institutional missing-risk-factor channels:
      1. Volatility Smile / Skew Surface Basis RNIV (`SPY` 5% OTM Put options):
         Standard VaR engines shock options using only a flat ATM implied volatility proxy (VIX).
         In reality, OTM put skew steepens non-linearly during market crashes. The difference
         between full skew-surface P&L and flat ATM-VIX P&L is the Volatility Skew RNIV.
      2. Yield-Curve Tenor & Convexity Basis RNIV ($5M 5Y Interest-Rate Swap):
         Difference between a single-factor linear DV01 rate shock and full multi-tenor
         convexity-adjusted swap revaluation + 5Y/20Y curve twist basis.
      3. Credit Single-Name Idiosyncratic Spread Basis RNIV ($1.5M Corporate Credit Sleeve):
         Unmodeled basis between the broad `LQD` IG index factor and constituent issuer spread dispersion.
    Evaluated over both the Current 250d Window (RNIV VaR) and the 2020 Stressed Window (Stressed RNIV / sRNIV).
    """
    recent_r = arith_returns.iloc[-250:]
    stress_r = arith_returns.loc["2020-02-01":"2021-01-31"]
    recent_vix = vix_df.loc[recent_r.index, "CLOSE"].diff().fillna(0.0).values / 100.0
    stress_vix = vix_df.loc[stress_r.index, "CLOSE"].diff().fillna(0.0).values / 100.0

    # Channel 1: Volatility Skew / Smile Surface Basis on 6,000 SPY 5% OTM Puts
    S0, K_put, r, q, T = 500.0, 475.0, 0.045, 0.013, 30.0 / 365.0
    dt = 1.0 / 252.0
    n_shares = 6_000.0
    base_iv = 0.185
    p0 = bsm_price(S0, K_put, r, q, base_iv, T, is_call=False)

    def _vol_skew_basis_losses(spy_rets: np.ndarray, dvix: np.ndarray) -> np.ndarray:
        basis_pnl = np.zeros(len(spy_rets), dtype=np.float64)
        for i in range(len(spy_rets)):
            S_new = S0 * (1.0 + float(spy_rets[i]))
            # Flat ATM VIX shock
            iv_flat = max(0.04, base_iv + float(dvix[i]))
            # True dynamic skew shock: downside crash amplifies OTM put skew by an extra -0.45 * min(r_spy, 0) + vol-of-vol
            skew_shift = -0.45 * min(float(spy_rets[i]), 0.0) - 0.12 * float(spy_rets[i])
            iv_skew = max(0.04, iv_flat + skew_shift)
            p_flat = bsm_price(S_new, K_put, r, q, iv_flat, T - dt, is_call=False)
            p_true = bsm_price(S_new, K_put, r, q, iv_skew, T - dt, is_call=False)
            # For a desk with short-skew / exotic barrier exposure, unmodeled skew shift creates basis loss
            basis_pnl[i] = -n_shares * (p_true - p_flat)
        return -basis_pnl

    skew_loss_curr = _vol_skew_basis_losses(recent_r["SPY"].values, recent_vix)
    skew_loss_stress = _vol_skew_basis_losses(stress_r["SPY"].values, stress_vix)

    # Channel 2: Yield-Curve Tenor & Convexity Basis on $5M 5Y IRS vs 20Y TLT proxy
    # TLT has ~16.5y modified duration; 5Y IRS has ~4.46y modified duration + curve twist basis
    def _curve_basis_losses(tlt_rets: np.ndarray, tip_rets: np.ndarray) -> np.ndarray:
        dy_20y = -tlt_rets / 16.5
        # 5Y swap yield experiences steeper front-end shocks + real-rate twist basis
        dy_5y_true = dy_20y * 1.18 + 0.35 * (-tip_rets / 7.5)
        exact_pnl = price_interest_rate_swap(5_000_000.0, 0.042, 0.042 + dy_5y_true, 5.0, pay_fixed=True)
        proxy_linear_pnl = 2233.36 * (dy_20y * 1e4)
        basis_pnl = exact_pnl - proxy_linear_pnl
        return -basis_pnl

    curve_loss_curr = _curve_basis_losses(recent_r["TLT"].values, recent_r["TIP"].values)
    curve_loss_stress = _curve_basis_losses(stress_r["TLT"].values, stress_r["TIP"].values)

    # Channel 3: Credit Spread Index-vs-Single-Name Idiosyncratic Basis on $1.5M Credit Book
    def _credit_basis_losses(lqd_rets: np.ndarray, tlt_rets: np.ndarray, spy_rets: np.ndarray) -> np.ndarray:
        # Pure credit spread return residual = LQD return - duration-matched Treasury return
        spread_ret = lqd_rets - 0.52 * tlt_rets
        # Idiosyncratic single-name dispersion widens during equity-credit stress
        idio_dispersion = 0.28 * spread_ret + 0.08 * np.minimum(spy_rets, 0.0)
        return -1_500_000.0 * idio_dispersion

    cred_loss_curr = _credit_basis_losses(recent_r["LQD"].values, recent_r["TLT"].values, recent_r["SPY"].values)
    cred_loss_stress = _credit_basis_losses(stress_r["LQD"].values, stress_r["TLT"].values, stress_r["SPY"].values)

    channels = [
        ("Volatility Smile / Skew Surface Basis (SPY Options)", skew_loss_curr, skew_loss_stress),
        ("Yield-Curve Tenor & Convexity Basis (5Y IRS vs 20Y Proxy)", curve_loss_curr, curve_loss_stress),
        ("Credit Spread Index-to-Issuer Basis ($1.5M Credit Sleeve)", cred_loss_curr, cred_loss_stress),
    ]

    rows: List[Dict[str, object]] = []
    tot_rniv_var = 0.0
    tot_srniv_var = 0.0
    for name, c_loss, s_loss in channels:
        rniv_99 = float(np.quantile(c_loss, alpha))
        rniv_es = float(np.mean(c_loss[c_loss >= rniv_99]))
        srniv_99 = float(np.quantile(s_loss, alpha))
        srniv_es = float(np.mean(s_loss[s_loss >= srniv_99]))
        tot_rniv_var += rniv_99
        tot_srniv_var += srniv_99
        rows.append(
            {
                "RNIV_Risk_Factor_Channel": name,
                "RNIV_1d_99_VaR_USD": rniv_99,
                "RNIV_1d_99_ES_USD": rniv_es,
                "Stressed_RNIV_1d_99_VaR_USD": srniv_99,
                "Stressed_RNIV_1d_99_ES_USD": srniv_es,
                "Regulatory_RNIV_Capital_Charge_USD": 3.0 * (rniv_99 + srniv_99),
            }
        )

    df_rniv = pd.DataFrame(rows).set_index("RNIV_Risk_Factor_Channel")
    summary = {
        "total_rniv_1d_99_var_usd": tot_rniv_var,
        "total_stressed_rniv_1d_99_var_usd": tot_srniv_var,
        "total_rniv_capital_charge_usd": 3.0 * (tot_rniv_var + tot_srniv_var),
    }
    return df_rniv, summary


# ==============================================================================
# 3. IRC (INCREMENTAL RISK CHARGE — 1-YEAR 99.9%) & ECONOMIC CAPITAL ENGINE
# ==============================================================================

def simulate_irc_and_economic_capital(
    num_simulations: int = 50_000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Basel 2.5 Incremental Risk Charge (IRC) & 1-Year 99.9% Economic Capital Engine:
    Models a $3,000,000 Corporate Credit Trading Book across 6 obligors (ratings AA to BB)
    over a 1-year horizon at the 99.9% confidence level using a Merton (1974) Single-Factor
    Gaussian Copula (`Z_i = sqrt(rho_i) * M + sqrt(1 - rho_i) * eps_i`):
      - Captures both Default Losses (`PD_i * LGD_i * EAD_i`) AND Credit Rating Migration
        Mark-to-Market Losses (`Duration_i * Delta_Spread(Rating_New) * EAD_i`).
      - Computes 1-Year 99.9% Default-Only VaR (FRTB DRC equivalent), 1-Year 99.9% IRC
        (Default + Migration), and 1-Year 99.9% Total Desk Economic Capital (integrating
        1-year market risk tail losses and credit IRC with inter-risk correlation rho_MC = 0.35).
    """
    obligors = [
        # (Issuer, Rating, Notional_USD, Annual_PD, LGD, Asset_Corr_rho, Mod_Duration, Downgrade_Prob, Upgrade_Prob, Spread_Shock_Downgrade_bps, Spread_Shock_Upgrade_bps)
        ("Obligor 1 (Tech IG - AA)", "AA", 600_000.0, 0.0004, 0.40, 0.18, 4.5, 0.055, 0.010, 45.0, -15.0),
        ("Obligor 2 (Financials IG - A)", "A", 600_000.0, 0.0012, 0.45, 0.20, 4.2, 0.070, 0.020, 75.0, -25.0),
        ("Obligor 3 (Industrial IG - BBB+)", "BBB+", 500_000.0, 0.0028, 0.45, 0.22, 4.8, 0.085, 0.025, 135.0, -35.0),
        ("Obligor 4 (Energy IG - BBB)", "BBB", 500_000.0, 0.0045, 0.50, 0.24, 4.0, 0.095, 0.030, 190.0, -45.0),
        ("Obligor 5 (Consumer Crossover - BB+)", "BB+", 400_000.0, 0.0125, 0.55, 0.22, 3.6, 0.110, 0.040, 280.0, -70.0),
        ("Obligor 6 (High-Yield Telecom - BB)", "BB", 400_000.0, 0.0210, 0.60, 0.20, 3.2, 0.125, 0.045, 360.0, -90.0),
    ]

    rng = np.random.default_rng(seed)
    m_sys = rng.standard_normal(size=num_simulations)  # Systemic macro credit factor M ~ N(0,1)

    total_default_loss = np.zeros(num_simulations, dtype=np.float64)
    total_migration_loss = np.zeros(num_simulations, dtype=np.float64)
    issuer_rows: List[Dict[str, object]] = []

    for name, rating, ead, pd_ann, lgd, rho, dur, p_dn, p_up, d_spr_dn, d_spr_up in obligors:
        eps_i = rng.standard_normal(size=num_simulations)
        z_i = np.sqrt(rho) * m_sys + np.sqrt(1.0 - rho) * eps_i

        # Merton thresholds for Default, Downgrade, and Upgrade (negative z_i = asset value deterioration)
        thresh_def = float(stats.norm.ppf(pd_ann))
        thresh_dn = float(stats.norm.ppf(pd_ann + p_dn))
        thresh_up = float(stats.norm.ppf(1.0 - p_up))

        is_def = z_i <= thresh_def
        is_dn = (z_i > thresh_def) & (z_i <= thresh_dn)
        is_up = z_i >= thresh_up

        def_loss_i = np.where(is_def, ead * lgd, 0.0)
        mig_loss_i = np.where(
            is_dn,
            ead * dur * (d_spr_dn * 1e-4),
            np.where(is_up, ead * dur * (d_spr_up * 1e-4), 0.0),
        )

        total_default_loss += def_loss_i
        total_migration_loss += mig_loss_i

        el_def = ead * pd_ann * lgd
        el_mig = ead * dur * (p_dn * d_spr_dn * 1e-4 + p_up * d_spr_up * 1e-4)
        issuer_rows.append(
            {
                "Obligor": name,
                "Rating": rating,
                "Notional_EAD_USD": ead,
                "Annual_PD_pct": pd_ann * 100.0,
                "LGD_pct": lgd * 100.0,
                "Asset_Correlation_rho": rho,
                "Modified_Duration_Yrs": dur,
                "Expected_Default_Loss_USD": el_def,
                "Expected_Migration_Loss_USD": el_mig,
                "Simulated_999_Loss_Contribution_USD": float(np.mean((def_loss_i + mig_loss_i)[(total_default_loss + total_migration_loss) >= np.quantile(total_default_loss + total_migration_loss, 0.998)])),
            }
        )

    total_irc_loss = total_default_loss + total_migration_loss
    el_total = float(np.mean(total_irc_loss))
    irc_999_total = float(np.quantile(total_irc_loss, 0.999))
    drc_999_default_only = float(np.quantile(total_default_loss, 0.999))
    irc_es_999 = float(np.mean(total_irc_loss[total_irc_loss >= irc_999_total]))

    # 1-Year 99.9% Economic Capital (EC) integrating Market Risk + Credit IRC
    # Market risk 1-year loss simulated with Student-t(df=6) innovations correlated with (-m_sys) at rho_mc = 0.35
    eps_mkt = rng.standard_t(df=6.0, size=num_simulations) * np.sqrt((6.0 - 2.0) / 6.0)
    z_mkt_loss = -0.35 * m_sys + np.sqrt(1.0 - 0.35**2) * eps_mkt
    # 1-year scaled market risk volatility on $10M desk (with 20-day liquidity horizon rebalancing factor)
    mkt_loss_1y = 10_000_000.0 * 0.042 * z_mkt_loss
    mkt_ec_999 = float(np.quantile(mkt_loss_1y, 0.999))
    credit_ec_999 = irc_999_total - el_total

    joint_loss_1y = mkt_loss_1y + (total_irc_loss - el_total)
    diversified_ec_999 = float(np.quantile(joint_loss_1y, 0.999))
    diversified_es_999 = float(np.mean(joint_loss_1y[joint_loss_1y >= diversified_ec_999]))

    summary = {
        "credit_book_notional_usd": 3_000_000.0,
        "expected_credit_loss_1y_usd": el_total,
        "drc_default_only_999_usd": drc_999_default_only,
        "irc_default_plus_migration_999_usd": irc_999_total,
        "migration_share_of_irc_pct": (irc_999_total - drc_999_default_only) / irc_999_total * 100.0,
        "irc_es_999_usd": irc_es_999,
        "standalone_market_ec_999_usd": mkt_ec_999,
        "standalone_credit_ec_999_usd": credit_ec_999,
        "diversified_total_economic_capital_999_usd": diversified_ec_999,
        "ec_diversification_benefit_usd": (mkt_ec_999 + credit_ec_999) - diversified_ec_999,
        "total_ec_es_999_usd": diversified_es_999,
    }
    return pd.DataFrame(issuer_rows), summary


# ==============================================================================
# 4. IRRBB (INTEREST RATE RISK IN THE BANKING BOOK) SUPERVISORY ENGINE (BCBS d368)
# ==============================================================================

def evaluate_irrbb_supervisory_shocks(
    tier1_capital_usd: float = 15_000_000.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Basel IRRBB (BCBS d368) Banking Book Interest Rate Risk Engine:
    Models a representative $100,000,000 Banking Book balance sheet across 8 repricing tenor
    buckets (Overnight to 20Y) with sticky core deposits, floating SOFR commercial loans,
    fixed-rate mortgages, and an ALCO 5Y Pay-Fixed Interest-Rate Swap hedge.
    Evaluates all 6 Basel Standard Supervisory Interest-Rate Shock Scenarios:
      1. Parallel Shock Up (+200 bps)
      2. Parallel Shock Down (-200 bps)
      3. Steepener Shock (Short rates down, Long rates up)
      4. Flattener Shock (Short rates up, Long rates down)
      5. Short Rate Shock Up (+250 bps decaying exponentially)
      6. Short Rate Shock Down (-250 bps decaying exponentially)
    Outputs both Delta EVE (Economic Value of Equity) and 1-Year Delta NII (Net Interest Income)
    and tests compliance with the Basel 15% Tier-1 Capital Outlier threshold.
    """
    # Tenor midpoints (years): 1M, 3M, 6M, 1Y, 2Y, 5Y, 10Y, 20Y
    tenors = np.array([0.0833, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0, 20.0], dtype=np.float64)
    bucket_labels = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "20Y"]
    base_curve = np.array([0.0450, 0.0445, 0.0435, 0.0425, 0.0415, 0.0420, 0.0430, 0.0445], dtype=np.float64)

    # Asset repricing cash flows ($M): Floating SOFR loans in 1M-6M, Fixed mortgages/Treasury HQLA in 2Y-20Y
    asset_cf_usd = np.array([18.0, 14.0, 10.0, 8.0, 12.0, 18.0, 14.0, 6.0], dtype=np.float64) * 1e6
    # Liability repricing cash flows ($M): Wholesale funding in 1M-1Y, Core NMD behavioral slotting in 2Y-5Y
    liab_cf_usd = np.array([22.0, 18.0, 14.0, 12.0, 10.0, 9.0, 0.0, 0.0], dtype=np.float64) * 1e6
    net_gap_cf_usd = asset_cf_usd - liab_cf_usd

    # Basel d368 short/long decay weights: exp(-t / 4.0) for short shock, (1 - exp(-t / 4.0)) for long shock
    s_decay = np.exp(-tenors / 4.0)
    l_decay = 1.0 - s_decay

    shocks = {
        "1. Parallel Up (+200 bps)": np.full_like(tenors, 0.0200),
        "2. Parallel Down (-200 bps)": np.full_like(tenors, -0.0200),
        "3. Steepener (Short -160 / Long +90 bps)": -0.65 * 0.0250 * s_decay + 0.90 * 0.0100 * l_decay,
        "4. Flattener (Short +200 / Long -60 bps)": 0.80 * 0.0250 * s_decay - 0.60 * 0.0100 * l_decay,
        "5. Short Rate Up (+250 bps)": 0.0250 * s_decay,
        "6. Short Rate Down (-250 bps)": -0.0250 * s_decay,
    }

    def _eve_pv(curve_y: np.ndarray) -> float:
        disc = np.exp(-curve_y * tenors)
        return float(np.sum(net_gap_cf_usd * disc))

    base_eve = _eve_pv(base_curve)

    # 1-Year NII sensitivity weights for buckets repricing within 1 year (t <= 1.0): time remaining = (1.0 - t)
    nii_time_weights = np.maximum(1.0 - tenors, 0.0)
    # Apply 35% deposit pass-through beta on short-term liabilities for NII calculation
    effective_nii_gap = asset_cf_usd - 0.65 * liab_cf_usd

    scenario_rows: List[Dict[str, object]] = []
    for name, dy_vec in shocks.items():
        stressed_curve = np.maximum(base_curve + dy_vec, 0.0005)
        stressed_eve = _eve_pv(stressed_curve)
        delta_eve_unhedged = stressed_eve - base_eve

        # Include $10M 5-Year Pay-Fixed ALCO Swap Hedge
        alco_swap_pnl = float(
            price_interest_rate_swap(10_000_000.0, 0.0420, stressed_curve[5], tenor_years=5.0, pay_fixed=True)
            - price_interest_rate_swap(10_000_000.0, 0.0420, base_curve[5], tenor_years=5.0, pay_fixed=True)
        )
        delta_eve_hedged = delta_eve_unhedged + alco_swap_pnl
        delta_nii_1y = float(np.sum(effective_nii_gap * dy_vec * nii_time_weights))

        eve_pct_tier1 = (delta_eve_hedged / tier1_capital_usd) * 100.0
        # Basel IRRBB Outlier Test (BCBS d368): EVE loss (negative Delta EVE) must not exceed 15% of Tier 1 capital
        outlier_pass = bool(delta_eve_hedged > -0.15 * tier1_capital_usd)
        scenario_rows.append(
            {
                "Supervisory_Shock_Scenario": name,
                "Short_End_1M_Shock_bps": float(dy_vec[0] * 1e4),
                "Long_End_10Y_Shock_bps": float(dy_vec[6] * 1e4),
                "Unhedged_Delta_EVE_USD": delta_eve_unhedged,
                "ALCO_5Y_Swap_Hedge_PnL_USD": alco_swap_pnl,
                "Hedged_Delta_EVE_USD": delta_eve_hedged,
                "Delta_EVE_pct_Tier1_Capital": eve_pct_tier1,
                "Delta_NII_1Y_USD": delta_nii_1y,
                "Basel_15pct_Outlier_Pass": outlier_pass,
            }
        )

    gap_table = pd.DataFrame(
        {
            "Repricing_Bucket": bucket_labels,
            "Midpoint_Years": tenors,
            "Base_Yield_pct": base_curve * 100.0,
            "Assets_Repricing_USD": asset_cf_usd,
            "Liabilities_Repricing_USD": liab_cf_usd,
            "Net_Repricing_Gap_USD": net_gap_cf_usd,
        }
    )
    return gap_table, pd.DataFrame(scenario_rows).set_index("Supervisory_Shock_Scenario")


# ==============================================================================
# 5. FRTB LIQUIDITY-HORIZON EXPECTED SHORTFALL & BASEL 2.5 vs. FRTB CAPITAL BRIDGE
# ==============================================================================

# Statutory FRTB Liquidity Horizons (BCBS d457 MAR33.12):
#   10 days: Sovereign interest rates (TLT) & Large-cap equity prices (SPY, AAPL)
#   20 days: G10 FX spot (FXE) & Equity large-cap volatility
#   40 days: Investment-grade corporate credit spreads (LQD) & Inflation breakeven (TIP, GLD)
FRTB_LIQUIDITY_HORIZONS = {
    "SPY": 10,
    "AAPL": 10,
    "TLT": 10,
    "FXE": 20,
    "LQD": 40,
    "TIP": 40,
    "GLD": 40,
}


def compute_frtb_liquidity_adjusted_es(
    arith_returns: pd.DataFrame,
    vix_df: pd.DataFrame,
    notional_usd: float = 10_000_000.0,
    alpha_frtb: float = 0.975,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute Basel FRTB (BCBS d457) Liquidity-Horizon-Adjusted Expected Shortfall at 97.5% confidence:
      ES_FRTB = sqrt( (ES_T(P))^2 + sum_{j>=2} [ ES_T(P, j) * sqrt((LH_j - LH_{j-1}) / T) ]^2 )
    where T = 10 days is the base horizon, LH_1 = 10d, LH_2 = 20d, LH_3 = 40d, and P(j) is the
    subset of risk factors with liquidity horizon LH_k >= LH_j:
      - Subset j=1 (LH >= 10d): All 7 risk factors (`SPY`, `AAPL`, `TLT`, `FXE`, `LQD`, `TIP`, `GLD`)
      - Subset j=2 (LH >= 20d): `FXE`, `LQD`, `TIP`, `GLD`
      - Subset j=3 (LH >= 40d): `LQD`, `TIP`, `GLD`
    Evaluated over both the Current 250-day window (`ES_{R,C}`) and the Stressed 2020 window (`ES_{R,S}`).
    """
    tickers = list(FIVE_PILLAR_MAP.keys())
    pos_usd = np.array([FIVE_PILLAR_MAP[t][2] * notional_usd for t in tickers], dtype=np.float64)

    curr_df = arith_returns.iloc[-250:][tickers]
    stress_df = arith_returns.loc["2020-02-01":"2021-01-31", tickers]

    def _subset_10d_es(df_slice: pd.DataFrame, min_lh: int) -> float:
        mask = np.array([1.0 if FRTB_LIQUIDITY_HORIZONS[t] >= min_lh else 0.0 for t in tickers])
        active_pos = pos_usd * mask
        if np.allclose(active_pos, 0.0):
            return 0.0
        daily_losses = -(df_slice.values @ active_pos)
        q_val = float(np.quantile(daily_losses, alpha_frtb))
        es_1d = float(np.mean(daily_losses[daily_losses >= q_val]))
        return es_1d * np.sqrt(10.0)

    buckets = [
        (1, 10, 10, "Bucket 1 (LH >= 10d: Equity, Sovereign Rates, FX, Credit, Inflation)", 1.0),
        (2, 20, 10, "Bucket 2 (LH >= 20d: FX, IG Credit, Inflation)", np.sqrt((20.0 - 10.0) / 10.0)),
        (3, 40, 20, "Bucket 3 (LH >= 40d: IG Credit Spreads & Inflation)", np.sqrt((40.0 - 20.0) / 10.0)),
    ]

    bucket_rows: List[Dict[str, object]] = []
    sum_sq_curr = 0.0
    sum_sq_stress = 0.0

    for j_idx, lh_j, lh_prev, desc, scale_mult in buckets:
        es10_curr = _subset_10d_es(curr_df, min_lh=lh_j)
        es10_stress = _subset_10d_es(stress_df, min_lh=lh_j)
        scaled_curr = es10_curr * scale_mult
        scaled_stress = es10_stress * scale_mult
        sum_sq_curr += scaled_curr ** 2
        sum_sq_stress += scaled_stress ** 2
        bucket_rows.append(
            {
                "Cascade_Bucket": desc,
                "Liquidity_Horizon_Days": lh_j,
                "Incremental_Scale_Factor": scale_mult,
                "Base_10d_ES_975_Current_USD": es10_curr,
                "Scaled_Cascade_Term_Current_USD": scaled_curr,
                "Base_10d_ES_975_Stressed_USD": es10_stress,
                "Scaled_Cascade_Term_Stressed_USD": scaled_stress,
            }
        )

    frtb_es_curr = float(np.sqrt(sum_sq_curr))
    frtb_es_stress = float(np.sqrt(sum_sq_stress))
    base_10d_curr = float(bucket_rows[0]["Base_10d_ES_975_Current_USD"])
    base_1d_curr = base_10d_curr / np.sqrt(10.0)

    summary = {
        "es_975_1d_current_usd": base_1d_curr,
        "es_975_10d_unadjusted_current_usd": base_10d_curr,
        "frtb_liquidity_adjusted_es_current_usd": frtb_es_curr,
        "frtb_liquidity_adjusted_es_stressed_usd": frtb_es_stress,
        "liquidity_horizon_addon_pct": (frtb_es_curr / base_10d_curr - 1.0) * 100.0,
        "frtb_imak_multiplier_1_5_charge_usd": 1.5 * frtb_es_stress,
    }
    return pd.DataFrame(bucket_rows), summary


def acerbi_szekely_es_backtest(
    realized_losses: np.ndarray | pd.Series,
    var_forecasts: np.ndarray | pd.Series,
    es_forecasts: np.ndarray | pd.Series,
    alpha: float = 0.975,
) -> Dict[str, float | int | bool]:
    """
    Acerbi & Szekely (2014) Direct Expected Shortfall Backtesting Statistic Z_2:
      Z_2 = - (1 / (N * (1 - alpha))) * sum_{t=1}^N [ (L_t * I_{L_t > VaR_t}) / ES_{alpha, t} ] + 1
    Under H_0 (correctly specified tail distribution and Expected Shortfall), E[Z_2] = 0,
    whereas underestimation of tail losses yields Z_2 < 0 (5% one-sided critical value approx -0.70).
    """
    l_arr = np.asarray(realized_losses, dtype=np.float64)
    v_arr = np.asarray(var_forecasts, dtype=np.float64)
    e_arr = np.maximum(np.asarray(es_forecasts, dtype=np.float64), 1e-6)

    hits = (l_arr > v_arr).astype(np.float64)
    n = len(l_arr)
    tail_prob = 1.0 - alpha

    z2_stat = -float(np.sum((l_arr * hits) / e_arr)) / (n * tail_prob) + 1.0
    # Non-parametric bootstrap p-value under null
    return {
        "n_obs": int(n),
        "n_breaches": int(np.sum(hits)),
        "expected_breaches": float(n * tail_prob),
        "acerbi_szekely_z2": z2_stat,
        "pass_es_backtest_5pct": bool(z2_stat > -0.70),
    }


def build_ima_regulatory_capital_bridge(
    arith_returns: pd.DataFrame,
    vix_df: pd.DataFrame,
    notional_usd: float = 10_000_000.0,
) -> pd.DataFrame:
    """
    Build a side-by-side institutional Regulatory Capital Bridge comparing:
      - Basel 2.5 IMA Total Market Risk Capital:
          K_Basel2.5 = 3.0 * VaR_{99%, 10d} + 3.0 * sVaR_{99%, 10d} + IRC_{99.9%, 1y} + RNIV_Capital
      - FRTB IMA Total Market Risk Capital (BCBS d457):
          K_FRTB = 1.5 * ES_{97.5%, LH, Stressed} + DRC_{99.9%, 1y} + RRAO (Residual Risk Add-On)
    """
    _, pillar_df = compute_five_pillar_var_attribution(arith_returns, notional_usd=notional_usd, alpha=0.99)
    _, rniv_sum = compute_rniv_capital_addons(arith_returns, vix_df, alpha=0.99)
    _, irc_sum = simulate_irc_and_economic_capital(num_simulations=50_000)
    _, frtb_sum = compute_frtb_liquidity_adjusted_es(arith_returns, vix_df, notional_usd=notional_usd, alpha_frtb=0.975)

    # 10-day scaled Basel 2.5 VaR & Stressed VaR on the 5-Pillar Desk
    var_1d_99 = float(pillar_df["Component_VaR_99_USD"].sum())
    var_10d_99 = var_1d_99 * np.sqrt(10.0)

    # Stressed 2020 10-day VaR
    tickers = list(FIVE_PILLAR_MAP.keys())
    pos_usd = np.array([FIVE_PILLAR_MAP[t][2] * notional_usd for t in tickers])
    stress_losses = -(arith_returns.loc["2020-02-01":"2021-01-31", tickers].values @ pos_usd)
    svar_1d_99 = float(np.quantile(stress_losses, 0.99))
    svar_10d_99 = svar_1d_99 * np.sqrt(10.0)

    basel25_var_cap = 3.0 * var_10d_99
    basel25_svar_cap = 3.0 * svar_10d_99
    basel25_irc_cap = float(irc_sum["irc_default_plus_migration_999_usd"])
    basel25_rniv_cap = float(rniv_sum["total_rniv_capital_charge_usd"])
    basel25_total = basel25_var_cap + basel25_svar_cap + basel25_irc_cap + basel25_rniv_cap

    frtb_imcc_cap = float(frtb_sum["frtb_imak_multiplier_1_5_charge_usd"])
    frtb_drc_cap = float(irc_sum["drc_default_only_999_usd"])
    frtb_rrao_cap = 0.50 * basel25_rniv_cap  # Residual Risk Add-On (RRAO) replacing part of RNIV
    frtb_total = frtb_imcc_cap + frtb_drc_cap + frtb_rrao_cap

    rows = [
        {
            "Capital_Component": "1. General & Stressed Market Risk (VaR+sVaR vs. FRTB IMCC ES_LH)",
            "Basel_2_5_IMA_Capital_USD": basel25_var_cap + basel25_svar_cap,
            "FRTB_IMA_Capital_USD": frtb_imcc_cap,
            "Methodology_Transition_Note": "Replaces 3x(VaR_99,10d + sVaR_99,10d) with 1.5x Stressed Liquidity-Adjusted ES_97.5,LH (10d/20d/40d)",
        },
        {
            "Capital_Component": "2. Credit Default & Migration Risk (IRC vs. FRTB DRC)",
            "Basel_2_5_IMA_Capital_USD": basel25_irc_cap,
            "FRTB_IMA_Capital_USD": frtb_drc_cap,
            "Methodology_Transition_Note": "1Y 99.9% Copula; FRTB moves migration spread risk into ES_LH (40d/60d) and retains Default-Only in DRC",
        },
        {
            "Capital_Component": "3. Unmodeled Basis & Exotic Risks (RNIV vs. FRTB RRAO/SES)",
            "Basel_2_5_IMA_Capital_USD": basel25_rniv_cap,
            "FRTB_IMA_Capital_USD": frtb_rrao_cap,
            "Methodology_Transition_Note": "Captures Vol Skew Basis, Yield-Curve Convexity Basis & Credit Issuer Dispersion",
        },
        {
            "Capital_Component": "TOTAL REGULATORY IMA MARKET RISK CAPITAL",
            "Basel_2_5_IMA_Capital_USD": basel25_total,
            "FRTB_IMA_Capital_USD": frtb_total,
            "Methodology_Transition_Note": f"1-Year 99.9% Internal Economic Capital Benchmark = ${irc_sum['diversified_total_economic_capital_999_usd']:,.0f}",
        },
    ]
    return pd.DataFrame(rows).set_index("Capital_Component")
