"""
Exploratory Data Analysis & Distributional Stylized Facts (`src/eda_diagnostics.py`).

Implements Phase 0 methodology:
1. Augmented Dickey-Fuller (ADF) stationarity verification on raw price levels vs. daily returns.
2. Distributional stylized facts: sample mean, annualized volatility, skewness, excess kurtosis,
   Jarque-Bera normality test, and empirical vs. Gaussian tail exceedance frequencies.
3. Quantile-Quantile (Q-Q) diagnostic coordinates against a standard normal distribution.
4. Rolling annualized volatility alongside Cboe VIX and 60-day rolling cross-asset correlation
   to diagnose regime shifts (2020 COVID-19 crash vs. 2022 inflation/rate-hike bear market).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller


def run_adf_stationarity_table(
    prices_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    significance_level: float = 0.05,
) -> pd.DataFrame:
    """
    Run the Augmented Dickey-Fuller (ADF) test on both price levels and daily returns
    for each asset, verifying that price levels contain a unit root (non-stationary)
    while daily returns reject the unit-root null hypothesis (stationary).
    """
    records: List[Dict[str, object]] = []
    common_cols = [c for c in prices_df.columns if c in returns_df.columns]

    for ticker in common_cols:
        # 1. Price Levels
        p_series = prices_df[ticker].dropna().values
        adf_p = adfuller(p_series, autolag="AIC", result_object=False)
        records.append(
            {
                "Asset": ticker,
                "Series": "Price Level (P_t)",
                "ADF_Statistic": float(adf_p[0]),
                "p_value": float(adf_p[1]),
                "Lags_Used": int(adf_p[2]),
                "Critical_1pct": float(adf_p[4]["1%"]),
                "Critical_5pct": float(adf_p[4]["5%"]),
                "Stationary_at_5pct": bool(adf_p[1] < significance_level),
            }
        )

        # 2. Daily Returns
        r_series = returns_df[ticker].dropna().values
        adf_r = adfuller(r_series, autolag="AIC", result_object=False)
        records.append(
            {
                "Asset": ticker,
                "Series": "Daily Return (r_t)",
                "ADF_Statistic": float(adf_r[0]),
                "p_value": float(adf_r[1]),
                "Lags_Used": int(adf_r[2]),
                "Critical_1pct": float(adf_r[4]["1%"]),
                "Critical_5pct": float(adf_r[4]["5%"]),
                "Stationary_at_5pct": bool(adf_r[1] < significance_level),
            }
        )

    return pd.DataFrame(records)


def compute_stylized_facts_table(
    returns_df: pd.DataFrame,
    portfolio_returns: Optional[pd.Series] = None,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Compute key distributional moments and tail-risk metrics across assets and portfolio:
    - Annualized Return & Annualized Volatility (sigma_daily * sqrt(252))
    - Skewness & Excess Kurtosis (Fisher definition, Normal = 0.0; Total Kurtosis = Excess + 3)
    - Jarque-Bera test statistic & p-value
    - Empirical vs. Gaussian >3-sigma tail event frequency ratio
    """
    df = returns_df.copy()
    if portfolio_returns is not None:
        df["PORTFOLIO"] = portfolio_returns

    rows: List[Dict[str, object]] = []
    gaussian_3sigma_prob = 2.0 * (1.0 - stats.norm.cdf(3.0))  # ~0.0026998

    for col in df.columns:
        s = df[col].dropna()
        mu_d = float(s.mean())
        sd_d = float(s.std(ddof=1))
        skew_val = float(stats.skew(s, bias=False))
        ex_kurt = float(stats.kurtosis(s, fisher=True, bias=False))
        tot_kurt = ex_kurt + 3.0
        jb_stat, jb_pval = stats.jarque_bera(s)

        z_scores = (s - mu_d) / sd_d
        emp_3sigma_freq = float(np.mean(np.abs(z_scores) > 3.0))
        tail_multiple = emp_3sigma_freq / gaussian_3sigma_prob if gaussian_3sigma_prob > 0 else np.nan

        rows.append(
            {
                "Asset": col,
                "Daily_Mean_bps": mu_d * 1e4,
                "Annualized_Return_pct": mu_d * trading_days * 100.0,
                "Daily_Vol_pct": sd_d * 100.0,
                "Annualized_Vol_pct": sd_d * np.sqrt(trading_days) * 100.0,
                "Skewness": skew_val,
                "Excess_Kurtosis": ex_kurt,
                "Total_Kurtosis": tot_kurt,
                "Min_Daily_Return_pct": float(s.min()) * 100.0,
                "Max_Daily_Return_pct": float(s.max()) * 100.0,
                "Empirical_3Sigma_Freq_pct": emp_3sigma_freq * 100.0,
                "Tail_Ratio_vs_Normal": tail_multiple,
                "Jarque_Bera_Stat": float(jb_stat),
                "Jarque_Bera_p_value": float(jb_pval),
            }
        )

    return pd.DataFrame(rows).set_index("Asset")


def compute_qq_coordinates(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (theoretical_normal_quantiles, standardized_empirical_quantiles)
    for plotting a Quantile-Quantile (Q-Q) chart against N(0, 1).
    """
    clean = series.dropna().values
    z_emp = (clean - np.mean(clean)) / np.std(clean, ddof=1)
    z_sorted = np.sort(z_emp)
    n = len(z_sorted)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo_q = stats.norm.ppf(probs)
    return theo_q, z_sorted


def compute_rolling_volatility_and_vix(
    returns_df: pd.DataFrame,
    portfolio_returns: pd.Series,
    vix_df: pd.DataFrame,
    window: int = 20,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Compute `window`-day rolling annualized volatility (%) for `SPY` and the
    aggregate portfolio alongside the Cboe VIX index.
    """
    spy_vol = returns_df["SPY"].rolling(window=window).std(ddof=1) * np.sqrt(trading_days) * 100.0
    port_vol = portfolio_returns.rolling(window=window).std(ddof=1) * np.sqrt(trading_days) * 100.0
    vix_close = vix_df["CLOSE"].reindex(returns_df.index)

    out = pd.DataFrame(
        {
            "SPY_Rolling_Vol_20d_pct": spy_vol,
            "Portfolio_Rolling_Vol_20d_pct": port_vol,
            "Cboe_VIX": vix_close,
        },
        index=returns_df.index,
    )
    return out.dropna(how="any")


def compute_rolling_stock_bond_correlation(
    returns_df: pd.DataFrame,
    equity_ticker: str = "SPY",
    bond_ticker: str = "TLT",
    window: int = 60,
) -> pd.Series:
    """
    Compute the `window`-day rolling correlation between equity (`SPY`) and
    long-duration Treasury bonds (`TLT`), highlighting the regime shift from
    negative (diversifying) in 2020 to positive (compounding losses) in 2022.
    """
    rolling_corr = (
        returns_df[equity_ticker]
        .rolling(window=window)
        .corr(returns_df[bond_ticker])
        .dropna()
    )
    rolling_corr.name = f"{equity_ticker}_{bond_ticker}_Rolling_Corr_{window}d"
    return rolling_corr


def summarize_regime_correlations(
    returns_df: pd.DataFrame,
    equity_ticker: str = "SPY",
    bond_ticker: str = "TLT",
) -> pd.DataFrame:
    """
    Summarize full-period and sub-regime correlations and annualized volatilities
    across key macro regimes (2017 Calm, 2020 COVID Crash, 2022 Rate-Hike Bear Market, 2023-2025).
    """
    regimes = [
        ("Full Sample (2015-2025)", "2015-01-01", "2025-12-31"),
        ("2017 Low-Vol Calm", "2017-01-01", "2017-12-31"),
        ("2020 COVID-19 Crisis (Q1-Q2)", "2020-02-01", "2020-05-31"),
        ("2022 Inflation & Rate-Hike Bear", "2022-01-01", "2022-12-31"),
        ("2023-2025 Post-Hike Regime", "2023-01-01", "2025-12-31"),
    ]
    records: List[Dict[str, object]] = []
    for label, start, end in regimes:
        sub = returns_df.loc[start:end]
        if len(sub) < 10:
            continue
        corr_val = float(sub[equity_ticker].corr(sub[bond_ticker]))
        spy_vol = float(sub[equity_ticker].std(ddof=1) * np.sqrt(252) * 100.0)
        tlt_vol = float(sub[bond_ticker].std(ddof=1) * np.sqrt(252) * 100.0)
        records.append(
            {
                "Regime": label,
                "Start": start,
                "End": end,
                "Trading_Days": len(sub),
                f"{equity_ticker}_{bond_ticker}_Correlation": corr_val,
                f"{equity_ticker}_Ann_Vol_pct": spy_vol,
                f"{bond_ticker}_Ann_Vol_pct": tlt_vol,
            }
        )
    return pd.DataFrame(records).set_index("Regime")
