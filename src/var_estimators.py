"""
Baseline Value-at-Risk (VaR) & Expected Shortfall (ES) Estimators (`src/var_estimators.py`).

Implements Phase 1 (Crawl) methodology:
1. Plain Historical Simulation (Non-Parametric) with rolling lookback windows (250d, 500d)
   and non-parametric bootstrap standard errors around the quantile estimate.
2. Parametric Variance-Covariance Estimators:
   - Normal VaR & ES
   - Lognormal VaR & ES
   - Student-t VaR & ES (MLE degrees of freedom nu, scaled sigma_scale = sigma * sqrt((nu-2)/nu))
   - Cornish-Fisher Semi-Parametric VaR & ES (skewness and excess kurtosis adjustment)
3. Cholesky Monte Carlo Simulation (M = 10,000 correlated multivariate scenarios).
4. Coherence & Subadditivity Verification:
   - Portfolio Diversification Benefit (Standalone Sum of VaRs vs. Diversified Portfolio VaR)
   - Discrete default/credit-event counterexample proving VaR violates subadditivity while ES is coherent.
5. Pre-Trade Marginal & Incremental VaR Calculator:
   - Exact binary-search trade sizer determining the maximum allowable `TSLA` notional addition
     under the CRO's $200,000 1-Day 99% VaR cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats


@dataclass
class RiskEstimate:
    """Standardized container for a 1-Day VaR and ES estimate."""

    model_name: str
    confidence_level: float
    var_usd: float
    es_usd: float
    var_pct: float
    es_pct: float
    var_bootstrap_se_usd: Optional[float] = None
    extra_info: Optional[Dict[str, float]] = None


def historical_simulation_var_es(
    returns_series: pd.Series | np.ndarray,
    alpha: float = 0.99,
    notional_usd: float = 10_000_000.0,
    lookback_days: Optional[int] = 250,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> RiskEstimate:
    """
    Plain Historical Simulation (Non-Parametric) VaR and ES.
    Uses order statistics on dollar losses L_t = -V_0 * r_{p,t}, identifying the
    [(1 - alpha)*n + 1]-th worst loss (or empirical alpha quantile) and averaging tail losses >= VaR.
    Also computes non-parametric bootstrap standard error for the VaR estimate.
    """
    arr = np.asarray(returns_series, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if lookback_days is not None and len(arr) > lookback_days:
        arr = arr[-lookback_days:]

    n = len(arr)
    losses_pct = -arr
    sorted_losses = np.sort(losses_pct)[::-1]  # Descending: index 0 is worst loss

    # Order statistic index k = floor((1 - alpha) * n) in 0-indexed descending array
    # corresponds to the [(1 - alpha)*n + 1]-th worst loss in 1-indexed notation
    k_idx = min(max(int(np.floor((1.0 - alpha) * n)), 0), n - 1)
    var_pct = float(sorted_losses[k_idx])
    tail_losses = sorted_losses[: k_idx + 1] if k_idx >= 0 else sorted_losses[:1]
    es_pct = float(np.mean(tail_losses))

    # Bootstrap standard error
    boot_se_usd = None
    if n_bootstrap > 0 and n > 10:
        rng = np.random.default_rng(seed)
        boot_idx = rng.integers(0, n, size=(n_bootstrap, n))
        boot_samples = np.sort(losses_pct[boot_idx], axis=1)[:, ::-1]
        boot_vars = boot_samples[:, k_idx] * notional_usd
        boot_se_usd = float(np.std(boot_vars, ddof=1))

    label = f"Historical Simulation ({n}d)"
    return RiskEstimate(
        model_name=label,
        confidence_level=alpha,
        var_usd=var_pct * notional_usd,
        es_usd=es_pct * notional_usd,
        var_pct=var_pct,
        es_pct=es_pct,
        var_bootstrap_se_usd=boot_se_usd,
        extra_info={"window_days": float(n), "order_rank_1idx": float(k_idx + 1)},
    )


def parametric_normal_var_es(
    asset_returns: pd.DataFrame,
    weights: Dict[str, float],
    alpha: float = 0.99,
    notional_usd: float = 10_000_000.0,
    lookback_days: Optional[int] = 250,
    zero_mean: bool = False,
) -> RiskEstimate:
    """
    Parametric Variance-Covariance (Gaussian) VaR and ES:
      sigma_p = sqrt(w^T Sigma w)
      VaR_alpha^Norm = V_0 * (-mu_p + sigma_p * z_alpha)
      ES_alpha^Norm  = V_0 * (-mu_p + sigma_p * phi(z_alpha) / (1 - alpha))
    """
    tickers = list(weights.keys())
    w_vec = np.array([weights[t] for t in tickers], dtype=np.float64)
    df = asset_returns[tickers].dropna()
    if lookback_days is not None and len(df) > lookback_days:
        df = df.iloc[-lookback_days:]

    cov_mat = df.cov(ddof=1).values
    mu_vec = df.mean().values
    mu_p = 0.0 if zero_mean else float(w_vec @ mu_vec)
    sigma_p = float(np.sqrt(w_vec.T @ cov_mat @ w_vec))

    z_alpha = float(stats.norm.ppf(alpha))
    phi_z = float(stats.norm.pdf(z_alpha))

    var_pct = -mu_p + sigma_p * z_alpha
    es_pct = -mu_p + sigma_p * (phi_z / (1.0 - alpha))

    return RiskEstimate(
        model_name="Parametric Normal (Var-Cov)",
        confidence_level=alpha,
        var_usd=var_pct * notional_usd,
        es_usd=es_pct * notional_usd,
        var_pct=var_pct,
        es_pct=es_pct,
        extra_info={"mu_p": mu_p, "sigma_p": sigma_p, "z_alpha": z_alpha},
    )


def parametric_lognormal_var_es(
    log_returns_series: pd.Series | np.ndarray,
    alpha: float = 0.99,
    notional_usd: float = 10_000_000.0,
    lookback_days: Optional[int] = 250,
) -> RiskEstimate:
    """
    Parametric Lognormal VaR & ES over geometric log returns R_t ~ N(mu_R, sigma_R^2):
      VaR_alpha^Lognorm = V_0 * (1 - exp(mu_R - sigma_R * z_alpha))
      ES_alpha^Lognorm  = V_0 * (1 - exp(mu_R + 0.5*sigma_R^2) * Phi(-z_alpha - sigma_R) / (1 - alpha))
    """
    arr = np.asarray(log_returns_series, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if lookback_days is not None and len(arr) > lookback_days:
        arr = arr[-lookback_days:]

    mu_r = float(np.mean(arr))
    sigma_r = float(np.std(arr, ddof=1))
    z_alpha = float(stats.norm.ppf(alpha))

    var_pct = 1.0 - float(np.exp(mu_r - sigma_r * z_alpha))
    expected_tail_factor = (
        np.exp(mu_r + 0.5 * (sigma_r ** 2))
        * stats.norm.cdf(-z_alpha - sigma_r)
        / (1.0 - alpha)
    )
    es_pct = 1.0 - float(expected_tail_factor)

    return RiskEstimate(
        model_name="Parametric Lognormal",
        confidence_level=alpha,
        var_usd=var_pct * notional_usd,
        es_usd=es_pct * notional_usd,
        var_pct=var_pct,
        es_pct=es_pct,
        extra_info={"mu_R": mu_r, "sigma_R": sigma_r},
    )


def parametric_student_t_var_es(
    returns_series: pd.Series | np.ndarray,
    alpha: float = 0.99,
    notional_usd: float = 10_000_000.0,
    lookback_days: Optional[int] = 250,
) -> RiskEstimate:
    """
    Parametric Student-t VaR and ES:
    Fits degrees of freedom nu via Maximum Likelihood (bounded nu > 2.5 for finite variance)
    and scales sigma_scale = sigma_p * sqrt((nu - 2) / nu):
      VaR_alpha^t = V_0 * (-mu_p + sigma_scale * t_{nu}^{-1}(alpha))
      ES_alpha^t  = V_0 * (-mu_p + sigma_scale * [g_{nu}(t_alpha) / (1 - alpha)] * [(nu + t_alpha^2) / (nu - 1)])
    """
    arr = np.asarray(returns_series, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if lookback_days is not None and len(arr) > lookback_days:
        arr = arr[-lookback_days:]

    mu_p = float(np.mean(arr))
    sigma_p = float(np.std(arr, ddof=1))

    # Fit Student-t degrees of freedom via MLE
    nu_mle, _, _ = stats.t.fit(arr, floc=mu_p)
    nu = float(np.clip(nu_mle, 2.5, 30.0))

    sigma_scale = sigma_p * np.sqrt((nu - 2.0) / nu)
    t_alpha = float(stats.t.ppf(alpha, df=nu))
    pdf_t = float(stats.t.pdf(t_alpha, df=nu))

    var_pct = -mu_p + sigma_scale * t_alpha
    es_pct = -mu_p + sigma_scale * (pdf_t / (1.0 - alpha)) * ((nu + t_alpha ** 2) / (nu - 1.0))

    return RiskEstimate(
        model_name=f"Parametric Student-t (df={nu:.1f})",
        confidence_level=alpha,
        var_usd=var_pct * notional_usd,
        es_usd=es_pct * notional_usd,
        var_pct=var_pct,
        es_pct=es_pct,
        extra_info={"nu_df": nu, "sigma_p": sigma_p, "sigma_scale": float(sigma_scale), "t_alpha": t_alpha},
    )


def cornish_fisher_var_es(
    returns_series: pd.Series | np.ndarray,
    alpha: float = 0.99,
    notional_usd: float = 10_000_000.0,
    lookback_days: Optional[int] = 250,
) -> RiskEstimate:
    """
    Cornish-Fisher Semi-Parametric Expansion for VaR & ES.
    Injects sample skewness (S of loss distribution = -skew(returns)) and excess kurtosis (K)
    into the standard normal quantile z_alpha:
      z_CF(alpha) = z + (1/6)(z^2 - 1)*S_loss + (1/24)(z^3 - 3z)*K - (1/36)(2z^3 - 5z)*S_loss^2
    """
    arr = np.asarray(returns_series, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if lookback_days is not None and len(arr) > lookback_days:
        arr = arr[-lookback_days:]

    losses = -arr
    mu_loss = float(np.mean(losses))
    sigma_loss = float(np.std(losses, ddof=1))
    s_loss = float(stats.skew(losses, bias=False))
    k_excess = float(stats.kurtosis(losses, fisher=True, bias=False))

    def _cf_quantile(p: float) -> float:
        z = float(stats.norm.ppf(p))
        return (
            z
            + (1.0 / 6.0) * (z ** 2 - 1.0) * s_loss
            + (1.0 / 24.0) * (z ** 3 - 3.0 * z) * k_excess
            - (1.0 / 36.0) * (2.0 * (z ** 3) - 5.0 * z) * (s_loss ** 2)
        )

    z_cf = _cf_quantile(alpha)
    var_pct = mu_loss + sigma_loss * z_cf

    # Numerical integration of Cornish-Fisher tail quantiles for Expected Shortfall
    tail_grid = np.linspace(alpha, 0.9995, 200)
    tail_z_cf = np.array([_cf_quantile(p) for p in tail_grid])
    es_pct = float(mu_loss + sigma_loss * np.mean(tail_z_cf))

    return RiskEstimate(
        model_name="Cornish-Fisher Semi-Parametric",
        confidence_level=alpha,
        var_usd=var_pct * notional_usd,
        es_usd=es_pct * notional_usd,
        var_pct=var_pct,
        es_pct=es_pct,
        extra_info={"z_cf": z_cf, "loss_skewness": s_loss, "excess_kurtosis": k_excess},
    )


def cholesky_monte_carlo_var_es(
    asset_returns: pd.DataFrame,
    weights: Dict[str, float],
    alpha: float = 0.99,
    notional_usd: float = 10_000_000.0,
    lookback_days: Optional[int] = 250,
    num_scenarios: int = 10_000,
    seed: int = 42,
) -> RiskEstimate:
    """
    Multivariate Cholesky Monte Carlo Simulation VaR & ES:
      1. Factorize sample covariance matrix Sigma = L @ L^T
      2. Draw M = 10,000 standard normal vectors Z ~ N(0, I)
      3. Generate correlated return scenarios r^{(m)} = mu + L @ Z^{(m)}
      4. Compute empirical VaR and ES from simulated portfolio losses.
    """
    tickers = list(weights.keys())
    w_vec = np.array([weights[t] for t in tickers], dtype=np.float64)
    df = asset_returns[tickers].dropna()
    if lookback_days is not None and len(df) > lookback_days:
        df = df.iloc[-lookback_days:]

    mu_vec = df.mean().values
    cov_mat = df.cov(ddof=1).values
    chol_L = np.linalg.cholesky(cov_mat)

    rng = np.random.default_rng(seed)
    # Antithetic variates for variance reduction
    half_m = num_scenarios // 2
    z_half = rng.standard_normal(size=(len(tickers), half_m))
    z_mat = np.concatenate([z_half, -z_half], axis=1)

    sim_asset_rets = mu_vec[:, None] + chol_L @ z_mat  # shape: (N_assets, M)
    sim_port_rets = w_vec @ sim_asset_rets
    sim_losses = -sim_port_rets

    var_pct = float(np.quantile(sim_losses, alpha))
    es_pct = float(np.mean(sim_losses[sim_losses >= var_pct]))

    return RiskEstimate(
        model_name=f"Cholesky Monte Carlo (M={num_scenarios:,})",
        confidence_level=alpha,
        var_usd=var_pct * notional_usd,
        es_usd=es_pct * notional_usd,
        var_pct=var_pct,
        es_pct=es_pct,
        extra_info={"num_scenarios": float(num_scenarios)},
    )


def build_phase1_comparison_table(
    asset_returns: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    weights: Dict[str, float],
    notional_usd: float = 10_000_000.0,
    cro_var_99_limit: float = 200_000.0,
    cro_es_99_limit: float = 280_000.0,
) -> pd.DataFrame:
    """
    Build a comprehensive dashboard comparing 95% and 99% 1-Day VaR and ES
    across all Phase 1 estimators, flagging CRO limit status at 99%.
    """
    port_arith = portfolio_df["portfolio_arith_return"]
    port_log = portfolio_df["portfolio_log_return"]

    rows: List[Dict[str, object]] = []
    for alpha in (0.95, 0.99):
        estimators = [
            historical_simulation_var_es(port_arith, alpha=alpha, notional_usd=notional_usd, lookback_days=250),
            historical_simulation_var_es(port_arith, alpha=alpha, notional_usd=notional_usd, lookback_days=500),
            parametric_normal_var_es(asset_returns, weights, alpha=alpha, notional_usd=notional_usd, lookback_days=250),
            parametric_lognormal_var_es(port_log, alpha=alpha, notional_usd=notional_usd, lookback_days=250),
            parametric_student_t_var_es(port_arith, alpha=alpha, notional_usd=notional_usd, lookback_days=250),
            cornish_fisher_var_es(port_arith, alpha=alpha, notional_usd=notional_usd, lookback_days=250),
            cholesky_monte_carlo_var_es(asset_returns, weights, alpha=alpha, notional_usd=notional_usd, lookback_days=250, num_scenarios=10_000),
        ]
        for est in estimators:
            rows.append(
                {
                    "Estimator": est.model_name,
                    "Confidence": f"{int(alpha * 100)}%",
                    "VaR_USD": est.var_usd,
                    "VaR_pct_NAV": est.var_pct * 100.0,
                    "ES_USD": est.es_usd,
                    "ES_pct_NAV": est.es_pct * 100.0,
                    "Bootstrap_SE_USD": est.var_bootstrap_se_usd if est.var_bootstrap_se_usd is not None else np.nan,
                    "CRO_VaR_Limit_Breach": (est.var_usd > cro_var_99_limit) if np.isclose(alpha, 0.99) else False,
                    "CRO_ES_Limit_Breach": (est.es_usd > cro_es_99_limit) if np.isclose(alpha, 0.99) else False,
                }
            )
    return pd.DataFrame(rows)


def compute_subadditivity_and_diversification(
    asset_returns: pd.DataFrame,
    weights: Dict[str, float],
    alpha: float = 0.99,
    notional_usd: float = 10_000_000.0,
    lookback_days: int = 250,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    1. Quantify Portfolio Diversification Benefit:
       Show that Standalone Sum of VaRs > Diversified Portfolio VaR for both Normal and Historical Simulation.
    2. Provide a rigorous discrete payoff counterexample (two independent default-prone bonds
       each with 0.8% default probability of losing $100,000) where 99% VaR violates subadditivity:
         VaR_0.99(L_1) = 0, VaR_0.99(L_2) = 0, but VaR_0.99(L_1 + L_2) = $100,000 > 0 + 0,
       while 99% Expected Shortfall satisfies subadditivity:
         ES_0.99(L_1 + L_2) <= ES_0.99(L_1) + ES_0.99(L_2).
    """
    tickers = list(weights.keys())
    df = asset_returns[tickers].dropna().iloc[-lookback_days:]
    z_alpha = float(stats.norm.ppf(alpha))

    standalone_rows: List[Dict[str, object]] = []
    sum_norm_var = 0.0
    sum_hs_var = 0.0
    sum_norm_es = 0.0
    sum_hs_es = 0.0

    for t in tickers:
        alloc_usd = weights[t] * notional_usd
        s = df[t].values
        mu_i = float(np.mean(s))
        sd_i = float(np.std(s, ddof=1))
        var_norm_i = alloc_usd * (-mu_i + sd_i * z_alpha)
        es_norm_i = alloc_usd * (-mu_i + sd_i * stats.norm.pdf(z_alpha) / (1.0 - alpha))

        hs_est = historical_simulation_var_es(s, alpha=alpha, notional_usd=alloc_usd, lookback_days=lookback_days, n_bootstrap=0)
        sum_norm_var += var_norm_i
        sum_norm_es += es_norm_i
        sum_hs_var += hs_est.var_usd
        sum_hs_es += hs_est.es_usd

        standalone_rows.append(
            {
                "Position": t,
                "Weight_pct": weights[t] * 100.0,
                "Allocation_USD": alloc_usd,
                "Normal_VaR_99_USD": var_norm_i,
                "Normal_ES_99_USD": es_norm_i,
                "HS_VaR_99_USD": hs_est.var_usd,
                "HS_ES_99_USD": hs_est.es_usd,
            }
        )

    port_ret = df.values @ np.array([weights[t] for t in tickers])
    div_norm = parametric_normal_var_es(df, weights, alpha=alpha, notional_usd=notional_usd, lookback_days=lookback_days)
    div_hs = historical_simulation_var_es(port_ret, alpha=alpha, notional_usd=notional_usd, lookback_days=lookback_days, n_bootstrap=0)

    standalone_rows.append(
        {
            "Position": "Undiversified Sum of Standalone",
            "Weight_pct": 100.0,
            "Allocation_USD": notional_usd,
            "Normal_VaR_99_USD": sum_norm_var,
            "Normal_ES_99_USD": sum_norm_es,
            "HS_VaR_99_USD": sum_hs_var,
            "HS_ES_99_USD": sum_hs_es,
        }
    )
    standalone_rows.append(
        {
            "Position": "Diversified Multi-Asset Portfolio",
            "Weight_pct": 100.0,
            "Allocation_USD": notional_usd,
            "Normal_VaR_99_USD": div_norm.var_usd,
            "Normal_ES_99_USD": div_norm.es_usd,
            "HS_VaR_99_USD": div_hs.var_usd,
            "HS_ES_99_USD": div_hs.es_usd,
        }
    )
    standalone_rows.append(
        {
            "Position": "Diversification Benefit ($ Saved)",
            "Weight_pct": 0.0,
            "Allocation_USD": 0.0,
            "Normal_VaR_99_USD": sum_norm_var - div_norm.var_usd,
            "Normal_ES_99_USD": sum_norm_es - div_norm.es_usd,
            "HS_VaR_99_USD": sum_hs_var - div_hs.var_usd,
            "HS_ES_99_USD": sum_hs_es - div_hs.es_usd,
        }
    )

    # Analytical coherence counterexample: two independent bonds with default prob p = 0.008 (0.8%)
    # Loss = $100,000 on default, $0 otherwise. At alpha = 0.99 (tail = 1.0%):
    p_def = 0.008
    loss_face = 100_000.0
    tail_prob = 1.0 - alpha  # 0.01

    # Single bond: P(L_1 > 0) = 0.008 < 0.01 => 99% quantile VaR_0.99(L_1) = 0
    var_l1 = 0.0
    var_l2 = 0.0
    # ES_0.99(L_1) integrates top 1% tail: 0.8% of the time loss is $100k, 0.2% of the time loss is $0
    es_l1 = (p_def / tail_prob) * loss_face  # 0.8 * 100,000 = $80,000
    es_l2 = es_l1

    # Combined portfolio L_1 + L_2:
    # P(L_1 + L_2 = 0) = (1 - 0.008)^2 = 0.984064 < 0.99!
    # Therefore at 99% confidence, VaR_0.99(L_1 + L_2) = $100,000 > VaR(L_1) + VaR(L_2) = $0!
    p_both = p_def ** 2
    p_one = 2.0 * p_def * (1.0 - p_def)
    var_combined = loss_face  # Since P(Loss >= $100k) = p_both + p_one = 0.015936 > 0.01
    # ES_0.99(L_1 + L_2) over the worst 1% tail:
    # Worst p_both = 0.000064 has loss $200k; remaining (0.01 - 0.000064) = 0.009936 has loss $100k
    es_combined = (p_both * (2.0 * loss_face) + (tail_prob - p_both) * loss_face) / tail_prob

    coherence_proof = {
        "var_bond1": var_l1,
        "var_bond2": var_l2,
        "var_sum_standalone": var_l1 + var_l2,
        "var_combined_portfolio": var_combined,
        "var_violates_subadditivity": bool(var_combined > (var_l1 + var_l2)),
        "es_bond1": es_l1,
        "es_bond2": es_l2,
        "es_sum_standalone": es_l1 + es_l2,
        "es_combined_portfolio": es_combined,
        "es_satisfies_subadditivity": bool(es_combined <= (es_l1 + es_l2) + 1e-9),
    }

    return pd.DataFrame(standalone_rows), coherence_proof


def compute_pretrade_marginal_var_sizing(
    asset_returns: pd.DataFrame,
    base_weights: Dict[str, float],
    candidate_ticker: str = "TSLA",
    base_notional_usd: float = 10_000_000.0,
    cro_var_99_limit_usd: float = 200_000.0,
    alpha: float = 0.99,
    lookback_days: int = 250,
) -> Dict[str, object]:
    """
    Pre-Trade Marginal & Incremental VaR Calculator:
    Evaluates adding a volatile position (`TSLA`) on top of the $10M baseline desk
    (or reallocating within the book) and finds the exact maximum allowable `TSLA`
    dollar notional delta_V_TSLA that keeps the desk at or below the CRO's $200,000
    1-Day 99% VaR cap (evaluated for both Parametric Normal VaR and Historical Simulation VaR).
    """
    core_tickers = list(base_weights.keys())
    all_tickers = core_tickers + [candidate_ticker]
    df = asset_returns[all_tickers].dropna().iloc[-lookback_days:]

    base_pos_usd = np.array([base_weights[t] * base_notional_usd for t in core_tickers], dtype=np.float64)
    cov_all = df.cov(ddof=1).values
    mu_all = df.mean().values
    z_alpha = float(stats.norm.ppf(alpha))

    # Baseline Parametric & HS VaR
    cov_core = cov_all[: len(core_tickers), : len(core_tickers)]
    mu_core = mu_all[: len(core_tickers)]
    base_sd_usd = float(np.sqrt(base_pos_usd.T @ cov_core @ base_pos_usd))
    base_mu_usd = float(base_pos_usd @ mu_core)
    base_norm_var = -base_mu_usd + z_alpha * base_sd_usd

    # Component & Marginal VaR of existing 4 assets
    cov_with_port = (cov_core @ base_pos_usd) / base_sd_usd  # d(sigma_p)/d(x_i)
    marginal_var_per_dollar = -mu_core + z_alpha * cov_with_port
    component_var_usd = base_pos_usd * marginal_var_per_dollar

    # Marginal VaR of candidate ticker (TSLA) at x_TSLA = 0
    cov_tsla_port = float(cov_all[-1, : len(core_tickers)] @ base_pos_usd) / base_sd_usd
    mu_tsla = float(mu_all[-1])
    tsla_marginal_var_per_dollar = -mu_tsla + z_alpha * cov_tsla_port

    def _norm_var_with_tsla(tsla_usd: float) -> float:
        pos_vec = np.append(base_pos_usd, tsla_usd)
        sd_usd = float(np.sqrt(pos_vec.T @ cov_all @ pos_vec))
        mu_usd = float(pos_vec @ mu_all)
        return -mu_usd + z_alpha * sd_usd

    def _hs_var_with_tsla(tsla_usd: float) -> float:
        pos_vec = np.append(base_pos_usd, tsla_usd)
        pnl_series = df.values @ pos_vec
        losses = -pnl_series
        sorted_losses = np.sort(losses)[::-1]
        k_idx = min(max(int(np.floor((1.0 - alpha) * len(losses))), 0), len(losses) - 1)
        return float(sorted_losses[k_idx])

    base_hs_var = _hs_var_with_tsla(0.0)

    # Solve for max allowable TSLA notional under Parametric Normal VaR cap ($200,000)
    if base_norm_var >= cro_var_99_limit_usd:
        max_tsla_norm_usd = 0.0
    else:
        max_tsla_norm_usd = float(
            optimize.brentq(lambda x: _norm_var_with_tsla(x) - cro_var_99_limit_usd, 0.0, 10_000_000.0)
        )

    # Solve for max allowable TSLA notional under Historical Simulation VaR cap ($200,000)
    if base_hs_var >= cro_var_99_limit_usd:
        max_tsla_hs_usd = 0.0
    else:
        max_tsla_hs_usd = float(
            optimize.brentq(lambda x: _hs_var_with_tsla(x) - cro_var_99_limit_usd, 0.0, 10_000_000.0)
        )

    # Build sensitivity curve across candidate TSLA trade sizes ($0 to $2,500,000)
    grid_usd = np.linspace(0.0, 2_500_000.0, 51)
    curve_rows = []
    for x_usd in grid_usd:
        n_var = _norm_var_with_tsla(float(x_usd))
        h_var = _hs_var_with_tsla(float(x_usd))
        curve_rows.append(
            {
                "TSLA_Trade_Size_USD": float(x_usd),
                "New_Portfolio_Normal_VaR_99_USD": n_var,
                "Incremental_Normal_VaR_USD": n_var - base_norm_var,
                "Linear_Marginal_Approx_VaR_USD": base_norm_var + tsla_marginal_var_per_dollar * float(x_usd),
                "New_Portfolio_HS_VaR_99_USD": h_var,
                "Incremental_HS_VaR_USD": h_var - base_hs_var,
            }
        )

    comp_df = pd.DataFrame(
        {
            "Asset": core_tickers,
            "Position_USD": base_pos_usd,
            "Marginal_VaR_per_Dollar": marginal_var_per_dollar,
            "Component_VaR_99_USD": component_var_usd,
            "Pct_Contribution_to_VaR": (component_var_usd / base_norm_var) * 100.0,
        }
    )

    return {
        "baseline_normal_var_99_usd": base_norm_var,
        "baseline_hs_var_99_usd": base_hs_var,
        "cro_var_99_limit_usd": cro_var_99_limit_usd,
        "tsla_marginal_var_per_dollar": tsla_marginal_var_per_dollar,
        "max_allowable_tsla_normal_usd": max_tsla_norm_usd,
        "max_allowable_tsla_hs_usd": max_tsla_hs_usd,
        "component_var_table": comp_df,
        "sizing_curve_df": pd.DataFrame(curve_rows),
    }
