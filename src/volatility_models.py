"""
Time-Series Volatility Regimes & Dynamic Historical Simulation (`src/volatility_models.py`).

Implements Phase 2 (Walk) methodology:
1. Diagnosing the 250-day Plain Historical Simulation "Ghost Effect" across 2020–2021.
2. RiskMetrics EWMA Volatility recursion with decay factor lambda = 0.94 (accelerated via C++17 kernel):
     sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2
3. GARCH(1,1) Conditional Volatility via the `arch` library (with both Normal and Student-t innovations):
     sigma_t^2 = omega + alpha_1 * eps_{t-1}^2 + beta_1 * sigma_{t-1}^2
   extracting 1-step-ahead conditional volatility forecasts sigma_{t|t-1} and standardized residuals eta_t.
4. Asymmetric EGARCH(1,1) Conditional Volatility & Filtered Historical Simulation (FHS) (Nelson, 1991):
     ln(sigma_t^2) = omega + alpha * (|z_{t-1}| - E|z|) + gamma * z_{t-1} + beta * ln(sigma_{t-1}^2)
   capturing the asymmetric leverage effect (gamma < 0) where negative return shocks amplify
   conditional volatility more than positive shocks of equal magnitude, and rescaling empirical
   standardized residuals (Filtered Historical Simulation) to forecast 1-day VaR and ES.
5. Dynamic Non-Parametric Upgrades:
   - Age-Weighted Historical Simulation (Boudoukh, Richardson & Whitelaw / BRW Hybrid):
       w_i = (lambda^{i-1} * (1 - lambda)) / (1 - lambda^n)
   - Volatility-Weighted Historical Simulation (Hull & White, 1998):
       r_{t,i}^* = r_{t,i} * (sigma_T / sigma_i)
     proving that Hull-White Volatility-Weighted HS can forecast losses strictly larger than the
     worst historical sample loss when conditional volatility spikes.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

from src.cpp_kernel import ewma_volatility_fast


def compute_rolling_std_volatility(
    returns_series: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Compute 1-step-ahead equally weighted rolling standard deviation forecast
    using observations up to t-1 (`shift(1)` for strict F_{t-1} conditioning).
    """
    vol = returns_series.rolling(window=window, min_periods=5).std(ddof=1).shift(1)
    vol = vol.bfill()
    vol.name = f"Rolling_StdDev_{window}d"
    return vol


def compute_ewma_volatility(
    returns_series: pd.Series,
    lambda_decay: float = 0.94,
    init_window: int = 20,
) -> pd.Series:
    """
    Compute RiskMetrics EWMA 1-step-ahead daily volatility forecast sigma_{t|t-1}
    via the C++17 kernel:
      sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2
    """
    clean = returns_series.dropna()
    arr = clean.values.astype(np.float64)
    init_var = float(np.var(arr[: min(init_window, len(arr))], ddof=1))
    vol_arr = ewma_volatility_fast(arr, lambda_decay=lambda_decay, init_var=init_var)
    return pd.Series(vol_arr, index=clean.index, name=f"EWMA_Vol_lambda_{lambda_decay:.2f}")


def fit_garch11_volatility(
    returns_series: pd.Series,
    dist: str = "normal",
    scale_factor: float = 100.0,
) -> Dict[str, object]:
    """
    Fit a GARCH(1,1) model using the `arch` library with either Normal (`dist='normal'`)
    or Student-t (`dist='t'`) innovations:
      sigma_t^2 = omega + alpha_1 * eps_{t-1}^2 + beta_1 * sigma_{t-1}^2
    Returns:
      - `cond_vol_in_sample`: Contemporaneous conditional daily volatility sigma_t
      - `cond_vol_forecast_1d`: Strict 1-step-ahead conditional volatility forecast sigma_{t|t-1}
        computed from the fitted GARCH recursion using info up to t-1
      - `std_residuals`: Standardized residuals eta_t = eps_t / sigma_t
      - `params`: Dictionary of fitted parameters (mu, omega, alpha[1], beta[1], persistence, nu)
    """
    clean = returns_series.dropna()
    scaled_rets = clean * scale_factor

    am = arch_model(
        scaled_rets,
        mean="Constant",
        vol="Garch",
        p=1,
        q=1,
        dist="t" if dist.lower() in ("t", "student", "student-t") else "normal",
        rescale=False,
    )
    res = am.fit(disp="off", show_warning=False)

    params = res.params.to_dict()
    mu_scaled = float(params.get("mu", 0.0))
    omega_scaled = float(params.get("omega", 0.01))
    alpha1 = float(params.get("alpha[1]", 0.10))
    beta1 = float(params.get("beta[1]", 0.85))
    nu_df = float(params.get("nu", np.nan))

    # In-sample conditional volatility from arch (in decimal return units)
    cond_vol_contemp = (res.conditional_volatility / scale_factor).rename(f"GARCH11_{dist}_contemp_vol")
    std_resid = pd.Series(res.std_resid, index=clean.index, name=f"GARCH11_{dist}_std_resid")

    # Strict 1-step-ahead forecast sigma_{t|t-1}:
    # On day t, sigma_{t|t-1}^2 = omega + alpha_1 * eps_{t-1}^2 + beta_1 * sigma_{t-1|t-2}^2
    eps_scaled = (scaled_rets - mu_scaled).values
    n = len(eps_scaled)
    h_1d_scaled = np.empty(n, dtype=np.float64)
    uncond_var = omega_scaled / max(1.0 - (alpha1 + beta1), 1e-3)
    h_1d_scaled[0] = uncond_var
    for t in range(1, n):
        h_1d_scaled[t] = omega_scaled + alpha1 * (eps_scaled[t - 1] ** 2) + beta1 * h_1d_scaled[t - 1]

    cond_vol_1d = pd.Series(
        np.sqrt(h_1d_scaled) / scale_factor,
        index=clean.index,
        name=f"GARCH11_{dist}_1d_vol",
    )

    return {
        "result": res,
        "cond_vol_in_sample": cond_vol_contemp,
        "cond_vol_forecast_1d": cond_vol_1d,
        "std_residuals": std_resid,
        "params": {
            "mu_daily": mu_scaled / scale_factor,
            "omega_daily_var": omega_scaled / (scale_factor ** 2),
            "alpha1": alpha1,
            "beta1": beta1,
            "persistence": alpha1 + beta1,
            "nu_df": nu_df,
            "aic": float(res.aic),
            "bic": float(res.bic),
            "loglikelihood": float(res.loglikelihood),
        },
    }


def fit_egarch11_volatility(
    returns_series: pd.Series,
    dist: str = "t",
    scale_factor: float = 100.0,
) -> Dict[str, object]:
    """
    Fit a Nelson (1991) Asymmetric Exponential GARCH(1,1,1) model (`vol='EGARCH'`, p=1, o=1, q=1):
      ln(sigma_t^2) = omega + alpha_1 * (|z_{t-1}| - E[|z|]) + gamma_1 * z_{t-1} + beta_1 * ln(sigma_{t-1}^2)
    where z_t = eps_t / sigma_t.
    When gamma_1 < 0, negative return innovations (z_{t-1} < 0) increase log-variance more than
    positive innovations of equal magnitude (the financial leverage / asymmetric volatility effect).
    """
    clean = returns_series.dropna()
    scaled_rets = clean * scale_factor

    am = arch_model(
        scaled_rets,
        mean="Constant",
        vol="EGARCH",
        p=1,
        o=1,
        q=1,
        dist="t" if dist.lower() in ("t", "student", "student-t") else "normal",
        rescale=False,
    )
    res = am.fit(disp="off", show_warning=False)

    params = res.params.to_dict()
    mu_scaled = float(params.get("mu", 0.0))
    omega = float(params.get("omega", 0.0))
    alpha1 = float(params.get("alpha[1]", 0.15))
    gamma1 = float(params.get("gamma[1]", -0.08))
    beta1 = float(params.get("beta[1]", 0.95))
    nu_df = float(params.get("nu", np.nan))

    cond_vol_contemp = (res.conditional_volatility / scale_factor).rename(f"EGARCH11_{dist}_contemp_vol")
    std_resid = pd.Series(res.std_resid, index=clean.index, name=f"EGARCH11_{dist}_std_resid")

    # Compute strict 1-step-ahead conditional volatility forecast sigma_{t|t-1}
    # using the EGARCH log-variance recursion conditioned on F_{t-1}:
    eps_scaled = (scaled_rets - mu_scaled).values
    n = len(eps_scaled)
    log_h = np.empty(n, dtype=np.float64)
    uncond_log_h = omega / max(1.0 - beta1, 1e-3)
    log_h[0] = uncond_log_h

    # Expected absolute value E[|z|] under N(0,1) is sqrt(2/pi) ~ 0.79788456
    e_abs_z = np.sqrt(2.0 / np.pi)
    if not np.isnan(nu_df) and nu_df > 2.1:
        from scipy.special import gamma as sp_gamma
        e_abs_z = float(
            2.0 * np.sqrt(nu_df - 2.0) * sp_gamma(0.5 * (nu_df + 1.0))
            / ((nu_df - 1.0) * np.sqrt(np.pi) * sp_gamma(0.5 * nu_df))
        )

    for t in range(1, n):
        sigma_prev = np.exp(0.5 * log_h[t - 1])
        z_prev = eps_scaled[t - 1] / max(sigma_prev, 1e-8)
        z_prev = float(np.clip(z_prev, -8.0, 8.0))
        log_h[t] = (
            omega
            + alpha1 * (abs(z_prev) - e_abs_z)
            + gamma1 * z_prev
            + beta1 * log_h[t - 1]
        )

    cond_vol_1d = pd.Series(
        np.exp(0.5 * log_h) / scale_factor,
        index=clean.index,
        name=f"EGARCH11_{dist}_1d_vol",
    )

    return {
        "result": res,
        "cond_vol_in_sample": cond_vol_contemp,
        "cond_vol_forecast_1d": cond_vol_1d,
        "std_residuals": std_resid,
        "params": {
            "mu_daily": mu_scaled / scale_factor,
            "omega_log_var": omega,
            "alpha1": alpha1,
            "gamma1_asymmetry": gamma1,
            "beta1": beta1,
            "persistence": beta1,
            "nu_df": nu_df,
            "aic": float(res.aic),
            "bic": float(res.bic),
            "loglikelihood": float(res.loglikelihood),
        },
    }


def egarch_filtered_hs_var_es(
    window_returns: np.ndarray | pd.Series,
    window_egarch_vols: np.ndarray | pd.Series,
    current_egarch_vol_forecast: float,
    mu_daily: float = 0.0,
    alpha: float = 0.99,
    notional_usd: float = 10_000_000.0,
) -> Tuple[float, float]:
    """
    Filtered Historical Simulation (FHS) using Asymmetric EGARCH(1,1) Volatility (Barone-Adesi et al., 1999):
      1. De-volatilize historical residuals over the rolling window:
           z_{t-i} = (r_{t-i} - mu) / sigma_{t-i}^{EGARCH}
      2. Rescale empirical standardized residuals by today's 1-step-ahead EGARCH volatility forecast sigma_{T|T-1}:
           r_{T, i}^{FHS} = mu + sigma_{T|T-1}^{EGARCH} * z_{t-i}
      3. Compute empirical non-parametric VaR and ES from the filtered loss distribution L_{T,i}^{FHS} = -V_0 * r_{T,i}^{FHS}.
    """
    r_arr = np.asarray(window_returns, dtype=np.float64)
    v_arr = np.maximum(np.asarray(window_egarch_vols, dtype=np.float64), 1e-8)

    std_resids = (r_arr - float(mu_daily)) / v_arr
    fhs_returns = float(mu_daily) + float(current_egarch_vol_forecast) * std_resids
    fhs_losses = -fhs_returns
    sorted_losses = np.sort(fhs_losses)[::-1]

    n = len(sorted_losses)
    k_idx = min(max(int(np.floor((1.0 - alpha) * n)), 0), n - 1)
    var_pct = float(sorted_losses[k_idx])
    es_pct = float(np.mean(sorted_losses[: k_idx + 1]))

    return var_pct * notional_usd, es_pct * notional_usd


def age_weighted_hs_var_es(
    window_returns: np.ndarray | pd.Series,
    alpha: float = 0.99,
    lambda_decay: float = 0.98,
    notional_usd: float = 10_000_000.0,
) -> Tuple[float, float]:
    """
    Age-Weighted Historical Simulation (Boudoukh, Richardson & Whitelaw / BRW Hybrid):
    Assigns exponentially decaying probability weights to the n historical observations
    where age i = 1 is the most recent observation (t - 1) and i = n is the oldest (t - n):
      w_i = [lambda^{i-1} * (1 - lambda)] / [1 - lambda^n]
    Sorts losses in descending order, accumulates probability mass until cumulative weight >= (1 - alpha),
    and returns (VaR_usd, ES_usd).
    """
    arr = np.asarray(window_returns, dtype=np.float64)
    n = len(arr)
    losses = -arr

    # Age index: arr[-1] has age i=1, arr[0] has age i=n
    ages = np.arange(n, 0, -1, dtype=np.float64)
    weights = (lambda_decay ** (ages - 1.0)) * (1.0 - lambda_decay) / (1.0 - (lambda_decay ** n))
    weights = weights / np.sum(weights)

    # Sort losses descending (worst loss first)
    order = np.argsort(losses)[::-1]
    sorted_losses = losses[order]
    sorted_weights = weights[order]
    cum_weights = np.cumsum(sorted_weights)

    tail_prob = 1.0 - alpha
    idx_hits = np.where(cum_weights >= tail_prob)[0]
    k_idx = int(idx_hits[0]) if len(idx_hits) > 0 else 0
    var_pct = float(sorted_losses[k_idx])

    # Expected Shortfall as probability-weighted expectation of the worst (1 - alpha) tail
    if k_idx == 0:
        es_pct = var_pct
    else:
        w_strict = sorted_weights[:k_idx]
        rem_w = max(tail_prob - float(np.sum(w_strict)), 0.0)
        es_pct = float((np.sum(w_strict * sorted_losses[:k_idx]) + rem_w * sorted_losses[k_idx]) / tail_prob)

    return var_pct * notional_usd, es_pct * notional_usd


def volatility_weighted_hs_var_es(
    window_returns: np.ndarray | pd.Series,
    window_vols: np.ndarray | pd.Series,
    current_vol_forecast: float,
    alpha: float = 0.99,
    notional_usd: float = 10_000_000.0,
) -> Tuple[float, float]:
    """
    Volatility-Weighted Historical Simulation (Hull & White, 1998):
    Rescales each historical return r_{t,i} by the ratio of today's conditional volatility
    forecast sigma_T to the historical conditional volatility sigma_i on day i:
      r_{t,i}^* = r_{t,i} * (sigma_T / sigma_i)
    Then computes empirical non-parametric VaR and ES on the volatility-updated losses L_{t,i}^* = -V_0 * r_{t,i}^*.
    """
    r_arr = np.asarray(window_returns, dtype=np.float64)
    v_arr = np.maximum(np.asarray(window_vols, dtype=np.float64), 1e-8)

    rescaled_returns = r_arr * (float(current_vol_forecast) / v_arr)
    rescaled_losses = -rescaled_returns
    sorted_losses = np.sort(rescaled_losses)[::-1]

    n = len(sorted_losses)
    k_idx = min(max(int(np.floor((1.0 - alpha) * n)), 0), n - 1)
    var_pct = float(sorted_losses[k_idx])
    es_pct = float(np.mean(sorted_losses[: k_idx + 1]))

    return var_pct * notional_usd, es_pct * notional_usd


def build_dynamic_var_series(
    portfolio_returns: pd.Series,
    alpha: float = 0.99,
    lookback_days: int = 250,
    notional_usd: float = 10_000_000.0,
    ewma_lambda: float = 0.94,
    age_lambda: float = 0.98,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Generate daily out-of-sample 1-step-ahead VaR forecasts across the entire sample
    (after an initial `lookback_days` burn-in) for:
      1. Plain Historical Simulation (250d equally weighted) — exhibits the 250-day "Ghost Effect"
      2. Parametric Rolling Normal VaR (250d)
      3. RiskMetrics EWMA Parametric VaR (lambda = 0.94)
      4. GARCH(1,1)-Normal Parametric VaR
      5. GARCH(1,1)-Student-t Parametric VaR
      6. EGARCH(1,1) Filtered Historical Simulation (FHS) — asymmetric leverage volatility + empirical residual scaling
      7. Age-Weighted Historical Simulation (BRW, lambda = 0.98)
      8. Hull-White Volatility-Weighted Historical Simulation (EWMA-scaled)
    All forecasts for date t strictly use returns and volatility states up to t-1.
    """
    clean_rets = portfolio_returns.dropna()
    dates = clean_rets.index
    r_vals = clean_rets.values.astype(np.float64)
    n_total = len(r_vals)

    # Compute 1-step-ahead volatility series
    ewma_vol_s = compute_ewma_volatility(clean_rets, lambda_decay=ewma_lambda)
    ewma_vals = ewma_vol_s.values

    garch_norm_fit = fit_garch11_volatility(clean_rets, dist="normal")
    garch_t_fit = fit_garch11_volatility(clean_rets, dist="t")
    egarch_t_fit = fit_egarch11_volatility(clean_rets, dist="t")

    garch_norm_vol = garch_norm_fit["cond_vol_forecast_1d"].values
    garch_t_vol = garch_t_fit["cond_vol_forecast_1d"].values
    egarch_t_vol = egarch_t_fit["cond_vol_forecast_1d"].values
    nu_garch_t = float(garch_t_fit["params"]["nu_df"])

    z_alpha = float(stats.norm.ppf(alpha))
    t_scale_mult = float(stats.t.ppf(alpha, df=nu_garch_t) * np.sqrt((nu_garch_t - 2.0) / nu_garch_t))

    out_records = []
    for t in range(lookback_days, n_total):
        win_rets = r_vals[t - lookback_days : t]
        win_ewma = ewma_vals[t - lookback_days : t]
        win_egarch = egarch_t_vol[t - lookback_days : t]
        win_losses = -win_rets
        sorted_losses = np.sort(win_losses)[::-1]
        k_idx = min(max(int(np.floor((1.0 - alpha) * lookback_days)), 0), lookback_days - 1)

        # 1. Plain HS (250d)
        plain_hs_var = float(sorted_losses[k_idx]) * notional_usd
        max_hist_loss_usd = float(sorted_losses[0]) * notional_usd

        # 2. Static Rolling Normal (250d)
        mu_w = float(np.mean(win_rets))
        sd_w = float(np.std(win_rets, ddof=1))
        rolling_norm_var = (-mu_w + z_alpha * sd_w) * notional_usd

        # 3. EWMA Parametric Normal VaR
        sigma_ewma_t = float(ewma_vals[t])
        ewma_norm_var = (z_alpha * sigma_ewma_t) * notional_usd

        # 4. GARCH(1,1) Normal VaR
        sigma_gn_t = float(garch_norm_vol[t])
        garch_norm_var = (-garch_norm_fit["params"]["mu_daily"] + z_alpha * sigma_gn_t) * notional_usd

        # 5. GARCH(1,1) Student-t VaR (with FHS standardized residual tail adjustment)
        sigma_gt_t = float(garch_t_vol[t])
        win_std_res = -r_vals[t - lookback_days : t] / np.maximum(garch_t_vol[t - lookback_days : t], 1e-8)
        emp_z_q = float(np.quantile(win_std_res, alpha))
        q_mult = max(t_scale_mult, emp_z_q)
        garch_t_var = (-garch_t_fit["params"]["mu_daily"] + q_mult * sigma_gt_t) * notional_usd

        # 6. Asymmetric EGARCH(1,1) Filtered Historical Simulation (FHS)
        sigma_eg_t = float(egarch_t_vol[t])
        egarch_fhs_var, egarch_fhs_es = egarch_filtered_hs_var_es(
            win_rets,
            win_egarch,
            current_egarch_vol_forecast=sigma_eg_t,
            mu_daily=float(egarch_t_fit["params"]["mu_daily"]),
            alpha=alpha,
            notional_usd=notional_usd,
        )

        # 7. Age-Weighted HS (BRW)
        age_hs_var, _ = age_weighted_hs_var_es(
            win_rets, alpha=alpha, lambda_decay=age_lambda, notional_usd=notional_usd
        )

        # 8. Hull-White Volatility-Weighted HS (using EWMA volatility ratios)
        hw_vwhs_var, _ = volatility_weighted_hs_var_es(
            win_rets, win_ewma, current_vol_forecast=sigma_ewma_t, alpha=alpha, notional_usd=notional_usd
        )

        out_records.append(
            {
                "Date": dates[t],
                "Realized_Return": r_vals[t],
                "Realized_Loss_USD": -r_vals[t] * notional_usd,
                "Max_Window_Loss_USD": max_hist_loss_usd,
                "Plain_HS_250d_VaR_USD": plain_hs_var,
                "Rolling_Normal_250d_VaR_USD": rolling_norm_var,
                "EWMA_Normal_VaR_USD": ewma_norm_var,
                "GARCH11_Normal_VaR_USD": garch_norm_var,
                "GARCH11_StudentT_VaR_USD": garch_t_var,
                "EGARCH11_FHS_VaR_USD": egarch_fhs_var,
                "EGARCH11_FHS_ES_USD": egarch_fhs_es,
                "Age_Weighted_HS_VaR_USD": age_hs_var,
                "Hull_White_VWHS_VaR_USD": hw_vwhs_var,
                "EWMA_Ann_Vol_pct": sigma_ewma_t * np.sqrt(252) * 100.0,
                "GARCH11_Ann_Vol_pct": sigma_gn_t * np.sqrt(252) * 100.0,
                "EGARCH11_Ann_Vol_pct": sigma_eg_t * np.sqrt(252) * 100.0,
                "Rolling_250d_Ann_Vol_pct": sd_w * np.sqrt(252) * 100.0,
            }
        )

    df_out = pd.DataFrame(out_records).set_index("Date")
    meta = {
        "garch_normal_params": garch_norm_fit["params"],
        "garch_t_params": garch_t_fit["params"],
        "egarch_t_params": egarch_t_fit["params"],
        "garch_normal_std_resid": garch_norm_fit["std_residuals"],
        "garch_t_std_resid": garch_t_fit["std_residuals"],
        "egarch_t_std_resid": egarch_t_fit["std_residuals"],
    }
    return df_out, meta
