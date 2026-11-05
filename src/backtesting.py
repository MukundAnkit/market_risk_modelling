"""
Model Validation & Statistical Backtesting Suite (`src/backtesting.py`).

Implements Phase 3 (Walk+) methodology:
1. Out-of-Sample Hit Sequence Generation:
     I_t = 1 if -X_t > VaR_{alpha, t} else 0
2. Basel Committee Traffic Light Framework (over rolling 250-day windows and latest 250-day window):
   - Green Zone: 0–4 breaches at 99% confidence (0.00 capital surcharge add-on)
   - Yellow/Amber Zone: 5–9 breaches (progressive multiplier penalty from +0.40 to +0.85)
   - Red Zone: >= 10 breaches (+1.00 multiplier penalty & model disqualification presumption)
3. Kupiec (1995) Proportion of Failures (POF) Unconditional Coverage Test:
     LR_UC = -2 ln [ (p^x * (1-p)^{N-x}) / ((x/N)^x * (1 - x/N)^{N-x}) ] ~ chi^2_1
4. Christoffersen (1998) Markov Independence & Conditional Coverage Tests:
     Counts adjacent transition states (n_00, n_01, n_10, n_11), transition probabilities
     pi_01, pi_11, pi, and evaluates LR_IND ~ chi^2_1 and LR_CC = LR_UC + LR_IND ~ chi^2_2.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


BASEL_YELLOW_ADDONS = {
    5: 0.40,
    6: 0.50,
    7: 0.65,
    8: 0.75,
    9: 0.85,
}


def generate_hit_sequence(
    realized_losses: np.ndarray | pd.Series,
    var_forecasts: np.ndarray | pd.Series,
) -> np.ndarray:
    """
    Compute the binary VaR exception indicator sequence:
      I_t = 1 if Realized_Loss_t > VaR_t else 0
    """
    losses_arr = np.asarray(realized_losses, dtype=np.float64)
    var_arr = np.asarray(var_forecasts, dtype=np.float64)
    return (losses_arr > var_arr).astype(np.int32)


def classify_basel_traffic_light(
    breach_count_250d: int,
    base_multiplier: float = 3.0,
) -> Dict[str, object]:
    """
    Classify a 250-day 99% VaR exception count into Basel Traffic Light zones
    and return the corresponding regulatory capital multiplier k = 3.0 + add_on.
    """
    n_b = int(breach_count_250d)
    if n_b <= 4:
        zone = "Green"
        addon = 0.00
    elif 5 <= n_b <= 9:
        zone = "Yellow"
        addon = BASEL_YELLOW_ADDONS[n_b]
    else:
        zone = "Red"
        addon = 1.00

    return {
        "breaches_250d": n_b,
        "zone": zone,
        "capital_addon": addon,
        "effective_multiplier": base_multiplier + addon,
    }


def kupiec_pof_test(
    hit_sequence: np.ndarray | pd.Series,
    alpha: float = 0.99,
    significance_level: float = 0.05,
) -> Dict[str, float | bool]:
    """
    Kupiec (1995) Proportion of Failures (POF) Likelihood-Ratio Test for Unconditional Coverage:
      H_0: Empirical failure probability pi_hat == p = 1 - alpha
      LR_UC = -2 * [ x*ln(p) + (N-x)*ln(1-p) - x*ln(x/N) - (N-x)*ln(1 - x/N) ] ~ chi^2_1
    """
    hits = np.asarray(hit_sequence, dtype=np.int32)
    n = len(hits)
    x = int(np.sum(hits))
    p = 1.0 - alpha

    if n == 0:
        raise ValueError("hit_sequence must not be empty.")

    pi_hat = x / n
    # Log-likelihood under null H_0
    ll_null = x * np.log(p) + (n - x) * np.log(1.0 - p)

    # Log-likelihood under alternative H_1 (handle boundary cases x=0 or x=n safely)
    if x == 0:
        ll_alt = 0.0
    elif x == n:
        ll_alt = 0.0
    else:
        ll_alt = x * np.log(pi_hat) + (n - x) * np.log(1.0 - pi_hat)

    lr_uc = float(max(-2.0 * (ll_null - ll_alt), 0.0))
    p_value = float(1.0 - stats.chi2.cdf(lr_uc, df=1))
    crit_val = float(stats.chi2.ppf(1.0 - significance_level, df=1))  # 3.8414588

    return {
        "n_obs": int(n),
        "n_breaches": x,
        "expected_breaches": float(n * p),
        "empirical_breach_rate_pct": float(pi_hat * 100.0),
        "nominal_tail_prob_pct": float(p * 100.0),
        "lr_uc": lr_uc,
        "p_value_uc": p_value,
        "critical_value_5pct": crit_val,
        "reject_uc_5pct": bool(lr_uc > crit_val),
    }


def christoffersen_markov_tests(
    hit_sequence: np.ndarray | pd.Series,
    alpha: float = 0.99,
    significance_level: float = 0.05,
) -> Dict[str, float | int | bool]:
    """
    Christoffersen (1998) Markov Independence Test (LR_IND ~ chi^2_1) and
    Joint Conditional Coverage Test (LR_CC = LR_UC + LR_IND ~ chi^2_2).
    """
    hits = np.asarray(hit_sequence, dtype=np.int32)
    if len(hits) < 2:
        raise ValueError("hit_sequence must contain at least 2 observations.")

    prev_s = hits[:-1]
    curr_s = hits[1:]

    n00 = int(np.sum((prev_s == 0) & (curr_s == 0)))
    n01 = int(np.sum((prev_s == 0) & (curr_s == 1)))
    n10 = int(np.sum((prev_s == 1) & (curr_s == 0)))
    n11 = int(np.sum((prev_s == 1) & (curr_s == 1)))

    n_trans = n00 + n01 + n10 + n11
    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi_bar = (n01 + n11) / n_trans if n_trans > 0 else 0.0

    def _safe_term(count: int, prob: float) -> float:
        if count == 0 or prob <= 0.0:
            return 0.0
        return float(count * np.log(prob))

    ll_ind_null = _safe_term(n00 + n10, 1.0 - pi_bar) + _safe_term(n01 + n11, pi_bar)
    ll_ind_alt = (
        _safe_term(n00, 1.0 - pi_01)
        + _safe_term(n01, pi_01)
        + _safe_term(n10, 1.0 - pi_11)
        + _safe_term(n11, pi_11)
    )

    lr_ind = float(max(-2.0 * (ll_ind_null - ll_ind_alt), 0.0))
    p_val_ind = float(1.0 - stats.chi2.cdf(lr_ind, df=1))
    crit_ind = float(stats.chi2.ppf(1.0 - significance_level, df=1))  # 3.8414588

    uc_res = kupiec_pof_test(hits, alpha=alpha, significance_level=significance_level)
    lr_uc = float(uc_res["lr_uc"])
    lr_cc = lr_uc + lr_ind
    p_val_cc = float(1.0 - stats.chi2.cdf(lr_cc, df=2))
    crit_cc = float(stats.chi2.ppf(1.0 - significance_level, df=2))  # 5.9914645

    return {
        **uc_res,
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "pi_01": float(pi_01),
        "pi_11": float(pi_11),
        "pi_uncond": float(pi_bar),
        "lr_ind": lr_ind,
        "p_value_ind": p_val_ind,
        "reject_ind_5pct": bool(lr_ind > crit_ind),
        "lr_cc": lr_cc,
        "p_value_cc": p_val_cc,
        "critical_cc_5pct": crit_cc,
        "reject_cc_5pct": bool(lr_cc > crit_cc),
    }


def run_champion_challenger_backtest(
    dyn_var_df: pd.DataFrame,
    alpha: float = 0.99,
    oos_days: int = 1500,
    basel_window: int = 250,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate all VaR models over the last `oos_days` out-of-sample trading days (default 1,500 days,
    spanning 2020–2025) and produce:
      1. Formal Champion-Challenger Model Validation Table with total exceptions, empirical breach %,
         Basel Traffic Light classifications (worst 250d crisis window & latest 250d window),
         Kupiec LR_UC, Christoffersen LR_IND, Conditional Coverage LR_CC, p-values, and validation verdict.
      2. Rolling 250-day breach count DataFrame across all models.
    """
    oos_df = dyn_var_df.iloc[-oos_days:].copy()
    realized_losses = oos_df["Realized_Loss_USD"].values

    model_mapping = {
        "Plain Historical Sim (250d)": "Plain_HS_250d_VaR_USD",
        "Static Rolling Normal (250d)": "Rolling_Normal_250d_VaR_USD",
        "RiskMetrics EWMA Normal (λ=0.94)": "EWMA_Normal_VaR_USD",
        "GARCH(1,1) Normal": "GARCH11_Normal_VaR_USD",
        "GARCH(1,1) Student-t": "GARCH11_StudentT_VaR_USD",
        "Age-Weighted HS (BRW, λ=0.98)": "Age_Weighted_HS_VaR_USD",
        "Hull-White Volatility-Weighted HS": "Hull_White_VWHS_VaR_USD",
    }

    summary_rows: List[Dict[str, object]] = []
    rolling_breaches_dict: Dict[str, pd.Series] = {}

    for label, col in model_mapping.items():
        var_series = oos_df[col].values
        hits = generate_hit_sequence(realized_losses, var_series)
        hits_s = pd.Series(hits, index=oos_df.index, name=label)

        roll_250 = hits_s.rolling(window=basel_window).sum().dropna()
        rolling_breaches_dict[label] = roll_250

        latest_250_breaches = int(hits_s.iloc[-basel_window:].sum())
        max_250_breaches = int(roll_250.max())

        basel_latest = classify_basel_traffic_light(latest_250_breaches)
        basel_peak = classify_basel_traffic_light(max_250_breaches)
        stats_res = christoffersen_markov_tests(hits, alpha=alpha)

        passed_all = (not stats_res["reject_uc_5pct"]) and (not stats_res["reject_ind_5pct"]) and (not stats_res["reject_cc_5pct"])
        if passed_all:
            verdict = "PASS (Champion Tier)"
        elif (not stats_res["reject_uc_5pct"]) and stats_res["reject_ind_5pct"]:
            verdict = "FAIL Independence (Clustered Breaches)"
        elif stats_res["reject_uc_5pct"] and (not stats_res["reject_ind_5pct"]):
            verdict = "FAIL Coverage (Excessive Breaches)"
        else:
            verdict = "FAIL Both Coverage & Independence"

        summary_rows.append(
            {
                "Model": label,
                "OOS_Days": int(stats_res["n_obs"]),
                "Expected_Breaches": float(stats_res["expected_breaches"]),
                "Total_Breaches": int(stats_res["n_breaches"]),
                "Breach_Rate_pct": float(stats_res["empirical_breach_rate_pct"]),
                "Clustered_Pairs_n11": int(stats_res["n11"]),
                "Latest_250d_Breaches": latest_250_breaches,
                "Latest_Basel_Zone": str(basel_latest["zone"]),
                "Peak_250d_Breaches": max_250_breaches,
                "Peak_Basel_Zone": str(basel_peak["zone"]),
                "LR_UC": float(stats_res["lr_uc"]),
                "p_val_UC": float(stats_res["p_value_uc"]),
                "LR_IND": float(stats_res["lr_ind"]),
                "p_val_IND": float(stats_res["p_value_ind"]),
                "LR_CC": float(stats_res["lr_cc"]),
                "p_val_CC": float(stats_res["p_value_cc"]),
                "Validation_Verdict": verdict,
            }
        )

    validation_table = pd.DataFrame(summary_rows).set_index("Model")
    rolling_breaches_df = pd.DataFrame(rolling_breaches_dict)
    return validation_table, rolling_breaches_df
