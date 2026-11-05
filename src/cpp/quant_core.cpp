// =============================================================================
// quant_core.cpp — C++17 Quantitative Risk & Derivatives Pricing Kernel
// Compiled as a native shared library (.dylib / .so) and bound via ctypes
// =============================================================================

#include <cmath>
#include <algorithm>
#include <vector>

namespace {

constexpr double INV_SQRT_2PI = 0.39894228040143267794;
constexpr double SQRT_2 = 1.41421356237309504880;

inline double norm_pdf(double x) {
    return INV_SQRT_2PI * std::exp(-0.5 * x * x);
}

inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / SQRT_2);
}

inline double bsm_price_scalar(double S, double K, double r, double q, double sigma, double T, int is_call) {
    if (T <= 0.0 || sigma <= 0.0) {
        double intrinsic = (is_call != 0) ? std::max(S - K, 0.0) : std::max(K - S, 0.0);
        return intrinsic;
    }
    double sqrt_T = std::sqrt(T);
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
    double d2 = d1 - sigma * sqrt_T;
    double df_r = std::exp(-r * T);
    double df_q = std::exp(-q * T);

    if (is_call != 0) {
        return S * df_q * norm_cdf(d1) - K * df_r * norm_cdf(d2);
    } else {
        return K * df_r * norm_cdf(-d2) - S * df_q * norm_cdf(-d1);
    }
}

} // namespace

extern "C" {

// 1. Scalar Black-Scholes-Merton Price
double cpp_bsm_price(double S, double K, double r, double q, double sigma, double T, int is_call) {
    return bsm_price_scalar(S, K, r, q, sigma, T, is_call);
}

// 2. Analytical Black-Scholes-Merton Greeks (Price, Delta, Gamma, Vega, Theta)
// Note: Vega is per 1.00 (100 vol points) change in sigma; Theta is per calendar year.
void cpp_bsm_greeks(
    double S, double K, double r, double q, double sigma, double T, int is_call,
    double* out_price, double* out_delta, double* out_gamma, double* out_vega, double* out_theta
) {
    if (T <= 1e-12 || sigma <= 1e-12) {
        *out_price = bsm_price_scalar(S, K, r, q, sigma, T, is_call);
        if (is_call != 0) {
            *out_delta = (S > K) ? 1.0 : 0.0;
        } else {
            *out_delta = (S < K) ? -1.0 : 0.0;
        }
        *out_gamma = 0.0;
        *out_vega = 0.0;
        *out_theta = 0.0;
        return;
    }

    double sqrt_T = std::sqrt(T);
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T);
    double d2 = d1 - sigma * sqrt_T;
    double df_r = std::exp(-r * T);
    double df_q = std::exp(-q * T);
    double pdf_d1 = norm_pdf(d1);

    *out_price = bsm_price_scalar(S, K, r, q, sigma, T, is_call);
    *out_gamma = (df_q * pdf_d1) / (S * sigma * sqrt_T);
    *out_vega = S * df_q * pdf_d1 * sqrt_T;

    if (is_call != 0) {
        double nd1 = norm_cdf(d1);
        double nd2 = norm_cdf(d2);
        *out_delta = df_q * nd1;
        *out_theta = -(S * df_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
                     - r * K * df_r * nd2
                     + q * S * df_q * nd1;
    } else {
        double n_minus_d1 = norm_cdf(-d1);
        double n_minus_d2 = norm_cdf(-d2);
        *out_delta = -df_q * n_minus_d1;
        *out_theta = -(S * df_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
                     + r * K * df_r * n_minus_d2
                     - q * S * df_q * n_minus_d1;
    }
}

// 3. Implied Volatility Solver via Safeguarded Newton-Raphson / Brent-Bisection
double cpp_bsm_implied_vol(
    double market_price, double S, double K, double r, double q, double T, int is_call,
    double tol, int max_iter
) {
    if (T <= 0.0 || market_price <= 0.0) {
        return -1.0;
    }
    double df_r = std::exp(-r * T);
    double df_q = std::exp(-q * T);
    double lower_bound = (is_call != 0)
        ? std::max(0.0, S * df_q - K * df_r)
        : std::max(0.0, K * df_r - S * df_q);
    double upper_bound = (is_call != 0) ? S * df_q : K * df_r;

    if (market_price < lower_bound - 1e-8 || market_price > upper_bound + 1e-8) {
        return -1.0;
    }

    double low = 1e-5;
    double high = 5.0;
    double sigma = 0.25; // initial guess

    for (int i = 0; i < max_iter; ++i) {
        double price = 0.0, delta = 0.0, gamma = 0.0, vega = 0.0, theta = 0.0;
        cpp_bsm_greeks(S, K, r, q, sigma, T, is_call, &price, &delta, &gamma, &vega, &theta);
        double diff = price - market_price;
        if (std::abs(diff) < tol) {
            return sigma;
        }
        if (diff > 0.0) {
            high = sigma;
        } else {
            low = sigma;
        }
        // Use Newton step if vega is well-behaved and step stays inside [low, high]
        if (vega > 1e-8) {
            double newton_next = sigma - diff / vega;
            if (newton_next > low && newton_next < high) {
                sigma = newton_next;
                continue;
            }
        }
        sigma = 0.5 * (low + high);
        if ((high - low) < tol) {
            return sigma;
        }
    }
    return sigma;
}

// 4. Vectorized Full Revaluation Monte Carlo for Option Portfolios
void cpp_bsm_mc_revaluation(
    const double* spot_scenarios,
    const double* vol_scenarios,
    int n_scenarios,
    double K,
    double r,
    double q,
    double T_new,
    int is_call,
    double base_price,
    double position_units,
    double* out_pnl
) {
    for (int i = 0; i < n_scenarios; ++i) {
        double p_sim = bsm_price_scalar(spot_scenarios[i], K, r, q, vol_scenarios[i], T_new, is_call);
        out_pnl[i] = position_units * (p_sim - base_price);
    }
}

// 5. RiskMetrics EWMA Variance Recursion
void cpp_ewma_volatility(
    const double* returns,
    int n,
    double lambda_decay,
    double init_var,
    double* out_vol
) {
    if (n <= 0) return;
    double var_t = init_var;
    out_vol[0] = std::sqrt(var_t);
    for (int t = 1; t < n; ++t) {
        double r_prev = returns[t - 1];
        var_t = lambda_decay * var_t + (1.0 - lambda_decay) * (r_prev * r_prev);
        out_vol[t] = std::sqrt(var_t);
    }
}

} // extern "C"
