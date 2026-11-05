# Project Plan: Market Risk Modelling, Backtesting & Derivatives Framework

## 1. Project Overview & Resume Alignment

- **Project Title:** Market Risk Modelling: VaR, Expected Shortfall & Monte Carlo Simulation
- **Classification:** Independent Quantitative Research Project (Aligned with GARP FRM Part I & Part II Market Risk Curriculum)
- **Core Objective:** Build a modular, reproducible Python risk engine that starts with foundational Exploratory Data Analysis (EDA) and baseline Value-at-Risk (VaR) / Expected Shortfall (ES) estimation, and progressively scales to dynamic time-series volatility regimes, statistical backtesting, and non-linear option portfolio stress testing.

### Resume Alignment Matrix

| Resume Bullet Point | Project Milestone | Core Technical Deliverables |
| :--- | :--- | :--- |
| Developing a Python-based market risk modelling framework to estimate portfolio VaR and ES using historical simulation, parametric approaches, and Monte Carlo simulation. | **Phase 0 (EDA) & Phase 1 (Crawl)** | Clean multi-asset return pipeline; Plain Historical Simulation, Parametric Variance-Covariance (Normal, Lognormal, Student-$t$), and Cholesky Monte Carlo engines. |
| Modelling financial asset returns and volatility using statistical and time-series techniques; evaluating the impact of changing market conditions and volatility regimes on portfolio risk estimates. | **Phase 2 (Walk)** | ADF stationarity tests; Rolling Volatility, RiskMetrics EWMA ($\lambda = 0.94$), and $\text{GARCH}(1,1)$ models; Hull-White Volatility-Weighted Historical Simulation across calm and crisis regimes. |
| Implementing backtesting procedures to assess VaR model performance, exception frequency, and model reliability under different market conditions. | **Phase 3 (Walk+)** | Out-of-sample rolling backtester; Basel Traffic Light zone counter; Kupiec POF unconditional coverage test; Christoffersen Markov independence and conditional coverage tests. |
| Extending the framework to evaluate derivative portfolio risk through simulated market risk factors and scenario-based exposure analysis. | **Phase 4 (Run)** | Black-Scholes-Merton pricing & analytical Greeks ($\Delta$, $\Gamma$, $\mathcal{V}$, $\Theta$); Delta-Normal vs. Delta-Gamma vs. Full Revaluation Monte Carlo; Historical Crisis Replay & Stress Grids. |

---

## 2. Business Context & Trading Desk Mandate

Quantitative risk management does not exist in a vacuum; risk metrics exist to enforce governance limits, size trades, and prevent insolvency during market panics. Throughout this project, the codebase operates as the risk engine for a **USD 10,000,000 Multi-Asset Macro Trading Desk**:

### Baseline Portfolio Allocation (USD 10,000,000 Notional)
1. **50% US Equities (USD 5,000,000):** Split between S&P 500 Index ETF (`SPY` – USD 3,000,000) and Apple Inc. (`AAPL` – USD 2,000,000) to represent broad market beta plus concentrated single-stock risk.
2. **30% Fixed Income (USD 3,000,000):** 20+ Year US Treasury Bond ETF (`TLT`) to capture interest-rate duration and stock-bond diversification.
3. **20% Commodities / Safe Haven (USD 2,000,000):** SPDR Gold Trust (`GLD`) to capture inflation and tail-crisis hedging behavior.

### Governance Mandate & Risk Limits
- The Chief Risk Officer (CRO) enforces a strict **1-Day 99% VaR limit of USD 200,000** (2.0% of portfolio NAV) and a **1-Day 99% Expected Shortfall (ES) limit of USD 280,000** (2.8% of portfolio NAV).

### Daily Business Decisions Supported by the Engine
- **Limit Monitoring & Escalation:** Flag daily limit breaches and diagnose whether a breach was driven by rising asset volatility or a breakdown in cross-asset correlations.
- **Marginal Risk & Pre-Trade Sizing:** When a portfolio manager proposes adding a volatile position (e.g., `TSLA` stock or an `SPY` option straddle), calculate the incremental/marginal risk contribution and determine the maximum allowable trade size that keeps the desk below its risk cap.
- **Model Selection for Capital Allocation:** Prove to the Model Validation Committee which risk estimator adapts fastest during regime transitions without generating excessive false alarms.

---

## 3. Data Strategy, Sources & Hygiene Rules

To ensure zero cost, 100% offline reproducibility, and no dependency on proprietary terminals, the project relies on public institutional-grade datasets cached locally as Apache Parquet files (`data/raw/*.parquet`):

| Data Layer | Source & Python Package | Tickers / Endpoints | Historical Window | Purpose in Framework |
| :--- | :--- | :--- | :--- | :--- |
| **Core Multi-Asset Prices** | Yahoo Finance (`yfinance`) | `SPY`, `AAPL`, `TLT`, `GLD`, `TSLA` | 2015-01-01 to 2026-01-01 (~2,750 trading days) | Covers tranquil markets (2017), COVID-19 crash (2020), inflation/rate-hike bear market (2022), and recent volatility. |
| **Market Implied Volatility** | Cboe Global Indices (Direct CSV) | [VIX History](https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv) | 2015-01-01 to 2026-01-01 (Daily OHLC) | Benchmarks statistical volatility forecasts (EWMA/GARCH) against option-implied market fear (VIX). |
| **Live Options Chain** | Yahoo Finance (`yfinance`) | `yf.Ticker("SPY").option_chain(expiry)` | Current EOD snapshot (Calls & Puts) | Provides real strike prices, bid/ask quotes, and implied volatilities for Phase 4 derivative stress testing. |
| **Optional Rates Extension** | Federal Reserve FRED (`pandas_datareader`) | `DGS1`, `DGS2`, `DGS5`, `DGS10` (CMT yields) | 2015-01-01 to 2026-01-01 | Used only in Phase 5 if extending from `TLT` ETF to direct bond cash-flow mapping. |

### Mandatory Data Hygiene Rules
- **Total-Return Adjusted Closes:** Always extract `Adj Close` for historical return calculations so stock splits and dividend payouts do not appear as artificial single-day price crashes.
- **Common Trading Calendar Alignment:** Inner-join all asset price series on NYSE trading dates and forward-fill at most 1 consecutive missing observation before computing returns.
- **Strict Point-in-Time Slicing (Zero Look-Ahead Bias):** Any risk forecast $\text{VaR}_t$ or volatility parameter $\hat{\sigma}_t$ estimated for trading day $t$ must strictly slice historical data up to index $t-1$ ($\mathcal{F}_{t-1}$).

---

## 4. Repository Structure

Organize your codebase cleanly from Day 1 so that configuration, data, core mathematical logic, and notebooks remain decoupled:

```text
market-risk-framework/
├── config/
│   └── portfolio.yaml                # Portfolio weights, notional ($10M), confidence levels (95%, 99%), lookback windows
├── data/
│   ├── raw/                          # Cached Parquet files from yfinance and Cboe VIX
│   └── processed/                    # Cleaned log/arithmetic return matrices and backtest logs
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Ingestion, calendar alignment, and return transformations
│   ├── eda_diagnostics.py            # ADF stationarity, Q-Q plots, rolling volatility, and correlation metrics
│   ├── var_estimators.py             # Phase 1: Historical Simulation, Parametric (Normal/t), and Basic Monte Carlo
│   ├── volatility_models.py          # Phase 2: Rolling StdDev, EWMA, GARCH(1,1), and Volatility-Weighted HS
│   ├── backtesting.py                # Phase 3: Traffic Light, Kupiec POF, and Christoffersen Markov tests
│   └── derivatives_stress.py         # Phase 4: Black-Scholes pricing, Greeks, Delta-Gamma, and Full Revaluation MC
├── notebooks/
│   ├── 00_data_hygiene_and_eda.ipynb
│   ├── 01_baseline_var_and_es.ipynb
│   ├── 02_volatility_regimes_and_ewma_garch.ipynb
│   ├── 03_model_backtesting_and_validation.ipynb
│   └── 04_derivatives_and_stress_testing.ipynb
├── tests/
│   └── test_risk_engine.py           # Unit tests verifying mathematical identities and edge cases
├── requirements.txt                  # numpy, pandas, scipy, statsmodels, arch, yfinance, matplotlib, pyyaml, pytest
└── README.md                         # Executive summary and key validation findings
```

---

## 5. Phased Implementation Roadmap (Crawl $\rightarrow$ Walk $\rightarrow$ Run)

### Phase 0: Data Engineering, Exploratory Data Analysis (EDA) & Stylized Facts
- **Objective:** Build the data ingestion pipeline, verify statistical stationarity, and empirically prove why naive normal distributions underestimate financial tail risk.
- **Technical Tasks & Methodology:**
  - **Return Transformations:** Compute both daily arithmetic returns:
    $$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$
    and geometric log returns:
    $$R_t = \ln(1 + r_t) = \ln\left(\frac{P_t}{P_{t-1}}\right)$$
    documenting how multi-period log returns sum additively while cross-sectional portfolio returns aggregate linearly via arithmetic returns ($r_{p,t} = \sum_{i=1}^N w_i r_{i,t}$).
  - **Stationarity Verification:** Run the Augmented Dickey-Fuller (ADF) test (`statsmodels.tsa.stattools.adfuller`) on raw price levels vs. daily returns. Confirm that prices have a unit root ($p > 0.05$, non-stationary), whereas daily returns reject the null hypothesis ($p < 0.01$, stationary).
  - **Distributional Stylized Facts (Fat Tails & Skewness):** Calculate sample mean, annualized volatility ($\sigma_{\text{daily}} \times \sqrt{252}$), skewness, and excess kurtosis for each asset and the aggregate portfolio. Generate a Quantile-Quantile (Q-Q) Plot against a standard normal distribution to visually highlight heavy tails (leptokurtosis) and negative skewness.
  - **Regime & Correlation Exploration:** Plot 20-day rolling annualized volatility alongside the Cboe VIX index to illustrate volatility clustering, and plot the 60-day rolling correlation between `SPY` and `TLT` to show how stock-bond correlation flipped from negative (diversifying) in 2020 to positive (compounding losses) in 2022.
- **Deliverable (`00_data_hygiene_and_eda.ipynb`):** A diagnostic report concluding that because portfolio returns exhibit excess kurtosis ($K > 3$) and time-varying volatility, static Gaussian risk models will systematically understate tail losses.

---

### Phase 1 (Crawl): Baseline VaR & Expected Shortfall Engine
- **Objective:** Implement the three foundational risk estimation methods at 95% and 99% confidence levels over a 1-day holding period, and connect the outputs to CRO limit governance.
- **Technical Tasks & Methodology:**
  - **Formal Loss & Risk Definitions:** Define daily portfolio dollar loss $L_t = -V_0 r_{p,t}$, where $V_0$ is USD 10,000,000. Define $\text{VaR}_\alpha$ as the $\alpha$-quantile of the loss distribution and Expected Shortfall as the conditional expectation of losses exceeding VaR:
    $$\text{ES}_\alpha = \mathbb{E}[L_t \mid L_t \ge \text{VaR}_\alpha]$$
  - **Estimator 1 — Plain Historical Simulation (Non-Parametric):** Using a rolling lookback window of $n = 250$ and $n = 500$ trading days, revalue current portfolio weights against historical return vectors, sort losses, identify the $[(1-\alpha)n + 1]\text{th}$ worst loss as $\text{VaR}_\alpha$, and compute $\text{ES}_\alpha$ as the average of all tail losses beyond that threshold. Also compute bootstrap standard errors around the quantile estimate.
  - **Estimator 2 — Parametric Variance-Covariance:** Compute the sample covariance matrix $\mathbf{\Sigma}$ across the 4 assets and portfolio standard deviation $\sigma_p = \sqrt{\mathbf{w}^\top \mathbf{\Sigma} \mathbf{w}}$. Implement:
    - **Normal VaR & ES:**
      $$\text{VaR}_\alpha^{\text{Norm}} = V_0 \left(-\mu_p + \sigma_p z_\alpha\right), \quad \text{ES}_\alpha^{\text{Norm}} = V_0 \left(-\mu_p + \sigma_p \frac{\phi(z_\alpha)}{1 - \alpha}\right)$$
    - **Lognormal VaR:**
      $$\text{VaR}_\alpha^{\text{Lognorm}} = V_0 \left(1 - \exp(\mu_R - \sigma_R z_\alpha)\right)$$
    - **Student-$t$ VaR & ES:** Fit degrees of freedom $\nu$ via maximum likelihood and scale $\sigma_{\text{scale}} = \sigma_p \sqrt{\frac{\nu-2}{\nu}}$ to capture fat tails parametrically.
    - **Cornish-Fisher Semi-Parametric VaR:** Inject sample skewness ($S$) and excess kurtosis ($K$) directly into the normal quantile $z_\alpha$:
      $$z_{\text{CF}}(\alpha) = z_\alpha + \frac{1}{6}(z_\alpha^2 - 1)S + \frac{1}{24}(z_\alpha^3 - 3z_\alpha)K - \frac{1}{36}(2z_\alpha^3 - 5z_\alpha)S^2$$
  - **Estimator 3 — Basic Monte Carlo Simulation:** Factorize the covariance matrix via Cholesky decomposition ($\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^\top$), draw $M = 10,000$ independent standard normal vectors $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, generate correlated asset return scenarios:
    $$\mathbf{r}^{(m)} = \boldsymbol{\mu} + \mathbf{L}\mathbf{Z}^{(m)}$$
    and compute empirical VaR and ES from the simulated P&L distribution.
  - **Coherence & Subadditivity Verification:** Demonstrate that Standalone Sum of VaRs exceeds Diversified Portfolio VaR (quantifying the portfolio diversification benefit), and include a standalone unit test proving why ES is a coherent risk measure whereas VaR can violate subadditivity ($\rho(L_1 + L_2) \le \rho(L_1) + \rho(L_2)$) on skewed/discrete payoffs.
- **Deliverable (`01_baseline_var_and_es.ipynb`):** A comparative dashboard showing 95% and 99% VaR and ES across all estimators, plus a pre-trade Marginal VaR calculator that determines how much `TSLA` can be added to the portfolio without breaching the CRO's USD 200,000 99% VaR cap.

---

### Phase 2 (Walk): Time-Series Volatility Regimes & Dynamic Historical Simulation
- **Objective:** Diagnose the structural flaws of equally weighted rolling windows ("ghost effects" and slow crisis reaction) and upgrade the risk engine using conditional time-series volatility models.
- **Technical Tasks & Methodology:**
  - **Diagnosing the "Ghost Effect":** Plot 250-day Plain Historical Simulation VaR across 2020–2021. Show how the March 2020 crash causes VaR to stay flat and artificially high for exactly 250 trading days, then drop abruptly on Day 251 when the crash observations roll out of the window.
  - **RiskMetrics EWMA Volatility:** Implement the Exponentially Weighted Moving Average variance recursion with decay factor $\lambda = 0.94$:
    $$\hat{\sigma}_t^2 = \lambda \hat{\sigma}_{t-1}^2 + (1 - \lambda) r_{t-1}^2$$
  - **$\text{GARCH}(1,1)$ Conditional Volatility:** Use the Python `arch` library to fit a $\text{GARCH}(1,1)$ model with both Normal and Student-$t$ innovations:
    $$\hat{\sigma}_t^2 = \omega + \alpha_1 \varepsilon_{t-1}^2 + \beta_1 \hat{\sigma}_{t-1}^2 \quad (\text{with } \alpha_1 + \beta_1 < 1)$$
    Extract 1-step-ahead conditional volatility forecasts $\hat{\sigma}_{t \mid t-1}$ and standardized residuals $\hat{\eta}_t = \varepsilon_t / \hat{\sigma}_t$.
  - **Dynamic Non-Parametric Upgrades:**
    - **Age-Weighted Historical Simulation (BRW Hybrid):** Assign exponentially decaying weights:
      $$w_i = \frac{\lambda^{i-1}(1-\lambda)}{1-\lambda^n}$$
      to historical observations so distant shocks smoothly lose influence.
    - **Volatility-Weighted Historical Simulation (Hull & White 1998):** Rescale each historical return $r_{t,i}$ by the ratio of today's EWMA/GARCH volatility forecast $\hat{\sigma}_T$ to the historical volatility $\hat{\sigma}_i$ on day $i$:
      $$r_{t,i}^* = r_{t,i} \frac{\hat{\sigma}_T}{\hat{\sigma}_i}$$
      Prove that unlike plain or age-weighted HS, Hull-White Volatility-Weighted HS can forecast losses larger than the worst historical loss in the sample when volatility spikes.
- **Deliverable (`02_volatility_regimes_and_ewma_garch.ipynb`):** Time-series overlay plots comparing Static Rolling VaR, EWMA VaR, $\text{GARCH}(1,1)$ VaR, and Hull-White Volatility-Weighted HS across calm markets (2017) and crisis regimes (2020 COVID crash and 2022 rate-hike cycle).

---

### Phase 3 (Walk+): Model Validation & Backtesting Suite
- **Objective:** Subject every VaR model built in Phases 1 and 2 to rigorous out-of-sample statistical backtesting to evaluate exception frequency, breach clustering, and regulatory capital penalties.
- **Technical Tasks & Methodology:**
  - **Out-of-Sample Hit Sequence Generation:** Over a 1,500-day out-of-sample test window (e.g., 2020–2025), compare each day's ex-ante 1-day VaR forecast $\text{VaR}_{\alpha, t}$ (computed using data up to $t-1$) against realized ex-post portfolio P&L $X_t$, recording the binary exception indicator:
    $$I_t = \begin{cases} 1 & \text{if } -X_t > \text{VaR}_{\alpha, t} \\ 0 & \text{otherwise} \end{cases}$$
  - **Basel Traffic Light Framework:** Count 99% VaR exceptions over rolling 250-day windows and classify model performance into:
    - **Green Zone:** 0–4 breaches (0% capital surcharge)
    - **Yellow/Amber Zone:** 5–9 breaches (progressive capital multiplier penalty)
    - **Red Zone:** $\ge 10$ breaches (model disqualification presumption)
  - **Kupiec (1995) Proportion of Failures (POF) Test (Unconditional Coverage):** Given $x = \sum_{t=1}^N I_t$ breaches over $N$ days and nominal tail probability $p = 1 - \alpha$, compute the likelihood-ratio test statistic:
    $$LR_{UC} = -2 \ln \left( \frac{p^x (1 - p)^{N - x}}{\left(\frac{x}{N}\right)^x \left(1 - \frac{x}{N}\right)^{N - x}} \right) \sim \chi^2_1$$
    Reject the model at the 5% significance level if $LR_{UC} > 3.841$ ($p\text{-value} < 0.05$).
  - **Christoffersen (1998) Independence & Conditional Coverage Tests:** Count the four adjacent-day transition states ($n_{00}, n_{01}, n_{10}, n_{11}$) in the hit sequence, compute transition probabilities:
    $$\hat{\pi}_{01} = \frac{n_{01}}{n_{00}+n_{01}}, \quad \hat{\pi}_{11} = \frac{n_{11}}{n_{10}+n_{11}}, \quad \hat{\pi} = \frac{n_{01}+n_{11}}{N}$$
    and evaluate:
    $$LR_{IND} = -2 \ln \left( \frac{(1 - \hat{\pi})^{n_{00} + n_{10}} \hat{\pi}^{n_{01} + n_{11}}}{(1 - \hat{\pi}_{01})^{n_{00}} \hat{\pi}_{01}^{n_{01}} (1 - \hat{\pi}_{11})^{n_{10}} \hat{\pi}_{11}^{n_{11}}} \right) \sim \chi^2_1$$
    $$LR_{CC} = LR_{UC} + LR_{IND} \sim \chi^2_2$$
- **Deliverable (`03_model_backtesting_and_validation.ipynb`):** A formal Champion-Challenger Model Validation Table reporting total exceptions, Basel Traffic Light zones, $LR_{UC}$, $LR_{IND}$, $LR_{CC}$, and $p$-values across all estimators—proving empirically that while Plain HS may pass Kupiec coverage over long windows, it fails Christoffersen independence due to volatility clustering, whereas GARCH / Volatility-Weighted HS passes both.

---

### Phase 4 (Run): Derivative Portfolio Risk & Scenario Stress Testing
- **Objective:** Extend the linear portfolio to include non-linear equity options on `SPY` (a protective put hedge and a delta-neutral straddle), demonstrating why linear VaR breaks down under convexity ($\Gamma$) and volatility surface shocks ($\mathcal{V}$).
- **Technical Tasks & Methodology:**
  - **Black-Scholes-Merton (BSM) Pricing & Implied Volatility Inversion:** Pull current `SPY` option chains via `yfinance`. Implement the exact BSM European call/put pricing formulas and use numerical root-finding (`scipy.optimize.brentq` or bisection) to solve for Implied Volatility:
    $$\sigma_{IV} = f^{-1}(P_{\text{mkt}}, S, K, r, T, q)$$
    plotting the empirical volatility skew across strike prices.
  - **Analytical Greeks Calculation:** Compute closed-form option sensitivities:
    - Delta: $\Delta = \frac{\partial V}{\partial S}$
    - Gamma: $\Gamma = \frac{\partial^2 V}{\partial S^2}$
    - Vega: $\mathcal{V} = \frac{\partial V}{\partial \sigma_{IV}}$
    - Theta: $\Theta = \frac{\partial V}{\partial t}$
  - **Comparing Three Derivative VaR Approaches:**
    - **Linear Delta-Normal Approximation:** Approximate option P&L as $\Delta V \approx \Delta \cdot \Delta S$. Show that for a delta-neutral straddle ($\Delta \approx 0$), Delta-Normal VaR erroneously reports near-zero risk.
    - **Quadratic Delta-Gamma-Vega Taylor Approximation:** Approximate P&L via second-order Taylor expansion:
      $$\Delta V \approx \Delta \, \Delta S + \frac{1}{2} \Gamma (\Delta S)^2 + \mathcal{V} \, \Delta \sigma_{IV} + \Theta \, \Delta t$$
    - **Full Revaluation Monte Carlo Simulation:** Simulate $M = 20,000$ joint scenarios of underlying `SPY` spot returns and implied volatility (VIX) shocks, and fully reprice the option portfolio along every path using the exact BSM equation:
      $$\Delta V^{(m)} = V_{\text{BSM}}(S_{t+1}^{(m)}, \sigma_{IV, t+1}^{(m)}, T - \Delta t) - V_{\text{BSM}}(S_t, \sigma_{IV, t}, T)$$
  - **Historical & Hypothetical Stress Testing:**
    - **Historical Crisis Replay:** Revalue the hedged vs. unhedged portfolio through the exact daily factor shocks of March 2020 (COVID crash) and 2022 (simultaneous stock-bond selloff).
    - **2D Hypothetical Vulnerability Grid:** Generate a heatmap of portfolio P&L across a $5 \times 5$ stress matrix combining underlying equity shocks ($-20\%$ to $+10\%$) with implied volatility shocks ($-5$ to $+25$ vol points).
- **Deliverable (`04_derivatives_and_stress_testing.ipynb`):** An executive risk memo quantifying the divergence between Delta-Normal, Delta-Gamma, and Full Revaluation Monte Carlo on option positions, alongside stress-test heatmaps demonstrating how protective puts alter the portfolio's tail loss profile.

---

### Phase 5 (Optional Scale-Up): Advanced FRM Part II Extensions
Once Phases 0 through 4 are complete and tested, your project already fulfills 100% of your resume claims. You can optionally add any of the following standalone modules to demonstrate advanced FRM Part II mastery:

- **Extension A — Fixed-Income Cash-Flow Mapping:** Replace the `TLT` ETF with two explicit US Treasury coupon bonds mapped to FRED zero-coupon vertices (`DGS1`–`DGS10`), proving why Cash-Flow Mapping VaR is strictly lower than Duration Mapping and Principal Mapping VaR.
- **Extension B — Extreme Value Theory (Peaks-Over-Threshold):** Fit a Generalized Pareto Distribution (GPD) to losses exceeding the 95th percentile threshold to estimate extreme 99.9% VaR and ES analytically.
- **Extension C — Backtesting Expected Shortfall:** Implement the Acerbi-Székely $Z_1$ and $Z_2$ test statistics to backtest 97.5% Expected Shortfall directly.
- **Extension D — FRTB Liquidity Horizon Cascade:** Assign portfolio risk factors to the 5 statutory Basel liquidity buckets (10, 20, 40, 60, and 120 days) and compute Liquidity-Adjusted Expected Shortfall.

---

## 6. Execution Checklist & Definition of Done

Use this checklist to track your progress from initial setup to final portfolio presentation:

- [ ] **Step 1 — Environment & Data Setup:** Initialize Git repository, create `config/portfolio.yaml`, and write `src/data_loader.py` to cache 10 years of `SPY`, `AAPL`, `TLT`, `GLD`, and Cboe `VIX` data to `data/raw/`.
- [ ] **Step 2 — EDA & Stylized Facts (`00_data_hygiene_and_eda.ipynb`):** Complete ADF stationarity table, Q-Q fat-tail plots, 20-day rolling volatility charts, and 60-day stock-bond rolling correlation analysis.
- [ ] **Step 3 — Baseline VaR & ES (`01_baseline_var_and_es.ipynb`):** Implement Plain HS, Parametric (Normal, Lognormal, Student-$t$, Cornish-Fisher), and Cholesky Monte Carlo; verify subadditivity and CRO limit checks.
- [ ] **Step 4 — Dynamic Volatility Regimes (`02_volatility_regimes_and_ewma_garch.ipynb`):** Implement EWMA ($\lambda = 0.94$), $\text{GARCH}(1,1)$, and Hull-White Volatility-Weighted HS; demonstrate elimination of the 250-day ghost effect.
- [ ] **Step 5 — Model Validation Suite (`03_model_backtesting_and_validation.ipynb`):** Run 1,500-day out-of-sample backtest; output Basel Traffic Light zones, Kupiec $LR_{UC}$, and Christoffersen $LR_{IND}$ / $LR_{CC}$ test results.
- [ ] **Step 6 — Derivatives & Stress Testing (`04_derivatives_and_stress_testing.ipynb`):** Implement BSM option pricer, implied volatility solver, and Greeks; compare Delta-Normal, Delta-Gamma, and Full Revaluation MC; run 2020/2022 historical stress replays.
- [ ] **Step 7 — Unit Testing & Executive Summary:** Add `pytest` suite (`tests/test_risk_engine.py`) verifying closed-form identities (e.g., $\text{VaR}_{0.99}^{\text{Norm}} \approx 2.326\sigma$ vs. $\text{ES}_{0.975}^{\text{Norm}} \approx 2.338\sigma$, and spaced vs. clustered Christoffersen test) and publish a concise 1-page Executive Validation Summary in `README.md`.