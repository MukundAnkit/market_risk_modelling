# Market Risk Modelling, Backtesting & Derivatives Stress-Testing Framework

[![Market Risk Engine CI](https://github.com/MukundAnkit/market_risk_modelling/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundAnkit/market_risk_modelling)

An institutional-grade **Python + C++17 Quantitative Market Risk Engine** aligned with the **GARP FRM Part I & Part II Market Risk Curriculum**. The framework models, backtests, and stress-tests a **USD 10,000,000 Multi-Asset Macro Trading Desk** (`SPY` 30%, `AAPL` 20%, `TLT` 30%, `GLD` 20%) operating under a Chief Risk Officer (CRO) governance mandate of **1-Day 99% VaR $\le$ USD 200,000** (2.0% of NAV) and **1-Day 99% Expected Shortfall (ES) $\le$ USD 280,000** (2.8% of NAV).

### Technical Documentation & Whitepaper (`docs/`)
- **Compiled PDF Whitepaper:** [`docs/market_risk_modelling_technical_whitepaper.pdf`](docs/market_risk_modelling_technical_whitepaper.pdf)
- **GitHub-Native Markdown Technical Report:** [`docs/model_development_technical_report.md`](docs/model_development_technical_report.md)
- **LaTeX Source Document:** [`docs/market_risk_modelling_technical_whitepaper.tex`](docs/market_risk_modelling_technical_whitepaper.tex)

---

## 1. System Architecture & Hybrid Language Design

The repository decouples configuration, data caching, compiled C++17 pricing/volatility kernels, modular Python risk estimators, interactive Jupyter showcase notebooks, and model documentation:

```text
market_risk_modelling/
├── .github/workflows/ci.yml          # GitHub Actions CI (compiles C++17 kernel & runs pytest)
├── .vscode/settings.json             # Antigravity / VS Code IDE workspace configuration
├── config/
│   └── portfolio.yaml                # $10M desk weights, CRO limits ($200k VaR / $280k ES), model parameters
├── data/
│   ├── raw/                          # Local Parquet caches (multi_asset_prices.parquet, cboe_vix.parquet)
│   └── processed/                    # Aligned arithmetic/log returns & portfolio P&L matrices
├── docs/
│   ├── figures/                      # High-DPI charts exported from Phases 0–4 notebooks
│   ├── market_risk_modelling_technical_whitepaper.pdf
│   ├── market_risk_modelling_technical_whitepaper.tex
│   └── model_development_technical_report.md
├── src/
│   ├── cpp/
│   │   └── quant_core.cpp            # Native C++17 shared-library kernel (BSM, Greeks, IV solver, MC, EWMA)
│   ├── cpp_kernel.py                 # Automatic clang++/g++ compilation & ctypes bridge with Python fallback
│   ├── data_loader.py                # Ingestion, NYSE calendar alignment, and strict F_{t-1} slicing
│   ├── eda_diagnostics.py            # Phase 0: ADF stationarity, stylized facts, Q-Q & regime correlations
│   ├── var_estimators.py             # Phase 1: Plain HS, Normal/Lognormal/Student-t/CF, Cholesky MC, Marginal VaR
│   ├── volatility_models.py          # Phase 2: Rolling Vol, EWMA, GARCH(1,1), Age-Weighted & Hull-White VWHS
│   ├── backtesting.py                # Phase 3: Basel Traffic Light, Kupiec POF, Christoffersen Markov tests
│   └── derivatives_stress.py         # Phase 4: BSM IV surface, Greeks, Delta-Normal vs Delta-Gamma-Vega vs Full MC
├── notebooks/
│   ├── 00_data_hygiene_and_eda.ipynb
│   ├── 01_baseline_var_and_es.ipynb
│   ├── 02_volatility_regimes_and_ewma_garch.ipynb
│   ├── 03_model_backtesting_and_validation.ipynb
│   └── 04_derivatives_and_stress_testing.ipynb
├── tests/
│   └── test_risk_engine.py           # Unit tests verifying mathematical identities, parity, and backtests
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 2. Executive Validation Summary (Phases 0–4)

### Phase 0: Data Hygiene, Stationarity & Distributional Stylized Facts (`00_data_hygiene_and_eda.ipynb`)
- **Stationarity (ADF Test):** All four core assets (`SPY`, `AAPL`, `TLT`, `GLD`) fail to reject a unit root in raw price levels ($p \in [0.389, 0.905]$, non-stationary) and reject the unit-root null at $p < 10^{-15}$ in daily returns (stationary).
- **Fat Tails & Asymmetry:** The $\$10\text{M}$ portfolio exhibits negative skewness ($S = -0.444$) and substantial excess kurtosis ($K_{\text{excess}} = 6.208$, total kurtosis $K = 9.208 > 3$). Extreme $>3\sigma$ tail events occur **$6.35\times$ more frequently** than predicted by a Gaussian distribution.
- **Regime Correlation Breakdown:** The 60-day rolling `SPY`–`TLT` correlation flipped from strongly negative ($\rho = -0.709$, diversifying flight-to-quality) during the Q1 2020 COVID-19 crash to strongly positive ($\rho = +0.483$, compounding stock-bond losses) during the 2022 inflation and rate-hike bear market.

### Phase 1 (Crawl): Baseline VaR, Expected Shortfall & Pre-Trade Sizing (`01_baseline_var_and_es.ipynb`)
- **1-Day 99% Risk Comparison ($10M Desk):**
  - **Plain Historical Simulation (250d):** $\text{VaR}_{0.99} = \$158,474$ | $\text{ES}_{0.99} = \$173,658$ (Bootstrap SE $= \$17,724$)
  - **Parametric Normal (Var-Cov):** $\text{VaR}_{0.99} = \$151,354$ | $\text{ES}_{0.99} = \$173,222$
  - **Parametric Student-$t$ (MLE $\hat{\nu} = 7.3$):** $\text{VaR}_{0.99} = \$164,126$ | $\text{ES}_{0.99} = \$204,959$
  - **Cornish-Fisher Semi-Parametric:** $\text{VaR}_{0.99} = \$156,794$ | $\text{ES}_{0.99} = \$185,878$
  - **Cholesky Monte Carlo ($M = 10,000$):** $\text{VaR}_{0.99} = \$151,617$ | $\text{ES}_{0.99} = \$173,974$
- **Diversification & Coherence:** Multi-asset diversification reduces 1-Day 99% Normal VaR by **$\$88,307$** ($36.8\%$) and Historical Simulation VaR by **$\$109,235$** ($40.8\%$) relative to the undiversified sum of standalone asset VaRs. Analytical and unit-tested counterexamples demonstrate how VaR violates subadditivity ($\rho(L_1+L_2) > \rho(L_1)+\rho(L_2)$) on discrete credit/default payoffs while Expected Shortfall remains strictly coherent.
- **Pre-Trade Marginal VaR Sizing (`TSLA`):** Adding volatile `TSLA` shares on top of the $\$10\text{M}$ desk has a Marginal VaR of $\$0.0397$ per dollar added, capping the maximum allowable pre-trade `TSLA` addition at **$\$961,274$** (Parametric Normal) and **$\$765,211$** (Historical Simulation) before breaching the CRO's $\$200,000$ 1-Day 99% VaR limit.

### Phase 2 (Walk): Time-Series Volatility Regimes & Dynamic HS (`02_volatility_regimes_and_ewma_garch.ipynb`)
- **Eliminating the 250-Day "Ghost Effect":** Plain 250-day Historical Simulation creates an artificial year-long plateau after March 2020 ($\text{VaR}_{0.99} = \$343,189$) that drops abruptly in March 2021. **Age-Weighted HS ($\lambda = 0.98$)**, **RiskMetrics EWMA ($\lambda = 0.94$)**, **$\text{GARCH}(1,1)$**, and **Hull-White (1998) Volatility-Weighted HS (VWHS)** adapt immediately to volatility spikes and decay smoothly as markets calm.
- **Exceeding Historical Sample Maxima:** During volatility spikes, Hull-White VWHS rescales historical returns by $\hat{\sigma}_T / \hat{\sigma}_i$, forecasting 99% VaR above the worst unscaled loss in the 250-day window on 19 crisis trading days.

### Phase 3 (Walk+): Champion–Challenger Model Validation Table (`03_model_backtesting_and_validation.ipynb`)
Over a **1,500-day out-of-sample window (2020–2025, expected 99% breaches = 15.0)**:

| Risk Estimator | Total Breaches | Breach Rate (%) | Clustered Pairs ($n_{11}$) | Latest Basel Zone | Kupiec $LR_{UC}$ ($p$-val) | Christoffersen $LR_{IND}$ ($p$-val) | Joint $LR_{CC}$ ($p$-val) | Model Validation Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Plain Historical Sim (250d)** | 22 | 1.47% | 2 | Green (4) | 2.885 (0.089) | 4.209 (**0.040**) | 7.093 (**0.029**) | **FAIL Independence (Clustered Breaches)** |
| **Static Rolling Normal (250d)** | 35 | 2.33% | 4 | Yellow (6) | 19.581 (<0.001) | 6.959 (0.008) | 26.540 (<0.001) | **FAIL Both Coverage & Independence** |
| **RiskMetrics EWMA Normal ($\lambda=0.94$)** | 29 | 1.93% | 0 | Green (4) | 10.369 (0.001) | 1.144 (0.285) | 11.513 (0.003) | **FAIL Coverage (Thin Gaussian Tails)** |
| **$\text{GARCH}(1,1)$ Normal** | 33 | 2.20% | 0 | Yellow (5) | 16.257 (<0.001) | 1.486 (0.223) | 17.743 (<0.001) | **FAIL Coverage (Thin Gaussian Tails)** |
| **$\text{GARCH}(1,1)$ Student-$t$ (FHS)** | 22 | 1.47% | 0 | Green (2) | 2.885 (0.089) | 0.655 (0.418) | 3.540 (0.170) | **PASS (Champion Tier)** |
| **Age-Weighted HS (BRW, $\lambda=0.98$)** | 21 | 1.40% | 0 | Green (2) | 2.156 (0.142) | 0.597 (0.440) | 2.753 (0.252) | **PASS (Champion Tier)** |
| **Hull-White Volatility-Weighted HS** | 18 | 1.20% | 0 | Green (1) | 0.570 (0.450) | 0.438 (0.508) | 1.007 (0.604) | **PASS (Champion Tier — Best Overall)** |

### Phase 4 (Run): Derivatives Risk & Scenario Stress Testing (`04_derivatives_and_stress_testing.ipynb`)
- **Failure of Linear Delta-Normal VaR on Non-Linear Options:** On a **Short ATM Delta-Neutral `SPY` Straddle** ($-6,000$ share equivalent, $\Delta_{\text{net}} = 0.00$), **Linear Delta-Normal VaR erroneously reports $\$0.00$ risk**, completely blind to negative convexity ($\Gamma < 0$) and short vega ($\mathcal{V} < 0$). By contrast, **Quadratic Delta-Gamma-Vega** ($\text{VaR}_{0.99} = \$34,185$) and **Full Revaluation Monte Carlo** ($M = 20,000$, $\text{VaR}_{0.99} = \$31,305$, $\text{ES}_{0.99} = \$38,162$) accurately capture the severe tail exposure.
- **Crisis Replay & 2D Stress Grid:** Hedging the $\$3,000,000$ `SPY` equity sleeve with a 5% OTM Protective Put generates **$+\$370,617$** in hedge gains during the March 2020 COVID-19 replay (cutting worst 1-day crash losses by **$33.04\%$**) and saves **$\$442,489$** ($47.6\%$) under the severe $(-20\%\text{ Spot}, +25\text{ IV pts})$ stress cell.

---

## 3. Quick Start & Reproducibility

```bash
# 1. Activate the project virtual environment
source .venv/bin/activate

# 2. Run the full unit test suite (automatically compiles the C++17 kernel on first run)
pytest tests/ -v

# 3. Inspect or re-run the interactive notebooks in Antigravity IDE or JupyterLab
jupyter lab notebooks/
```
