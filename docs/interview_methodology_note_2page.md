# Multi-Asset VaR & ES Engine with Backtesting, Derivatives Revaluation & IMA Capital (RNIV / IRC / IRRBB / FRTB)
**2-Page Quantitative Model Development, Validation & Interview Walkthrough Note**
**Author:** Mukund Ankit | **Portfolio Mandate:** USD 10,000,000 Multi-Asset Macro Desk | **CRO Limits:** 1d 99% VaR $\le$ $200k, ES $\le$ $280k

---

## 1. Executive Framing & How to Pitch This Project in an Interview
**Why this project exists:** At ANZ Bank, I develop Wholesale Credit Risk regulatory capital models (PD/LGD under APS 113 / Basel III), and my IGIDR Master's thesis modeled daily interbank interest-rate spreads using **VARX and asymmetric $\text{EGARCH}$ volatility models**. I built this production-grade **Python + C++17 Market Risk & Regulatory Capital Engine** to demonstrate end-to-end mastery of **Internal Models Approach (IMA) Market Risk** across **VaR/ES, RNIV, IRC, Economic Capital, IRRBB, and the FRTB transition**.

**Complete Coverage of the 5-Phase IMA Mandate:**
1. **Five VaR Risk-Factor Pillars (`Rates, FX, Credit, Inflation, Equity`):** Point-in-time ($\mathcal{F}_{t-1}$) pipeline over 2015–2025 ($N=2,797$ days) covering **Equity (`SPY`, `AAPL`, `TSLA`)**, **Rates (`TLT` + 5Y IRS)**, **FX (`FXE` / EURUSD)**, **Credit (`LQD` + 6-Obligor Credit Book)**, and **Inflation (`TIP` + `GLD`)**. Euler Component VaR attributes the $\$10\text{M}$ 5-pillar book's $\$119,246$ 1-Day 99% VaR across Equity ($\$82.2\text{k}$, $68.9\%$), Credit ($\$12.1\text{k}$, $10.1\%$), Rates ($\$10.8\text{k}$, $9.1\%$), FX ($\$7.8\text{k}$, $6.6\%$), and Inflation ($\$6.3\text{k}$, $5.3\%$), saving $\$73,286$ in cross-asset diversification.
2. **VaR/ES Engine & Asymmetric $\text{EGARCH}(1,1)$ Filtered Historical Simulation (FHS):** Plain HS (250d/500d + bootstrap SE), Parametric (Normal, Lognormal, MLE Student-$t$, Cornish-Fisher), Cholesky Monte Carlo ($M=10,000$), **$\text{EGARCH}(1,1)$ FHS** (leveraging Nelson's 1991 asymmetric log-variance specification $\ln \sigma_t^2 = \omega + \alpha_1(|z_{t-1}| - \mathbb{E}|z|) + \gamma_1 z_{t-1} + \beta_1 \ln \sigma_{t-1}^2$ with $\hat{\gamma}_1 = -0.0129 < 0$ from my thesis), and **Hull-White (1998) Volatility-Weighted HS**.
3. **Out-of-Sample Model Validation & Backtesting (1,500 Days):** Basel Traffic-Light zones, **Kupiec (1995) POF ($LR_{UC}$)**, **Christoffersen (1998) Markov Independence ($LR_{IND}, LR_{CC}$)**, and **Acerbi & Székely (2014) $Z_2$ ES Backtesting**.
4. **Derivatives Revaluation (`SPY` Options + 5Y Swap) & Stressed VaR ($\text{sVaR}$):** C++17 Black-Scholes-Merton kernel + 5-Year $\$5\text{M}$ Pay-Fixed USD Interest-Rate Swap (`IRS`); compares Full Revaluation Monte Carlo ($M=20,000$) against Linear Delta-Normal / DV01 and Quadratic Delta-Gamma-Vega / Convexity, plus **Basel 2.5 Stressed VaR ($\text{sVaR}_{0.99}$)** over 2020 COVID ($3.49\times$ Normal VaR) and 2022 Rate-Hike ($2.69\times$ Normal VaR) windows.
5. **IMA Capital Pillars — RNIV, IRC (1Y 99.9%), Economic Capital, IRRBB & FRTB Transition (`src/regulatory_capital_frtb.py`):** Full implementation of **RNIV** basis add-ons, **1-Year 99.9% Merton Copula IRC & Economic Capital**, **Basel d368 IRRBB ($\Delta\text{EVE}$ & $\Delta\text{NII}$ across 6 supervisory shocks)**, and **FRTB Liquidity-Horizon-Adjusted $97.5\%$ ES ($LH \in \{10, 20, 40\}\text{d}$)**.

---

## 2. Out-of-Sample VaR Backtesting (1,500 Days) & Derivatives / Stressed VaR Evidence

### A. 1,500-Day Champion–Challenger Backtest Table ($\alpha = 99\%$, Expected Breaches $= 15.0$)
| Risk Estimator | Breaches ($x$) | Rate ($\hat{\pi}$) | Clustered ($n_{11}$) | Latest / Peak Basel Zone | Kupiec $LR_{UC}$ ($p$) | Christoffersen $LR_{IND}$ ($p$) | Joint $LR_{CC}$ ($p$) | Validation Verdict & Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Plain Historical Sim (250d)** | 22 | 1.47% | **2** | Green (4) / **Red (10)** | 2.885 (0.089) | **4.209 (0.040)** | **7.093 (0.029)** | **FAIL Independence:** Ghost plateau hides average rate, but slow crisis onset causes consecutive clustered breaches ($n_{11}=2$). |
| **Static Rolling Normal (250d)** | 35 | 2.33% | **4** | Yellow (6) / **Red (17)** | **19.581 (<0.001)** | **6.959 (0.008)** | **26.540 (<0.001)** | **FAIL Both:** Static window + thin Gaussian quantile ($2.326\sigma$) fails coverage and clusters in crises. |
| **RiskMetrics EWMA Normal ($\lambda=0.94$)** | 29 | 1.93% | 0 | Green (4) / Yellow (9) | **10.369 (0.001)** | 1.144 (0.285) | **11.513 (0.003)** | **FAIL Coverage:** Fast variance update eliminates clustering ($n_{11}=0$), but Gaussian tail is too thin. |
| **$\text{GARCH}(1,1)$ Student-$t$** | **22** | **1.47%** | **0** | **Green (2)** / Yellow (6) | **2.885 (0.089)** | **0.655 (0.418)** | **3.540 (0.170)** | **PASS (Champion Tier):** Dynamic variance + fat-tailed quantiles eliminate clustering and pass coverage. |
| **$\text{EGARCH}(1,1)$ Filtered HS (FHS)** | **18** | **1.20%** | **0** | **Green (1)** / Yellow (6) | **0.570 (0.450)** | **0.438 (0.508)** | **1.007 (0.604)** | **PASS (Champion — Tied Best):** Asymmetric log-vol + empirical FHS residuals achieve 18 breaches and $n_{11}=0$. |
| **Hull-White Volatility-Weighted HS** | **18** | **1.20%** | **0** | **Green (1)** / Yellow (6) | **0.570 (0.450)** | **0.438 (0.508)** | **1.007 (0.604)** | **PASS (Champion — Tied Best):** Rescales returns by $\hat{\sigma}_T/\hat{\sigma}_i$; exceeds max window loss on 19 spike days. |

### B. Where Linear Delta-Normal / DV01 Approximations Break Down vs. Full Revaluation ($M = 20,000$)
* **Short ATM Delta-Neutral `SPY` Straddle ($\Delta_{\text{net}} = 0$):** Call Delta ($+0.499$) and Put Delta ($-0.499$) cancel out, so **Linear Delta-Normal VaR reports $\$0.00$ risk**, completely blind to negative Gamma ($\Gamma < 0$) and short Vega ($\mathcal{V} < 0$). **C++17 Full Revaluation Monte Carlo** reveals **1-Day 99% VaR $= \$31,305.43$** and **99% ES $= \$38,162.12$** (Quadratic Delta-Gamma-Vega $= \$34,184.63$).
* **5-Year $\$5\text{M}$ Pay-Fixed Interest-Rate Swap ($\text{DV01} = \$2,233.36/\text{bp}$):** Under a stressed **$-300\text{ bps}$ parallel rate shock**, discount-curve convexity causes **Linear DV01 to understate the swap loss by $\$55,649.96$** ($-\$670,183$ linear vs. $-\$725,833$ Full Revaluation), whereas **Quadratic DV01 + Convexity** reduces the error by **$94.3\%$** to $\$3,184.03$.
* **Basel 2.5 Stressed VaR ($\text{sVaR}_{0.99}$):** Calibrating over the **2020 COVID-19 stress window** yields $\text{sVaR}_{0.99} = \$338,666$ (**$3.49\times$** the 2017 Normal-Period VaR of $\$97,075$), and calibrating over the **2022 Rate-Hike bear market** yields $\text{sVaR}_{0.99} = \$261,257$ (**$2.69\times$** Normal VaR).

---

## 3. IMA Capital Pillars: RNIV, IRC (99.9%), Economic Capital, IRRBB & FRTB Transition

### A. RNIV (Risks Not In VaR) Capital Add-On (`compute_rniv_capital_addons`)
Quantifies 99% tail losses from unmodeled basis risks omitted by standard VaR engines across Current (2025) and Stressed (2020) windows:
* **Channel 1 — Volatility Smile/Skew Surface Basis (`SPY` Options):** $\$4,001$ (Current 1d 99% RNIV) | $\$12,148$ (Stressed RNIV) $\implies \$48,446$ Capital.
* **Channel 2 — Yield-Curve Tenor & Convexity Basis (5Y IRS vs. 20Y Proxy):** $\$13,081$ (Current) | $\$41,711$ (Stressed) $\implies \$164,375$ Capital.
* **Channel 3 — Credit Index-to-Single-Name Dispersion Basis (`LQD` Sleeve):** $\$7,270$ (Current) | $\$20,841$ (Stressed) $\implies \$84,331$ Capital.
* **Total RNIV Capital Add-On ($3 \times (\text{RNIV} + \text{sRNIV})$):** **$\$297,152.32$** (Current 1d RNIV $= \$24,352$, Stressed 1d sRNIV $= \$74,699$).

### B. IRC (Incremental Risk Charge — 1Y 99.9%) & Enterprise Economic Capital (`simulate_irc_and_economic_capital`)
Directly leverages credit risk modeling (Merton 1974 Single-Factor Gaussian Copula $Z_i = \sqrt{\rho_i}M_{\text{sys}} + \sqrt{1-\rho_i}\varepsilon_i$, $M=50,000$ trials) across a $\$3,000,000$ 6-obligor corporate bond portfolio (ratings `AA` to `BB`, $\rho_i \in [0.18, 0.24]$) over a 1-year constant-level-of-risk horizon at **99.9% confidence**:
* **1-Year 99.9% Default-Only Charge (FRTB DRC Equivalent):** **$\$460,000.00$** (Expected 1Y Credit Loss $= \$26,768.82$).
* **1-Year 99.9% Basel 2.5 IRC (Default + Rating Migration Spread Widening):** **$\$503,000.00$** ($99.9\%$ IRC ES $= \$577,305.69$; credit rating downgrades contribute **$8.55\%$** of total IRC).
* **1-Year 99.9% Enterprise Economic Capital (Market Risk + Credit IRC, $\rho_{\text{Mkt,Cred}} = 0.35$):** Combines Standalone 1Y 99.9% Market Risk EC ($\$1,738,782$) and Standalone Credit UL ($\$476,231$) into a **Diversified Total Economic Capital of $\$1,814,058.64$** (saving **$\$400,954.44$** in inter-risk diversification).

### C. IRRBB (Interest Rate Risk in the Banking Book — BCBS d368) (`evaluate_irrbb_supervisory_shocks`)
Evaluates a $\$100\text{M}$ Banking Book (sticky deposits, floating SOFR loans, fixed mortgages across 8 repricing buckets `1M`–`20Y`, Tier-1 Capital $= \$15\text{M}$) across all **6 Basel Supervisory Shocks**. Under **Parallel Up ($+200\text{ bps}$)**, Unhedged $\Delta\text{EVE}$ is **$-\$3,088,346$ ($-20.59\%$ of Tier-1 Capital, breaching the Basel $-15\%$ Outlier limit)** and 1Y $\Delta\text{NII}$ is **$+\$111,336$**. Adding a **$\$10\text{M}$ 5Y Pay-Fixed ALCO Swap Hedge** generates **$+\$848,683$** in hedge gain, bringing **Hedged $\Delta\text{EVE}$ to $-\$2,239,662$ ($-14.93\%$ of Tier-1 Capital)**—passing the Basel 15% outlier test across all 6 supervisory scenarios.

### D. Transition to FRTB (BCBS d457) Liquidity-Horizon Cascade & Total Capital Bridge
Scales 97.5% Expected Shortfall across statutory Basel liquidity horizons ($LH = 10\text{d}$ for Equity/Sovereign Rates, $20\text{d}$ for FX, $40\text{d}$ for IG Credit Spreads & Inflation):
$$\text{ES}_{\text{FRTB}} = \sqrt{ \text{ES}_{10\text{d}}^2(P) + \sum_{j \ge 2} \left( \text{ES}_{10\text{d}}(P, j)\sqrt{\frac{LH_j - LH_{j-1}}{10}} \right)^2 } = \$427,390.53 \text{ (Current)} \;\;|\;\; \$962,410.19 \text{ (Stressed)}$$
Adding the 20d/40d liquidity-horizon cascade raises 97.5% ES by **$+10.52\%$** over unadjusted 10-day ES ($\$386,715$). Across the full desk, transitioning from **Basel 2.5 IMA Capital ($\$4,318,744$)** to **FRTB IMA Capital ($\$2,052,191$)** eliminates the double-counting of normal + stressed VaR and aligns regulatory capital closely with **1-Year 99.9% Internal Economic Capital ($\$1,814,059$)**.
