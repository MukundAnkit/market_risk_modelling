# Multi-Asset VaR & ES Engine with Backtesting, Derivatives Revaluation & IMA Regulatory Capital (`RNIV`, `IRC`, `IRRBB`, `FRTB`)

An institutional-grade **Market Risk Modelling, Backtesting, Derivatives Revaluation, and IMA Regulatory Capital Framework** governing a **$10,000,000 Multi-Asset Trading Desk** across all five regulatory risk factor pillars (**Equities**, **Interest Rates**, **Foreign Exchange (`FX`)**, **Credit Spread**, and **Inflation / Commodities**), paired with a **$100,000,000 Banking Book (`IRRBB`)** and a **compiled C++17 quantitative pricing kernel**.

---

## 1. Key Deliverables & Documentation

* **2-Page Interview Methodology Note (PDF & Markdown):**
  * [`docs/interview_methodology_note_2page.pdf`](docs/interview_methodology_note_2page.pdf) | [`docs/interview_methodology_note_2page.md`](docs/interview_methodology_note_2page.md)
  * Concise 2-page executive note walking through the 5-pillar data design, VaR/ES + asymmetric $\text{EGARCH}(1,1)$ Filtered Historical Simulation (FHS), 1,500-day Kupiec/Christoffersen backtesting, options & 5Y Interest-Rate Swap revaluation, Basel 2.5 Stressed VaR ($\text{sVaR}$), **RNIV**, **1-Year 99.9% Merton Copula IRC & Economic Capital**, **6-Shock Basel IRRBB ($\Delta\text{EVE}$ & $\Delta\text{NII}$)**, and **FRTB Liquidity-Adjusted 97.5% ES**.
* **Full Model Development & Validation Technical Whitepaper (PDF, Markdown & LaTeX):**
  * [`docs/market_risk_modelling_technical_whitepaper.pdf`](docs/market_risk_modelling_technical_whitepaper.pdf) | [`docs/model_development_technical_report.md`](docs/model_development_technical_report.md) | [`docs/market_risk_modelling_technical_whitepaper.tex`](docs/market_risk_modelling_technical_whitepaper.tex)
* **End-to-End Interactive Showcase Notebooks (`notebooks/`):**
  * [`notebooks/05_end_to_end_var_es_backtesting_derivatives_showcase.ipynb`](notebooks/05_end_to_end_var_es_backtesting_derivatives_showcase.ipynb) — Runs the 4-phase core workflow end-to-end.
  * [`notebooks/06_ima_regulatory_capital_rniv_irc_irrbb_frtb.ipynb`](notebooks/06_ima_regulatory_capital_rniv_irc_irrbb_frtb.ipynb) — Runs the **5-Pillar IMA Attribution, RNIV, 99.9% IRC & Economic Capital, 6-Shock IRRBB ($\Delta\text{EVE}/\Delta\text{NII}$), and FRTB Liquidity-Adjusted ES Capital Bridge**.
  * Deep-Dive Phase Notebooks:
    * [`notebooks/00_data_hygiene_and_eda.ipynb`](notebooks/00_data_hygiene_and_eda.ipynb) — Phase 0: ADF stationarity, leptokurtosis ($K=9.21$), Q-Q plots, and the 2020 vs. 2022 `SPY`–`TLT` correlation flip.
    * [`notebooks/01_baseline_var_and_es.ipynb`](notebooks/01_baseline_var_and_es.ipynb) — Phase 1: Plain HS, Normal, Lognormal, MLE Student-$t$, Cornish-Fisher, Cholesky MC, subadditivity counterexample, and pre-trade `TSLA` Marginal VaR sizing.
    * [`notebooks/02_volatility_regimes_and_ewma_garch.ipynb`](notebooks/02_volatility_regimes_and_ewma_garch.ipynb) — Phase 2: 250-day COVID-19 "Ghost Effect" diagnosis, RiskMetrics EWMA, $\text{GARCH}(1,1)$, **Asymmetric $\text{EGARCH}(1,1)$ Filtered Historical Simulation (FHS)**, Age-Weighted HS, and Hull-White VWHS.
    * [`notebooks/03_model_backtesting_and_validation.ipynb`](notebooks/03_model_backtesting_and_validation.ipynb) — Phase 3: 1,500-day out-of-sample Basel Traffic-Light zones, Kupiec POF ($LR_{UC}$), and Christoffersen Markov Independence ($LR_{IND}, LR_{CC}$).
    * [`notebooks/04_derivatives_and_stress_testing.ipynb`](notebooks/04_derivatives_and_stress_testing.ipynb) — Phase 4: `SPY` option chain IV skew, **5-Year $5M Pay-Fixed USD Interest-Rate Swap**, Delta-Normal / DV01 vs. Delta-Gamma-Vega / Convexity vs. C++17 Full Revaluation MC, and **Basel 2.5 Stressed VaR ($\text{sVaR}_{0.99}$)**.

---

## 2. Summary of Key Empirical Results

### A. 1,500-Day Out-of-Sample Model Validation (99% 1-Day VaR, Expected Breaches = 15.0)

| Risk Estimator | Breaches ($x$) | Rate ($\hat{\pi}$) | Clustered Pairs ($n_{11}$) | Latest 250d Zone | Kupiec $LR_{UC}$ ($p$) | Christoffersen $LR_{IND}$ ($p$) | Joint $LR_{CC}$ ($p$) | Validation Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Plain Historical Sim (250d)** | 22 | 1.467% | **2** | Green (4) / Peak Red (10) | 2.885 (0.089) | **4.209 (0.040)** | **7.093 (0.029)** | **FAIL Independence** |
| **Static Rolling Normal (250d)** | 35 | 2.333% | **4** | Yellow (6) / Peak Red (17) | **19.581 (0.000)** | **6.959 (0.008)** | **26.540 (0.000)** | **FAIL Coverage & Ind.** |
| **RiskMetrics EWMA Normal** | 29 | 1.933% | 0 | Green (4) | **10.369 (0.001)** | 1.144 (0.285) | **11.513 (0.003)** | **FAIL Coverage** |
| **$\text{GARCH}(1,1)$ Student-$t$** | **22** | **1.467%** | **0** | **Green (2)** | **2.885 (0.089)** | **0.655 (0.418)** | **3.540 (0.170)** | **PASS (Champion)** |
| **$\text{EGARCH}(1,1)$ Filtered HS (FHS)** | **18** | **1.200%** | **0** | **Green (1)** | **0.570 (0.450)** | **0.438 (0.508)** | **1.007 (0.604)** | **PASS (Tied #1 Champion)** |
| **Hull-White Volatility-Weighted HS** | **18** | **1.200%** | **0** | **Green (1)** | **0.570 (0.450)** | **0.438 (0.508)** | **1.007 (0.604)** | **PASS (Tied #1 Champion)** |

### B. IMA Regulatory Capital Bridge (`VaR + sVaR + IRC + RNIV` vs. `FRTB IMCC + DRC + RRAO`)

| Regulatory Capital Component | Basel 2.5 IMA Capital (USD) | FRTB IMA (BCBS d457) Counterpart | FRTB IMA Capital (USD) |
| :--- | :---: | :--- | :---: |
| **1. General & Specific Market Risk** | **$3,518,591.29** ($m_c\sqrt{10}\,\text{VaR}_{99\%} + m_s\sqrt{10}\,\text{sVaR}_{99\%}$) | **IMCC** ($1.5 \times \text{ES}_{97.5\%, LH}^{\text{Stressed}}$ across $LH \in \{10,20,40\}\text{d}$) | **$1,443,615.28** |
| **2. Credit Migration & Default Risk** | **$503,000.00** (**IRC**: 1Y 99.9% Default + Rating Migration) | **DRC** (1Y 99.9% Default-Only Copula) | **$460,000.00** |
| **3. Unmodeled / Residual Basis Risk** | **$297,152.32** (**RNIV**: Skew + Curve Tenor + Credit Basis) | **RRAO + NMRF** (Residual Risk Add-On) | **$148,576.16** |
| **Total IMA Pillar-1 Capital** | **$4,318,743.60** | **Total FRTB IMA Pillar-1 Capital** | **$2,052,191.44** |
| **Enterprise Economic Capital (1Y 99.9%)** | **$1,814,058.64** (Diversified Joint Market + Credit Tail Loss; saves **$400,954.44** vs. standalone sum) | **IRRBB Hedged Worst $\Delta\text{EVE}$ (+200 bps)** | **-$2,239,662.19** (**-14.93% Tier-1, PASS** vs. -20.59% Unhedged) |

---

## 3. Repository Structure & Quickstart

```text
market_risk_modelling/
├── config/
│   └── portfolio.yaml                         # 5-pillar IMA weights, IRC obligors, IRRBB balance sheet & FRTB horizons
├── src/
│   ├── cpp/
│   │   ├── quant_core.cpp                     # Native C++17 BSM pricing, Greeks, IV solver, MC & EWMA
│   │   └── libquant_core.dylib                # Compiled shared library (-O3 -std=c++17)
│   ├── cpp_kernel.py                          # Python ctypes bridge to C++17 shared library
│   ├── data_loader.py                         # Multi-asset data pipeline (SPY, AAPL, TLT, FXE, LQD, TIP, GLD, TSLA, VIX)
│   ├── eda_diagnostics.py                     # Phase 0: ADF stationarity, JB tests, moments & rolling correlation
│   ├── var_estimators.py                      # Phase 1: HS, Normal, Lognormal, Student-t, Cornish-Fisher, Cholesky MC
│   ├── volatility_models.py                   # Phase 2: EWMA, GARCH(1,1), Asymmetric EGARCH(1,1) FHS, BRW & Hull-White VWHS
│   ├── backtesting.py                         # Phase 3: Basel Traffic Light, Kupiec POF & Christoffersen Markov tests
│   ├── derivatives_stress.py                  # Phase 4: SPY Options, 5Y Interest-Rate Swap, Delta-Gamma & Stressed VaR
│   └── regulatory_capital_frtb.py             # Phase 5: 5-Pillar Attribution, RNIV, 99.9% IRC & EC, IRRBB & FRTB ES
├── notebooks/
│   ├── 00_data_hygiene_and_eda.ipynb          # Phase 0 showcase notebook
│   ├── 01_baseline_var_and_es.ipynb           # Phase 1 showcase notebook
│   ├── 02_volatility_regimes_and_ewma_garch.ipynb # Phase 2 showcase notebook
│   ├── 03_model_backtesting_and_validation.ipynb  # Phase 3 showcase notebook
│   ├── 04_derivatives_and_stress_testing.ipynb    # Phase 4 showcase notebook
│   ├── 05_end_to_end_var_es_backtesting_derivatives_showcase.ipynb # End-to-End Master Notebook
│   └── 06_ima_regulatory_capital_rniv_irc_irrbb_frtb.ipynb         # Phase 5 IMA, RNIV, IRC, IRRBB & FRTB Notebook
├── docs/
│   ├── interview_methodology_note_2page.pdf   # 2-Page Interview Methodology Note (PDF)
│   ├── interview_methodology_note_2page.md    # 2-Page Interview Methodology Note (Markdown)
│   ├── market_risk_modelling_technical_whitepaper.pdf # Full Technical Whitepaper (PDF)
│   ├── model_development_technical_report.md  # Full Technical Whitepaper (Markdown)
│   └── figures/                               # 14 high-DPI publication figures
└── tests/
    └── test_risk_engine.py                    # Automated pytest verification suite (8 comprehensive tests)
```

```bash
# Run the full unit test suite (including C++17 kernel, EGARCH FHS, 5Y IRS, RNIV, IRC, IRRBB & FRTB checks)
.venv/bin/pytest tests/test_risk_engine.py -v
```
