# Multi-Asset VaR & ES Engine with Backtesting and Derivatives Revaluation
**2-Page Quantitative Model Development, Validation & Interview Walkthrough Note**
**Author:** Mukund Ankit | **Portfolio Mandate:** USD 10,000,000 Multi-Asset Macro Desk | **CRO Limits:** 1d 99% VaR $\le$ $200k, ES $\le$ $280k

---

## 1. Executive Framing & How to Pitch This Project in an Interview
**Why this project exists:** While my core experience at ANZ Bank focuses on Wholesale Credit Risk Modelling (PD/LGD regulatory capital under APS 113 / Basel III) and my IGIDR Master's thesis modeled daily interbank spreads using **VARX and asymmetric $\text{EGARCH}$ volatility models**, I built this production-grade **Python + C++17 Market Risk Engine** to demonstrate hands-on mastery of Trading Book Market Risk (GARP FRM / Basel 2.5 / FRTB).

**The 4-Phase Narrative Arc:**
1. **Phase 1 — Data & Multi-Asset Risk Factors (Equities, Rates, FX, Gold):** Built a point-in-time ($\mathcal{F}_{t-1}$) pipeline across US Equities (`SPY`, `AAPL`), Long-Duration Rates (`TLT`), Foreign Exchange (`FXE` / EURUSD), and Gold (`GLD`) over 2015–2025 ($N=2,797$ trading days). Verified unit-root non-stationarity in price levels ($p > 0.38$) vs. $I(0)$ stationarity in daily returns ($p < 10^{-15}$) via ADF tests, quantified severe leptokurtosis ($K = 9.21 > 3.0$; $>3\sigma$ tail events occur **$6.35\times$ more often** than under a Gaussian bell curve), and diagnosed the 2020 vs. 2022 stock-bond correlation flip ($\hat{\rho}_{\text{SPY,TLT}}$ shifting from $-0.71$ in Q1 2020 to $+0.48$ in 2022).
2. **Phase 2 — VaR/ES Engine & Asymmetric $\text{EGARCH}(1,1)$ Filtered Historical Simulation (FHS):** Implemented Plain Historical Simulation (250d/500d + bootstrap SE), Parametric Variance-Covariance (Normal, Lognormal, MLE Student-$t$, Cornish-Fisher), Cholesky Monte Carlo ($M=10,000$), and **$\text{EGARCH}(1,1)$ Filtered Historical Simulation** (drawing directly on my thesis work with Nelson's 1991 asymmetric log-variance specification) alongside **Hull-White (1998) Volatility-Weighted HS**.
3. **Phase 3 — Out-of-Sample Model Validation & Backtesting (1,500 Days):** Designed and executed an out-of-sample backtesting suite (2020–2025) applying the **Basel Traffic-Light zones**, **Kupiec (1995) POF unconditional coverage ($LR_{UC}$)**, and **Christoffersen (1998) Markov independence & conditional coverage ($LR_{IND}, LR_{CC}$)** tests. Proved *why* static rolling windows and Gaussian models fail and *why* $\text{EGARCH}(1,1)$ FHS and Hull-White VWHS pass.
4. **Phase 4 — Derivatives Revaluation (Options + 5Y Interest-Rate Swap) & Stressed VaR ($\text{sVaR}$):** Priced an `SPY` option book (5% OTM Protective Put & ATM Delta-Neutral Straddle) via a native **C++17 Black-Scholes-Merton kernel** and a **$\$5,000,000$ 5-Year Pay-Fixed USD Interest-Rate Swap (IRS)** via exact discount factors. Demonstrated where Linear Delta-Normal / DV01 and Delta-Gamma approximations break down relative to **Full Revaluation Monte Carlo ($M=20,000$)**, and computed **Basel 2.5 Stressed VaR ($\text{sVaR}_{0.99}$)** across the 2020 COVID and 2022 Rate-Hike crisis windows versus the 2017 normal period.

---

## 2. Core Mathematical Specifications & Why Each Upgrade Was Needed

### Phase 1 & 2: From Static Gaussian VaR to Asymmetric $\text{EGARCH}(1,1)$ Filtered Historical Simulation
* **Portfolio Loss & Coherence:** For portfolio return $r_{p,t} = \mathbf{w}^\top \mathbf{r}_t$ and notional $V_0 = \$10,000,000$, daily dollar loss is $L_t = -V_0 r_{p,t}$. Value-at-Risk is the $\alpha$-quantile $\text{VaR}_\alpha(L) = F_L^{-1}(\alpha)$, while Expected Shortfall averages losses in the worst $(1-\alpha)$ tail: $\text{ES}_\alpha(L) = \mathbb{E}[L_t \mid L_t \ge \text{VaR}_\alpha(L)]$. In `src/var_estimators.py`, we prove analytically via a two-bond credit counterexample ($p = 0.8\%$ default probability) why $\text{VaR}_{0.99}$ violates subadditivity ($\text{VaR}_{0.99}(L_1+L_2) = \$100\text{k} > \$0 + \$0$) while $\text{ES}_{0.99}$ remains coherent ($\$100.64\text{k} \le \$160\text{k}$).
* **Diagnosing the 250-Day "Ghost Effect":** Plain 250-day Historical Simulation weights all 250 observations equally ($1/250$). After the March 2020 COVID crash, 99% Plain HS VaR locks onto a rigid plateau at **$\$343,189$** for exactly 250 trading days—even after markets calm down—before dropping off a cliff on Day 251 (March 2021).
* **Thesis Connection — Asymmetric $\text{EGARCH}(1,1)$ Filtered Historical Simulation (FHS):** Standard $\text{GARCH}(1,1)$ ($\sigma_t^2 = \omega + \alpha_1 \varepsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2$) treats positive and negative return shocks symmetrically because $\varepsilon_{t-1}$ is squared. Drawing on my Master's thesis, `src/volatility_models.py` estimates [Nelson's (1991) Exponential GARCH](https://doi.org/10.2307/2938260) with Student-$t$ innovations ($\hat{\nu} = 7.09$):

$$\ln(\sigma_t^2) = \omega + \alpha_1 \left(|z_{t-1}| - \mathbb{E}|z|\right) + \gamma_1 z_{t-1} + \beta_1 \ln(\sigma_{t-1}^2), \qquad z_t = \frac{r_{p,t} - \mu}{\sigma_t}$$

  where $\hat{\gamma}_1 = -0.0129 < 0$ captures the **asymmetric leverage effect** (negative shocks $z_{t-1} < 0$ amplify log-variance more than positive shocks) and $\hat{\beta}_1 = 0.9830$ captures volatility persistence without requiring non-negativity constraints on parameters. Under **Filtered Historical Simulation (FHS)**, historical returns are de-volatilized into standardized empirical residuals $\hat{z}_{t-i} = (r_{p, t-i} - \hat{\mu})/\hat{\sigma}_{t-i}^{\text{EGARCH}}$ and rescaled by today's 1-step-ahead forecast $\hat{\sigma}_{T \mid T-1}^{\text{EGARCH}}$: $r_{T, i}^{\text{FHS}} = \hat{\mu} + \hat{\sigma}_{T \mid T-1}^{\text{EGARCH}} \hat{z}_{t-i}$. This preserves the empirical fat-tailed, negatively skewed residual distribution while adapting instantaneously to current volatility regimes.

---

## 3. Phase 3: Out-of-Sample Backtesting Evidence (1,500 Trading Days, 2020–2025)
Over $N = 1,500$ out-of-sample days at $\alpha = 99\%$ ($\mathbb{E}[\text{breaches}] = 15.0$), we evaluate the hit sequence $I_t = \mathbb{I}(L_t > \text{VaR}_{0.99, t \mid t-1})$ using three formal regulatory tests:
1. **Basel Traffic-Light Zones (Rolling 250d Window):** Green ($0\text{–}4$ breaches, $+0.00$ capital surcharge), Yellow ($5\text{–}9$ breaches, $+0.40$ to $+0.85$ penalty), Red ($\ge 10$ breaches, $+1.00$ penalty & model disqualification presumption).
2. **Kupiec (1995) POF Unconditional Coverage ($LR_{UC} \sim \chi^2_1$, 5% crit $= 3.841$):** Tests whether the breach rate $\hat{\pi} = x/N$ equals $p = 1\%$.
3. **Christoffersen (1998) Markov Independence ($LR_{IND} \sim \chi^2_1$) & Conditional Coverage ($LR_{CC} = LR_{UC} + LR_{IND} \sim \chi^2_2$, 5% crit $= 5.991$):** Tests whether breaches cluster on consecutive days ($n_{11} > 0$, i.e., $\pi_{11} \ne \pi_{01}$).

| Risk Estimator | Total Breaches ($x$) | Breach Rate ($\hat{\pi}$) | Clustered Pairs ($n_{11}$) | Latest / Peak 250d Basel Zone | Kupiec $LR_{UC}$ ($p$-val) | Christoffersen $LR_{IND}$ ($p$-val) | Joint $LR_{CC}$ ($p$-val) | Model Validation Verdict & Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Plain Historical Sim (250d)** | 22 | 1.47% | **2** | Green (4) / **Red (10)** | 2.885 (0.089) | **4.209 (0.040)** | **7.093 (0.029)** | **FAIL Independence:** Ghost plateau hides average rate, but slow crisis onset causes consecutive clustered breaches ($n_{11}=2$). |
| **Static Rolling Normal (250d)** | 35 | 2.33% | **4** | Yellow (6) / **Red (17)** | **19.581 (<0.001)** | **6.959 (0.008)** | **26.540 (<0.001)** | **FAIL Both:** Static window + thin Gaussian quantile ($2.326\sigma$) fails coverage and clusters in crises. |
| **RiskMetrics EWMA Normal ($\lambda=0.94$)** | 29 | 1.93% | 0 | Green (4) / Yellow (9) | **10.369 (0.001)** | 1.144 (0.285) | **11.513 (0.003)** | **FAIL Coverage:** Fast variance update eliminates clustering ($n_{11}=0$), but Gaussian tail is too thin. |
| **$\text{GARCH}(1,1)$ Normal** | 33 | 2.20% | 0 | Yellow (5) / Yellow (9) | **16.257 (<0.001)** | 1.486 (0.223) | **17.743 (<0.001)** | **FAIL Coverage:** Zero clustering ($n_{11}=0$), but Gaussian innovations underestimate leptokurtic tails. |
| **$\text{GARCH}(1,1)$ Student-$t$** | **22** | **1.47%** | **0** | **Green (2)** / Yellow (6) | **2.885 (0.089)** | **0.655 (0.418)** | **3.540 (0.170)** | **PASS (Champion Tier):** Dynamic variance + fat-tailed quantiles eliminate clustering and pass coverage. |
| **$\text{EGARCH}(1,1)$ Filtered HS (FHS)** | **18** | **1.20%** | **0** | **Green (1)** / Yellow (6) | **0.570 (0.450)** | **0.438 (0.508)** | **1.007 (0.604)** | **PASS (Champion — Tied Best):** Asymmetric log-vol + empirical FHS residuals achieve 18 breaches and $n_{11}=0$. |
| **Hull-White Volatility-Weighted HS** | **18** | **1.20%** | **0** | **Green (1)** / Yellow (6) | **0.570 (0.450)** | **0.438 (0.508)** | **1.007 (0.604)** | **PASS (Champion — Tied Best):** Rescales returns by $\hat{\sigma}_T/\hat{\sigma}_i$; exceeds max window loss on 19 spike days. |

---

## 4. Phase 4: Derivatives Revaluation (Options + 5Y IRS) & Basel 2.5 Stressed VaR ($\text{sVaR}$)

### A. Where Delta-Normal / DV01 and Delta-Gamma Break Down vs. Full Revaluation ($M = 20,000$ Scenarios)
* **Option Straddle Breakdown ($\Delta_{\text{net}} = 0$):** On a **Short ATM Delta-Neutral `SPY` Straddle** ($-6,000$ share equivalent), Call Delta ($+0.499$) and Put Delta ($-0.499$) cancel out to $\Delta_{\text{net}} = 0.00$. Consequently, **Linear Delta-Normal VaR ($\Delta V \approx \Delta \cdot \Delta S$) reports exactly $\$0.00$ of risk**, completely blind to negative Gamma ($\Gamma < 0$) and short Vega ($\mathcal{V} < 0$). Repricing all 20,000 correlated spot-volatility scenarios through our C++17 Black-Scholes kernel reveals **Full Revaluation 99% VaR $= \$31,305.43$** and **99% ES $= \$38,162.12$** (closely approximated by Quadratic Delta-Gamma-Vega at $\$34,184.63$).
* **Hedged `SPY` Protective Put Convexity Cushion:** On the **$\$3\text{M}$ `SPY` + 5% OTM Protective Put** book, Linear Delta-Normal VaR **overstates** 99% VaR by **$+\$9,304.85$ ($+16.6\%$)** ($\$65,198$ vs. $\$55,893$ Full Revaluation MC) because it ignores the positive Gamma and Vega cushion as `SPY` drops toward the put strike.
* **5-Year $\$5,000,000$ Pay-Fixed Interest-Rate Swap (IRS) Breakdown:** Priced via exact semi-annual discount factors $V_{\text{swap}}(y) = N - \sum_{j=1}^{10} \frac{c N / 2}{(1 + y/2)^j} - \frac{N}{(1 + y/2)^{10}}$ ($\text{DV01} = \$2,233.36/\text{bp}$), Linear DV01 ($\Delta V \approx \text{DV01} \cdot \Delta y$) works well for small daily fluctuations ($\sigma_y = 7.8\text{ bps/day}$, where 99% DV01 VaR is $\$40,767$ vs. $\$40,962$ Full Revaluation). However, under a stressed **$-300\text{ bps}$ parallel rate cut**, discount-curve convexity causes **Linear DV01 to understate the swap loss by $\$55,649.96$** ($-\$670,183$ linear vs. $-\$725,833$ exact Full Revaluation), whereas **Quadratic DV01 + Convexity** reduces the error by $94.3\%$ to $\$3,184.03$.

### B. Basel 2.5 Stressed VaR ($\text{sVaR}_{0.99}$) vs. Normal-Period VaR ($10M Macro Desk)
Under Basel 2.5, banks must compute **Stressed VaR ($\text{sVaR}$)** over a continuous 1-year historical period of significant financial stress alongside current VaR:

| Calibration Window | Mean VIX | Annualized Portfolio Vol | Parametric Normal 99% VaR | Historical Sim 99% VaR / $\text{sVaR}$ | Historical Sim 99% ES / $\text{sES}$ | $\text{sVaR} / \text{Normal VaR}$ Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal Period (2017 Low-Vol Calm)** | 11.36 | 5.66% | $80,684.01 | **$97,075.31** | $115,804.58 | **1.00x (Baseline)** |
| **Current Window (Latest 250d, 2025)** | 17.09 | 10.24% | $151,353.73 | **$158,473.92** | $173,658.09 | **1.63x** |
| **Stressed Window A (2020 COVID-19 Crash)** | 26.47 | 15.91% | $230,331.08 | **$338,666.16** | **$442,361.32** | **3.49x Normal VaR** |
| **Stressed Window B (2022 Rate-Hike Bear)** | 22.86 | 16.82% | $251,737.38 | **$261,257.27** | **$287,551.37** | **2.69x Normal VaR** |

**Key Takeaway:** Calibrating the $\$10\text{M}$ desk over the 2020 COVID stress window yields a 1-Day 99% **$\text{sVaR}$ of $\$338,666$ ($3.49\times$ the 2017 Normal-Period VaR of $\$97,075$)**, and calibrating over the 2022 Rate-Hike stock-bond selloff yields **$\$261,257$ ($2.69\times$ Normal VaR)**—demonstrating why Basel 2.5 requires capital buffers anchored to stressed regimes ($k \cdot \text{VaR}_t + k_s \cdot \text{sVaR}_t$) to prevent pro-cyclical capital depletion during calm years.
