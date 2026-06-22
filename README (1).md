# Bayesian European Equity Factor Model

A cross-sectional equity selection model for European public equities, estimated using Bayesian logistic regression via Metropolis-Hastings MCMC. The model predicts the probability that a given stock will outperform the STOXX Europe 600 benchmark over the following quarter, conditional on a set of sector-neutral fundamental and price-based factors.

This repository contains the full implementation, the underlying research methodology, and a complete account of the iterative development process, including approaches that were tested and discarded.

---

## Overview

Predicting individual stock returns directly is difficult given the low signal-to-noise ratio characteristic of equity markets. This project instead frames the prediction task as a binary classification problem: rather than estimating the magnitude of a stock's future return, the model estimates the probability that the stock will outperform a broad market benchmark over a fixed horizon. This formulation is more robust to outliers and aligns with recent evidence in the empirical asset pricing literature suggesting that classification-based approaches can match or exceed continuous regression methods for cross-sectional equity selection (Wolff, 2024; Jiang et al., 2025).

The model is estimated within a Bayesian framework rather than by maximum likelihood. This choice allows the incorporation of prior beliefs grounded in decades of empirical asset pricing research, produces a full posterior distribution over each factor's coefficient rather than a single point estimate, and yields a posterior predictive probability that can be used directly for position sizing rather than requiring an arbitrary classification threshold.

Over the evaluation period (2016 to 2024), the production model (referred to throughout as V3) generated approximately 34 percentage points of cumulative outperformance against the STOXX 600 benchmark, with a quarterly hit rate of 62.9 percent across 26 walk-forward validation periods.

A subsequent investigation, documented in full in the Data Integrity Investigation section below, found that this result depends on a fundamental dataset that is not point-in-time, and that the coefficient sign for at least one factor is not stable across executions performed on different dates. Readers are encouraged to read that section before drawing conclusions from the headline figures.

---

## Methodology

### Problem formulation

For each stock i in quarter q, the binary outcome variable is defined as

  y(i,q) = 1 if r(i,q) > r(benchmark,q), else 0

where r(i,q) is the realised return of stock i over quarter q and r(benchmark,q) is the contemporaneous return of the iShares STOXX Europe 600 UCITS ETF. The model estimates P(y = 1 | x) as a logistic function of a vector of factors x, observed strictly as of the beginning of the quarter being predicted.

### Factors

Five factors are used, each grounded in a distinct stream of the empirical asset pricing literature.

Earnings yield, computed as EBITDA divided by enterprise value, captures the value premium documented since Fama and French (1993) and operationalised in this model following the earnings-yield specification used by Greenblatt (2010).

Return on equity captures the profitability premium identified by Novy-Marx (2013) and incorporated into the Fama-French five-factor model as the RMW factor.

Six-month price momentum, computed with a one-month skip to avoid contamination from short-term reversal, follows the specification of Jegadeesh and Titman (1993).

Realised volatility, computed over a trailing twelve-month window, is included to capture the low-volatility anomaly documented by Ang, Hodrick, Xing and Zhang (2006).

One-month reversal captures the short-horizon mean-reversion effect documented by Jegadeesh (1990).

The two fundamental factors, earnings yield and return on equity, are sector-neutralised prior to estimation: each stock's factor value is expressed as a z-score relative to its eleven-sector peer group rather than the full cross-section. This addresses a specific failure mode identified during development, in which factors computed across the full universe were found to reflect persistent sector-level valuation differences rather than genuine within-sector mispricing. A detailed account of this finding is provided in the development log.

### Prior specification

Coefficients are assigned independent Cauchy priors following the weakly informative default prior recommended by Gelman, Jakulin, Pittau and Su (2008) for logistic regression models. The prior location for each coefficient is shifted away from zero in the direction implied by the relevant empirical asset pricing result; the prior scale is held at 2.5 for all factor coefficients and 10 for the intercept, consistent with Gelman et al.'s recommendation for standardised predictors.

### Estimation

The posterior distribution is sampled using a random-walk Metropolis-Hastings algorithm. The chain is initialised at the maximum likelihood estimate, and the proposal covariance is set following the optimal scaling result of Roberts, Gelman and Gilks (1997), which specifies a proposal covariance of (2.38 squared divided by the number of parameters) multiplied by the observed Fisher information at the maximum likelihood estimate. Each walk-forward fold is estimated with 15,000 iterations, of which the first 3,000 are discarded as burn-in.

### Validation

The model is validated using an expanding-window walk-forward procedure rather than k-fold cross-validation. This distinction is material: financial returns exhibit serial dependence and regime-dependent behaviour that violate the independence assumption underlying k-fold cross-validation, and naive cross-validation on financial panel data risks introducing lookahead bias. Under the walk-forward protocol, the model is trained at each step using only data available prior to the quarter being predicted, and the training window expands by one quarter at each step. Scaling constants used to standardise predictors are computed exclusively on the training fold and applied to the validation fold, never the reverse.

---

## Data

The investable universe comprises 59 constituents of the STOXX Europe 600 index, drawn from the IBEX 35, FTSE 100, DAX 40 and CAC 40, selected to provide liquid, large-capitalisation coverage across the four principal continental European and UK equity markets. Price data and fundamental snapshots are obtained from Yahoo Finance. The benchmark series is the iShares STOXX Europe 600 UCITS ETF.

A specific limitation of this data source, and its consequences for the results reported below, is examined in detail in the following section. Separately, return on equity was found to contain a small number of materially erroneous values for European issuers, most notably an apparent return on equity in excess of 600 percent for one industrial issuer, attributable to near-zero or negative book equity. These cases were identified through manual inspection and corrected using verified company-reported figures; the correction process is documented in full in the development log.

---

## Data integrity investigation: point-in-time fundamentals

This section documents a methodological finding that materially qualifies how the results in this repository should be interpreted, and is presented prominently rather than as a footnote, since it is arguably the most consequential finding of the project.

### The observation

The model specification described above was executed on two separate occasions several weeks apart, using identical code and an identical investable universe. The posterior coefficient for earnings yield differed not only in magnitude but in sign between the two runs: approximately plus 0.41 in the first execution and approximately minus 0.23 in the second, with both 95 percent credible intervals excluding zero. No change had been made to the model, the priors, or the estimation procedure between the two runs.

### Diagnosis

The cause was identified as a data construction error rather than an estimation or modelling error. The fundamental dataset (earnings yield and return on equity for each stock) was downloaded once per execution, reflecting whatever values were current on Yahoo Finance at the moment of download, and this single static snapshot was applied identically to every one of the 35 quarters spanning 2016 to 2024 in the backtest. In effect, the model was evaluating, for example, the earnings yield of a stock in the first quarter of 2017 using a value that would not exist for another nine years. Because this snapshot differs depending on the date on which the script happens to be executed, the resulting coefficient estimate is not a stable property of the data-generating process but an artefact of execution timing.

### Remediation

The fundamental data pipeline was rebuilt to enforce point-in-time discipline. For each stock, a full history of reported quarterly and annual financial statements was retrieved, and each reported figure was assigned an availability date equal to the statement's period-end date plus a 90-day reporting lag, intended to approximate the delay between a fiscal period ending and the corresponding results becoming publicly available. At each quarter in the backtest, the model performs an as-of lookup against this history, retrieving only the most recent figure that would genuinely have been available at that point in time, and using no information that had not yet been published. Sector-neutral standardisation was correspondingly moved from a single calculation performed once across the full sample to a calculation performed independently within each quarter, using only the stocks with valid point-in-time data in that quarter.

A second, more subtle error was identified and corrected during this work. The initial implementation of the quarterly sector-neutral standardisation filled sector-quarter groups containing no valid observations with a value of zero rather than leaving them undefined. This had the effect of silently fabricating plausible-looking factor values for periods in which no genuine data existed, which masked the very problem under investigation and produced an apparently complete dataset extending back to 2016 despite the underlying data not supporting this. Correcting this revealed the true extent of the limitation.

### Finding

Under genuine point-in-time discipline, only 17.3 percent of stock-quarter observations have a valid earnings yield figure and 19.8 percent have a valid return on equity figure, concentrated almost entirely in the period from the third quarter of 2022 to the third quarter of 2024. Of the 35 quarters spanning the full price history, only 9 have usable point-in-time fundamental data. This is a direct consequence of the depth of free historical financial statement data available through Yahoo Finance, which typically extends to the last four to eight reported quarters and the last four reported fiscal years, rather than any property of the model.

Nine quarters is not a sufficient sample for a walk-forward validation procedure to produce statistically meaningful inference, and no performance metrics are reported for the point-in-time specification on this basis. The headline results reported elsewhere in this README, including the 34 percentage point cumulative outperformance figure, are derived from the non-point-in-time specification and should accordingly be read as a demonstration of the modelling and validation methodology rather than as a validated estimate of live performance. The reproducibility failure documented above indicates that the magnitude and even the sign of these specific results should not be relied upon.

### Implication for further work

This finding identifies the acquisition of institutional-grade point-in-time fundamental data, for example through Bloomberg, Refinitiv, or a lower-cost alternative such as Simfin+, as a prerequisite for any future attempt to validate this model's performance with statistical confidence, rather than a desirable enhancement. The point-in-time data pipeline implemented during this investigation is retained in the codebase and requires no further modification beyond substituting the underlying data source once one with sufficient historical depth is available.

---

## Results

The figures below are derived from the non-point-in-time specification described in the Methodology section, for the reasons set out in the preceding Data Integrity Investigation. They demonstrate that the modelling and validation pipeline functions as intended and produces internally consistent diagnostics, but should not be read as a validated estimate of achievable performance.

The walk-forward validated model achieves a mean quarterly classification accuracy of approximately 61 percent and a mean area under the ROC curve of approximately 0.61, evaluated out-of-sample across 26 quarters between 2018 and 2024. All five factor coefficients carry posterior 95 percent credible intervals consistent with the direction implied by their associated prior, and none of the five posterior distributions include zero within the credible interval at the final walk-forward fold.

A long-only portfolio constructed by investing equally in the top quartile of stocks ranked by posterior predictive probability, rebalanced quarterly, generated a cumulative outperformance of approximately 34 percentage points against the benchmark over the evaluation period, with positive excess returns in 178 of 283 evaluated quarters.

These figures should be read as gross, backtested results. No transaction costs, bid-ask spread, or market impact have been modelled; net-of-cost performance in a live implementation would be expected to be materially lower, plausibly by 30 to 50 percent depending on rebalancing frequency and position turnover. The evaluation period also coincides with a generally favourable environment for European equities; performance during a sustained drawdown has not been tested.

---

## Repository structure

```
bayesian-european-factor-model/
    README.md
    RESEARCH_LOG.md
    src/
        v3_production_model.py
    paper/
        bayesian_european_equity_paper.html
    figures/
        fig1_factor_distributions.png
        fig2_factor_vs_outcome.png
        fig3_correlation_matrix.png
        fig4_trace_plots.png
        fig5_posterior_densities.png
        fig6_portfolio_performance.png
        fig7_validation_metrics.png
```

The script in src/ is self-contained and reproduces the full pipeline end to end, from data download through posterior estimation, walk-forward validation, and portfolio construction. It is designed to be run in Google Colab without local environment configuration, though it will run in any standard Python environment with the dependencies listed below installed.

---

## Running the model

```
pip install yfinance pandas numpy matplotlib scipy scikit-learn seaborn
python src/v3_production_model.py
```

Total runtime is approximately 15 minutes, dominated by data download and the repeated MCMC estimation across walk-forward folds.

---

## Limitations and further work

The most consequential data limitation, the absence of point-in-time fundamental data in the free data source used throughout this project, is addressed in detail in the Data Integrity Investigation section above rather than repeated here. An attempt to expand the investable universe from 60 to 150 stocks using the same free data source resulted in degraded performance, attributable to declining data quality in smaller-capitalisation names rather than any property of the model itself; this finding is documented in the research log and informed the decision to retain the smaller, higher-quality universe.

An earnings revision factor, constructed from the ratio of forward to trailing consensus EPS estimates, was found to carry the strongest univariate predictive signal of any factor tested, but could not be incorporated into the production model because the data source provides only the current estimate rather than a point-in-time history, which would otherwise introduce a data-staleness bias into the backtest. This factor is the highest-priority candidate for future inclusion once point-in-time fundamental data becomes available.

Position sizing proportional to posterior conviction, following the Bayesian Kelly criterion, was also tested and is documented in the research log. The naive implementation underperformed equal weighting due to the small posterior standard deviations produced by a well-converged MCMC chain; a volatility-adjusted variant improved on this but did not surpass the equal-weighted benchmark on a cumulative return basis, while achieving a materially higher hit rate.

The model has not yet been validated against live trading. All performance figures reported here are derived from historical backtesting.

---

## References

Ang, A., Hodrick, R. J., Xing, Y. and Zhang, X. (2006). The cross-section of volatility and expected returns. Journal of Finance, 61(1), 259-299.

Fama, E. F. and French, K. R. (1993). Common risk factors in the returns on stocks and bonds. Journal of Financial Economics, 33(1), 3-56.

Fama, E. F. and French, K. R. (2015). A five-factor asset pricing model. Journal of Financial Economics, 116(1), 1-22.

Gelman, A., Jakulin, A., Pittau, M. G. and Su, Y.-S. (2008). A weakly informative default prior distribution for logistic and other regression models. Annals of Applied Statistics, 2(4), 1360-1383.

Greenblatt, J. (2010). The Little Book That Still Beats the Market. Wiley.

Gu, S., Kelly, B. and Xiu, D. (2020). Empirical asset pricing via machine learning. Review of Financial Studies, 33(5), 2223-2273.

Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. Journal of Finance, 45(3), 881-898.

Jegadeesh, N. and Titman, S. (1993). Returns to buying winners and selling losers. Journal of Finance, 48(1), 65-91.

Jiang, J., Yang, C., Wang, X. and Li, B. (2025). Why regression? Binary encoding classification brings confidence to stock market index price prediction. arXiv:2506.03153.

Novy-Marx, R. (2013). The other side of value: the gross profitability premium. Journal of Financial Economics, 108(1), 1-28.

Roberts, G. O., Gelman, A. and Gilks, W. R. (1997). Weak convergence and optimal scaling of random walk Metropolis algorithms. Annals of Applied Probability, 7(1), 110-120.

Wolff, D. (2024). Stock picking with machine learning. Journal of Forecasting, 43(5).

---

This is an independent research project and does not constitute investment advice.
