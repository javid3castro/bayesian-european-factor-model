# Predicting European Equity Outperformance: A Bayesian Logistic Factor Model Estimated by MCMC

*Binary cross-sectional stock selection on the STOXX Europe 600*

**Javier de Castro**
Independent research project · 2026 · Python implementation

> **Note on this paper's results.** A subsequent investigation, conducted after the analysis below was completed, found that the fundamental dataset used here is not point-in-time and that at least one coefficient sign is not stable across executions performed on different dates. The methodology and derivations below remain valid; the specific numerical results should be read as a demonstration of the pipeline rather than a validated estimate of performance. The full account of this finding is documented in the project's [development log](RESEARCH_LOG.md) and [README](README.md).

---

## Abstract

I develop a Bayesian logistic regression model to predict whether individual European equities will outperform the STOXX Europe 600 benchmark over the subsequent quarter. My binary outcome formulation — superior to direct return regression in the presence of heavy noise — is estimated via a Metropolis-Hastings MCMC algorithm, sampling the posterior distribution of five factor coefficients: earnings yield, return on equity, six-month price momentum, realised volatility, and short-term reversal. Priors are specified following Gelman et al. (2008) and grounded in decades of empirical asset pricing literature. I validate using expanding-window walk-forward methodology to respect temporal ordering and eliminate lookahead bias — a critical departure from k-fold cross-validation used in non-financial settings. Applied to 60 STOXX 600 constituents over 2016 to 2024, my model achieves mean quarterly prediction accuracy of approximately 57 to 60 percent, generating statistically significant alpha over the benchmark when stocks are selected by posterior probability of outperformance. I find that Bayesian and frequentist logistic regression produce nearly identical parameter estimates with large samples, confirming that data likelihood dominates priors — but the Bayesian framework provides the additional advantage of quantified uncertainty, enabling principled position sizing proportional to posterior probability mass.

**Keywords:** Bayesian inference, logistic regression, MCMC, factor investing, European equities, STOXX 600, Metropolis-Hastings, walk-forward validation, earnings yield, momentum, binary classification

---

## Contents

1. [Introduction](#introduction)
2. [Literature Review](#literature-review)
3. [Methodology and Theoretical Framework](#1-methodology-and-theoretical-framework)
   - 1.1 [The Bayesian Approach](#11-the-bayesian-approach)
   - 1.2 [The Bayesian Logistic Regression](#12-the-bayesian-logistic-regression-for-equity-outperformance)
   - 1.3 [The Metropolis-Hastings Algorithm](#13-the-metropolis-hastings-algorithm)
   - 1.4 [Walk-Forward Validation](#14-walk-forward-validation)
4. [Exploratory Data Analysis](#2-exploratory-data-analysis)
5. [Building the Model](#3-building-the-model)
   - 3.1 [Selecting Factors](#31-selecting-factors)
   - 3.2 [Choosing a Prior](#32-choosing-a-prior)
   - 3.3 [Coding the Model](#33-coding-the-model)
   - 3.4 [Results and Diagnostics](#34-results-and-diagnostics)
6. [Portfolio Construction and Performance](#4-portfolio-construction-and-performance)
7. [Conclusion](#5-conclusion)
8. [References](#references)

---

## Introduction

A fundamental question in quantitative equity management is whether a stock will outperform its benchmark over the next investment horizon. Unlike sports analytics, where outcomes are observed at high frequency under well-defined rules, financial returns are contaminated by extraordinary levels of noise. The signal-to-noise ratio in stock returns is notoriously low, and this structural feature has profound implications for how I should model the problem.

The dominant paradigm in empirical asset pricing — exemplified by Gu, Kelly, and Xiu (2020) — treats return prediction as a regression problem, estimating continuous expected excess returns as a function of firm characteristics. This approach is theoretically elegant but practically fragile: predicting that Volkswagen will return exactly 8.3 percent next quarter is a fundamentally different task from predicting that Volkswagen will outperform the DAX. The former is nearly impossible; the latter is tractable. Wolff (2024) formalises this insight, demonstrating that binary classification — predicting whether a stock exceeds the cross-sectional median — generates substantial and robust outperformance on the STOXX Europe 600, with simple regularised logistic models matching or exceeding complex machine learning alternatives.

I build on this evidence by introducing a *Bayesian* logistic regression framework, estimated via Metropolis-Hastings MCMC, applied to European equity selection. The Bayesian formulation offers three advantages over its frequentist counterpart. First, it allows the incorporation of prior economic knowledge — the value premium, momentum effect, and profitability factor are not statistical accidents; they are documented phenomena spanning decades and dozens of markets, and a sensible model should reflect this. Second, it produces a full posterior distribution over model parameters, enabling principled uncertainty quantification. Third, the posterior predictive probability $\hat{p}(x)$ serves directly as a position-sizing signal: a stock assigned $\hat{p} = 0.72$ receives a larger position than one assigned $\hat{p} = 0.54$, without arbitrary discretisation.

My model operates on 60 STOXX Europe 600 constituents drawn from four exchanges (IBEX, FTSE, DAX, CAC), validated quarterly over 2016 to 2024 using an expanding walk-forward window that strictly prevents lookahead bias. I document prediction accuracy, examine the posterior evidence for each factor, and translate the model's output into a quarterly rebalanced long portfolio benchmarked against the STOXX 600 ETF.

---

## Literature Review

The academic literature on factor-based equity return prediction is vast. The foundational framework is due to Fama and French (1993), who identify market beta, size (SMB), and value (HML) as systematic drivers of cross-sectional return variation. The value factor — the tendency of high book-to-market stocks to outperform — remains one of the most replicated findings in finance, though its magnitude has declined since publication, consistent with market learning (McLean and Pontiff, 2016). The profitability factor is formalised by Novy-Marx (2013) and incorporated into Fama and French's (2015) five-factor model, while momentum — the persistence of recent winner and loser status over six to twelve month horizons — is documented by Jegadeesh and Titman (1993) and incorporated in Carhart's (1997) four-factor model. Greenblatt (2010) synthesises value and quality into the "Magic Formula", ranking stocks by earnings yield and return on capital; European backtests show this two-factor combination outperforming the market by over 150 percent over twelve years (Quant Investing, 2024).

The question of whether to frame return prediction as regression or classification has received increasing attention. Gu, Kelly, and Xiu (2020) use OLS regression to model continuous expected excess returns on the S&P 500, finding that neural networks achieve the highest out-of-sample $R^2$ — but note explicitly the low signal-to-noise ratio in equity returns as a binding constraint on all methods. A concurrent literature argues for binary reformulation. Wolff (2024) trains classifiers to predict whether each stock exceeds the cross-sectional median return in the following week, applying this to both S&P 500 and STOXX Europe 600 constituents, and finds that logistic regression achieves results comparable to random forests and gradient boosting. Jiang et al. (2025), in a paper provocatively titled "Why Regression?", demonstrate on stock index prediction tasks that converting the continuous regression target to binary encoding consistently outperforms direct regression — a finding echoed across multiple international markets.

Bayesian methods in financial prediction remain relatively rare despite their natural fit. The closest precedents are in Bayesian portfolio allocation (Black and Litterman, 1992) and Bayesian factor modelling (Pastor and Stambaugh, 2000), rather than in the cross-sectional classification problem I study here. The methodological framework most directly relevant to my implementation is Gelman et al. (2008), who recommend Cauchy priors with scale 2.5 on standardised predictors in logistic regression, noting that these are weakly informative while accommodating sparse solutions better than Gaussian priors due to their heavier tails. For MCMC implementation, I follow Roberts, Gelman, and Gilks (1997), who derive the optimal scaling of the random-walk Metropolis proposal covariance as $(2.38^2/d) \times \hat{\Sigma}$, targeting an acceptance rate near 23.4 percent for high-dimensional targets.

---

## 1. Methodology and Theoretical Framework

### 1.1 The Bayesian Approach

The fundamental distinction between Bayesian and frequentist statistics lies in the treatment of model parameters. In the frequentist framework, parameters are fixed but unknown constants estimated from data. In the Bayesian framework, parameters are treated as random variables with probability distributions, updated upon observing data.

Consider a model with outcome $y$ and unknown parameter vector $\boldsymbol{\beta}$. Bayesian inference combines:

- A *prior distribution* $p(\boldsymbol{\beta})$, encoding beliefs about $\boldsymbol{\beta}$ before observing data.
- A *likelihood function* $p(y \mid X, \boldsymbol{\beta})$, describing how data is generated given $\boldsymbol{\beta}$.

After observing data, Bayes' theorem yields the *posterior distribution*:

$$p(\boldsymbol{\beta} \mid y, X) = \frac{p(y \mid X, \boldsymbol{\beta})\, p(\boldsymbol{\beta})}{p(y)}, \qquad p(y) = \int p(y \mid X, \boldsymbol{\beta})\,p(\boldsymbol{\beta})\, d\boldsymbol{\beta}$$

The marginal likelihood $p(y)$ is a normalising constant. Since it is independent of $\boldsymbol{\beta}$, I work with the unnormalised posterior:

$$p(\boldsymbol{\beta} \mid y, X) \propto p(y \mid X, \boldsymbol{\beta})\cdot p(\boldsymbol{\beta})$$

In the logistic regression setting, the integral defining $p(y)$ is intractable in closed form, motivating MCMC simulation. The key point estimates are the posterior mean $\mathbb{E}[\boldsymbol{\beta} \mid y]$ and posterior median, while uncertainty is summarised by the posterior variance and 95 percent credible intervals formed by the 2.5th and 97.5th quantiles of each marginal posterior.

### 1.2 The Bayesian Logistic Regression for Equity Outperformance

#### 1.2.1 My Data and Binary Outcome

My response variable is whether stock $i$ outperforms the STOXX 600 benchmark over the subsequent quarter. For each stock-quarter pair $(i, q)$, I define:

$$y_{i,q} = \begin{cases} 1 & \text{if } r_{i,q} > r_{\text{STOXX},q} \\ 0 & \text{otherwise} \end{cases}$$

where $r_{i,q}$ is the return of stock $i$ during quarter $q$ and $r_{\text{STOXX},q}$ is the contemporaneous return of the iShares STOXX Europe 600 ETF (EXSA.DE). This formulation eliminates market-wide movements and focuses model capacity on genuine cross-sectional alpha generation.

> **Why binary rather than continuous?** The signal-to-noise ratio in individual stock returns is extremely low — typical OLS $R^2$ values are 1 to 5 percent even with sophisticated factor sets (Gu, Kelly, Xiu, 2020). Outliers, such as a stock dropping 60 percent on a profit warning, severely distort continuous regression. The binary formulation is robust to such events — an extreme negative return simply becomes $y = 0$ — and is directly actionable: I rank stocks by $\hat{p}(y=1)$ and invest in the top quartile.

Let $\boldsymbol{x}_{i,q} = (x_{i,q,1}, \ldots, x_{i,q,p})^\top$ denote the vector of factors computed *as of the beginning of quarter $q$* (strictly historical, no lookahead). My objective is to model:

$$\mathbb{P}(y_{i,q} = 1 \mid \boldsymbol{x}_{i,q}) \equiv p_{i,q}$$

#### 1.2.2 The Logistic Function

A linear model $p_{i,q} = \boldsymbol{x}_{i,q}^\top \boldsymbol{\beta}$ would allow predicted probabilities outside $[0,1]$. The logit transformation maps probabilities to the real line via the log-odds, enabling linear modelling of the transformed quantity:

$$\log\left(\frac{p_{i,q}}{1 - p_{i,q}}\right) = \boldsymbol{x}_{i,q}^\top \boldsymbol{\beta} = \beta_0 + \beta_1 x_{i,q,1} + \cdots + \beta_p x_{i,q,p}$$

Solving for $p_{i,q}$ yields the logistic (sigmoid) function:

$$p_{i,q} = \sigma(\boldsymbol{x}_{i,q}^\top \boldsymbol{\beta}) = \frac{1}{1 + e^{-\boldsymbol{x}_{i,q}^\top \boldsymbol{\beta}}}$$

This ensures $p_{i,q} \in (0,1)$ for all $\boldsymbol{x}$ and $\boldsymbol{\beta}$. Each coefficient $\beta_j$ represents the change in log-odds of outperformance from a one-unit increase in the $j$-th (Gelman-scaled) factor: $\beta_j > 0$ implies higher values of the factor increase the probability of outperforming.

#### 1.2.3 The Likelihood Function

The outcome $y_{i,q}$ follows a Bernoulli distribution with success probability $p_{i,q}$. Under conditional independence across observations, the likelihood is:

$$\mathcal{L}(\boldsymbol{\beta}) = p(y \mid X, \boldsymbol{\beta}) = \prod_{i,q} p_{i,q}^{y_{i,q}} (1 - p_{i,q})^{1 - y_{i,q}}$$

For numerical stability, I work with the log-likelihood. Using the identity $\log\sigma(\eta) = \eta - \log(1 + e^\eta)$ and $\log(1 - \sigma(\eta)) = -\log(1 + e^\eta)$:

$$\log \mathcal{L}(\boldsymbol{\beta}) = \sum_{i,q} \left[ y_{i,q}\, \eta_{i,q} - \log(1 + e^{\eta_{i,q}}) \right], \quad \eta_{i,q} = \boldsymbol{x}_{i,q}^\top \boldsymbol{\beta}$$

#### 1.2.4 Moving from Frequentist to Bayesian

In the classical setting, $\boldsymbol{\beta}$ is estimated by maximising the likelihood (MLE). In my Bayesian formulation, I treat $\boldsymbol{\beta}$ as a random vector with prior distribution $p(\boldsymbol{\beta})$ and seek the full posterior:

$$p(\boldsymbol{\beta} \mid y, X) \propto \mathcal{L}(\boldsymbol{\beta})\cdot p(\boldsymbol{\beta}) = \left(\prod_{i,q} p_{i,q}^{y_{i,q}} (1-p_{i,q})^{1-y_{i,q}}\right) p(\boldsymbol{\beta})$$

Because the logistic function is nonlinear, the posterior does not belong to any standard distribution family, and the normalising constant requires a high-dimensional integral that is not tractable analytically. I therefore rely on MCMC to draw samples from the posterior.

### 1.3 The Metropolis-Hastings Algorithm

#### 1.3.1 Procedure

I define the unnormalised target density as $\tilde{\pi}(\boldsymbol{\beta}) = \mathcal{L}(\boldsymbol{\beta}) \cdot p(\boldsymbol{\beta})$, so that the posterior is $p(\boldsymbol{\beta} \mid y, X) = \tilde{\pi}(\boldsymbol{\beta}) / C$ where $C$ is unknown. The Metropolis-Hastings algorithm constructs a Markov chain whose stationary distribution is the target $\pi(\boldsymbol{\beta})$.

I employ a Gaussian random-walk proposal, initialised from the MLE with proposal covariance calibrated via the observed Fisher information following Roberts, Gelman, and Gilks (1997):

$$\boldsymbol{\beta}' = \boldsymbol{\beta}^{(t)} + \boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}\!\left(\mathbf{0},\; \frac{2.38^2}{d} \hat{\Sigma}\right)$$

where $d$ is the number of parameters and $\hat{\Sigma} = (\mathbf{X}^\top \mathbf{W} \mathbf{X})^{-1}$ is the inverse observed Fisher information, with $\mathbf{W} = \mathrm{diag}(p_{i,q}(1-p_{i,q}))$ evaluated at the MLE. The scaling constant $2.38^2/d$ is chosen to target an optimal acceptance rate of approximately 23.4 percent in high dimensions.

#### 1.3.2 The Acceptance Probability

Given the current state $\boldsymbol{\beta}^{(t)}$, a proposed value $\boldsymbol{\beta}'$ is accepted with probability:

$$\alpha(\boldsymbol{\beta}' \mid \boldsymbol{\beta}^{(t)}) = \min\!\left\{1,\; \frac{\tilde{\pi}(\boldsymbol{\beta}')}{\tilde{\pi}(\boldsymbol{\beta}^{(t)})} \cdot \frac{Q(\boldsymbol{\beta}^{(t)} \mid \boldsymbol{\beta}')}{Q(\boldsymbol{\beta}' \mid \boldsymbol{\beta}^{(t)})}\right\}$$

Because the Gaussian random-walk proposal is symmetric — $Q(\boldsymbol{\beta}' \mid \boldsymbol{\beta}) = Q(\boldsymbol{\beta} \mid \boldsymbol{\beta}')$ — the proposal ratio cancels and the acceptance simplifies to a ratio of log-posteriors:

$$\log \alpha = \log \tilde{\pi}(\boldsymbol{\beta}') - \log \tilde{\pi}(\boldsymbol{\beta}^{(t)}) = \left[\log \mathcal{L}(\boldsymbol{\beta}') + \log p(\boldsymbol{\beta}')\right] - \left[\log \mathcal{L}(\boldsymbol{\beta}^{(t)}) + \log p(\boldsymbol{\beta}^{(t)})\right]$$

I draw $u \sim \mathrm{Uniform}(0,1)$ and set $\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}'$ if $\log u < \log \alpha$, else $\boldsymbol{\beta}^{(t+1)} = \boldsymbol{\beta}^{(t)}$. One can verify that this acceptance probability satisfies the detailed balance condition $\pi(\boldsymbol{\beta}) P(\boldsymbol{\beta}' \mid \boldsymbol{\beta}) = \pi(\boldsymbol{\beta}') P(\boldsymbol{\beta} \mid \boldsymbol{\beta}')$, guaranteeing convergence of the chain to the target posterior.

After running $N = 15{,}000$ iterations, I discard the first $B = 3{,}000$ as burn-in. The remaining $S = 12{,}000$ post-burn-in draws $\{\boldsymbol{\beta}^{(s)}\}_{s=1}^S$ are treated as samples from $p(\boldsymbol{\beta} \mid y, X)$.

### 1.4 Walk-Forward Validation

A crucial design choice is the adoption of expanding-window walk-forward validation rather than k-fold cross-validation. The distinction is not cosmetic: many machine learning benchmarks rely on observations that are approximately independent of each other. Financial returns are *not* independent in this way — return distributions are non-stationary, factor premia vary with the business cycle, and regime changes such as the 2020 COVID shock or the 2022 rate cycle affect all stocks simultaneously.

Standard k-fold cross-validation on financial data allows future information to contaminate the training set, artificially inflating apparent out-of-sample performance. Walk-forward validation eliminates this by construction: at each evaluation quarter $q$, the model is trained exclusively on data from quarters $1, \ldots, q-1$, then evaluated on quarter $q$. The window expands as additional data becomes available.

$$\text{For } q = q_{\min}, \ldots, Q: \quad \text{train on } \{(X_{i,q'}, y_{i,q'}): q' < q\}, \quad \text{validate on } \{(X_{i,q}, y_{i,q})\}$$

Predicted probabilities are formed by averaging the sigmoid over posterior draws:

$$\hat{p}(\boldsymbol{x}) = \frac{1}{S} \sum_{s=1}^S \sigma\!\left(\boldsymbol{x}^\top \boldsymbol{\beta}^{(s)}\right)$$

I require a minimum of $q_{\min} = 10$ training quarters (2.5 years) before the first validation quarter, ensuring sufficient data for stable MCMC convergence. Scaling constants (Gelman means and standard deviations) are computed exclusively on the training fold and applied to the validation fold, never the reverse.

---

## 2. Exploratory Data Analysis

My dataset comprises 60 STOXX Europe 600 constituents drawn from four national markets: Spain (IBEX 35), the United Kingdom (FTSE 100), Germany (DAX 40), and France (CAC 40). Price data spans January 2016 to December 2024, yielding 36 calendar quarters and approximately 1,800 to 2,000 stock-quarter observations after filtering for complete factor data. The benchmark is the iShares STOXX Europe 600 UCITS ETF (ticker: EXSA.DE), which provides a liquid, total-return measure of the broad European market.

The base rate of the binary outcome $y = 1$ is close to 50 percent by construction, since the benchmark is defined relative to the broader STOXX 600 rather than my 60-stock universe, ranging from 44 to 56 percent across quarters. This near-balanced class distribution means that a naive classifier predicting always $y = 0$ or always $y = 1$ achieves approximately 50 percent accuracy — my minimum threshold for claiming predictive power.

### Factor Distributions

**Earnings yield.** The distribution of earnings yield (EBITDA divided by enterprise value) is right-skewed, with most stocks clustering in the 4 to 12 percent range. The IBEX stocks (energy, banks, telecoms) tend to exhibit higher yields than the premium-valued CAC luxury and technology names. The bimodal structure visible in some samples reflects this IBEX versus CAC bifurcation — Spain trades at structurally lower multiples than France.

**Momentum.** Six-month price momentum follows an approximately normal distribution centred near zero with fat tails, consistent with the well-documented momentum crash phenomenon: most periods are quiet, but occasional reversals are extreme. The spike at very negative values corresponds to the 2020 COVID shock period.

**Realised volatility.** Annualised realised volatility ranges from roughly 12 to 55 percent, with a pronounced right skew. Financial and energy stocks (Santander, Bayer, BP) cluster in the high-volatility tail, while defensive staples and utilities (Unilever, Diageo, Engie) concentrate at lower volatility levels.

### Factor versus Outcome

Examining mean outperformance probability across factor deciles reveals the expected monotone relationships. Stocks in the top decile of earnings yield — the cheapest by valuation — outperform with frequency approximately 5 to 8 percentage points above the bottom decile, consistent with the European value premium documented by Wolff (2024) and Greenblatt (2010). The momentum relationship is similarly monotone: stocks with the highest trailing six-month returns continue to outperform in the following quarter, while past losers continue to lag.

Realised volatility exhibits a negative relationship with outperformance — the low-volatility anomaly first documented by Ang et al. (2006) appears in my European sample. High-volatility stocks consistently underperform on a net basis, consistent with investors overpaying for lottery-like payoffs.

### Correlation Structure

The correlation matrix reveals that the five factors are largely orthogonal, with pairwise correlations below 0.25 in absolute value, supporting their joint inclusion in the model without multicollinearity concerns. The one notable exception is momentum and realised volatility, which exhibit modest positive correlation of approximately 0.18: recent winners tend to have slightly higher volatility, as strong price movements mechanically inflate short-window standard deviations. This is not sufficient to exclude either factor, as their relationships with the outcome operate through distinct economic mechanisms.

---

## 3. Building the Model

### 3.1 Selecting Factors

Factor selection follows three criteria: each factor must be *observable before the prediction quarter*, with no lookahead; each factor must have a *clear economic mechanism* linking it to future outperformance; and each factor must exhibit a *statistically and economically significant* univariate relationship with the outcome in my exploratory analysis.

| Factor | Economic Mechanism | Expected Sign | Source |
|---|---|---|---|
| Earnings Yield (EY) | High EY → cheap vs earnings → value premium | + | Greenblatt (2010), Fama-French HML |
| Return on Equity (ROE) | High ROE → efficient capital use → profitability premium | + | Novy-Marx (2013), FF RMW |
| Momentum 6m | Price persistence over 6–12m horizon | + | Jegadeesh & Titman (1993) |
| Realised Volatility | Investors overpay for high-vol lottery stocks | − | Ang et al. (2006), low-vol anomaly |
| 1-Month Reversal | Short-term mean reversion after microstructure effects | − | Jegadeesh (1990) |

I exclude leverage (debt to equity) from the primary specification: while leverage predicts stock risk, it proxies for endogenous financing decisions that correlate with earnings yield and ROE, and including it risks double-counting the value signal rather than adding independent information.

### 3.2 Choosing a Prior

Following Gelman et al. (2008), I first scale all continuous predictors to have mean 0 and standard deviation 0.5, making prior scales directly comparable across coefficients:

$$x_j^* = \frac{x_j - \bar{x}_j}{2 \cdot \mathrm{sd}(x_j)}$$

I adopt independent Cauchy priors, preferred over Gaussian priors because their heavier tails allow occasional large coefficients (when one factor dominates) while still shrinking small effects toward zero. Following Gelman et al. (2008), the baseline weakly informative prior for standardised logistic coefficients is Cauchy(0, 2.5), with the intercept given Cauchy(0, 10).

For factors where decades of empirical evidence establish a directional prior belief, I shift the location parameter away from zero. A shift of $\pm 0.5$ on the Gelman scale implies that a two-standard-deviation increase in the factor changes the odds by a factor of $e^{\pm 0.5} \approx 0.61$ or $1.65$ — a meaningful but not extreme prior belief:

| Coefficient | Prior | Interpretation |
|---|---|---|
| Intercept ($\beta_0$) | Cauchy(0, 10) | Diffuse — no base-rate belief |
| Earnings Yield ($\beta_1$) | Cauchy(+0.5, 2.5) | 2 SD ↑ in EY → odds × 1.65 of outperform |
| ROE ($\beta_2$) | Cauchy(+0.4, 2.5) | 2 SD ↑ in ROE → odds × 1.49 of outperform |
| Momentum 6m ($\beta_3$) | Cauchy(+0.4, 2.5) | 2 SD ↑ in mom → odds × 1.49 of outperform |
| Realised Vol ($\beta_4$) | Cauchy(−0.3, 2.5) | 2 SD ↑ in vol → odds × 0.74 of outperform |
| 1m Reversal ($\beta_5$) | Cauchy(−0.2, 2.5) | Recent 1m winner → mildly less likely to outperform |

The Cauchy prior density for coefficient $\beta_j$ is:

$$p(\beta_j) = \frac{1}{\pi s_j \left[1 + \left(\frac{\beta_j - m_j}{s_j}\right)^2\right]}$$

and the joint log-prior under independence is:

$$\log p(\boldsymbol{\beta}) = \sum_{j=0}^p \left[-\log(\pi s_j) - \log\!\left(1 + \left(\frac{\beta_j - m_j}{s_j}\right)^2\right)\right]$$

### 3.3 Coding the Model

#### 3.3.1 Data Preparation and Gelman Scaling

```python
# Gelman scaling: x* = (x - mean) / (2 × sd)
# Critical: scaling constants computed on the training fold only
def gelman_scale(train_df, val_df, cols):
    info = {}
    X_train = train_df[cols].copy()
    X_val   = val_df[cols].copy()
    for col in cols:
        mu  = train_df[col].mean()    # training mean only
        sd  = train_df[col].std()     # training sd only
        info[col] = {"mean": mu, "sd": sd}
        X_train[col] = (train_df[col] - mu) / (2 * sd)
        X_val[col]   = (val_df[col]   - mu) / (2 * sd)  # apply same scale
    X_train.insert(0, "intercept", 1.0)
    X_val.insert(0, "intercept", 1.0)
    return X_train.values, X_val.values, info
```

#### 3.3.2 Log-Posterior (Target Distribution)

```python
def log_likelihood(beta, X, y):
    # Numerically stable: sum(y * eta - log1p(exp(eta)))
    eta = X @ beta
    return np.sum(y * eta - np.log1p(np.exp(np.clip(eta, -500, 500))))

def log_prior_cauchy(beta, loc, scale):
    # Independent Cauchy: -log(pi*s) - log(1 + ((beta-m)/s)^2)
    z = (beta - loc) / scale
    return np.sum(-np.log(np.pi * scale) - np.log1p(z ** 2))

def log_posterior(beta, X, y, prior_loc, prior_scale):
    return log_likelihood(beta, X, y) + log_prior_cauchy(beta, prior_loc, prior_scale)
```

#### 3.3.3 Metropolis-Hastings Step

```python
def mh_step(beta_curr, lp_curr, X, y, U, prior_loc, prior_scale):
    d         = len(beta_curr)
    z         = np.random.standard_normal(d)
    eps       = U.T @ z                       # correlated proposal draw
    beta_prop = beta_curr + eps
    lp_prop   = log_posterior(beta_prop, X, y, prior_loc, prior_scale)
    log_alpha = lp_prop - lp_curr             # symmetric proposal cancels

    if np.log(np.random.uniform()) < log_alpha:
        return beta_prop, lp_prop, True       # accept
    return beta_curr, lp_curr, False          # reject
```

#### 3.3.4 Walk-Forward Loop

```python
for q_val_idx in range(min_train_q, n_quarters):
    q_val   = quarters[q_val_idx]
    q_train = quarters[:q_val_idx]            # all prior quarters

    train_df = panel[panel["date"].isin(q_train)]
    val_df   = panel[panel["date"] == q_val]

    X_train, X_val, _ = gelman_scale(train_df, val_df, FACTOR_COLS)
    y_train = train_df["y"].values

    # Run MCMC on training data
    post_draws, accept_rate, _ = run_mh_chain(
        X_train, y_train, n_iter=15000, burn_in=3000
    )

    # Posterior predictive: average sigmoid over posterior draws
    eta_mat = X_val @ post_draws.T            # (n_val, S)
    p_hat   = expit(eta_mat).mean(axis=1)     # posterior mean P(y=1)
```

#### 3.3.5 Acceptance Rate and Convergence

The Metropolis-Hastings chain is initialised at the MLE estimate, with proposal covariance $\hat{\Sigma} = (X^\top W X)^{-1}$ scaled by $2.38^2/d$. The resulting acceptance rates across walk-forward folds average 0.24 to 0.28, within the optimal range of 0.20 to 0.30 for $d = 6$ parameters (Roberts et al., 1997).

### 3.4 Results and Diagnostics

I present posterior summaries from the final walk-forward fold (most training data, hence most informative posteriors). All coefficients are on the Gelman scale.

| Coefficient | Post. Mean | Post. SD | CI 2.5% | CI 97.5% | P(β>0) | Signal |
|---|---|---|---|---|---|---|
| Intercept | −0.031 | 0.032 | −0.093 | 0.031 | 0.168 | Unclear |
| Earnings Yield (EY) | +0.412 | 0.048 | +0.318 | +0.507 | 0.998 | Positive |
| Return on Equity (ROE) | +0.218 | 0.041 | +0.138 | +0.298 | 0.997 | Positive |
| Momentum 6m | +0.317 | 0.038 | +0.242 | +0.390 | 1.000 | Positive |
| Realised Volatility | −0.184 | 0.044 | −0.270 | −0.098 | 0.002 | Negative |
| 1m Reversal | −0.089 | 0.039 | −0.165 | −0.013 | 0.012 | Negative |

All five factors carry statistically significant evidence consistent with their prior direction. Earnings yield is the strongest predictor, with a posterior mean of +0.412 and a 95 percent credible interval entirely above zero. This is consistent with the value premium in European equities: cheap stocks by earnings yield outperform expensive ones. Momentum (6m) is the second strongest signal, consistent with Jegadeesh and Titman (1993) and the extensive Carhart (1997) literature. The low-volatility anomaly appears through the negative coefficient on realised volatility, and short-term reversal is mildly but significantly negative.

As noted at the top of this paper, a subsequent reproducibility check found that the sign of the earnings yield coefficient is not stable when the underlying fundamental data is re-downloaded on a different date, which is documented in full in the project's development log. The interpretation above should accordingly be read as a demonstration of how to interpret the model's output, conditional on the data snapshot used, rather than as a settled empirical conclusion.

#### 3.4.1 Trace Plots

The trace plots for all six parameters show chains oscillating in stable horizontal bands with no upward or downward drift, indicating that the sampler has reached the stationary distribution after burn-in. Frequent up-and-down movement, without long flat stretches, confirms adequate mixing. The stationarity is consistent across all walk-forward folds, indicating that the proposal covariance calibration from the MLE is effective throughout the sample period.

#### 3.4.2 Model Comparison: Bayesian versus Frequentist

I compare my Bayesian estimates with standard frequentist MLE logistic regression. The two approaches produce point estimates and uncertainty bounds that are nearly identical, confirming that with sample sizes of 1,000 to 2,000 observations, the data likelihood dominates the prior. This is not surprising: with large $n$, the Bernstein–von Mises theorem guarantees that the posterior concentrates around the MLE, and the prior's influence is negligible.

The Bayesian advantage is not in the point estimates but in the richer inferential output: the full posterior distribution over $\boldsymbol{\beta}$ yields natural probability statements, such as "the momentum coefficient is positive with 99.9 percent posterior probability", and the posterior predictive distribution $\hat{p}(\boldsymbol{x})$ carries principled uncertainty that can be propagated into position sizing.

---

## 4. Portfolio Construction and Performance

The model's output — a posterior probability $\hat{p}_{i,q}$ of outperforming the STOXX 600 for each stock $i$ in quarter $q$ — translates directly into portfolio construction. At each quarter-end, I rank all stocks in the validation universe by $\hat{p}_{i,q}$ and invest equally in the top quartile (approximately 15 stocks), rebalancing quarterly.

This approach embeds a natural Kelly-inspired position-sizing logic: stocks with higher posterior probabilities receive the same capital allocation within the portfolio but are selected precisely because the model assigns them the highest probability mass above 0.5. A more refined implementation would weight positions proportionally to $\hat{p}_{i,q}$, which I implement as an alternative strategy (probability-weighted portfolio) and discuss further in the development log.

### Performance Metrics

| Strategy | Ann. Return | Sharpe Ratio | Max Drawdown | Hit Rate vs STOXX |
|---|---|---|---|---|
| **Bayesian Top-Quartile** | +11.4% | 0.82 | −22.1% | 61.5% |
| Probability-Weighted | +10.2% | 0.74 | −23.8% | 58.3% |
| Frequentist Top-Quartile | +10.8% | 0.78 | −22.4% | 59.6% |
| Equal-Weight Universe | +8.1% | 0.63 | −28.4% | 52.1% |
| STOXX 600 Benchmark | +7.2% | 0.59 | −30.1% | — |

The Bayesian top-quartile portfolio delivers an annualised return of approximately 11.4 percent versus the benchmark's 7.2 percent, generating roughly 4.2 percentage points of excess annual return. The Sharpe ratio of 0.82 compares favourably to the benchmark's 0.59, indicating that the excess return is not simply a compensation for additional risk. The hit rate of 61.5 percent — meaning the Bayesian portfolio beat the benchmark in 61.5 percent of quarters — is consistent with the model's mean prediction accuracy of approximately 57 to 60 percent.

The frequentist top-quartile strategy delivers comparable but slightly inferior performance (Sharpe 0.78 versus 0.82), suggesting that the Bayesian framework provides modest but consistent gains through better uncertainty quantification in the prediction step.

---

## 5. Conclusion

I have demonstrated that a Bayesian logistic regression model estimated via Metropolis-Hastings MCMC can generate economically interpretable signals for cross-sectional European equity selection. Several conclusions emerge from this analysis.

**Binary formulation outperforms continuous return prediction in practice.** The structural signal-to-noise ratio of equity returns makes direct return regression unreliable — an $R^2$ of 1 to 5 percent implies 95 to 99 percent of return variation is unexplained. Reformulating the problem as "does this stock outperform the benchmark?" focuses model capacity on the directional question that matters for portfolio construction, and proves more robust to outliers and regime shifts.

**The five-factor structure captures distinct economic mechanisms.** Earnings yield (value premium), return on equity (profitability premium), six-month momentum (price persistence), realised volatility (low-volatility anomaly), and short-term reversal all carry posterior evidence consistent with prior economic expectations and with 95 percent credible intervals excluding zero in the expected direction, in the specification examined here.

**Bayesian and frequentist estimates converge with large samples.** Posterior estimates are nearly identical to MLE estimates once training samples exceed approximately 1,000 observations. The Bayesian value-add lies not in different point estimates but in principled prior encoding of empirical asset pricing knowledge, full posterior uncertainty for position sizing, and natural probability statements about factor significance.

**Walk-forward validation is non-negotiable in finance.** Financial returns exhibit autocorrelation and regime-dependence that k-fold cross-validation does not respect; applying it naively to financial data induces lookahead bias that artificially inflates apparent performance. My expanding-window walk-forward protocol eliminates this by design.

Several important limitations deserve acknowledgment, beyond the point-in-time data finding noted at the top of this paper. The factor set, while grounded in the literature, is far smaller than the 94-characteristic space of Gu, Kelly, and Xiu (2020). The model assumes factor relationships are stable across regimes, an assumption tested by the 2020 COVID shock and the 2022 rate normalisation. Future work should incorporate stock-specific fixed effects, time-varying factor exposures via state-space models, and richer alternative data, alongside the institutional-grade point-in-time fundamental data identified in the development log as the highest-priority requirement for any further validation of this model's performance.

---

## References

Ang, A., Hodrick, R. J., Xing, Y., and Zhang, X. (2006). The cross-section of volatility and expected returns. *Journal of Finance*, 61(1), 259–299.

Black, F. and Litterman, R. (1992). Global portfolio optimization. *Financial Analysts Journal*, 48(5), 28–43.

Carhart, M. M. (1997). On persistence in mutual fund performance. *Journal of Finance*, 52(1), 57–82.

Fama, E. F. and French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.

Fama, E. F. and French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1–22.

Gelman, A., Jakulin, A., Pittau, M. G., and Su, Y.-S. (2008). A weakly informative default prior distribution for logistic and other regression models. *Annals of Applied Statistics*, 2(4), 1360–1383. [Link](https://sites.stat.columbia.edu/gelman/research/unpublished/priors11.pdf)

Greenblatt, J. (2010). *The Little Book That Still Beats the Market*. Wiley.

Gu, S., Kelly, B., and Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273. [Link](https://dachxiu.chicagobooth.edu/download/ML.pdf)

Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. *Journal of Finance*, 45(3), 881–898.

Jegadeesh, N. and Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1), 65–91.

Jiang, J., Yang, C., Wang, X., and Li, B. (2025). Why regression? Binary encoding classification brings confidence to stock market index price prediction. arXiv:2506.03153.

McLean, R. D. and Pontiff, J. (2016). Does academic research destroy stock return predictability? *Journal of Finance*, 71(1), 5–32.

Novy-Marx, R. (2013). The other side of value: the gross profitability premium. *Journal of Financial Economics*, 108(1), 1–28.

Quant Investing (2024). Magic Formula complete guide: European backtests 1999–2024. [Link](https://www.quant-investing.com/blog/magic-formula-complete-guide)

Roberts, G. O., Gelman, A., and Gilks, W. R. (1997). Weak convergence and optimal scaling of random walk Metropolis algorithms. *Annals of Applied Probability*, 7(1), 110–120.

Wolff, D. (2024). Stock picking with machine learning. *Journal of Forecasting*, 43(5).

---

*This is an independent research project and does not constitute investment advice. Code available in this repository's [`src/`](src/) directory.*
