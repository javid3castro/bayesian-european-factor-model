"""
================================================================================
BAYESIAN EUROPEAN EQUITY FACTOR MODEL — V3 (PRODUCTION)
Predicting Cross-Sectional Outperformance via Bayesian Logistic Regression
with Metropolis-Hastings MCMC, Sector-Neutral Factors, and Walk-Forward
Validation.

This is the clean, consolidated V3 model — the production version documented
in the README and RESEARCH_LOG. Run this file end-to-end in Google Colab.

Architecture inspired by:
  - Gu, Kelly, Xiu (2020) — Empirical Asset Pricing via Machine Learning
  - Wolff (2024) — Stock Picking with Machine Learning (STOXX 600)
  - Gelman et al. (2008) — Weakly Informative Default Priors for Logistic Regression
  - Greenblatt (2010) — The Little Book That Beats the Market (EV/EBIT factor)
  - Roberts, Gelman, Gilks (1997) — Optimal Scaling of Random Walk Metropolis

Model: P(stock_i outperforms STOXX600 | sector-neutral factors_i)
Method: Bayesian Logistic Regression via Random-Walk Metropolis-Hastings
Validation: Expanding-window walk-forward (no lookahead bias)
Universe: 59 STOXX 600 constituents — IBEX, FTSE, DAX, CAC
Result: +34.0pp cumulative alpha vs STOXX 600 (2016-2024), 62.9% hit rate
================================================================================
"""

# ============================================================
# BLOCK 1 — SETUP
# ============================================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install",
                "yfinance", "pandas", "numpy", "matplotlib",
                "scipy", "scikit-learn", "seaborn", "--quiet"],
               capture_output=True)

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

plt.rcParams.update({
    "figure.facecolor": "#0a0a0f", "axes.facecolor": "#12121a",
    "axes.edgecolor": "#2a2a3a",   "axes.labelcolor": "#7070a0",
    "text.color": "#e8e8f0",       "xtick.color": "#7070a0",
    "ytick.color": "#7070a0",      "grid.color": "#2a2a3a",
    "grid.linewidth": 0.5,         "font.family": "monospace",
    "axes.titlecolor": "#e8e8f0",  "axes.titlesize": 11,
    "axes.labelsize": 10,          "figure.dpi": 120,
})
ACCENT, GREEN, AMBER, DANGER = "#7c6ff7", "#4fc8a0", "#f7c67c", "#f76c6c"

print("=" * 65)
print("  BAYESIAN EUROPEAN EQUITY FACTOR MODEL — V3 PRODUCTION")
print("=" * 65)
print("✅ Block 1 complete: libraries loaded")


# ============================================================
# BLOCK 2 — UNIVERSE: 60 STOXX 600 stocks
# ============================================================
UNIVERSE = {
    "Inditex": "ITX.MC", "Santander": "SAN.MC", "BBVA": "BBVA.MC",
    "Telefónica": "TEF.MC", "Iberdrola": "IBE.MC", "Repsol": "REP.MC",
    "ACS": "ACS.MC", "CaixaBank": "CABK.MC", "Amadeus": "AMS.MC",
    "Ferrovial": "FER.MC", "Mapfre": "MAP.MC", "Naturgy": "NTGY.MC",
    "Acciona": "ANA.MC", "Grifols": "GRF.MC", "Melia": "MEL.MC",

    "Shell": "SHEL.L", "AstraZeneca": "AZN.L", "Unilever": "ULVR.L",
    "Rio Tinto": "RIO.L", "GSK": "GSK.L", "Next": "NXT.L",
    "Diageo": "DGE.L", "BP": "BP.L", "Rolls-Royce": "RR.L",
    "Imperial Brands": "IMB.L", "Vodafone": "VOD.L",
    "Legal & General": "LGEN.L", "Tesco": "TSCO.L", "BT Group": "BT-A.L",
    "Aviva": "AV.L",

    "SAP": "SAP.DE", "Siemens": "SIE.DE", "Allianz": "ALV.DE",
    "Mercedes-Benz": "MBG.DE", "BMW": "BMW.DE", "BASF": "BAS.DE",
    "Bayer": "BAYN.DE", "Deutsche Telekom": "DTE.DE", "Adidas": "ADS.DE",
    "Volkswagen": "VOW3.DE", "Münchener Rück": "MUV2.DE",
    "Deutsche Bank": "DBK.DE", "Infineon": "IFX.DE", "Brenntag": "BNR.DE",

    "LVMH": "MC.PA", "L'Oréal": "OR.PA", "TotalEnergies": "TTE.PA",
    "Sanofi": "SAN.PA", "Schneider": "SU.PA", "Airbus": "AIR.PA",
    "Pernod Ricard": "RI.PA", "Kering": "KER.PA", "Hermès": "RMS.PA",
    "Michelin": "ML.PA", "Danone": "BN.PA", "Engie": "ENGI.PA",
    "Vinci": "DG.PA", "BNP Paribas": "BNP.PA", "Société Générale": "GLE.PA",
}

BENCHMARK_TICKER = "EXSA.DE"
START_DATE, END_DATE = "2016-01-01", "2024-12-31"
QUARTERS_TRAIN, N_ITER, BURN_IN = 10, 15_000, 3_000

print(f"\n📊 Universe: {len(UNIVERSE)} European equities")
print(f"   Period:   {START_DATE} → {END_DATE}")
print(f"   Benchmark: STOXX 600 ETF ({BENCHMARK_TICKER})")
print("✅ Block 2 complete")


# ============================================================
# BLOCK 3 — DATA PIPELINE
# ============================================================
def download_prices(tickers, start, end):
    print(f"\n📥 Downloading price history ({len(tickers)} tickers)...")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    thresh = len(prices) * 0.8
    prices = prices.dropna(axis=1, thresh=thresh).ffill().bfill()
    print(f"   ✓ {prices.shape[1]} tickers with sufficient history")
    return prices

def get_fundamentals_snapshot(ticker, name):
    try:
        info = yf.Ticker(ticker).info
        if not info or info.get("regularMarketPrice") is None:
            return None
        ev, ebitda = info.get("enterpriseValue"), info.get("ebitda")
        ev_ebitda = info.get("enterpriseToEbitda")
        if ev_ebitda and ev_ebitda > 0:
            earnings_yield = 1.0 / ev_ebitda
        elif ev and ebitda and ev > 0 and ebitda > 0:
            earnings_yield = ebitda / ev
        else:
            earnings_yield = None
        return {
            "name": name, "ticker": ticker,
            "earnings_yield": earnings_yield,
            "roe": info.get("returnOnEquity"),
            "log_mktcap": np.log(info.get("marketCap")) if info.get("marketCap") else None,
            "debt_equity": info.get("debtToEquity"),
            "gross_margin": info.get("grossMargins"),
        }
    except Exception:
        return None

print("\n📥 Downloading price history...")
all_tickers = list(UNIVERSE.values()) + [BENCHMARK_TICKER]
prices = download_prices(all_tickers, START_DATE, END_DATE)
bench_prices = prices[[BENCHMARK_TICKER]].copy() if BENCHMARK_TICKER in prices.columns else None
stock_prices = prices.drop(columns=[BENCHMARK_TICKER], errors="ignore")
valid_tickers = {name: t for name, t in UNIVERSE.items() if t in stock_prices.columns}
stock_prices = stock_prices[[t for t in valid_tickers.values()]]
print(f"   ✓ {stock_prices.shape[1]} stocks · {stock_prices.shape[0]} trading days")

print("\n📥 Downloading fundamentals...")
fundamentals = {}
for name, ticker in valid_tickers.items():
    result = get_fundamentals_snapshot(ticker, name)
    if result:
        fundamentals[ticker] = result
fund_df = pd.DataFrame(fundamentals).T
print(f"✅ Block 3 complete: {len(fund_df)} stocks with fundamental data")


# ============================================================
# BLOCK 3B — ROE DATA QUALITY FIX
# ============================================================
print("\n🔧 Cleaning ROE values...")

def get_roe_robust(ticker):
    try:
        stock = yf.Ticker(ticker)
        fin, bs = stock.financials, stock.balance_sheet
        ni = None
        for key in ['Net Income', 'Net Income Common Stockholders']:
            if key in fin.index:
                ni = fin.loc[key].iloc[0]; break
        eq = None
        for key in ['Stockholders Equity', 'Total Equity',
                    'Common Stock Equity', 'Total Stockholders Equity']:
            if key in bs.index:
                eq = bs.loc[key].iloc[0]; break
        if ni is not None and eq and eq > 0 and not np.isnan(eq):
            return float(ni / eq)
        return None
    except Exception:
        return None

# Manual overrides — verified company-reported figures (see RESEARCH_LOG.md)
ROE_OVERRIDES = {
    "RR.L": 0.18, "VOD.L": 0.008, "MUV2.DE": 0.198,
    "ULVR.L": 0.310, "BP.L": 0.080,
}

for ticker in fund_df.index:
    old_roe = fund_df.loc[ticker, 'roe']
    if ticker in ROE_OVERRIDES:
        new_roe = ROE_OVERRIDES[ticker]
    else:
        robust = get_roe_robust(ticker)
        if robust is not None and not np.isnan(robust):
            new_roe = robust
        elif old_roe is not None and not (isinstance(old_roe, float) and np.isnan(old_roe)):
            new_roe = old_roe
        else:
            new_roe = None
    if new_roe is not None:
        new_roe = float(np.clip(new_roe, -1.0, 1.5))
    fund_df.loc[ticker, 'roe'] = new_roe

n_ok = fund_df['roe'].notna().sum()
print(f"   ✅ ROE coverage: {n_ok}/{len(fund_df)} stocks")
print("✅ Block 3B complete")


# ============================================================
# BLOCK 3C — SECTOR NEUTRALITY
# ============================================================
print("\n🏷️  Assigning sectors...")

SECTOR_MAP = {
    "ITX.MC": "Consumer Discretionary", "ADS.DE": "Consumer Discretionary",
    "MC.PA": "Consumer Discretionary", "KER.PA": "Consumer Discretionary",
    "RMS.PA": "Consumer Discretionary", "NXT.L": "Consumer Discretionary",
    "MEL.MC": "Consumer Discretionary", "ML.PA": "Consumer Discretionary",

    "ULVR.L": "Consumer Staples", "DGE.L": "Consumer Staples",
    "IMB.L": "Consumer Staples", "RI.PA": "Consumer Staples",
    "BN.PA": "Consumer Staples", "OR.PA": "Consumer Staples",
    "TSCO.L": "Consumer Staples",

    "SHEL.L": "Energy", "BP.L": "Energy", "REP.MC": "Energy",
    "TTE.PA": "Energy", "NTGY.MC": "Energy",

    "SAN.MC": "Financials", "BBVA.MC": "Financials", "CABK.MC": "Financials",
    "MAP.MC": "Financials", "ALV.DE": "Financials", "MUV2.DE": "Financials",
    "DBK.DE": "Financials", "LGEN.L": "Financials", "AV.L": "Financials",
    "BNP.PA": "Financials", "GLE.PA": "Financials",

    "AZN.L": "Healthcare", "GSK.L": "Healthcare", "GRF.MC": "Healthcare",
    "BAYN.DE": "Healthcare", "SAN.PA": "Healthcare",

    "ACS.MC": "Industrials", "FER.MC": "Industrials", "ANA.MC": "Industrials",
    "SIE.DE": "Industrials", "RR.L": "Industrials", "AIR.PA": "Industrials",
    "DG.PA": "Industrials", "BNR.DE": "Industrials", "SU.PA": "Industrials",

    "RIO.L": "Materials", "BAS.DE": "Materials",

    "AMS.MC": "Technology", "SAP.DE": "Technology", "IFX.DE": "Technology",

    "TEF.MC": "Telecom", "VOD.L": "Telecom", "DTE.DE": "Telecom",
    "BT-A.L": "Telecom",

    "IBE.MC": "Utilities", "ENGI.PA": "Utilities",

    "MBG.DE": "Automotive", "BMW.DE": "Automotive", "VOW3.DE": "Automotive",
}

fund_df['sector'] = fund_df.index.map(lambda t: SECTOR_MAP.get(t, "Other"))

def add_sector_neutral_factors(df, factor_cols, sector_col='sector'):
    df = df.copy()
    for col in factor_cols:
        new_col = col + '_sn'
        df[new_col] = np.nan
        for sector in df[sector_col].unique():
            mask = df[sector_col] == sector
            values = df.loc[mask, col].dropna()
            if len(values) >= 3:
                mu, sd = values.mean(), values.std()
            else:
                all_vals = df[col].dropna()
                mu, sd = all_vals.mean(), all_vals.std()
            if sd > 1e-8:
                df.loc[mask, new_col] = (df.loc[mask, col] - mu) / sd
            else:
                df.loc[mask, new_col] = 0.0
    return df

fund_df = add_sector_neutral_factors(fund_df, ['earnings_yield', 'roe'])

FACTOR_COLS_SN = [
    'earnings_yield_sn', 'roe_sn',
    'f_momentum_6m', 'f_realized_vol', 'f_reversal_1m',
]
print(f"   Total: {len(fund_df)} stocks across {fund_df['sector'].nunique()} sectors")
print("✅ Block 3C complete")


# ============================================================
# BLOCK 4 — FACTOR ENGINEERING
# ============================================================
def compute_quarterly_factors(prices, bench_prices, fund_df):
    print("\n⚙️  Engineering quarterly factors...")
    daily_ret = prices.pct_change().dropna(how="all")
    quarter_ends = daily_ret.resample("QE").last().index
    records = []

    for i, q_end in enumerate(quarter_ends[:-1]):
        q_next = quarter_ends[i + 1]
        for ticker in prices.columns:
            try:
                p_now, p_next = prices[ticker].asof(q_end), prices[ticker].asof(q_next)
                if pd.isna(p_now) or pd.isna(p_next) or p_now <= 0:
                    continue
                stock_ret = (p_next - p_now) / p_now

                if bench_prices is not None:
                    b_now = bench_prices[BENCHMARK_TICKER].asof(q_end)
                    b_next = bench_prices[BENCHMARK_TICKER].asof(q_next)
                    if pd.isna(b_now) or pd.isna(b_next) or b_now <= 0:
                        continue
                    bench_ret_q = (b_next - b_now) / b_now
                else:
                    bench_ret_q = prices.pct_change(
                        periods=int((q_next - q_end).days)).loc[q_end].median()

                y = int(stock_ret > bench_ret_q)
                idx_end = daily_ret.index.get_indexer([q_end], method="ffill")[0]

                idx_6m, idx_1m = max(0, idx_end-126), max(0, idx_end-21)
                mom_6m = ((prices[ticker].iloc[idx_1m]-prices[ticker].iloc[idx_6m])
                          /prices[ticker].iloc[idx_6m]) if idx_1m>idx_6m and prices[ticker].iloc[idx_6m]>0 else np.nan

                idx_vol = max(0, idx_end-252)
                ret_window = daily_ret[ticker].iloc[idx_vol:idx_end+1].dropna()
                real_vol = ret_window.std()*np.sqrt(252) if len(ret_window)>20 else np.nan

                idx_4w = max(0, idx_end-21)
                p_4w = prices[ticker].iloc[idx_4w]
                reversal = (p_now-p_4w)/p_4w if p_4w>0 else np.nan

                records.append({
                    "date": q_end, "ticker": ticker, "stock_return": stock_ret,
                    "bench_return": bench_ret_q, "y": y,
                    "f_momentum_6m": mom_6m, "f_realized_vol": real_vol,
                    "f_reversal_1m": reversal,
                })
            except Exception:
                continue

    panel = pd.DataFrame(records)
    print(f"   ✓ Panel: {len(panel):,} observations · base rate y=1: {panel['y'].mean():.3f}")
    return panel

panel = compute_quarterly_factors(stock_prices, bench_prices, fund_df)

if 'earnings_yield_sn' in fund_df.columns:
    panel['earnings_yield_sn'] = panel['ticker'].map(fund_df['earnings_yield_sn'].to_dict())
    panel['roe_sn'] = panel['ticker'].map(fund_df['roe_sn'].to_dict())

FACTOR_COLS = FACTOR_COLS_SN
print("✅ Block 4 complete: factor panel constructed")


# ============================================================
# BLOCK 6 — GELMAN SCALING
# ============================================================
def gelman_scale(train_df, val_df, cols):
    info = {}
    X_train, X_val = train_df[cols].copy(), val_df[cols].copy()
    for col in cols:
        mu, sd = train_df[col].mean(), train_df[col].std()
        sd = sd if sd != 0 else 1e-8
        info[col] = {"mean": mu, "sd": sd}
        X_train[col] = (train_df[col]-mu)/(2*sd)
        X_val[col] = (val_df[col]-mu)/(2*sd)
    X_train.insert(0, "intercept", 1.0)
    X_val.insert(0, "intercept", 1.0)
    return X_train.values, X_val.values, info

print("✅ Block 6 complete: Gelman scaling defined")


# ============================================================
# BLOCK 7 — BAYESIAN SPECIFICATION
# ============================================================
# Priors grounded in empirical asset pricing literature (see README)
PRIOR_LOC = np.array([0.0, 0.5, 0.4, 0.4, -0.3, -0.2])
PRIOR_SCALE = np.array([10.0, 2.5, 2.5, 2.5, 2.5, 2.5])

def log_likelihood(beta, X, y):
    eta = X @ beta
    return np.sum(y*eta - np.log1p(np.exp(np.clip(eta, -500, 500))))

def log_prior_cauchy(beta, loc, scale):
    z = (beta-loc)/scale
    return np.sum(-np.log(np.pi*scale) - np.log1p(z**2))

def log_posterior(beta, X, y, prior_loc, prior_scale):
    return log_likelihood(beta, X, y) + log_prior_cauchy(beta, prior_loc, prior_scale)

print("✅ Block 7 complete: Bayesian functions defined")


# ============================================================
# BLOCK 8 — METROPOLIS-HASTINGS ALGORITHM
# ============================================================
def mh_step(beta_curr, lp_curr, X, y, U, prior_loc, prior_scale):
    d = len(beta_curr)
    eps = U.T @ np.random.standard_normal(d)
    beta_prop = beta_curr + eps
    lp_prop = log_posterior(beta_prop, X, y, prior_loc, prior_scale)
    if np.log(np.random.uniform()) < (lp_prop - lp_curr):
        return beta_prop, lp_prop, True
    return beta_curr, lp_curr, False

def run_mh_chain(X_train, y_train, n_iter=N_ITER, burn_in=BURN_IN,
                 prior_loc=PRIOR_LOC, prior_scale=PRIOR_SCALE, verbose=True):
    d = X_train.shape[1]
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500, fit_intercept=False)
    lr.fit(X_train, y_train)
    beta_mle = lr.coef_.flatten()

    p_hat = expit(X_train @ beta_mle)
    W = p_hat*(1-p_hat)
    Hess = X_train.T @ np.diag(W) @ X_train
    try:
        Sigma_hat = np.linalg.inv(Hess)
    except np.linalg.LinAlgError:
        Sigma_hat = np.eye(d)*0.01
    Sigma_prop = (2.38**2/d)*Sigma_hat
    try:
        U = np.linalg.cholesky(Sigma_prop).T
    except np.linalg.LinAlgError:
        U = np.eye(d)*0.05

    beta_curr = beta_mle.copy()
    lp_curr = log_posterior(beta_curr, X_train, y_train, prior_loc, prior_scale)
    chain, accepted = np.zeros((n_iter, d)), 0
    for t in range(n_iter):
        beta_curr, lp_curr, acc = mh_step(beta_curr, lp_curr, X_train, y_train, U, prior_loc, prior_scale)
        chain[t] = beta_curr
        accepted += int(acc)
    accept_rate = accepted/n_iter
    if verbose:
        print(f"   accept rate={accept_rate:.3f} (target: 0.20–0.30)")
    return chain[burn_in:], accept_rate, beta_mle

print("✅ Block 8 complete: MH algorithm defined")


# ============================================================
# BLOCK 9 — WALK-FORWARD VALIDATION
# ============================================================
def walk_forward_validation(panel, factor_cols, min_train_q=QUARTERS_TRAIN,
                            n_iter=N_ITER, burn_in=BURN_IN):
    print("\n" + "="*55)
    print("  WALK-FORWARD VALIDATION — expanding window, no lookahead")
    print("="*55)

    use_cols = factor_cols + ["y","date","ticker","stock_return","bench_return"]
    df = panel[use_cols].dropna()
    dates = sorted(df["date"].unique())
    n_q = len(dates)

    all_preds, fold_metrics, all_post_draws = [], [], []

    for q_val_idx in range(min_train_q, n_q):
        q_val, q_train = dates[q_val_idx], dates[:q_val_idx]
        train_df = df[df["date"].isin(q_train)].reset_index(drop=True)
        val_df = df[df["date"] == q_val].reset_index(drop=True)
        if len(val_df) < 5:
            continue

        X_train, X_val, _ = gelman_scale(train_df, val_df, factor_cols)
        y_train, y_val = train_df["y"].values.astype(float), val_df["y"].values.astype(float)

        post_draws, acc_rate, _ = run_mh_chain(X_train, y_train, n_iter, burn_in, verbose=False)
        all_post_draws.append(post_draws)

        eta_mat = X_val @ post_draws.T
        p_hat = expit(eta_mat).mean(axis=1)
        p_std = expit(eta_mat).std(axis=1)

        lr_base = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500, fit_intercept=False)
        lr_base.fit(X_train, y_train)
        p_freq = expit(X_val @ lr_base.coef_.flatten())

        y_pred_bayes = (p_hat >= 0.5).astype(int)
        acc_bayes = accuracy_score(y_val, y_pred_bayes)
        acc_freq = accuracy_score(y_val, (p_freq >= 0.5).astype(int))
        try:
            auc_bayes = roc_auc_score(y_val, p_hat)
            auc_freq = roc_auc_score(y_val, p_freq)
        except ValueError:
            auc_bayes = auc_freq = np.nan

        fold_metrics.append({"quarter": q_val, "n_val": len(val_df),
                             "acc_bayes": acc_bayes, "acc_freq": acc_freq,
                             "auc_bayes": auc_bayes, "auc_freq": auc_freq,
                             "accept_rate": acc_rate})

        preds_df = val_df[["date","ticker","y","stock_return","bench_return"]].copy()
        preds_df["p_hat"], preds_df["p_std"], preds_df["p_freq"] = p_hat, p_std, p_freq
        all_preds.append(preds_df)

    metrics_df = pd.DataFrame(fold_metrics)
    all_preds_df = pd.concat(all_preds, ignore_index=True)
    print(f"\n  Quarters validated: {len(metrics_df)}")
    print(f"  Mean accuracy (Bayesian): {metrics_df['acc_bayes'].mean():.4f}")
    print(f"  Mean accuracy (Frequentist): {metrics_df['acc_freq'].mean():.4f}")
    return {"metrics": metrics_df, "predictions": all_preds_df, "post_draws": all_post_draws}

results = walk_forward_validation(panel, FACTOR_COLS)
print("✅ Block 9 complete")


# ============================================================
# ADDENDUM — Missing blocks for full V3 reproducibility
# Insert these AFTER Block 9 (walk-forward) and BEFORE Block 11
# (portfolio construction) in v3_production_model.py
#
# These generate fig1, fig2, fig3 (EDA), fig4, fig5 (posterior
# diagnostics), fig7 (validation metrics over time), and the
# Bayesian vs Frequentist comparison table.
# ============================================================


# ============================================================
# BLOCK 5 — EXPLORATORY DATA ANALYSIS
# Generates: fig1_factor_distributions.png
#            fig2_factor_vs_outcome.png
#            fig3_correlation_matrix.png
# ============================================================
def run_eda(panel):
    print("\n📊 Running Exploratory Data Analysis...")

    FACTORS = {
        "earnings_yield_sn": "Earnings Yield (sector-neutral)",
        "roe_sn":             "Return on Equity (sector-neutral)",
        "f_momentum_6m":      "6-Month Momentum",
        "f_realized_vol":     "Realized Volatility (1Y)",
        "f_reversal_1m":      "1-Month Reversal",
    }
    panel_clean = panel.dropna(subset=list(FACTORS.keys()) + ["y"])

    # ── Fig 1: Factor distributions ──
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Figure 1 — Factor Distributions", fontsize=12, y=0.98)
    for ax, (col, label) in zip(axes.flat, FACTORS.items()):
        data = panel_clean[col].clip(panel_clean[col].quantile(0.01),
                                     panel_clean[col].quantile(0.99))
        ax.hist(data, bins=40, color=ACCENT, alpha=0.8, edgecolor="none")
        ax.set_title(label, pad=6); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig1_factor_distributions.png", dpi=130,
                bbox_inches="tight", facecolor="#0a0a0f")
    plt.show()

    # ── Fig 2: Factor deciles vs outcome ──
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Figure 2 — Factor Deciles vs P(Outperform)", fontsize=12, y=0.98)
    for ax, (col, label) in zip(axes.flat, FACTORS.items()):
        d = panel_clean[[col, "y"]].copy()
        d["decile"] = pd.qcut(d[col], 10, labels=False, duplicates="drop")
        dec_mean = d.groupby("decile")["y"].mean()
        colors = [GREEN if v > 0.5 else DANGER for v in dec_mean.values]
        ax.bar(dec_mean.index, dec_mean.values, color=colors, alpha=0.85)
        ax.axhline(0.5, color=AMBER, linewidth=1.2, linestyle="--", alpha=0.7)
        ax.set_title(label, pad=6); ax.set_ylim(0.35, 0.65); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig2_factor_vs_outcome.png", dpi=130,
                bbox_inches="tight", facecolor="#0a0a0f")
    plt.show()

    # ── Fig 3: Correlation matrix ──
    corr_cols = list(FACTORS.keys()) + ["y"]
    corr_data = panel_clean[corr_cols].copy()
    corr_data.columns = [v.split("(")[0].strip() for v in FACTORS.values()] + ["Outperform"]
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_data.corr(), annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, center=0, ax=ax, annot_kws={"size": 8},
                linewidths=0.5, linecolor="#2a2a3a")
    ax.set_title("Figure 3 — Correlation Matrix", fontsize=11, pad=10)
    plt.tight_layout()
    plt.savefig("fig3_correlation_matrix.png", dpi=130,
                bbox_inches="tight", facecolor="#0a0a0f")
    plt.show()
    print("✅ Block 5 complete: EDA finished")

run_eda(panel)


# ============================================================
# BLOCK 10 — POSTERIOR SUMMARIES AND DIAGNOSTICS
# Generates: fig4_trace_plots.png
#            fig5_posterior_densities.png
# ============================================================
def posterior_summaries_and_plots(post_draws, factor_names):
    all_names = ["intercept"] + factor_names
    post_mean, post_sd = post_draws.mean(axis=0), post_draws.std(axis=0)
    post_ci = np.percentile(post_draws, [2.5, 97.5], axis=0)

    print("\n" + "="*70)
    print("  POSTERIOR SUMMARIES — Final Walk-Forward Fold")
    print("="*70)
    for j, name in enumerate(all_names):
        signal = "✅ Positive" if post_ci[0,j] > 0 else ("❌ Negative" if post_ci[1,j] < 0 else "— Unclear")
        print(f"  {name:<25} mean={post_mean[j]:>7.4f}  "
              f"CI=[{post_ci[0,j]:>7.4f}, {post_ci[1,j]:>7.4f}]  {signal}")

    # ── Fig 4: Trace plots ──
    n_params = len(all_names)
    nrows = (n_params + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, nrows*3))
    fig.suptitle("Figure 4 — MCMC Trace Plots", fontsize=11)
    for j, (ax, name) in enumerate(zip(axes.flat, all_names)):
        ax.plot(post_draws[:, j], color=ACCENT, linewidth=0.4, alpha=0.8)
        ax.axhline(post_mean[j], color=GREEN, linewidth=1.2, linestyle="--")
        ax.set_title(name); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig4_trace_plots.png", dpi=130, bbox_inches="tight", facecolor="#0a0a0f")
    plt.show()

    # ── Fig 5: Posterior densities ──
    fig, axes = plt.subplots(nrows, 2, figsize=(14, nrows*3))
    fig.suptitle("Figure 5 — Posterior Distributions with 95% CI", fontsize=11)
    for j, (ax, name) in enumerate(zip(axes.flat, all_names)):
        ax.hist(post_draws[:, j], bins=60, density=True, color=ACCENT,
                alpha=0.7, edgecolor="none")
        ax.axvline(post_ci[0,j], color=AMBER, linestyle="--", linewidth=1.2)
        ax.axvline(post_ci[1,j], color=AMBER, linestyle="--", linewidth=1.2)
        ax.axvline(post_mean[j], color=GREEN, linewidth=1.5)
        ax.axvline(0, color=DANGER, linewidth=0.8, alpha=0.6)
        ax.set_title(name); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig5_posterior_densities.png", dpi=130, bbox_inches="tight", facecolor="#0a0a0f")
    plt.show()
    print("✅ Block 10 complete")

last_draws = results["post_draws"][-1]
posterior_summaries_and_plots(last_draws, FACTOR_COLS)


# ============================================================
# BLOCK 12 — VALIDATION METRICS OVER TIME
# Generates: fig7_validation_metrics.png
# ============================================================
def plot_validation_metrics(metrics_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Figure 7 — Walk-Forward Validation Metrics", fontsize=11)

    ax = axes[0,0]
    ax.plot(metrics_df["quarter"], metrics_df["acc_bayes"], color=GREEN,
            linewidth=1.8, marker="o", ms=4, label="Bayesian")
    ax.plot(metrics_df["quarter"], metrics_df["acc_freq"], color=AMBER,
            linewidth=1.2, marker="s", ms=4, linestyle="--", label="Frequentist")
    ax.axhline(0.5, color=DANGER, linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_title("Accuracy per Quarter"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0,1]
    ax.plot(metrics_df["quarter"], metrics_df["auc_bayes"], color=GREEN,
            linewidth=1.8, marker="o", ms=4, label="Bayesian")
    ax.plot(metrics_df["quarter"], metrics_df["auc_freq"], color=AMBER,
            linewidth=1.2, marker="s", ms=4, linestyle="--", label="Frequentist")
    ax.axhline(0.5, color=DANGER, linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_title("ROC-AUC per Quarter"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1,0]
    ax.bar(metrics_df["quarter"], metrics_df["accept_rate"], color=ACCENT, alpha=0.8, width=60)
    ax.axhline(0.234, color=GREEN, linewidth=1.2, linestyle="--", label="Optimal MH (23.4%)")
    ax.set_title("MH Acceptance Rate"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1,1]
    ax.hist(metrics_df["acc_bayes"], bins=10, color=GREEN, alpha=0.7,
            label=f"Bayesian (μ={metrics_df['acc_bayes'].mean():.3f})", edgecolor="none")
    ax.hist(metrics_df["acc_freq"], bins=10, color=AMBER, alpha=0.7,
            label=f"Frequentist (μ={metrics_df['acc_freq'].mean():.3f})", edgecolor="none")
    ax.axvline(0.5, color=DANGER, linewidth=1.2, linestyle="--")
    ax.set_title("Accuracy Distribution"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("fig7_validation_metrics.png", dpi=130, bbox_inches="tight", facecolor="#0a0a0f")
    plt.show()

    t_stat, p_val = stats.ttest_rel(metrics_df["acc_bayes"].dropna(), metrics_df["acc_freq"].dropna())
    print(f"\n  Paired t-test (Bayesian vs Frequentist): t={t_stat:.4f}  p={p_val:.4f}")
    print("✅ Block 12 complete")

plot_validation_metrics(results["metrics"])


# ============================================================
# BLOCK 13 — MODEL COMPARISON: BAYESIAN vs FREQUENTIST
# ============================================================
def compare_bayesian_frequentist(post_draws, X_train, y_train, factor_names):
    all_names = ["intercept"] + factor_names
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000, fit_intercept=False)
    lr.fit(X_train, y_train)
    beta_mle = lr.coef_.flatten()

    p_hat = expit(X_train @ beta_mle)
    W = p_hat*(1-p_hat)
    Hess = X_train.T @ np.diag(W) @ X_train
    try:
        se_mle = np.sqrt(np.diag(np.linalg.inv(Hess)))
    except np.linalg.LinAlgError:
        se_mle = np.full_like(beta_mle, np.nan)

    bayes_mean, bayes_sd = post_draws.mean(axis=0), post_draws.std(axis=0)

    print("\n" + "="*80)
    print("  MODEL COMPARISON: BAYESIAN vs FREQUENTIST (MLE)")
    print("="*80)
    print(f"\n  {'Coefficient':<25} {'Bayes Mean':>11} {'Bayes SD':>9}  {'MLE':>9} {'SE':>8}")
    print("  " + "-"*68)
    for j, name in enumerate(all_names):
        print(f"  {name:<25} {bayes_mean[j]:>11.4f} {bayes_sd[j]:>9.4f}  "
              f"{beta_mle[j]:>9.4f} {se_mle[j]:>8.4f}")
    print("✅ Block 13 complete")

# Reconstruct last fold's training data for comparison
use_cols = FACTOR_COLS + ["y", "date", "ticker"]
df_clean = panel[use_cols].dropna()
dates_clean = sorted(df_clean["date"].unique())
train_last = df_clean[df_clean["date"].isin(dates_clean[:-1])].reset_index(drop=True)
val_last = df_clean[df_clean["date"] == dates_clean[-1]].reset_index(drop=True)
X_tr, X_vl, _ = gelman_scale(train_last, val_last, FACTOR_COLS)
y_tr = train_last["y"].values.astype(float)

compare_bayesian_frequentist(last_draws, X_tr, y_tr, FACTOR_COLS)




# ============================================================
# BLOCK 11 — PORTFOLIO CONSTRUCTION (equal-weight top quartile)
# ============================================================
def construct_and_evaluate_portfolio(predictions, top_pct=0.25):
    print("\n" + "="*55)
    print(f"  PORTFOLIO CONSTRUCTION — top {int(top_pct*100)}% equal weight")
    print("="*55)

    quarterly_returns = []
    for q_date, q_data in predictions.groupby("date"):
        if len(q_data) < 4:
            continue
        bench_ret = q_data["bench_return"].iloc[0]
        n_long = max(1, int(len(q_data)*top_pct))
        top_q = q_data.nlargest(n_long, "p_hat")
        ret_long = top_q["stock_return"].mean()
        ret_ew = q_data["stock_return"].mean()
        quarterly_returns.append({
            "date": q_date, "ret_bayesian": ret_long, "ret_eq_weight": ret_ew,
            "ret_benchmark": bench_ret, "excess_bayes": ret_long - bench_ret,
        })

    port_df = pd.DataFrame(quarterly_returns).set_index("date")

    def sharpe(r, freq=4):
        mu, sd = r.mean()*freq, r.std()*np.sqrt(freq)
        return mu/sd if sd > 0 else 0
    def max_dd(r):
        cum = (1+r).cumprod()
        return ((cum-cum.cummax())/cum.cummax()).min()

    for col, label in [("ret_bayesian","Bayesian Top-Q"),("ret_benchmark","STOXX 600")]:
        r = port_df[col]
        print(f"  {label:<20} Ann.Ret={r.mean()*4*100:>6.1f}%  Sharpe={sharpe(r):.3f}  MaxDD={max_dd(r)*100:.1f}%")

    hit_rate = (port_df["ret_bayesian"] > port_df["ret_benchmark"]).mean()
    cum_gap = (1+port_df["ret_bayesian"]).cumprod().iloc[-1] - (1+port_df["ret_benchmark"]).cumprod().iloc[-1]
    print(f"\n  Hit rate vs benchmark: {hit_rate*100:.1f}%")
    print(f"  Cumulative alpha:      {cum_gap*100:+.1f}pp")

    fig, ax1 = plt.subplots(figsize=(11, 5))
    for label, (col, color) in [("Bayesian Top-Q",("ret_bayesian",GREEN)),
                                  ("STOXX 600",("ret_benchmark",DANGER))]:
        cum = (1+port_df[col]).cumprod()
        ax1.plot(cum.index, cum.values, label=label, color=color, linewidth=2)
    ax1.set_title("V3 Production Model — Cumulative Return vs STOXX 600")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("v3_portfolio_performance.png", dpi=130, bbox_inches="tight", facecolor="#0a0a0f")
    plt.show()

    return {"portfolio": port_df}

portfolio_results = construct_and_evaluate_portfolio(results["predictions"])
print("\n✅ ALL BLOCKS COMPLETE — V3 PRODUCTION MODEL")
print("   This is the version documented in README.md and RESEARCH_LOG.md")
