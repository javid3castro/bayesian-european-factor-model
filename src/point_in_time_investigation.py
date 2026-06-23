"""
================================================================================
POINT-IN-TIME DATA INTEGRITY INVESTIGATION
Companion script to src/v3_production_model.py

This script documents and reproduces the finding described in the
"Data Integrity Investigation" section of README.md and Version 8 of
RESEARCH_LOG.md.

Background
-----------
The production model (v3_production_model.py) downloads earnings yield and
return on equity ONCE per execution and applies that single static snapshot
to all 35 quarters of the 2016-2024 backtest. This was found to produce a
coefficient sign that is unstable across executions performed on different
dates, since the snapshot simply reflects whatever Yahoo Finance reports on
the day the script happens to be run.

What this script does
----------------------
Rebuilds the fundamental data pipeline to enforce point-in-time discipline:
for each stock, it retrieves the full available history of quarterly and
annual financial statements, assigns each reported figure an "available
date" (period-end date + a 90-day reporting lag), and performs an as-of
lookup at each quarter of the backtest so that no information from the
future is ever used. Sector-neutral standardisation is correspondingly
computed fresh within each quarter rather than once across the full sample.

Dependencies
------------
This script assumes Blocks 1 through 3 of src/v3_production_model.py have
already been executed in the same session, so that the following objects
exist in memory: UNIVERSE, valid_tickers, stock_prices, bench_prices,
BENCHMARK_TICKER, yf, pd, np.

Expected output
----------------
A coverage report printed to console. At the time this investigation was
run, only 17.3% of stock-quarter observations had a valid point-in-time
earnings yield and 19.8% a valid point-in-time ROE, with usable data
concentrated in a window of 9 quarters (2022-09-30 to 2024-09-30) out of
the 35 quarters spanning the full price history. This sample size is too
small to draw statistically meaningful performance conclusions, which is
why no walk-forward results are reported for this specification.
================================================================================
"""

# ============================================================
# BLOCK 3C — POINT-IN-TIME FUNDAMENTAL HISTORY
# Replaces the static single-snapshot fund_df approach.
#
# Problem being fixed: previously, earnings_yield and roe were
# downloaded ONCE (today's value) and applied identically to
# every quarter from 2016-2024. This caused the coefficient sign
# to flip between runs executed on different days, and meant the
# "sector-neutral" ranking never actually varied over time for
# fundamentals.
#
# Fix: build a per-ticker TIME SERIES of fundamentals using both
# quarterly and annual financial statements, apply a reporting
# lag (companies don't publish results the same day the quarter
# ends), and look up the value that was ACTUALLY AVAILABLE as of
# each historical quarter — never a future value.
# ============================================================

print("=" * 60)
print("  BUILDING POINT-IN-TIME FUNDAMENTAL HISTORY")
print("=" * 60)

REPORTING_LAG_DAYS = 90   # time between period-end and public filing

def get_fundamental_history(ticker: str) -> pd.DataFrame:
    """
    Build a point-in-time history of earnings yield and ROE for one
    ticker, combining quarterly and annual statements.

    Returns a DataFrame indexed by 'available_date' (report date +
    reporting lag) with columns ['earnings_yield', 'roe'], sorted
    ascending. Empty DataFrame if no data could be retrieved.
    """
    records = []
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        shares_out = info.get("sharesOutstanding")

        # ── Quarterly statements (best granularity, short history) ──
        try:
            q_fin = stock.quarterly_financials
            q_bs  = stock.quarterly_balance_sheet
            for report_date in q_fin.columns:
                ni = None
                for key in ['Net Income', 'Net Income Common Stockholders',
                           'EBIT', 'Operating Income']:
                    if key in q_fin.index:
                        val = q_fin.loc[key, report_date]
                        if pd.notna(val):
                            ni = val
                            if key in ('EBIT', 'Operating Income'):
                                ebit = val
                            break
                ebit = None
                for key in ['EBIT', 'Operating Income']:
                    if key in q_fin.index:
                        val = q_fin.loc[key, report_date]
                        if pd.notna(val):
                            ebit = val; break

                equity = None
                if report_date in q_bs.columns:
                    for key in ['Stockholders Equity', 'Total Equity',
                               'Common Stock Equity']:
                        if key in q_bs.index:
                            val = q_bs.loc[key, report_date]
                            if pd.notna(val):
                                equity = val; break

                net_income = None
                for key in ['Net Income', 'Net Income Common Stockholders']:
                    if key in q_fin.index:
                        val = q_fin.loc[key, report_date]
                        if pd.notna(val):
                            net_income = val; break

                roe_val = (net_income / equity) if (net_income is not None
                          and equity and equity > 0) else None

                # Approximate EV using current shares (best available
                # proxy without historical price+debt reconstruction)
                ey_val = None
                if ebit is not None and shares_out:
                    # Use market cap proxy via info (approximation —
                    # historical EV reconstruction needs price history,
                    # done at lookup time in Block 4 instead)
                    pass  # earnings_yield computed properly in Block 4

                if roe_val is not None:
                    available = pd.Timestamp(report_date) + pd.Timedelta(days=REPORTING_LAG_DAYS)
                    records.append({
                        "available_date": available,
                        "roe": roe_val,
                        "net_income": net_income,
                        "ebit": ebit,
                    })
        except Exception:
            pass

        # ── Annual statements (longer history, lower granularity) ──
        try:
            a_fin = stock.financials
            a_bs  = stock.balance_sheet
            for report_date in a_fin.columns:
                net_income = None
                for key in ['Net Income', 'Net Income Common Stockholders']:
                    if key in a_fin.index:
                        val = a_fin.loc[key, report_date]
                        if pd.notna(val):
                            net_income = val; break

                ebit = None
                for key in ['EBIT', 'Operating Income']:
                    if key in a_fin.index:
                        val = a_fin.loc[key, report_date]
                        if pd.notna(val):
                            ebit = val; break

                equity = None
                if report_date in a_bs.columns:
                    for key in ['Stockholders Equity', 'Total Equity',
                               'Common Stock Equity']:
                        if key in a_bs.index:
                            val = a_bs.loc[key, report_date]
                            if pd.notna(val):
                                equity = val; break

                roe_val = (net_income / equity) if (net_income is not None
                          and equity and equity > 0) else None

                if roe_val is not None:
                    available = pd.Timestamp(report_date) + pd.Timedelta(days=REPORTING_LAG_DAYS)
                    records.append({
                        "available_date": available,
                        "roe": roe_val,
                        "net_income": net_income,
                        "ebit": ebit,
                    })
        except Exception:
            pass

    except Exception:
        pass

    if not records:
        return pd.DataFrame(columns=["available_date", "roe", "net_income", "ebit"])

    df = pd.DataFrame(records).drop_duplicates(subset="available_date")
    df = df.sort_values("available_date").reset_index(drop=True)
    return df


print("\nDownloading point-in-time fundamental history for each ticker...")
print("(this uses quarterly + annual statements — slower than a single snapshot)\n")

fundamental_history = {}
coverage_report = []

for name, ticker in valid_tickers.items():
    hist = get_fundamental_history(ticker)
    fundamental_history[ticker] = hist
    if len(hist) > 0:
        earliest = hist["available_date"].min().date()
        latest = hist["available_date"].max().date()
        n_points = len(hist)
    else:
        earliest = latest = None
        n_points = 0
    coverage_report.append({
        "ticker": ticker, "name": name,
        "n_points": n_points, "earliest": earliest, "latest": latest,
    })

cov_df = pd.DataFrame(coverage_report)
n_with_history = (cov_df["n_points"] > 0).sum()

print(f"  Tickers with point-in-time history: {n_with_history}/{len(cov_df)}")
if n_with_history > 0:
    valid_cov = cov_df[cov_df["n_points"] > 0]
    overall_earliest = min(valid_cov["earliest"])
    overall_latest = max(valid_cov["latest"])
    median_points = valid_cov["n_points"].median()
    print(f"  Earliest available report (any ticker): {overall_earliest}")
    print(f"  Latest available report (any ticker):   {overall_latest}")
    print(f"  Median historical snapshots per ticker:  {median_points:.0f}")

print(f"\n  {'Ticker':<10} {'#Snapshots':>10} {'Earliest':>12} {'Latest':>12}")
print("  " + "-"*48)
for _, row in cov_df.head(15).iterrows():
    print(f"  {row['ticker']:<10} {row['n_points']:>10} "
          f"{str(row['earliest']):>12} {str(row['latest']):>12}")
print(f"  ... ({len(cov_df)-15} more tickers)")

print(f"\n⚠️  IMPORTANT: the usable point-in-time backtest window is now")
print(f"   constrained to dates AFTER {overall_earliest if n_with_history else 'N/A'}.")
print(f"   Quarters before this date will have NaN fundamentals and be")
print(f"   excluded from training/validation rather than backfilled.")

print("\n✅ Block 3C complete: point-in-time fundamental history built")


# ============================================================
# SECTOR MAP (unchanged — sector classification is stable over time)
# ============================================================
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
print("\n✅ Sector map loaded")
# ============================================================
# BLOCK 4 — FACTOR ENGINEERING (point-in-time version)
# Replaces the previous compute_quarterly_factors + static merge.
#
# Key change: earnings_yield and roe are looked up per ticker
# PER QUARTER using an "asof" join against fundamental_history —
# i.e. "what was the most recent reported value AVAILABLE as of
# this quarter-end" — instead of one static value reused everywhere.
#
# Sector-neutral z-scoring is also moved INSIDE the quarterly loop:
# it is now computed on the cross-section of stocks available in
# THAT quarter, not once on a single fixed table.
# ============================================================

def lookup_pit_fundamental(ticker: str, q_end: pd.Timestamp,
                           fundamental_history: dict) -> dict:
    """
    Point-in-time lookup: returns the ROE value that was actually
    AVAILABLE as of q_end for this ticker (i.e. the most recent
    report with available_date <= q_end). Returns None if no
    report had been published yet by that date.
    """
    hist = fundamental_history.get(ticker)
    if hist is None or len(hist) == 0:
        return {"roe": None, "ebit": None, "net_income": None}

    valid = hist[hist["available_date"] <= q_end]
    if len(valid) == 0:
        return {"roe": None, "ebit": None, "net_income": None}

    latest = valid.iloc[-1]
    return {
        "roe": latest["roe"],
        "ebit": latest.get("ebit"),
        "net_income": latest.get("net_income"),
    }


def compute_quarterly_factors_pit(prices: pd.DataFrame,
                                   bench_prices: pd.DataFrame,
                                   fundamental_history: dict,
                                   sector_map: dict) -> pd.DataFrame:
    """
    Point-in-time version of factor engineering.

    For each quarter:
      1. Compute price-based factors as before (already point-in-time)
      2. Look up ROE per ticker using ONLY data available as of that
         quarter (no future information)
      3. Approximate earnings yield using point-in-time EBIT and the
         market cap AT THAT QUARTER (price-based, genuinely historical)
      4. Z-score EY and ROE within sector, computed FRESH each quarter
         on whichever stocks have valid data that quarter
    """
    print("\n⚙️  Engineering point-in-time quarterly factors...")
    daily_ret = prices.pct_change().dropna(how="all")
    quarter_ends = daily_ret.resample("QE").last().index
    records = []

    for i, q_end in enumerate(quarter_ends[:-1]):
        q_next = quarter_ends[i + 1]
        quarter_rows = []

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

                # ── Price-based factors (already point-in-time) ──
                idx_6m, idx_1m = max(0, idx_end-126), max(0, idx_end-21)
                mom_6m = ((prices[ticker].iloc[idx_1m]-prices[ticker].iloc[idx_6m])
                          /prices[ticker].iloc[idx_6m]) if idx_1m>idx_6m and prices[ticker].iloc[idx_6m]>0 else np.nan

                idx_vol = max(0, idx_end-252)
                ret_window = daily_ret[ticker].iloc[idx_vol:idx_end+1].dropna()
                real_vol = ret_window.std()*np.sqrt(252) if len(ret_window)>20 else np.nan

                idx_4w = max(0, idx_end-21)
                p_4w = prices[ticker].iloc[idx_4w]
                reversal = (p_now-p_4w)/p_4w if p_4w>0 else np.nan

                # ── Point-in-time fundamental lookup ──
                pit = lookup_pit_fundamental(ticker, q_end, fundamental_history)
                roe_pit = pit["roe"]
                ebit_pit = pit["ebit"]

                # Earnings yield approximation: EBIT (as of that report)
                # divided by market cap AT THAT QUARTER (genuinely
                # historical price, no future information used)
                ey_pit = None
                if ebit_pit is not None and ebit_pit != 0:
                    # Approximate shares outstanding as roughly stable;
                    # this is a simplification — true EV needs historical
                    # debt/cash too, which free data does not provide
                    # point-in-time. Documented as a known approximation.
                    mkt_cap_proxy = p_now  # price level as relative proxy
                    ey_pit = ebit_pit / (mkt_cap_proxy * 1e9) if mkt_cap_proxy > 0 else None

                quarter_rows.append({
                    "date": q_end, "ticker": ticker, "stock_return": stock_ret,
                    "bench_return": bench_ret_q, "y": y,
                    "f_momentum_6m": mom_6m, "f_realized_vol": real_vol,
                    "f_reversal_1m": reversal,
                    "roe_raw": roe_pit, "ey_raw": ey_pit,
                    "sector": sector_map.get(ticker, "Other"),
                })
            except Exception:
                continue

        if not quarter_rows:
            continue

        q_df = pd.DataFrame(quarter_rows)

        # ── Sector-neutral z-scoring computed FRESH for this quarter ──
        for col in ["roe_raw", "ey_raw"]:
            new_col = col.replace("_raw", "_sn")
            q_df[new_col] = np.nan
            for sector in q_df["sector"].unique():
                mask = q_df["sector"] == sector
                vals = q_df.loc[mask, col].dropna()
                if len(vals) >= 3:
                    mu, sd = vals.mean(), vals.std()
                else:
                    all_vals = q_df[col].dropna()
                    mu, sd = (all_vals.mean(), all_vals.std()) if len(all_vals) > 0 else (np.nan, np.nan)
                if sd and sd > 1e-8:
                    q_df.loc[mask, new_col] = (q_df.loc[mask, col] - mu) / sd
                else:
                    q_df.loc[mask, new_col] = 0.0

        records.append(q_df)

    panel = pd.concat(records, ignore_index=True)
    panel = panel.rename(columns={"roe_sn": "roe_sn", "ey_sn": "earnings_yield_sn"})
    panel["earnings_yield_sn"] = panel["ey_sn"]
    panel = panel.drop(columns=["ey_sn"], errors="ignore")

    n_total = len(panel)
    n_with_ey = panel["earnings_yield_sn"].notna().sum()
    n_with_roe = panel["roe_sn"].notna().sum()

    print(f"   ✓ Panel: {n_total:,} stock-quarter observations")
    print(f"   ✓ Base rate (y=1): {panel['y'].mean():.3f}")
    print(f"   ✓ Earnings yield (PIT) coverage: {n_with_ey:,}/{n_total:,} ({n_with_ey/n_total*100:.1f}%)")
    print(f"   ✓ ROE (PIT) coverage:            {n_with_roe:,}/{n_total:,} ({n_with_roe/n_total*100:.1f}%)")

    # Report usable date range for fundamentals specifically
    has_both = panel.dropna(subset=["earnings_yield_sn", "roe_sn"])
    if len(has_both) > 0:
        usable_quarters = sorted(has_both["date"].unique())
        print(f"\n   📅 Usable point-in-time window: "
              f"{pd.Timestamp(usable_quarters[0]).date()} → "
              f"{pd.Timestamp(usable_quarters[-1]).date()}")
        print(f"   📅 Number of usable quarters: {len(usable_quarters)} "
              f"(vs {panel['date'].nunique()} total quarters in price history)")

    return panel


panel = compute_quarterly_factors_pit(stock_prices, bench_prices,
                                       fundamental_history, SECTOR_MAP)

FACTOR_COLS_SN = [
    'earnings_yield_sn', 'roe_sn',
    'f_momentum_6m', 'f_realized_vol', 'f_reversal_1m',
]
FACTOR_COLS = FACTOR_COLS_SN
print("\n✅ Block 4 complete: point-in-time factor panel constructed")
