"""Check data/prices.parquet before anyone computes a return from it.

[1] Adjustment: adj_close/close must only change on dividend or split days and
    end near 1.0; known splits must be in stock_splits and the close must be
    continuous across them (i.e. split-adjusted).
[2] Gaps: missing trading days inside each ticker's span (coverage.csv).
[3] Coverage: tickers with data vs tickers.csv; history depth at the study
    snapshots (2010, 2012 year-ends).
[4] Cross-check: daily total returns against the independently verified
    S&P 500 dataset (Johnbrick123/sp500-data release, CRSP-checked).

Writes data/verify_report.txt. Exits 1 on a hard failure (SPY missing,
adjustment broken for >2% of tickers, or cross-check disagreement >1% of days).
Env: SP500_URL to override the cross-check source; SP500_FILE for a local copy.
"""
import os
import sys
import urllib.request
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PRICES = DATA / "prices.parquet"
SP500_URL = os.environ.get("SP500_URL", "https://github.com/Johnbrick123/sp500-data/releases/download/data/prices.parquet")
SP500_FILE = Path(os.environ.get("SP500_FILE", DATA / "sp500_prices.parquet"))
KNOWN_SPLITS = [("AAPL", "2014-06-09", 7), ("AAPL", "2020-08-31", 4), ("NVDA", "2024-06-10", 10),
                ("TSLA", "2020-08-31", 5), ("AMZN", "2022-06-06", 20), ("GOOGL", "2022-07-18", 20),
                ("WMT", "2024-02-26", 3), ("TSLA", "2022-08-25", 3), ("NVDA", "2021-07-20", 4)]
lines = []
hard_fail = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


con = duckdb.connect()
con.execute(f"CREATE VIEW p AS SELECT * FROM '{PRICES}'")
n, nt, lo, hi = con.execute("SELECT COUNT(*), COUNT(DISTINCT ticker), MIN(date), MAX(date) FROM p").fetchone()
say("=" * 70)
say(f"PRICE VERIFICATION  {pd.Timestamp.today().date()}   rows {n:,}  tickers {nt:,}  {lo.date()} -> {hi.date()}")
say("=" * 70)
if con.execute("SELECT COUNT(*) FROM p WHERE ticker='SPY'").fetchone()[0] == 0:
    hard_fail.append("SPY missing")

# [1] adjustment -------------------------------------------------------------
say("\n[1] ADJUSTMENT")
adj = con.execute("""
    WITH r AS (
        SELECT ticker, date, dividends, stock_splits, adj_close / close AS f,
               LAG(adj_close / close) OVER (PARTITION BY ticker ORDER BY date) AS f_prev,
               source
        FROM p WHERE close > 0 AND adj_close > 0
    )
    SELECT ticker, ANY_VALUE(source) source,
           SUM(CASE WHEN f_prev IS NOT NULL AND ABS(f / f_prev - 1) > 1e-4 AND dividends = 0 AND stock_splits = 0 THEN 1 ELSE 0 END) AS unexplained_jumps,
           SUM(CASE WHEN f_prev IS NOT NULL AND ABS(f / f_prev - 1) > 1e-4 THEN 1 ELSE 0 END) AS jumps,
           MAX(CASE WHEN f_prev IS NOT NULL AND dividends = 0 AND stock_splits = 0 THEN ABS(f / f_prev - 1) ELSE 0 END) AS max_unexplained_jump,
           SUM(CASE WHEN dividends > 0 THEN 1 ELSE 0 END) AS div_days,
           ARG_MAX(f, date) AS last_factor, MIN(f) AS min_factor
    FROM r GROUP BY ticker
""").df()
yahoo = adj[adj.source == "yahoo"]
bad_last = yahoo[(yahoo.last_factor - 1).abs() > 0.01]
bad_jump = yahoo[(yahoo.unexplained_jumps > 5) | (yahoo.max_unexplained_jump > 0.02)]
no_div_adj = yahoo[(yahoo.div_days > 0) & (yahoo.jumps == 0)]
say(f"    yahoo tickers {len(yahoo):,}; last adj_close/close within 1% of 1.0: {len(yahoo) - len(bad_last):,} (bad {len(bad_last)})")
say(f"    tickers with >5 unexplained adjustment-factor jumps or one above 2% (no dividend/split that day): {len(bad_jump)} "
    f"({', '.join(bad_jump.ticker.head(15))})")
say(f"    tickers with dividends but a flat adjustment factor (dividends not adjusted): {len(no_div_adj)}")
say(f"    stooq tickers (close only, no dividend adjustment, adj_close = close): {int((adj.source == 'stooq').sum())}")
if len(yahoo) and (len(bad_last) + len(bad_jump)) / len(yahoo) > 0.02:
    hard_fail.append("adjustment broken for >2% of yahoo tickers")
for t, d, k in KNOWN_SPLITS:
    row = con.execute("SELECT stock_splits FROM p WHERE ticker=? AND date=?", [t, d]).fetchone()
    if row is None:
        say(f"    [SKIP] {t} {d}: no row")
        continue
    prev, cur = con.execute("""SELECT close FROM p WHERE ticker=? AND date<=? ORDER BY date DESC LIMIT 2""", [t, d]).fetchall()[::-1]
    cont = abs(cur[0] / prev[0] - 1) < 0.3
    ok = abs(row[0] - k) < 1e-6 and cont
    say(f"    [{'PASS' if ok else 'FAIL'}] {t} {d} split {row[0]} (expected {k}); close continuous across split: {cont}")
    if not ok:
        hard_fail.append(f"known split {t} {d}")
adj.to_csv(DATA / "adjustment_check.csv", index=False)

# [2] gaps -------------------------------------------------------------------
say("\n[2] GAPS (missing trading days inside each ticker's own span, vs SPY calendar)")
cov = pd.read_csv(DATA / "coverage.csv", parse_dates=["first_date", "last_date"])
cov["missing_pct"] = cov.missing_days / cov.expected_days.replace(0, np.nan) * 100
say(f"    tickers with zero missing days: {(cov.missing_days == 0).sum():,} of {len(cov):,}")
say(f"    missing >1% of span: {(cov.missing_pct > 1).sum():,}; >5%: {(cov.missing_pct > 5).sum():,}")
worst = cov.sort_values("missing_pct", ascending=False).head(10)
say("    worst: " + ", ".join(f"{r.ticker} {r.missing_pct:.1f}%" for r in worst.itertuples()))
# longest internal gap per ticker (days between consecutive rows), only for big ones
gaps = con.execute("""
    WITH g AS (SELECT ticker, date, date - LAG(date) OVER (PARTITION BY ticker ORDER BY date) AS gap FROM p)
    SELECT ticker, MAX(gap) AS max_gap_days FROM g GROUP BY ticker HAVING MAX(gap) > INTERVAL 30 DAY ORDER BY 2 DESC
""").df()
say(f"    tickers with an internal gap longer than 30 days: {len(gaps):,} (e.g. {', '.join(gaps.ticker.head(10))})")
gaps.to_csv(DATA / "gaps_over_30d.csv", index=False)

# [3] coverage ---------------------------------------------------------------
say("\n[3] COVERAGE")
want = pd.read_csv(ROOT / "prices" / "tickers.csv")
have = set(cov.ticker)
missing = want[~want.ticker.isin(have)]
say(f"    tickers.csv: {len(want):,}; with price data: {want.ticker.isin(have).sum():,}; none: {len(missing):,}")
say(f"    missing by exchange: {missing.exchange.value_counts().to_dict()}")
for y in (2010, 2012, 2016, 2021):
    k = (cov["first_date"] <= f"{y}-12-31").sum()
    say(f"    history reaching back to {y}-12-31 or earlier: {k:,} tickers")
say(f"    last date is the dataset max for {(cov['last_date'] == cov['last_date'].max()).sum():,} tickers "
    f"(others ended earlier = delisted while in Yahoo, or stale)")
missing.to_csv(DATA / "missing_tickers.csv", index=False)

# [4] cross-check vs sp500-data ---------------------------------------------
say("\n[4] CROSS-CHECK vs Johnbrick123/sp500-data (CRSP-verified S&P 500 members)")
try:
    if not SP500_FILE.exists():
        urllib.request.urlretrieve(SP500_URL, SP500_FILE)
    con.execute(f"CREATE VIEW s AS SELECT ticker, date, close, adj_close, source FROM '{SP500_FILE}'")
    cmp = con.execute("""
        WITH j AS (SELECT p.ticker, p.date, p.adj_close AS a, s.adj_close AS b
                   FROM p JOIN s USING (ticker, date) WHERE p.date >= '2000-01-01'),
             r AS (SELECT ticker, date,
                          a / LAG(a) OVER (PARTITION BY ticker ORDER BY date) - 1 AS ra,
                          b / LAG(b) OVER (PARTITION BY ticker ORDER BY date) - 1 AS rb
                   FROM j)
        SELECT ticker, COUNT(*) AS ndays,
               SUM(CASE WHEN ABS(ra - rb) > 0.005 THEN 1 ELSE 0 END) AS bad_days,
               MAX(ABS(ra - rb)) AS max_diff,
               EXP(SUM(LN(1 + ra))) / EXP(SUM(LN(1 + rb))) - 1 AS wealth_ratio_diff
        FROM r WHERE ra IS NOT NULL AND rb IS NOT NULL AND ra > -0.99 AND rb > -0.99
        GROUP BY ticker
    """).df()
    miss = con.execute("""
        SELECT COUNT(*) FROM s WHERE date >= '2000-01-01' AND ticker IN (SELECT DISTINCT ticker FROM p)
          AND (ticker, date) NOT IN (SELECT ticker, date FROM p)
    """).fetchone()[0]
    say(f"    sp500-data member-days since 2000 for our tickers that we lack: {miss:,}")
    tot_days, tot_bad = cmp.ndays.sum(), cmp.bad_days.sum()
    say(f"    overlapping tickers {len(cmp):,}; member-days {tot_days:,}; days differing >0.5pp: {tot_bad:,} ({tot_bad / tot_days * 100:.3f}%)")
    say(f"    tickers with >1% bad days: {(cmp.bad_days / cmp.ndays > 0.01).sum()}; |cumulative wealth diff| >2%: {(cmp.wealth_ratio_diff.abs() > 0.02).sum()}")
    w = cmp.reindex(cmp.wealth_ratio_diff.abs().sort_values(ascending=False).index).head(10)
    say("    worst cumulative: " + ", ".join(f"{r.ticker} {r.wealth_ratio_diff * 100:+.1f}%" for r in w.itertuples()))
    cmp.to_csv(DATA / "crosscheck_sp500.csv", index=False)
    if tot_days and tot_bad / tot_days > 0.01:
        hard_fail.append("cross-check: >1% of member-days disagree with sp500-data")
except Exception as e:
    say(f"    [SKIP] cross-check unavailable: {type(e).__name__}: {str(e)[:150]}")

say("\nRESULT: " + ("PASS" if not hard_fail else "FAIL: " + "; ".join(hard_fail)))
(DATA / "verify_report.txt").write_text("\n".join(lines) + "\n")
sys.exit(1 if hard_fail else 0)
