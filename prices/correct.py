"""Correct Yahoo's adjusted close for spin-offs (audit finding F4.1).
Usage: python -I correct.py prices.parquet sp500_reference.parquet prices_corrected.parquet corrections.csv

Yahoo books spin-offs inconsistently: sometimes as a non-integer "split"
factor, sometimes as a split AND a cash dividend of the spun value (double
count: DHR 2016-07-05 shows +61% on a day the holder made +2%), sometimes
as nothing. Three treatments, recorded per ticker in `adj_source`:
  reference   : ticker overlaps the CRSP-verified sp500-data set. Daily
                total-return factors are taken from the reference wherever
                both series have the date pair; Yahoo factors fill the rest.
                The corrected series is chained backward from Yahoo's last
                adj_close so the latest level is unchanged.
  yahoo_fixed : no reference, but a day carries a non-simple split factor AND
                a dividend above 10% of the prior close: the dividend's
                adjustment is undone (all earlier adj_close divided by
                (1 - div / prev_close)), the split factor kept.
  yahoo       : unchanged.
Tickers with a non-simple split factor and no reference remain uncertain
(flag `spin_uncertain`); they are reported with and without downstream.
"""
import sys

import duckdb
import numpy as np
import pandas as pd

prices, ref, out, out_csv = sys.argv[1:5]
con = duckdb.connect()
r = con.execute(f"SELECT ticker, date, adj_close AS ref_adj FROM '{ref}' WHERE ticker NOT LIKE '%-______' AND adj_close > 0 ORDER BY ticker, date").df()
r["date"] = pd.to_datetime(r.date)
ref_by_ticker = {t: g.set_index("date").ref_adj for t, g in r.groupby("ticker")}
del r
tickers = [x[0] for x in con.execute(f"SELECT DISTINCT ticker FROM '{prices}' ORDER BY 1").fetchall()]
simple = sorted({n / m for n in range(1, 11) for m in range(1, 11)} | {n for n in range(1, 101)} | {1 / n for n in range(1, 101)})


def is_simple(f):
    return abs(f - 1) < 0.02 or any(abs(f / s - 1) < 0.001 for s in simple)


ref_tickers = set(ref_by_ticker)
rows = []
import pyarrow as pa
import pyarrow.parquet as pq
schema = pa.schema([("ticker", pa.string()), ("date", pa.timestamp("us")), ("adj_close", pa.float64()),
                    ("adj_close_yahoo", pa.float64()), ("adj_source", pa.string()), ("spin_uncertain", pa.bool_())])
writer = pq.ParquetWriter(out + ".adj.parquet", schema, compression="zstd")


def emit(t, dates, adj_y, corr, source, unc):
    writer.write_table(pa.table({"ticker": [t] * len(dates), "date": pd.to_datetime(dates), "adj_close": corr,
                                 "adj_close_yahoo": adj_y, "adj_source": [source] * len(dates), "spin_uncertain": [unc] * len(dates)}, schema=schema))


for t in tickers:
    g = con.execute("SELECT date, close, adj_close, dividends, stock_splits FROM read_parquet(?) WHERE ticker = ? ORDER BY date", [prices, t]).df()
    g["date"] = pd.to_datetime(g.date)
    n = len(g)
    adj = g.adj_close.values.astype(float); close = g.close.values.astype(float)
    div = g.dividends.values.astype(float); spl = g.stock_splits.values.astype(float); dates = g.date.values
    odd = np.array([s > 0 and not is_simple(s) for s in spl])
    note = ""
    if t in ref_tickers:
        rg = ref_by_ticker[t]
        # Yahoo factor between consecutive Yahoo dates; replace by reference factor when both dates are in the reference
        f = np.ones(n)
        f[1:] = adj[1:] / adj[:-1]
        in_ref = pd.Index(dates).isin(rg.index)
        both = in_ref[1:] & in_ref[:-1]
        if both.any():
            ref_vals = rg.reindex(pd.DatetimeIndex(dates)).values
            rf = ref_vals[1:] / ref_vals[:-1]
            f[1:][both] = rf[both]
        # chain backward from Yahoo's last level
        corr = np.empty(n); corr[-1] = adj[-1]
        for i in range(n - 2, -1, -1):
            corr[i] = corr[i + 1] / f[i + 1]
        emit(t, dates, adj, corr, "reference", False)
        note = f"reference factors on {int(both.sum())} of {n - 1} day pairs"
        rows.append(dict(ticker=t, adj_source="reference", odd_splits=int(odd.sum()), double_counts=0, note=note))
        continue
    # no reference: undo dividend on odd-split days with a large same-day dividend
    corr = adj.copy(); dc = 0
    for i in np.where(odd)[0]:
        if i > 0 and div[i] > 0 and close[i - 1] > 0 and div[i] / close[i - 1] > 0.10:
            factor = 1.0 - div[i] / close[i - 1]
            if 0 < factor < 1:
                corr[:i] = corr[:i] / factor
                dc += 1
    emit(t, dates, adj, corr, "yahoo_fixed" if dc else "yahoo", bool(odd.any()))
    rows.append(dict(ticker=t, adj_source="yahoo_fixed" if dc else "yahoo", odd_splits=int(odd.sum()), double_counts=dc,
                     note=f"dividend undone on {dc} double-count day(s)" if dc else ""))

writer.close()
con.execute(f"""
    COPY (SELECT p.ticker, p.date, p.open, p.high, p.low, p.close, a.adj_close, p.volume, p.dividends, p.stock_splits, p.source,
                 a.adj_close_yahoo, a.adj_source, a.spin_uncertain
          FROM '{prices}' p JOIN '{out}.adj.parquet' a USING (ticker, date) ORDER BY p.ticker, p.date)
    TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")
import os
os.remove(out + ".adj.parquet")
C = pd.DataFrame(rows)
C.to_csv(out_csv, index=False)
print(C.adj_source.value_counts().to_dict(), "double-count days undone:", int(C.double_counts.sum()),
      "uncertain (odd split, no reference):", int(((C.adj_source != 'reference') & (C.odd_splits > 0)).sum()))
