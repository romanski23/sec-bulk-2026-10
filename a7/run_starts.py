"""A7 amendment A2: run start, dead money and entry offsets per winner.
Usage: python -I run_starts.py prices.parquet winners.parquet run_starts.parquet

Monthly adj_close (last trading day of each month). For each winner (first
qualifying snapshot), searching month-ends from the 36 months before the snapshot to the end
of its window (A2 says "earliest month-end"; a winner's first qualifying
year-end often falls after its run began):
  run_start  = earliest month-end m with (a) forward 36-month total return
               annualising >= 40% and (b) no close in the following 12 months
               more than 25% below adj_close[m].
  dead_money = number of month-ends between snapshot and run_start (exclusive)
               whose forward 24-month annualised return < 20%; also their max
               drawdown from the snapshot price.
  entry offsets: annualised return to window end and share of the run's total
               gain retained, entering at run_start -12, -6, 0, +6, +12 months.
Winners whose window end is after the last price (incomplete windows) are
dated on the available history and flagged.
"""
import sys

import duckdb
import numpy as np
import pandas as pd

prices, winners, out = sys.argv[1:4]
con = duckdb.connect()
me = con.execute(f"""
    SELECT ticker, date, adj_close FROM (
        SELECT ticker, date, adj_close, ROW_NUMBER() OVER (PARTITION BY ticker, date_trunc('month', date) ORDER BY date DESC) rn
        FROM '{prices}') WHERE rn = 1 ORDER BY ticker, date
""").df()
me["date"] = pd.to_datetime(me.date)
LAST = me.date.max()
W = pd.read_parquet(winners)
series = {t: g.set_index("date").adj_close for t, g in me.groupby("ticker")}


def ann(r, months):
    return (1 + r) ** (12 / months) - 1


rows = []
for w in W.itertuples():
    s = series.get(w.ticker)
    if s is None:
        continue
    horizon = 10 if w.tag == "win_10x10y" else 5
    end = min(pd.Timestamp(f"{w.snapshot + horizon}-12-31"), LAST)
    start = pd.Timestamp(w.snap_date) - pd.DateOffset(months=36)
    s = s[s.index >= start]                      # full history for the forward-36m criterion
    v = s.values; idx = s.index
    n_all = len(v)
    n = int(np.searchsorted(idx.values, np.datetime64(end), side="right"))   # window end index (exclusive)
    snap_i = int(np.searchsorted(idx.values, np.datetime64(pd.Timestamp(w.snap_date)), side="right") - 1)
    run_start = None
    for i in range(n):
        if i + 36 >= n_all:
            break
        fwd36 = v[i + 36] / v[i] - 1
        if ann(fwd36, 36) < 0.40:
            continue
        if v[i + 1:i + 13].min() < 0.75 * v[i]:
            continue
        run_start = i
        break
    rec = dict(ticker=w.ticker, tag=w.tag, snapshot=w.snapshot, snap_date=w.snap_date, window_end=end,
               window_complete=end < LAST, months_available=n_all, run_start=None)
    if run_start is None:
        rec["run_start_found"] = False
        rows.append(rec)
        continue
    rec["run_start_found"] = True
    rec["run_start"] = idx[run_start]
    rec["months_to_run_start"] = run_start - snap_i  # negative = run began before the snapshot
    # dead money: month-ends from the snapshot to the run start (A2); empty when the run began before the snapshot
    dm = 0
    for i in range(snap_i, run_start):
        j = min(i + 24, n_all - 1)
        if j > i and ann(v[j] / v[i] - 1, j - i) < 0.20:
            dm += 1
    rec["dead_money_months"] = dm
    rec["drawdown_before_run"] = v[snap_i:run_start + 1].min() / v[snap_i] - 1 if run_start > snap_i else 0.0
    rec["run_total_gain"] = v[n - 1] / v[run_start] - 1
    rec["run_peak_mult"] = v[run_start:n].max() / v[run_start]
    for off in (-12, -6, 0, 6, 12):
        k = run_start + off
        if 0 <= k < n - 1:
            months = n - 1 - k
            rec[f"entry_{off:+d}_ann"] = ann(v[n - 1] / v[k] - 1, months)
            rec[f"entry_{off:+d}_gain_share"] = (v[n - 1] / v[k] - 1) / rec["run_total_gain"] if rec["run_total_gain"] > 0 else np.nan
        else:
            rec[f"entry_{off:+d}_ann"] = np.nan
            rec[f"entry_{off:+d}_gain_share"] = np.nan
    rows.append(rec)
R = pd.DataFrame(rows)
R.to_parquet(out, index=False)
f = R[R.run_start_found]
print(f"winners {len(R)}, run start found {len(f)} ({R.run_start_found.mean():.0%}); windows complete {R.window_complete.mean():.0%}")
for tag, g in f.groupby("tag"):
    print(f"\n{tag}: n={len(g)}")
    print(f"  months from snapshot to run start: median {g.months_to_run_start.median():.0f}, "
          f"share >= 12 months: {(g.months_to_run_start >= 12).mean():.0%}")
    print(f"  dead-money months before run start: median {g.dead_money_months.median():.0f}, "
          f"share with >= 12: {(g.dead_money_months >= 12).mean():.0%}")
    print(f"  drawdown from snapshot before run start: median {g.drawdown_before_run.median():.0%}")
    for off in (-12, -6, 0, 6, 12):
        a = g[f"entry_{off:+d}_ann"]; gs = g[f"entry_{off:+d}_gain_share"]
        print(f"  entry at run start {off:+d}m: median annualised {a.median():.0%}, median share of run gain kept {gs.median():.0%} (n={a.notna().sum()})")
