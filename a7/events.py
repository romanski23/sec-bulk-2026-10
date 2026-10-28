"""A7 amendment A2: events E1-E7 for every company-quarter, dated to publication.
Usage: python -I events.py quarterly.parquet prices.parquet tickers_primary.csv events.parquet

E1  TTM operating income turns positive after >= 4 negative quarters
E2  TTM free cash flow turns positive after >= 4 negative quarters
E3  TTM revenue growth (y/y) accelerates by >= 10 points vs a year earlier
E4  TTM gross margin up >= 3 points y/y for two consecutive quarters
E5  incremental operating margin (dTTM op income / dTTM revenue, y/y) > 25% for two consecutive quarters
E6  share-count growth (y/y, cover-page shares) falls below 3% after >= 4 quarters above 5%
E7  price makes a new 52-week high after >= 12 months without one (monthly closes), dated to that month-end
Each fundamental event fires at `avail` of the quarter that completes the condition.
Output: one row per (cik, ticker, event_date, event) plus the quarter end.
"""
import sys

import duckdb
import numpy as np
import pandas as pd

quarterly, prices, tickers_csv, out = sys.argv[1:5]
q = pd.read_parquet(quarterly, columns=["cik", "end", "filed", "avail", "ttm_revenue", "ttm_op_income", "ttm_fcf", "ttm_gross_profit", "cover_shares"])
q = q.rename(columns={"cover_shares": "shares_out"})   # E6 on one share source only (audit F6.8)
q = q[q.avail.notna() & (q.avail >= q["end"])].sort_values(["cik", "end"]).reset_index(drop=True)
g = q.groupby("cik", sort=False)


def lag(col, k):
    return g[col].shift(k)


# consecutive-quarter logic uses positional lags; guard that lag-4 is ~1 year back
gap4 = (q["end"] - lag("end", 4)).dt.days
ok4 = (gap4 >= 330) & (gap4 <= 400)
gap1 = (q["end"] - lag("end", 1)).dt.days
ok1 = (gap1 >= 60) & (gap1 <= 120)

neg_oi = (q.ttm_op_income < 0)
prev4_neg_oi = pd.concat([lag("ttm_op_income", k) < 0 for k in (1, 2, 3, 4)], axis=1).all(axis=1)
q["E1"] = (q.ttm_op_income > 0) & prev4_neg_oi & ok4

prev4_neg_fcf = pd.concat([lag("ttm_fcf", k) < 0 for k in (1, 2, 3, 4)], axis=1).all(axis=1)
q["E2"] = (q.ttm_fcf > 0) & prev4_neg_fcf & ok4

gap8 = (q["end"] - lag("end", 8)).dt.days
ok8 = (gap8 >= 690) & (gap8 <= 770)
REV_FLOOR = 10e6   # bases below US$10M make growth rates meaningless (audit F6.3)
rev_g = q.ttm_revenue / lag("ttm_revenue", 4) - 1
rev_g_prev = lag("ttm_revenue", 4) / lag("ttm_revenue", 8) - 1
q["rev_growth"] = rev_g
q["E3"] = ((rev_g - rev_g_prev) >= 0.10) & ok4 & ok8 & (q.ttm_revenue > REV_FLOOR) & (lag("ttm_revenue", 4) > REV_FLOOR) & (lag("ttm_revenue", 8) > REV_FLOOR)

gm = q.ttm_gross_profit / q.ttm_revenue
gm_yoy = gm - (lag("ttm_gross_profit", 4) / lag("ttm_revenue", 4))
gm_yoy_prev = gm_yoy.groupby(q.cik).shift(1)
q["E4"] = (gm_yoy >= 0.03) & (gm_yoy_prev >= 0.03) & ok4 & ok1

d_oi = q.ttm_op_income - lag("ttm_op_income", 4)
d_rev = q.ttm_revenue - lag("ttm_revenue", 4)
inc_m = np.where(d_rev > 0, d_oi / d_rev, np.nan)
inc_m = pd.Series(inc_m, index=q.index)
inc_m_prev = inc_m.groupby(q.cik).shift(1)
q["E5"] = (inc_m > 0.25) & (inc_m_prev > 0.25) & ok4 & ok1

sh_g = q.shares_out / lag("shares_out", 4) - 1
sh_prev_high = pd.concat([sh_g.groupby(q.cik).shift(k) > 0.05 for k in (1, 2, 3, 4)], axis=1).all(axis=1)
q["E6"] = (sh_g < 0.03) & sh_prev_high & ok4

ev = []
for e in ["E1", "E2", "E3", "E4", "E5", "E6"]:
    sub = q.loc[q[e].fillna(False), ["cik", "end", "avail"]].rename(columns={"end": "quarter_end", "avail": "event_date"})
    sub["event"] = e
    ev.append(sub)
F = pd.concat(ev, ignore_index=True)
tk = pd.read_csv(tickers_csv)
tk = tk[tk.is_primary][["cik", "ticker"]]
F = F.merge(tk, on="cik", how="left")

# E7 from monthly closes
con = duckdb.connect()
me = con.execute(f"""
    SELECT ticker, date, adj_close FROM (
        SELECT ticker, date, adj_close, ROW_NUMBER() OVER (PARTITION BY ticker, date_trunc('month', date) ORDER BY date DESC) rn
        FROM '{prices}') WHERE rn = 1 ORDER BY ticker, date
""").df()
me["date"] = pd.to_datetime(me.date)
me = me[me.date < me.date.max().to_period("M").to_timestamp()]   # drop the partial current month (audit F6.6)
rows = []
for t, s in me.groupby("ticker"):
    v = s.adj_close.values; d = s.date.values
    if len(v) < 24:
        continue
    hi12 = pd.Series(v).rolling(12, min_periods=12).max().shift(1).values  # prior 12 month-ends
    is_high = v > hi12
    last_high = -10**6
    for i in range(12, len(v)):
        if is_high[i]:
            if i - last_high >= 12:
                rows.append((t, d[i]))
            last_high = i
E7 = pd.DataFrame(rows, columns=["ticker", "event_date"]).merge(tk, on="ticker", how="inner")   # primary tickers only (audit F3.1)
E7["event"] = "E7"; E7["quarter_end"] = pd.NaT
E7["event_date"] = pd.to_datetime(E7.event_date)
F = pd.concat([F, E7[["cik", "quarter_end", "event_date", "event", "ticker"]]], ignore_index=True)
F["cik"] = F.cik.astype("Int64")
F.to_parquet(out, index=False)
print(F.groupby("event").agg(n=("cik", "size"), companies=("cik", "nunique")).to_string())
