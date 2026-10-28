"""A7 step 4a: snapshot universe, forward returns, winner tags.
Usage: python -I build_universe.py prices.parquet quarterly.parquet outcomes.parquet tickers.csv out_dir

Per year-end snapshot Y (2010..2021):
- Universe: tickers with a price on the last trading day of Y, a CIK, and
  market cap >= 300M. Cap = unadjusted close * shares outstanding, where
  "unadjusted close" = Yahoo split-adjusted close * cumulative split factor of
  all splits AFTER the snapshot, and shares = latest SEC cover-page count whose
  filing (`avail`) is public by the snapshot date.
- Forward total return from adj_close at the snapshot to the max adj_close
  within the window? NO: winner = >=10x total return at any month-end within
  10 years, or >=5x within 5 years (pre-registration §1 says "within"; the
  peak-based definition is recorded as such and the end-of-window multiple is
  stored alongside so both can be reported).
- Dead companies: price history ends before the window end and outcomes.parquet
  shows a bankruptcy 8-K / Form 15 / Form 25 -> flagged; return from the last
  price is treated per amendment A1 downstream.
- One count per winner: first qualifying snapshot.
Writes universe.parquet (one row per ticker-snapshot) and winners.parquet.
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

prices, quarterly, outcomes, tickers_csv, out = sys.argv[1:6]
cik_links = sys.argv[6] if len(sys.argv) > 6 else None
out = Path(out); out.mkdir(parents=True, exist_ok=True)
STALE_DAYS = 455      # fundamentals older than 15 months before the snapshot are treated as missing (audit F6.2)
con = duckdb.connect()
con.execute(f"CREATE VIEW p AS SELECT * FROM '{prices}'")
cal = con.execute("SELECT date FROM p WHERE ticker='SPY' ORDER BY date").df()["date"]
LAST_DATE = cal.max()
SNAPS = list(range(2010, 2022))
snap_dates = {y: cal[cal <= f"{y}-12-31"].max() for y in SNAPS}

# month-end adj_close per ticker (for forward returns at monthly resolution)
me = con.execute("""
    SELECT ticker, date, adj_close, close FROM (
        SELECT ticker, date, adj_close, close,
               ROW_NUMBER() OVER (PARTITION BY ticker, date_trunc('month', date) ORDER BY date DESC) rn
        FROM p) WHERE rn = 1
""").df()
me["date"] = pd.to_datetime(me["date"])
me = me[me.date < LAST_DATE.to_period("M").to_timestamp()]   # drop the partial current month (audit F6.6)
me = me.sort_values(["ticker", "date"])
# corrected-price flags (prices/correct.py): adj_source per ticker and spin_uncertain
pflags = con.execute("SELECT ticker, ANY_VALUE(adj_source) AS adj_source, BOOL_OR(spin_uncertain) AS spin_uncertain FROM p GROUP BY ticker").df() \
    if "adj_source" in [c[0] for c in con.execute("DESCRIBE p").fetchall()] else None
# cumulative later split factor: product of stock_splits (>0) strictly after date
splits = con.execute("SELECT ticker, date, stock_splits FROM p WHERE stock_splits > 0 ORDER BY ticker, date").df()
splits["date"] = pd.to_datetime(splits["date"])
# liquidity: median daily dollar volume over the 63 trading days ending at each snapshot
liq = con.execute("""
    SELECT ticker, snap, MEDIAN(close * volume) AS dollar_vol_63d FROM (
        SELECT p.ticker, s.snap, p.close, p.volume,
               ROW_NUMBER() OVER (PARTITION BY p.ticker, s.snap ORDER BY p.date DESC) rn
        FROM p JOIN (SELECT UNNEST(?) AS snap) s ON p.date <= s.snap AND p.date > s.snap - INTERVAL 120 DAY
    ) WHERE rn <= 63 GROUP BY ticker, snap
""", [[pd.Timestamp(d) for d in snap_dates.values()]]).df()
liq["snap"] = pd.to_datetime(liq["snap"])
# special dividends / spin-offs booked as dividends: any single dividend above 25% of that day's close
bigdiv = con.execute("SELECT ticker, MIN(date) AS first_big_div, MAX(dividends / close) AS max_div_ratio FROM p WHERE close > 0 AND dividends / close > 0.25 GROUP BY ticker").df()
bigdiv["first_big_div"] = pd.to_datetime(bigdiv["first_big_div"])
# spin-offs booked by Yahoo as split factors (e.g. DELL 1.973 for VMware, GE 1.253 for the 2023 spin). Ordinary splits,
# reverse splits and stock dividends are simple ratios (n/m with n, m <= 10; integers and 1/integers to 100; within 2% of 1); anything else is flagged.
allsplits = con.execute("SELECT ticker, date, stock_splits FROM p WHERE stock_splits > 0").df()
simple = sorted({n / m for n in range(1, 11) for m in range(1, 11)} | {n for n in range(1, 101)} | {1 / n for n in range(1, 101)})
def is_simple(f):
    return abs(f - 1) < 0.02 or any(abs(f / r - 1) < 0.001 for r in simple)
allsplits["odd"] = ~allsplits.stock_splits.map(is_simple)
oddsplit = allsplits[allsplits.odd].groupby("ticker").agg(first_odd_split=("date", "min"), n_odd_splits=("date", "size")).reset_index()
oddsplit["first_odd_split"] = pd.to_datetime(oddsplit["first_odd_split"])
# last price per ticker
lastpx = con.execute("SELECT ticker, MAX(date) last_date, MIN(date) first_date FROM p GROUP BY ticker").df()
lastpx["last_date"] = pd.to_datetime(lastpx["last_date"]); lastpx["first_date"] = pd.to_datetime(lastpx["first_date"])

tk = pd.read_csv(tickers_csv)
if "is_primary" in tk:
    tk = tk[tk.is_primary]
tk = tk[["ticker", "cik", "exchange", "name", "sic"]]
tk["cik"] = tk["cik"].astype(int)
links = pd.read_csv(cik_links) if cik_links else pd.DataFrame(columns=["ticker", "pred_cik", "succ_first_10k"])
links = links[links.pred_cik.notna()]
links["pred_cik"] = links.pred_cik.astype(int)
links["succ_first_10k"] = pd.to_datetime(links.succ_first_10k)
q = pd.read_parquet(quarterly, columns=["cik", "end", "filed", "bs_filed", "avail", "shares_out", "cover_shares", "public_float", "cover_date", "cover_filed",
                                        "ttm_revenue", "ttm_op_income", "ttm_fcf",
                                        "ttm_net_income", "ttm_cfo", "ttm_capex", "ttm_pretax_income", "assets", "equity", "cash", "lt_debt", "st_debt",
                                        "q_cfo", "q_capex", "ttm_sbc", "ttm_rnd", "ttm_gross_profit", "q_diluted_shares", "q_revenue_src", "q_op_income_src"])
floats = q.loc[q.public_float.notna() & q.cover_filed.notna() & (q.public_float > 0), ["cik", "cover_date", "cover_filed", "public_float"]].drop_duplicates()
q = q[q.filed.notna() & (q.filed >= q["end"])]
q["bs_filed"] = q.bs_filed.where(q.bs_filed.notna(), q.filed)
STOCK_COLS = ["cik", "end", "bs_filed", "shares_out", "cover_shares", "public_float", "assets", "equity", "cash", "lt_debt", "st_debt", "q_diluted_shares", "q_cfo", "q_capex"]
FLOW_COLS = ["cik", "end", "avail", "ttm_revenue", "ttm_op_income", "ttm_fcf", "ttm_net_income", "ttm_cfo", "ttm_capex", "ttm_pretax_income", "ttm_sbc", "ttm_rnd", "ttm_gross_profit", "q_revenue_src", "q_op_income_src"]
oc = pd.read_parquet(outcomes)


def later_split_factors(date):
    s = splits[splits.date > date]
    return s.groupby("ticker").stock_splits.prod()


rows = []
for y in SNAPS:
    d = snap_dates[y]
    px = me[me.date == d][["ticker", "adj_close", "close"]]
    u = px.merge(tk, on="ticker", how="inner")
    # predecessor CIK for snapshots before the successor's first 10-K (audit F5.3)
    u["cik_listed"] = u.cik
    lk = links[links.succ_first_10k > d][["ticker", "pred_cik"]]
    u = u.merge(lk, on="ticker", how="left")
    u["cik"] = u.pred_cik.fillna(u.cik).astype(int)
    u["cik_source"] = np.where(u.pred_cik.notna(), "predecessor", "listed")
    u = u.drop(columns=["pred_cik"])
    # latest fundamentals public by d, not older than STALE_DAYS (audit F6.2)
    qs = q.loc[(q.bs_filed <= d) & (q["end"] >= d - pd.Timedelta(days=STALE_DAYS)), STOCK_COLS] \
          .sort_values(["cik", "end"]).drop_duplicates("cik", keep="last").rename(columns={"end": "bs_end"})
    qf = q.loc[(q.avail <= d) & (q["end"] >= d - pd.Timedelta(days=STALE_DAYS)), FLOW_COLS].dropna(subset=["ttm_revenue", "ttm_op_income", "ttm_cfo"], how="all") \
          .sort_values(["cik", "end"]).drop_duplicates("cik", keep="last").rename(columns={"end": "ttm_end"})
    fl_d = floats[(floats.cover_filed <= d) & (floats.cover_date >= d - pd.Timedelta(days=730))].sort_values(["cik", "cover_date"]).drop_duplicates("cik", keep="last")
    u = u.merge(qs, on="cik", how="left").merge(qf, on="cik", how="left").merge(fl_d[["cik", "public_float"]].rename(columns={"public_float": "float_latest"}), on="cik", how="left")
    u["split_factor_after"] = u.ticker.map(later_split_factors(d)).fillna(1.0)
    u["close_unadj"] = u.close * u.split_factor_after
    # share count chain (audit F5.1/F5.2): cover count -> balance-sheet count -> diluted weighted average -> public float as a
    # cap proxy for multi-class issuers whose total is not in companyfacts (labelled approximate)
    bs_shares = u.shares_out.where(u.cover_shares.isna(), np.nan)
    u["shares_used"] = u.cover_shares.where(u.cover_shares.notna(), bs_shares.where(bs_shares.notna(), u.q_diluted_shares))
    u["shares_source"] = np.where(u.cover_shares.notna(), "cover", np.where(bs_shares.notna(), "balance_sheet",
                         np.where(u.q_diluted_shares.notna(), "diluted_wavg", np.where(u.float_latest.notna(), "public_float_approx", "none"))))
    u["mcap"] = u.close_unadj * u.shares_used
    u["mcap"] = u.mcap.where(u.mcap.notna(), u.float_latest)
    u["mcap_approx"] = u.shares_source == "public_float_approx"
    u["snapshot"] = y
    u["snap_date"] = d
    u = u.merge(liq[liq.snap == d][["ticker", "dollar_vol_63d"]], on="ticker", how="left")
    # forward paths
    fut = me[(me.date > d)]
    for horizon, label in ((5, "5y"), (10, "10y")):
        end = pd.Timestamp(f"{y + horizon}-12-31")
        f = fut[fut.date <= end].groupby("ticker").agg(peak=("adj_close", "max"), last_date=("date", "max"),
                                                       last_adj=("adj_close", "last"))
        f = f.rename(columns={"peak": f"peak_{label}", "last_date": f"lastdate_{label}", "last_adj": f"end_adj_{label}"})
        u = u.merge(f, left_on="ticker", right_index=True, how="left")
        u[f"mult_peak_{label}"] = u[f"peak_{label}"] / u.adj_close
        u[f"mult_end_{label}"] = u[f"end_adj_{label}"] / u.adj_close
        u[f"window_complete_{label}"] = end <= LAST_DATE
    rows.append(u)
U = pd.concat(rows, ignore_index=True)
U = U.merge(lastpx, on="ticker", how="left")
U = U.merge(oc[["cik", "first_10k", "last_10kq", "form15", "form25", "bankruptcy_8k"]], on="cik", how="left")
U = U.merge(bigdiv, on="ticker", how="left").merge(oddsplit, on="ticker", how="left")
if pflags is not None:
    U = U.merge(pflags, on="ticker", how="left")
else:
    U["adj_source"] = "yahoo"; U["spin_uncertain"] = U.first_odd_split.notna()
U["first_10k"] = pd.to_datetime(U.first_10k); U["last_10kq"] = pd.to_datetime(U.last_10kq)
U["filer_active"] = (U.first_10k <= U.snap_date) & (U.last_10kq >= U.snap_date - pd.Timedelta(days=400))
U["liquid"] = U.dollar_vol_63d >= 1e6
U["big_dividend"] = U.first_big_div.notna()
U["odd_split"] = U.first_odd_split.notna()
U["corp_action_flag"] = U.big_dividend | U.odd_split
U["eligible"] = (U.mcap >= 300e6) & U.mcap.notna() & U.filer_active & U.liquid
U["win_10x10y"] = U.eligible & (U.mult_peak_10y >= 10)
U["win_5x5y"] = U.eligible & (U.mult_peak_5y >= 5)
# headline sample (audit F3.3): complete windows only; incomplete snapshots are right-censored and reported separately
U["headline_10y"] = U.eligible & U.window_complete_10y
U["headline_5y"] = U.eligible & U.window_complete_5y
# flags inside the forward window (corporate actions that could distort the outcome), reported with and without
U["flag_in_window_10y"] = ((U.first_big_div > U.snap_date) & (U.first_big_div <= U.snap_date + pd.DateOffset(years=10))) | \
                          ((U.first_odd_split > U.snap_date) & (U.first_odd_split <= U.snap_date + pd.DateOffset(years=10)))
U["flag_in_window_5y"] = ((U.first_big_div > U.snap_date) & (U.first_big_div <= U.snap_date + pd.DateOffset(years=5))) | \
                         ((U.first_odd_split > U.snap_date) & (U.first_odd_split <= U.snap_date + pd.DateOffset(years=5)))
U["outcome_uncertain"] = (U.adj_source != "reference") & (U.flag_in_window_10y | U.flag_in_window_5y)
U["dead"] = (U.last_date < LAST_DATE - pd.Timedelta(days=10))
U.to_parquet(out / "universe.parquet", index=False)

# first qualifying snapshot per ticker
w = []
for tag in ("win_10x10y", "win_5x5y"):
    ww = U[U[tag]].sort_values(["ticker", "snapshot"]).drop_duplicates("ticker")
    ww = ww.assign(tag=tag)
    w.append(ww)
W = pd.concat(w, ignore_index=True)
W.to_parquet(out / "winners.parquet", index=False)

# summary
print("snapshot  priced  cap>=300M  eligible  no_shares  float_approx  pred_cik  win10x10y  win5x5y  10y_complete")
for y in SNAPS:
    s = U[U.snapshot == y]
    print(f"{y}  {len(s):6d}  {int((s.mcap >= 300e6).sum()):6d}  {int(s.eligible.sum()):6d}  {int((s.shares_source == 'none').sum()):6d}  "
          f"{int((s.eligible & s.mcap_approx).sum()):6d}  {int((s.eligible & (s.cik_source == 'predecessor')).sum()):5d}  "
          f"{int(s.win_10x10y.sum()):5d}  {int(s.win_5x5y.sum()):5d}  {bool(s.window_complete_10y.iloc[0]) if len(s) else None}")
print("unique winners:", W.groupby("tag").ticker.nunique().to_dict(),
      " headline (complete windows):", {t: int(W[(W.tag == t) & W["window_complete_10y" if t == "win_10x10y" else "window_complete_5y"]].ticker.nunique()) for t in W.tag.unique()})
print("outcome_uncertain (no reference series and a corporate-action flag inside the window):", W[W.outcome_uncertain].groupby("tag").ticker.nunique().to_dict(),
      " adj_source:", W.adj_source.value_counts().to_dict())
