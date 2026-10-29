"""A7 amendment A7: point-in-time feature set for arbitrary (ticker, cik, obs_date) rows.
Usage (as a module): build_monthly(prices) -> monthly table; features(obs, quarterly, monthly, filings) -> DataFrame.
Everything uses only information public at obs_date: fundamentals via avail/bs_filed, prices up to the month-end,
8-K filings by filing date.
"""
import duckdb
import numpy as np
import pandas as pd

STALE = 455


def build_monthly(prices):
    """Month-end price features per ticker (last trading day of each calendar month, partial month dropped)."""
    con = duckdb.connect()
    m = con.execute(f"""
        WITH d AS (
            SELECT ticker, date, close, adj_close, close * volume AS dv, stock_splits, date_trunc('month', date) AS mo
            FROM '{prices}' WHERE close > 0 AND adj_close > 0),
        me AS (
            SELECT ticker, mo, ARG_MAX(close, date) AS close, ARG_MAX(adj_close, date) AS adj, MAX(adj_close) AS adj_hi_m, SUM(dv) AS dv_m, MAX(date) AS month_end
            FROM d GROUP BY ticker, mo)
        SELECT * FROM me ORDER BY ticker, mo
    """).df()
    m["mo"] = pd.to_datetime(m.mo).astype("datetime64[ns]"); m["month_end"] = pd.to_datetime(m.month_end).astype("datetime64[ns]")
    last = m.mo.max(); m = m[m.mo < last]
    g = m.groupby("ticker", sort=False)
    for k in (1, 3, 6, 12):
        m[f"ret_{k}m"] = m.adj / g.adj.shift(k) - 1
    r1 = m.adj / g.adj.shift(1) - 1
    m["vol_12m"] = r1.groupby(m.ticker).rolling(12, min_periods=9).std().reset_index(level=0, drop=True) * np.sqrt(12)
    m["hi_36m"] = g.adj.rolling(36, min_periods=12).max().reset_index(level=0, drop=True)
    m["hi_12m"] = g.adj.rolling(12, min_periods=6).max().reset_index(level=0, drop=True)
    m["dd_from_36m_high"] = m.adj / m.hi_36m - 1
    m["dist_52w_high"] = m.adj / m.hi_12m - 1
    dv6 = g.dv_m.rolling(6, min_periods=4).sum().reset_index(level=0, drop=True)
    m["dv_6m_change"] = dv6 / dv6.groupby(m.ticker).shift(6) - 1
    m["dv_6m"] = dv6
    # later split factor: product of splits strictly after the month end (to un-adjust Yahoo's close)
    sp = con.execute(f"SELECT ticker, date, stock_splits FROM '{prices}' WHERE stock_splits > 0 ORDER BY ticker, date").df()
    sp["date"] = pd.to_datetime(sp.date).astype("datetime64[ns]")
    sp["cum_after"] = sp[::-1].groupby("ticker").stock_splits.cumprod()[::-1]   # product of this and all later splits
    sp = sp.sort_values("date")
    m = m.sort_values("month_end")
    m = pd.merge_asof(m, sp[["ticker", "date", "cum_after"]].rename(columns={"date": "split_date"}), left_on="month_end", right_on="split_date",
                      by="ticker", direction="forward", allow_exact_matches=False)
    m["split_after"] = m.cum_after.fillna(1.0)
    m["close_unadj"] = m.close * m.split_after
    return m.drop(columns=["cum_after", "split_date"]).sort_values(["ticker", "mo"])


def features(obs, quarterly, monthly, filings):
    """obs: DataFrame with ticker, cik, obs_date (month-end-ish; snapped to the month)."""
    obs = obs.copy()
    obs["mo"] = pd.to_datetime(obs.obs_date).dt.to_period("M").dt.to_timestamp().astype("datetime64[ns]")
    obs["obs_date"] = pd.to_datetime(obs.obs_date)
    q = pd.read_parquet(quarterly) if isinstance(quarterly, str) else quarterly
    q = q.sort_values(["cik", "end"]).reset_index(drop=True)
    for c in ("end", "avail", "bs_filed", "filed"):
        if c in q:
            q[c] = pd.to_datetime(q[c]).astype("datetime64[ns]")
    obs["obs_date"] = obs.obs_date.astype("datetime64[ns]")
    obs["cik"] = obs.cik.astype(int)
    g = q.groupby("cik", sort=False)
    gap = (q["end"] - g["end"].shift(4)).dt.days
    ok = gap.between(330, 400)
    # one-year-earlier values for changes
    for c in ["ttm_revenue", "ttm_gross_profit", "ttm_op_income", "ttm_fcf", "ttm_sbc", "ttm_rnd", "cover_shares", "assets"]:
        q[c + "_py"] = g[c].shift(4).where(ok)
    gap8 = (q["end"] - g["end"].shift(8)).dt.days
    q["ttm_revenue_p2y"] = g.ttm_revenue.shift(8).where(gap8.between(690, 770))
    # quarters since TTM operating income / FCF last negative
    for c, name in (("ttm_op_income", "oi"), ("ttm_fcf", "fcf")):
        neg = (q[c] < 0)
        grp = (neg != neg.groupby(q.cik).shift()).cumsum()
        q[f"q_since_{name}_pos"] = np.where(q[c] > 0, q.groupby([q.cik, grp]).cumcount() + 1, 0)
    flow_cols = ["cik", "end", "avail", "ttm_revenue", "ttm_gross_profit", "ttm_op_income", "ttm_net_income", "ttm_cfo", "ttm_capex", "ttm_fcf",
                 "ttm_sbc", "ttm_rnd", "ttm_buybacks", "ttm_issuance", "ttm_revenue_py", "ttm_gross_profit_py", "ttm_op_income_py", "ttm_fcf_py",
                 "ttm_sbc_py", "ttm_rnd_py", "ttm_revenue_p2y", "q_since_oi_pos", "q_since_fcf_pos", "q_revenue_src", "q_op_income_src"]
    bs_cols = ["cik", "end", "bs_filed", "assets", "equity", "cash", "lt_debt", "st_debt", "cover_shares", "cover_shares_py", "assets_py", "q_diluted_shares", "public_float"]
    qf = q.loc[q.avail.notna(), flow_cols].sort_values("avail")
    qb = q.loc[q.bs_filed.notna(), bs_cols].sort_values("bs_filed")
    o = obs.sort_values("obs_date")
    o["cik"] = o.cik.astype("int64"); qf["cik"] = qf.cik.astype("int64"); qb["cik"] = qb.cik.astype("int64")
    o = pd.merge_asof(o, qf, left_on="obs_date", right_on="avail", by="cik", tolerance=pd.Timedelta(days=STALE), direction="backward")
    o = o.rename(columns={"end": "ttm_end"})
    o = pd.merge_asof(o.sort_values("obs_date"), qb, left_on="obs_date", right_on="bs_filed", by="cik", tolerance=pd.Timedelta(days=STALE), direction="backward")
    o = o.rename(columns={"end": "bs_end"})
    # prices
    mcols = ["ticker", "mo", "close_unadj", "adj", "ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_12m", "dd_from_36m_high", "dist_52w_high", "dv_6m_change", "dv_6m"]
    o = o.merge(monthly[mcols], on=["ticker", "mo"], how="left")
    # 8-K activity in the prior 6 months
    f = pd.read_parquet(filings, columns=["cik", "form", "filingDate", "items"]) if isinstance(filings, str) else filings
    f = f[f.form.str.startswith("8-K")].copy(); f["filingDate"] = pd.to_datetime(f.filingDate).astype("datetime64[ns]")
    con = duckdb.connect(); con.register("o", o[["cik", "obs_date"]].drop_duplicates()); con.register("f", f)
    k8 = con.execute("""
        SELECT o.cik, o.obs_date, COUNT(*) AS n_8k_6m,
               MAX(CASE WHEN f.items LIKE '%1.01%' THEN 1 ELSE 0 END) AS k8_101,
               MAX(CASE WHEN f.items LIKE '%5.02%' THEN 1 ELSE 0 END) AS k8_502,
               MAX(CASE WHEN f.items LIKE '%2.02%' THEN 1 ELSE 0 END) AS k8_202,
               MAX(CASE WHEN f.items LIKE '%8.01%' THEN 1 ELSE 0 END) AS k8_801
        FROM o JOIN f ON f.cik = o.cik AND f.filingDate <= o.obs_date AND f.filingDate > o.obs_date - INTERVAL 183 DAY
        GROUP BY 1, 2""").df()
    k8["obs_date"] = pd.to_datetime(k8.obs_date).astype("datetime64[ns]")
    o = o.merge(k8, on=["cik", "obs_date"], how="left")
    for c in ["n_8k_6m", "k8_101", "k8_502", "k8_202", "k8_801"]:
        o[c] = o[c].fillna(0)
    # derived features
    sh = o.cover_shares.where(o.cover_shares > 0, o.q_diluted_shares.where(o.q_diluted_shares > 0))
    o["mcap"] = o.close_unadj * sh
    o["log_mcap"] = np.log(o.mcap.where(o.mcap > 0))
    rev = o.ttm_revenue.where(o.ttm_revenue > 0)
    o["rev_growth"] = np.where((o.ttm_revenue > 0) & (o.ttm_revenue_py > 0), o.ttm_revenue / o.ttm_revenue_py - 1, np.nan)
    rev_g_py = np.where((o.ttm_revenue_py > 0) & (o.ttm_revenue_p2y > 0), o.ttm_revenue_py / o.ttm_revenue_p2y - 1, np.nan)
    o["rev_accel"] = o.rev_growth - rev_g_py
    o["gross_margin"] = o.ttm_gross_profit / rev
    o["gross_margin_chg"] = o.gross_margin - o.ttm_gross_profit_py / o.ttm_revenue_py.where(o.ttm_revenue_py > 0)
    o["op_margin"] = o.ttm_op_income / rev
    o["op_margin_chg"] = o.op_margin - o.ttm_op_income_py / o.ttm_revenue_py.where(o.ttm_revenue_py > 0)
    d_rev = o.ttm_revenue - o.ttm_revenue_py
    o["incr_op_margin"] = np.where(d_rev > 0, (o.ttm_op_income - o.ttm_op_income_py) / d_rev, np.nan)
    o["fcf_margin"] = o.ttm_fcf / rev
    o["oi_positive"] = (o.ttm_op_income > 0).astype(float).where(o.ttm_op_income.notna())
    o["fcf_positive"] = (o.ttm_fcf > 0).astype(float).where(o.ttm_fcf.notna())
    o["rnd_intensity"] = o.ttm_rnd / rev
    o["sbc_intensity"] = o.ttm_sbc / rev
    o["capex_intensity"] = o.ttm_capex / rev
    o["net_debt_to_assets"] = (o.lt_debt.fillna(0) + o.st_debt.fillna(0) - o.cash.fillna(0)) / o.assets.where(o.assets > 0)
    o["cash_to_assets"] = o.cash / o.assets.where(o.assets > 0)
    o["asset_growth"] = o.assets / o.assets_py.where(o.assets_py > 0) - 1
    o["share_change"] = o.cover_shares / o.cover_shares_py.where(o.cover_shares_py > 0) - 1
    o["buyback_yield"] = o.ttm_buybacks / o.mcap.where(o.mcap > 0)
    o["issuance_yield"] = o.ttm_issuance / o.mcap.where(o.mcap > 0)
    o["ev_to_sales"] = (o.mcap + o.lt_debt.fillna(0) + o.st_debt.fillna(0) - o.cash.fillna(0)) / rev
    o["pe"] = np.where(o.ttm_net_income > 0, o.mcap / o.ttm_net_income, np.nan)
    o["pb"] = o.mcap / o.equity.where(o.equity > 0)
    o["p_fcf"] = np.where(o.ttm_fcf > 0, o.mcap / o.ttm_fcf, np.nan)
    o["roa"] = o.ttm_op_income / o.assets.where(o.assets > 0)
    o["roe"] = o.ttm_net_income / o.equity.where(o.equity > 0)
    o["sector1"] = o.get("sic", pd.Series(index=o.index, dtype=object)).astype(str).str.zfill(4).str[0] if "sic" in o else None
    o["fund_age_days"] = (o.obs_date - o.ttm_end).dt.days
    return o


FEATURES = ["log_mcap", "rev_growth", "rev_accel", "gross_margin", "gross_margin_chg", "op_margin", "op_margin_chg", "incr_op_margin", "fcf_margin",
            "oi_positive", "fcf_positive", "q_since_oi_pos", "q_since_fcf_pos", "rnd_intensity", "sbc_intensity", "capex_intensity",
            "net_debt_to_assets", "cash_to_assets", "asset_growth", "share_change", "buyback_yield", "issuance_yield",
            "ev_to_sales", "pe", "pb", "p_fcf", "roa", "roe",
            "ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_12m", "dd_from_36m_high", "dist_52w_high", "dv_6m_change",
            "n_8k_6m", "k8_101", "k8_502", "k8_202", "k8_801"]
