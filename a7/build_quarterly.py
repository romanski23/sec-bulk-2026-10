"""Point-in-time quarterly fundamentals from facts.parquet.
Usage: python -I build_quarterly.py facts.parquet quarterly.parquet

Rules (no look-ahead):
- Only original 10-K/10-Q family filings (no amendments, no 20-F/40-F).
- Each (cik, tag, start, end) value is taken from its FIRST filing (as first
  reported; later restatements ignored). `filed` = date it became public.
- Flow tags: quarterly value = YTD(start,end) - YTD(start,prev_end) when the
  filing only gives year-to-date figures; `filed` = later of the two filings.
- TTM = sum of 4 consecutive quarters whose first and last period ends are
  255-290 days apart; `avail` = latest flow filing date among the four (balance
  sheet and cover fields have their own `filed`). Nothing is used before its date.
- Q4 for discrete-quarter filers: FY value minus the three discrete quarters
  inside the fiscal year when no same-start 9-month YTD exists (audit F6.1).
- Cover share count and public float carry their own filing date (audit F2.3).
"""
import sys

import duckdb
import numpy as np
import pandas as pd

FLOW = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet", "RevenuesNetOfInterestExpense"],
    "interest_income": ["InterestAndDividendIncomeOperating"],
    "noninterest_income": ["NoninterestIncome"],
    "net_interest_income": ["InterestIncomeExpenseNet"],
    "premiums": ["PremiumsEarnedNet"],
    "pretax_income": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                      "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
    "total_costs": ["CostsAndExpenses", "OperatingExpenses"],
    "gross_profit": ["GrossProfit"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "op_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss", "ProfitLoss", "IncomeLossFromContinuingOperations"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "sbc": ["ShareBasedCompensation"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization"],
    "interest": ["InterestExpense"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock"],
    "dividends_paid": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "issuance": ["ProceedsFromIssuanceOfCommonStock"],
    "diluted_shares": ["WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic"],
    "dps": ["CommonStockDividendsPerShareDeclared", "CommonStockDividendsPerShareCashPaid"],
}
STOCK = {
    "assets": ["Assets"], "assets_current": ["AssetsCurrent"], "liabilities": ["Liabilities"],
    "liabilities_current": ["LiabilitiesCurrent"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "lt_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "st_debt": ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"],
    "inventory": ["InventoryNet"], "receivables": ["AccountsReceivableNetCurrent"],
    "goodwill": ["Goodwill"], "intangibles": ["IntangibleAssetsNetExcludingGoodwill"], "ppe": ["PropertyPlantAndEquipmentNet"],
    "shares_out": ["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"],
    "public_float": ["EntityPublicFloat"],
}
FORMS = ("10-K", "10-Q", "10-K405", "10-KT", "10-QT")
src, out = sys.argv[1], sys.argv[2]
con = duckdb.connect()
con.execute("SET memory_limit='2GB'"); con.execute("SET threads=1"); con.execute("SET preserve_insertion_order=false"); con.execute(f"SET temp_directory='{out}.tmp'")
forms = ",".join(f"'{f}'" for f in FORMS)
alltags = [t for v in FLOW.values() for t in v] + [t for v in STOCK.values() for t in v]
tags = ",".join(f"'{t}'" for t in alltags)
# first-filed value per (cik, tag, start, end); shares/USD units only
con.execute(f"""
    COPY (
    SELECT cik, tag, TRY_CAST(start AS TIMESTAMP) AS start, TRY_CAST("end" AS TIMESTAMP) AS "end", val, TRY_CAST(filed AS TIMESTAMP) AS filed FROM (
        SELECT cik, tag, start, "end", val, filed,
               ROW_NUMBER() OVER (PARTITION BY cik, tag, start, "end" ORDER BY filed, accn) rn
        FROM '{src}'
        WHERE form IN ({forms}) AND tag IN ({tags}) AND val IS NOT NULL
          AND unit IN ('USD', 'shares', 'USD/shares')
          AND TRY_CAST("end" AS DATE) BETWEEN DATE '2007-01-01' AND DATE '2030-12-31'
          AND TRY_CAST(filed AS DATE) BETWEEN DATE '2007-01-01' AND DATE '2030-12-31'
          AND (start IS NULL OR TRY_CAST(start AS DATE) BETWEEN DATE '1990-01-01' AND DATE '2030-12-31')
    ) WHERE rn = 1
    ) TO '{out}.first.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
""")
con.close()
import pyarrow.parquet as pq
df = pq.read_table(f"{out}.first.parquet").to_pandas(strings_to_categorical=True, self_destruct=True)
import os; os.remove(f"{out}.first.parquet")
for c in ("end", "filed", "start"):
    df[c] = pd.to_datetime(df[c]).astype("datetime64[ns]")
df["val"] = df.val.astype("float64")
import gc, resource
gc.collect()
def mem(tag):
    print(f"[mem] {tag}: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB", flush=True)
mem("loaded")
print("facts (first filed):", len(df), flush=True)

flow_tags = {t: k for k, v in FLOW.items() for t in v}
stock_tags = {t: k for k, v in STOCK.items() for t in v}
pri = {t: i for v in list(FLOW.values()) + list(STOCK.values()) for i, t in enumerate(v)}
df["field"] = df.tag.map({**flow_tags, **stock_tags}).astype("category")
df["pri"] = df.tag.map(pri).astype("int8")

mem("before flows")
# ---- flows: derive quarterly values ---------------------------------------
fl = df[df.tag.isin(flow_tags)].copy()
fl["days"] = (fl["end"] - fl["start"]).dt.days
fl = fl[(fl.days >= 80) & (fl.days <= 380)]
# for each (cik, tag, start) sort by end; quarter = this - previous cumulative with same start
fl = fl.sort_values(["cik", "tag", "start", "end"])
g = fl.groupby(["cik", "tag", "start"], sort=False, observed=True)
fl["prev_end"] = g["end"].shift(1)
fl["prev_val"] = g["val"].shift(1)
fl["prev_filed"] = g["filed"].shift(1)
gap = (fl["end"] - fl["prev_end"]).dt.days
direct_q = (fl.days >= 80) & (fl.days <= 100)
derived = fl.prev_end.notna() & (gap >= 80) & (gap <= 100)
POINT_FIELDS = {"diluted_shares", "dps"}          # weighted averages and per-share figures are not additive (audit N4)
is_point = fl.field.isin(POINT_FIELDS).values
fl["qval"] = np.where(direct_q, fl.val, np.where(derived & ~is_point, fl.val - fl.prev_val, np.nan))
fl["qfiled"] = np.where(derived & ~direct_q, np.maximum(fl.filed, fl.prev_filed), fl.filed)
mem("after derive"); fl["qfiled"] = pd.to_datetime(fl["qfiled"])
# Q4 = FY - (Q1 + Q2 + Q3) for filers that tag discrete quarters but no 9-month YTD (audit F6.1)
fy = fl[(fl.days >= 350) & (fl.days <= 380) & fl.qval.isna() & ~fl.field.isin(POINT_FIELDS)][["cik", "tag", "start", "end", "val", "filed"]]
dq = fl[direct_q.reindex(fl.index, fill_value=False)][["cik", "tag", "start", "end", "qval", "qfiled"]]
if len(fy) and len(dq):
    _c = duckdb.connect()
    _c.register("fy", fy); _c.register("dq", dq)
    agg = _c.execute("""
        SELECT f.cik, f.tag, f.start, f."end", COUNT(*) AS n, SUM(q.qval) AS qsum, MAX(q.qfiled) AS qf,
               MIN(q.start) AS first_q_start, MAX(q."end") AS last_q_end
        FROM fy f JOIN dq q ON q.cik = f.cik AND q.tag = f.tag AND q.start >= f.start AND q."end" < f."end"
        GROUP BY 1, 2, 3, 4
    """).df()
    agg = agg[(agg.n == 3) & ((agg.first_q_start - agg.start).dt.days.abs() <= 3) & ((agg["end"] - agg.last_q_end).dt.days.between(80, 100))]
    fy4 = fy.merge(agg, on=["cik", "tag", "start", "end"])
    fy4["qval"] = fy4.val - fy4.qsum
    fy4["qfiled"] = fy4[["filed", "qf"]].max(axis=1)
    fy4["start"] = fy4.last_q_end + pd.Timedelta(days=1)
    fy4["days"] = (fy4["end"] - fy4["start"]).dt.days
    fy4 = fy4[["cik", "tag", "start", "end", "val", "filed", "days", "qval", "qfiled"]]
    fy4["tag"] = fy4.tag.astype(fl.tag.dtype)
    fy4["field"] = fy4.tag.astype(str).map(flow_tags).astype(fl.field.dtype); fy4["pri"] = fy4.tag.astype(str).map(pri).astype("int8")
    direct_q = pd.concat([direct_q, pd.Series(False, index=fy4.index + fl.index.max() + 1)])
    fy4.index = fy4.index + fl.index.max() + 1
    fl = pd.concat([fl, fy4])
    print("Q4 derived from FY minus three discrete quarters:", len(fy4))
fl = fl.loc[fl.qval.notna(), ["cik", "field", "end", "pri", "qval", "qfiled"]].copy()
gc.collect(); mem("after Q4 concat")
# pick best tag per (cik, field, end): lowest priority index, then direct over derived
fl["derived"] = ~direct_q.reindex(fl.index, fill_value=False).values
fl = fl.sort_values(["cik", "field", "end", "qfiled", "derived", "pri"]).drop_duplicates(["cik", "field", "end"])
fl["field"] = fl.field.astype(str)
q = fl.pivot(index=["cik", "end"], columns="field", values="qval")
qf = fl.pivot(index=["cik", "end"], columns="field", values="qfiled")
del fl; gc.collect()
base = qf.min(axis=1)
late = qf.gt(base.values[:, None] + pd.Timedelta(days=120))
q = q.mask(late)
qf = qf.mask(late)
q.columns = [f"q_{c}" for c in q.columns]
qfiled = qf.max(axis=1).rename("q_filed")
print("quarterly flow rows:", len(q), flush=True); mem("after flow pivot")

# ---- stocks ---------------------------------------------------------------
cover = df[df.tag.isin(["EntityCommonStockSharesOutstanding", "EntityPublicFloat"])].sort_values(["cik", "field", "end", "filed"]).drop_duplicates(["cik", "field", "end"])
cv = cover.pivot(index=["cik", "end"], columns="field", values="val").reset_index().rename(columns={"end": "cover_date"})
cvf = cover.pivot(index=["cik", "end"], columns="field", values="filed").max(axis=1).rename("cover_filed").reset_index().rename(columns={"end": "cover_date"})
cv = cv.merge(cvf, on=["cik", "cover_date"])
df = df[~df.tag.isin(["EntityCommonStockSharesOutstanding", "EntityPublicFloat"])]
st = df[df.tag.isin(stock_tags)].sort_values(["cik", "field", "end", "filed", "pri"]).drop_duplicates(["cik", "field", "end"])
del df; gc.collect()
st["field"] = st.field.astype(str)
s = st.pivot(index=["cik", "end"], columns="field", values="val")
sf = st.pivot(index=["cik", "end"], columns="field", values="filed")
sbase = sf.min(axis=1)
slate = sf.gt(sbase.values[:, None] + pd.Timedelta(days=120))
s = s.mask(slate)
sfiled = sf.mask(slate).max(axis=1).rename("s_filed")

# ---- merge on (cik, end); snap stock dates to flow quarter ends within 7 days -------------
qq = q.join(qfiled).reset_index()
ss = s.join(sfiled).reset_index()
qq = qq.sort_values("end"); ss = ss.sort_values("end")
m = pd.merge_asof(qq, ss, on="end", by="cik", tolerance=pd.Timedelta(days=7), direction="nearest")
m = m.sort_values("end")
cv = cv.sort_values("cover_date")
# two-pass cover merge (audit N2/N18): first a cover dated on/after the quarter end (within 75 d), else one dated up to 7 days before it
cvr = cv.rename(columns={"shares_out": "cover_shares", "public_float": "cover_float"}).sort_values("cover_date")
m1 = pd.merge_asof(m, cvr, left_on="end", right_on="cover_date", by="cik", tolerance=pd.Timedelta(days=75), direction="forward")
miss = m1.cover_date.isna()
m2 = pd.merge_asof(m[miss.values].sort_values("end"), cvr, left_on="end", right_on="cover_date", by="cik", tolerance=pd.Timedelta(days=7), direction="backward")
m = pd.concat([m1[~miss.values], m2], ignore_index=True).sort_values("end")
m["shares_out"] = m["cover_shares"].where(m["cover_shares"].notna(), m.get("shares_out"))
m["public_float"] = m["cover_float"].where(m["cover_float"].notna(), m.get("public_float"))
m = m.sort_values(["cik", "end"]).reset_index(drop=True)
m["filed"] = m[["q_filed", "s_filed", "cover_filed"]].max(axis=1)   # date by which every field in the row was public
m["bs_filed"] = m[["s_filed", "cover_filed"]].max(axis=1)           # balance sheet and share count only

mem("before TTM")
# ---- TTM -------------------------------------------------------------------
m["fcf_q"] = m["q_cfo"] - m["q_capex"].fillna(0)
# operating income fallback (audit coverage): companies that never tag OperatingIncomeLoss get pre-tax income, flagged per company
has_oi = m.groupby("cik").q_op_income.transform(lambda x: x.notna().any())
m["q_op_income_src"] = np.where(has_oi, "operating_income", np.where(m.q_pretax_income.notna(), "pretax_income", "none"))
m["q_op_income"] = m.q_op_income.where(has_oi, m.q_pretax_income)
# bank/insurer revenue proxy when no revenue tag: interest and dividend income + non-interest income (gross), else premiums
bank_gross = (m["q_interest_income"] + m["q_noninterest_income"]).where(m["q_interest_income"].notna() & m["q_noninterest_income"].notna())
cands = {"revenue": m["q_revenue"], "bank_gross": bank_gross, "premiums": m["q_premiums"]}
cov = pd.DataFrame({k: v.notna().groupby(m.cik).transform("sum") for k, v in cands.items()})
pick = cov.idxmax(axis=1).where(cov.max(axis=1) > 0, "none")       # one revenue definition per company: the longest coverage (audit N9)
m["q_revenue_src"] = pick
m["q_revenue"] = np.select([pick == "revenue", pick == "bank_gross", pick == "premiums"], [m["q_revenue"], bank_gross, m["q_premiums"]], np.nan)
ttm_fields = ["revenue", "gross_profit", "op_income", "net_income", "cfo", "capex", "sbc", "rnd", "da", "buybacks", "dividends_paid", "issuance",
              "pretax_income", "interest_income", "noninterest_income", "net_interest_income"]
gm = m.groupby("cik", sort=False)
span = (m["end"] - gm["end"].shift(3)).dt.days
ok4 = (span >= 255) & (span <= 290)
for f in ttm_fields:
    col = f"q_{f}"
    if col not in m:
        continue
    roll = gm[col].rolling(4, min_periods=4).sum().reset_index(level=0, drop=True)
    m[f"ttm_{f}"] = np.where(ok4, roll, np.nan)
m["ttm_fcf"] = m["ttm_cfo"] - m["ttm_capex"].fillna(0).where(m["ttm_cfo"].notna())
# avail: date the TTM flows were fully public (flow filings only, audit F2.4)
fi = pd.Series(m["q_filed"].values.astype("datetime64[ns]").astype("int64"), index=m.index).astype(float).where(m["q_filed"].notna())
roll_f = fi.groupby(m["cik"]).rolling(4, min_periods=4).max().reset_index(level=0, drop=True)
m["avail"] = pd.to_datetime(np.where(ok4, roll_f, fi), unit="ns")
m = m[(m["end"] >= "2008-01-01")]
m.to_parquet(out, index=False)
print("quarterly rows:", len(m), "ciks:", m.cik.nunique(), "with TTM revenue:", m.ttm_revenue.notna().sum())
pd.set_option("display.width", 200); print(m[m.cik == 1045810][["end", "filed", "avail", "q_revenue", "ttm_revenue", "ttm_op_income", "ttm_fcf", "shares_out", "cover_date"]].tail(8))
