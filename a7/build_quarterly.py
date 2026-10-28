"""Point-in-time quarterly fundamentals from facts.parquet.
Usage: python -I build_quarterly.py facts.parquet quarterly.parquet

Rules (no look-ahead):
- Only original 10-K/10-Q family filings (no amendments, no 20-F/40-F).
- Each (cik, tag, start, end) value is taken from its FIRST filing (as first
  reported; later restatements ignored). `filed` = date it became public.
- Flow tags: quarterly value = YTD(start,end) - YTD(start,prev_end) when the
  filing only gives year-to-date figures; `filed` = later of the two filings.
- TTM = sum of 4 consecutive quarters spanning 350-380 days; `avail` = latest
  filed date among the four. Nothing is used before its `avail`.
"""
import sys

import duckdb
import numpy as np
import pandas as pd

FLOW = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"],
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
forms = ",".join(f"'{f}'" for f in FORMS)
alltags = [t for v in FLOW.values() for t in v] + [t for v in STOCK.values() for t in v]
tags = ",".join(f"'{t}'" for t in alltags)
# first-filed value per (cik, tag, start, end); shares/USD units only
df = con.execute(f"""
    SELECT cik, tag, start, "end", val, filed FROM (
        SELECT cik, tag, start, "end", val, filed,
               ROW_NUMBER() OVER (PARTITION BY cik, tag, start, "end" ORDER BY filed, accn) rn
        FROM '{src}'
        WHERE form IN ({forms}) AND tag IN ({tags}) AND val IS NOT NULL
          AND unit IN ('USD', 'shares', 'USD/shares')
    ) WHERE rn = 1
""").df()
df["end"] = pd.to_datetime(df["end"]); df["filed"] = pd.to_datetime(df["filed"]); df["start"] = pd.to_datetime(df["start"])
print("facts (first filed):", len(df))

flow_tags = {t: k for k, v in FLOW.items() for t in v}
stock_tags = {t: k for k, v in STOCK.items() for t in v}
pri = {t: i for v in list(FLOW.values()) + list(STOCK.values()) for i, t in enumerate(v)}
df["field"] = df.tag.map({**flow_tags, **stock_tags})
df["pri"] = df.tag.map(pri)

# ---- flows: derive quarterly values ---------------------------------------
fl = df[df.tag.isin(flow_tags)].copy()
fl["days"] = (fl["end"] - fl["start"]).dt.days
fl = fl[(fl.days >= 80) & (fl.days <= 380)]
# for each (cik, tag, start) sort by end; quarter = this - previous cumulative with same start
fl = fl.sort_values(["cik", "tag", "start", "end"])
g = fl.groupby(["cik", "tag", "start"], sort=False)
fl["prev_end"] = g["end"].shift(1)
fl["prev_val"] = g["val"].shift(1)
fl["prev_filed"] = g["filed"].shift(1)
gap = (fl["end"] - fl["prev_end"]).dt.days
direct_q = (fl.days >= 80) & (fl.days <= 100)
derived = fl.prev_end.notna() & (gap >= 80) & (gap <= 100)
fl["qval"] = np.where(direct_q, fl.val, np.where(derived, fl.val - fl.prev_val, np.nan))
fl["qfiled"] = np.where(derived & ~direct_q, np.maximum(fl.filed, fl.prev_filed), fl.filed)
fl = fl[fl.qval.notna()].copy()
fl["qfiled"] = pd.to_datetime(fl["qfiled"])
# also: annual (FY) value minus 9M YTD handled above (same start); FY-only filers get no quarters
# pick best tag per (cik, field, end): lowest priority index, then direct over derived
fl["derived"] = ~direct_q[fl.index]
fl = fl.sort_values(["cik", "field", "end", "qfiled", "derived", "pri"]).drop_duplicates(["cik", "field", "end"])
q = fl.pivot(index=["cik", "end"], columns="field", values="qval")
qf = fl.pivot(index=["cik", "end"], columns="field", values="qfiled")
base = qf.min(axis=1)
late = qf.gt(base.values[:, None] + pd.Timedelta(days=120))
q = q.mask(late)
qf = qf.mask(late)
q.columns = [f"q_{c}" for c in q.columns]
qfiled = qf.max(axis=1).rename("q_filed")
print("quarterly flow rows:", len(q))

# ---- stocks ---------------------------------------------------------------
cover = df[df.tag.isin(["EntityCommonStockSharesOutstanding", "EntityPublicFloat"])].sort_values(["cik", "field", "end", "filed"]).drop_duplicates(["cik", "field", "end"])
cv = cover.pivot(index=["cik", "end"], columns="field", values="val").reset_index().rename(columns={"end": "cover_date"})
df = df[~df.tag.isin(["EntityCommonStockSharesOutstanding", "EntityPublicFloat"])]
st = df[df.tag.isin(stock_tags)].sort_values(["cik", "field", "end", "filed", "pri"]).drop_duplicates(["cik", "field", "end"])
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
m = pd.merge_asof(m, cv.rename(columns={"shares_out": "cover_shares", "public_float": "cover_float"}), left_on="end", right_on="cover_date", by="cik",
                  tolerance=pd.Timedelta(days=75), direction="forward")
m["shares_out"] = m["cover_shares"].where(m["cover_shares"].notna(), m.get("shares_out"))
m["public_float"] = m["cover_float"].where(m["cover_float"].notna(), m.get("public_float"))
m = m.sort_values(["cik", "end"]).reset_index(drop=True)
m["filed"] = m[["q_filed", "s_filed"]].max(axis=1)

# ---- TTM -------------------------------------------------------------------
m["fcf_q"] = m["q_cfo"] - m["q_capex"].fillna(0)
ttm_fields = ["revenue", "gross_profit", "op_income", "net_income", "cfo", "capex", "sbc", "rnd", "da", "buybacks", "dividends_paid", "issuance"]
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
fi = pd.Series(m["filed"].values.astype("datetime64[ns]").astype("int64"), index=m.index).astype(float).where(m["filed"].notna())
roll_f = fi.groupby(m["cik"]).rolling(4, min_periods=4).max().reset_index(level=0, drop=True)
m["avail"] = pd.to_datetime(np.where(ok4, roll_f, fi), unit="ns")
m = m[(m["end"] >= "2008-01-01")]
m.to_parquet(out, index=False)
print("quarterly rows:", len(m), "ciks:", m.cik.nunique(), "with TTM revenue:", m.ttm_revenue.notna().sum())
pd.set_option("display.width", 200); print(m[m.cik == 1045810][["end", "filed", "avail", "q_revenue", "ttm_revenue", "ttm_op_income", "ttm_fcf", "shares_out", "cover_date"]].tail(8))
