"""Extract a tag whitelist from SEC companyfacts.zip into a long parquet table.
Usage: python -I extract_facts.py companyfacts.zip out.parquet
Columns: cik, entity, taxonomy, tag, unit, start, end, val, accn, fy, fp, form, filed, frame
"""
import json
import sys
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

TAGS = {
    "us-gaap": [
        "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet",
        "GrossProfit", "CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
        "OperatingIncomeLoss", "NetIncomeLoss", "ProfitLoss", "IncomeLossFromContinuingOperations",
        "NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInInvestingActivities",
        "NetCashProvidedByUsedInFinancingActivities", "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "Assets", "AssetsCurrent", "Liabilities", "LiabilitiesCurrent", "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "LongTermDebt", "LongTermDebtNoncurrent", "LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings",
        "CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "ResearchAndDevelopmentExpense", "SellingGeneralAndAdministrativeExpense", "ShareBasedCompensation",
        "DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "InterestExpense",
        "IncomeTaxExpenseBenefit", "InventoryNet", "AccountsReceivableNetCurrent", "Goodwill",
        "IntangibleAssetsNetExcludingGoodwill", "PropertyPlantAndEquipmentNet",
        "WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic",
        "CommonStockSharesOutstanding", "EarningsPerShareDiluted", "EarningsPerShareBasic",
        "PaymentsForRepurchaseOfCommonStock", "PaymentsOfDividends", "PaymentsOfDividendsCommonStock",
        "CommonStockDividendsPerShareDeclared", "CommonStockDividendsPerShareCashPaid",
        "ProceedsFromIssuanceOfCommonStock",
    ],
    "dei": ["EntityCommonStockSharesOutstanding", "EntityPublicFloat"],
}
schema = pa.schema([
    ("cik", pa.int64()), ("entity", pa.string()), ("taxonomy", pa.string()), ("tag", pa.string()),
    ("unit", pa.string()), ("start", pa.string()), ("end", pa.string()), ("val", pa.float64()),
    ("accn", pa.string()), ("fy", pa.int32()), ("fp", pa.string()), ("form", pa.string()),
    ("filed", pa.string()), ("frame", pa.string()),
])

z = zipfile.ZipFile(sys.argv[1])
writer = pq.ParquetWriter(sys.argv[2], schema, compression="zstd")
buf = {k.name: [] for k in schema}
n_files = 0


def flush():
    if buf["cik"]:
        writer.write_table(pa.table(buf, schema=schema))
        for k in buf:
            buf[k].clear()


for name in z.namelist():
    if not name.endswith(".json"):
        continue
    d = json.loads(z.read(name))
    if not d or d.get("cik") is None:
        continue
    cik, entity = int(d.get("cik")), d.get("entityName")
    facts = d.get("facts") or {}
    for tax, tags in TAGS.items():
        tf = facts.get(tax) or {}
        for tag in tags:
            node = tf.get(tag)
            if not node:
                continue
            for unit, items in (node.get("units") or {}).items():
                for it in items:
                    buf["cik"].append(cik); buf["entity"].append(entity); buf["taxonomy"].append(tax)
                    buf["tag"].append(tag); buf["unit"].append(unit); buf["start"].append(it.get("start"))
                    buf["end"].append(it.get("end")); buf["val"].append(float(it.get("val")) if it.get("val") is not None else None)
                    buf["accn"].append(it.get("accn")); buf["fy"].append(it.get("fy")); buf["fp"].append(it.get("fp"))
                    buf["form"].append(it.get("form")); buf["filed"].append(it.get("filed")); buf["frame"].append(it.get("frame"))
    n_files += 1
    if len(buf["cik"]) > 2_000_000:
        flush()
    if n_files % 2000 == 0:
        print(n_files, flush=True)
flush()
writer.close()
print("done", n_files)
