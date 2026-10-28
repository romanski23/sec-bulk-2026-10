"""A7 tier C denominator, amendment A1: registrants that stopped filing.
Usage: python -I build_dead.py filings.parquet outcomes.parquet quarterly.parquet universe.parquet dead_universe.parquet

For each year-end snapshot Y (2010-2021): 10-K filers that (a) had filed a 10-K
before Y-12-31, (b) filed a 10-K/10-Q within 400 days after it, (c) have NO
live price in universe.parquet at that snapshot (otherwise they are already
counted), (d) stopped filing before 2025-06-01, and (e) whose latest public
float (10-K cover, dei:EntityPublicFloat) public by the snapshot is >= US$300M.
Public float understates market cap (excludes insiders), so the cap is labelled
approximate and the count of float-based members is printed with every tier C
statistic.

Outcome, by filing trail after the snapshot (priority order):
  bankrupt  : an 8-K with item 1.03 after the snapshot           -> return -100% (A1)
  acquired  : DEFM14A or SC 13E3 within 18 months before the last 10-K/10-Q -> excluded from returns (A1)
  delisted  : Form 25 or Form 15 without the above                 -> -100% unless a later price is found (A1)
  stopped   : none of the above (went dark)                         -> -100%, counted separately
  spac_liquidated : SIC 6770 blank-check companies                   -> trust redeemed (~0%), excluded from return figures
outcome_date = last 10-K/10-Q filing date (the price path after it is unknown).
"""
import sys

import pandas as pd

filings, outcomes, quarterly, universe, out = sys.argv[1:6]
f = pd.read_parquet(filings, columns=["cik", "form", "filingDate"])
f["filingDate"] = pd.to_datetime(f.filingDate)
oc = pd.read_parquet(outcomes)
for c in ("first_10k", "last_10kq", "form15", "form25", "bankruptcy_8k"):
    oc[c] = pd.to_datetime(oc[c])
q = pd.read_parquet(quarterly, columns=["cik", "end", "filed", "public_float"])
q = q[q.public_float.notna() & (q.filed >= q["end"])]
U = pd.read_parquet(universe, columns=["cik", "snapshot", "mcap"])
live = set(zip(U.cik, U.snapshot))
merger = f[f.form.isin(["DEFM14A", "SC 13E3"])].groupby("cik").filingDate.max().rename("merger_proxy")
oc = oc.merge(merger, on="cik", how="left")
dead = oc[(oc.last_10kq < "2025-06-01") & (oc.n_10kq >= 4)].copy()

rows = []
for y in range(2010, 2022):
    d = pd.Timestamp(f"{y}-12-31")
    cand = dead[(dead.first_10k <= d) & (dead.last_10kq >= d - pd.Timedelta(days=400)) & (dead.last_10kq > d)]
    cand = cand[[(c, y) not in live for c in cand.cik]]
    fl = q[q.filed <= d].sort_values(["cik", "end"]).drop_duplicates("cik", keep="last")[["cik", "public_float", "end"]]
    cand = cand.merge(fl.rename(columns={"public_float": "cap_proxy", "end": "float_asof"}), on="cik", how="inner")
    cand = cand[cand.cap_proxy >= 300e6]
    bk = cand.bankruptcy_8k.notna() & (cand.bankruptcy_8k > d)
    acq = cand.merger_proxy.notna() & (cand.merger_proxy >= cand.last_10kq - pd.Timedelta(days=548)) & (cand.merger_proxy > d)
    dl = cand.form25.notna() | cand.form15.notna()
    cand["outcome"] = "stopped"
    cand.loc[dl, "outcome"] = "delisted"
    cand.loc[acq, "outcome"] = "acquired"
    cand.loc[bk, "outcome"] = "bankrupt"
    cand.loc[cand.sic.astype(str) == "6770", "outcome"] = "spac_liquidated"  # trust returned, ~0% not -100%; excluded from returns like acquired
    cand["snapshot"] = y
    cand["outcome_date"] = cand.last_10kq
    cand["cap_source"] = "public_float_approx"
    rows.append(cand[["cik", "name", "sic", "snapshot", "cap_proxy", "float_asof", "cap_source", "outcome", "outcome_date",
                      "bankruptcy_8k", "merger_proxy", "form25", "form15"]])
D = pd.concat(rows, ignore_index=True)
D.to_parquet(out, index=False)
print(D.groupby(["snapshot", "outcome"]).size().unstack(fill_value=0).to_string())
print("unique dead registrants:", D.cik.nunique())
