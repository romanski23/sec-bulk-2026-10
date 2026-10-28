"""A7 tier C denominator, amendment A1 (rebuilt after audit F5.3/F5.4).
Usage: python -I build_dead.py filings.parquet outcomes.parquet quarterly.parquet universe.parquet dead_universe.parquet

For each year-end snapshot Y: 10-K registrants that were ACTIVE at the snapshot
(a 10-K filed on or before it, a 10-K/10-Q within the 400 days before it, and at
least one 10-K/10-Q after it), that are NOT in the live universe at that snapshot
(by CIK, predecessor CIKs included), and whose latest public float (10-K cover,
dei:EntityPublicFloat) public by the snapshot is >= US$300M, no older than 730 days (a 10-K filed in March reports float as of the prior June, so 18 months is the normal age at a December snapshot; longer is stale)
days, and plausible (below 20x the largest live market cap of that snapshot;
XBRL scale errors otherwise pass the floor). Public float understates market cap,
so `cap_source = public_float_approx`.

Outcome, from the filing trail AFTER the snapshot (priority order):
  bankrupt        : 8-K item 1.03 after the snapshot                         -> -100% (A1)
  acquired        : DEFM14A or SC 13E3 after the snapshot and within 18 months before the last 10-K/10-Q -> excluded from returns (A1)
  spac_liquidated : SIC 6770                                                 -> trust redeemed (~0%), excluded from returns
  delisted        : Form 25 or Form 15 after the snapshot, no merger proxy   -> -100% unless a later price is found (A1)
  still_filing    : last 10-K/10-Q on/after 2025-06-01 but no exchange-listed price today
                    (acquired after the price cut-off, OTC-demoted, or unmapped successor) -> counted, excluded from returns, printed
  stopped         : none of the above (went dark)                            -> -100%, counted separately
Foreign private issuers (20-F/40-F only) are outside both tables by construction; stated with every tier C figure.
"""
import sys

import pandas as pd

filings, outcomes, quarterly, universe, out = sys.argv[1:6]
f = pd.read_parquet(filings, columns=["cik", "form", "filingDate"])
f["filingDate"] = pd.to_datetime(f.filingDate)
tenk = f[f.form.isin(["10-K", "10-Q", "10-K405", "10-KT", "10-QT"])]
oc = pd.read_parquet(outcomes)
for c in ("first_10k", "last_10kq"):
    oc[c] = pd.to_datetime(oc[c])
q = pd.read_parquet(quarterly, columns=["cik", "end", "filed", "public_float"])
q = q[q.public_float.notna() & (q.filed >= q["end"]) & (q.public_float > 0)]
U = pd.read_parquet(universe, columns=["cik", "cik_listed", "snapshot", "mcap", "eligible"])
live = set(zip(U.cik, U.snapshot)) | set(zip(U.cik_listed, U.snapshot))
bk = pd.read_parquet(filings, columns=["cik", "form", "filingDate", "items"])
bk = bk[bk.form.str.startswith("8-K") & bk["items"].fillna("").str.contains("1.03")]
bk["filingDate"] = pd.to_datetime(bk.filingDate)
f15 = f[f.form.str.startswith("15")]
f25 = f[f.form.isin(["25", "25-NSE"])]


def first_after(df, d):
    x = df[df.filingDate > d].groupby("cik").filingDate.min()
    return x


rows = []
for y in range(2010, 2022):
    d = pd.Timestamp(f"{y}-12-31")
    recent = tenk[(tenk.filingDate <= d) & (tenk.filingDate >= d - pd.Timedelta(days=400))].cik.unique()
    after = tenk[tenk.filingDate > d].cik.unique()
    cand = oc[(oc.first_10k <= d) & oc.cik.isin(recent) & oc.cik.isin(after)].copy()
    cand = cand[[(c, y) not in live for c in cand.cik]]
    fl = q[(q.filed <= d) & (q["end"] >= d - pd.Timedelta(days=730))].sort_values(["cik", "end"]).drop_duplicates("cik", keep="last")[["cik", "public_float", "end"]]
    cand = cand.merge(fl.rename(columns={"public_float": "cap_proxy", "end": "float_asof"}), on="cik", how="inner")
    cap_max = U[(U.snapshot == y) & U.eligible].mcap.max()
    cand = cand[(cand.cap_proxy >= 300e6) & (cand.cap_proxy <= 20 * cap_max)]
    mp = first_after(f[f.form.isin(["DEFM14A", "SC 13E3"])], d).rename("merger_proxy")
    bkd = first_after(bk, d).rename("bankruptcy_8k")
    d15 = first_after(f15, d).rename("form15"); d25 = first_after(f25, d).rename("form25")
    cand = cand.drop(columns=[c for c in ["bankruptcy_8k", "form15", "form25", "merger_proxy"] if c in cand]) \
               .merge(mp, on="cik", how="left").merge(bkd, on="cik", how="left").merge(d15, on="cik", how="left").merge(d25, on="cik", how="left")
    cand["outcome"] = "stopped"
    cand.loc[cand.last_10kq >= "2025-06-01", "outcome"] = "still_filing"
    dl = cand.form25.notna() | cand.form15.notna()
    cand.loc[dl & (cand.outcome == "stopped"), "outcome"] = "delisted"
    acq = cand.merger_proxy.notna() & (cand.merger_proxy >= cand.last_10kq - pd.Timedelta(days=548))
    cand.loc[acq, "outcome"] = "acquired"
    cand.loc[cand.sic.astype(str) == "6770", "outcome"] = "spac_liquidated"
    cand.loc[cand.bankruptcy_8k.notna(), "outcome"] = "bankrupt"
    cand["snapshot"] = y
    cand["outcome_date"] = cand[["bankruptcy_8k", "merger_proxy", "form25", "form15"]].min(axis=1).fillna(cand.last_10kq)
    cand["cap_source"] = "public_float_approx"
    rows.append(cand[["cik", "name", "sic", "snapshot", "cap_proxy", "float_asof", "cap_source", "outcome", "outcome_date",
                      "last_10kq", "bankruptcy_8k", "merger_proxy", "form25", "form15"]])
D = pd.concat(rows, ignore_index=True)
D.to_parquet(out, index=False)
print(D.groupby(["snapshot", "outcome"]).size().unstack(fill_value=0).to_string())
print("unique dead/unlisted registrants:", D.cik.nunique(), " float age days median:", int((pd.to_datetime(D.snapshot.astype(str) + "-12-31") - D.float_asof).dt.days.median()))
