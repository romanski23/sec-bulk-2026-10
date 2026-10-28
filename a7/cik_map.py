"""Map primary tickers whose current CIK is a recent successor (reincorporation,
holding-company reorganisation) to the predecessor CIK that filed before it
(audit F5.3; XOM -> CIK 2115436 with first 10-K 2026-08-03, history under 34088).
Usage: python -I cik_map.py tickers_primary.csv outcomes.parquet prices.parquet cik_links.csv

Candidate when the current CIK's first 10-K is 2008 or later and the ticker's
price history starts more than 2 years before it. Predecessor = a different CIK with >= 4 10-K/10-Q
filings whose last 10-K/10-Q falls within [-550, +120] days of the successor's
first 10-K, and whose normalised name (spaces and corporate suffixes removed; former names included)
equals, or is a prefix of, the successor's. With several matches, an exact name match wins, then the
most filings; the basis is recorded.
"""
import json
import re
import sys

import duckdb
import pandas as pd

tickers_csv, outcomes, prices, out = sys.argv[1:5]
tk = pd.read_csv(tickers_csv)
tk = tk[tk.is_primary]
oc = pd.read_parquet(outcomes)
oc["first_10k"] = pd.to_datetime(oc.first_10k); oc["last_10kq"] = pd.to_datetime(oc.last_10kq)
first_px = duckdb.connect().execute(f"SELECT ticker, MIN(date) first_px FROM '{prices}' GROUP BY 1").df()
first_px["first_px"] = pd.to_datetime(first_px.first_px)
STOP = {"inc", "corp", "corporation", "co", "company", "ltd", "limited", "plc", "holdings", "holding", "group", "the", "llc", "lp", "sa", "nv", "ag",
        "new", "de", "delaware", "trust", "incorporated", "and", "of"}


def norm(n):
    n = re.sub(r"[^a-z0-9 ]", " ", str(n).lower().replace("&", " and "))
    return "".join(w for w in n.split() if w not in STOP)


def names(row):
    s = {norm(row["name"])}
    try:
        for f in json.loads(row.formerNames or "[]"):
            s.add(norm(f.get("name", "")))
    except Exception:
        pass
    return {x for x in s if len(x) >= 5}


oc["names"] = oc.apply(names, axis=1)
cand = tk.merge(oc[["cik", "first_10k", "last_10kq", "n_10kq", "names"]], on="cik", how="left").merge(first_px, on="ticker", how="left")
cand = cand[cand.first_10k.notna() & cand.first_px.notna() & ((cand.first_10k - cand.first_px).dt.days > 730) & (cand.first_10k >= "2008-01-01")]
pool = oc[(oc.n_10kq >= 4)].copy()
rows = []
for r in cand.itertuples():
    lo, hi = r.first_10k - pd.Timedelta(days=550), r.first_10k + pd.Timedelta(days=120)
    p = pool[(pool.cik != r.cik) & (pool.last_10kq >= lo) & (pool.last_10kq <= hi) & (pool.first_10k < r.first_10k)]
    def match(a, b):
        return any(x == y or (len(y) >= 8 and x.startswith(y)) or (len(x) >= 8 and y.startswith(x)) for x in a for y in b)
    hits = [x for x in p.itertuples() if match(x.names, r.names)]
    if len(hits) == 1:
        h = hits[0]
        rows.append(dict(ticker=r.ticker, cik=r.cik, pred_cik=h.cik, pred_first_10k=h.first_10k, pred_last_10kq=h.last_10kq,
                         succ_first_10k=r.first_10k, match_basis="name", pred_name=h.name))
    elif len(hits) > 1:
        exact = [x for x in hits if x.names & r.names]
        pick = exact if len(exact) == 1 else sorted(exact or hits, key=lambda x: -x.n_10kq)[:1]
        h = pick[0]
        rows.append(dict(ticker=r.ticker, cik=r.cik, pred_cik=h.cik, pred_first_10k=h.first_10k, pred_last_10kq=h.last_10kq,
                         succ_first_10k=r.first_10k, match_basis=f"name_tiebreak:{len(hits)}", pred_name=h.name))
L = pd.DataFrame(rows)
# manual overrides (cik_overrides.csv next to this script) replace or add rows
from pathlib import Path
ov_path = Path(__file__).resolve().parent / "cik_overrides.csv"
if ov_path.exists():
    ov = pd.read_csv(ov_path)
    L = L[~L.ticker.isin(ov.ticker)]
    ov = ov.merge(tk[["ticker", "cik"]], on="ticker", how="left").merge(oc[["cik", "first_10k", "last_10kq", "name"]].rename(
        columns={"cik": "pred_cik", "first_10k": "pred_first_10k", "last_10kq": "pred_last_10kq", "name": "pred_name"}), on="pred_cik", how="left")
    ov = ov.merge(oc[["cik", "first_10k"]].rename(columns={"first_10k": "succ_first_10k"}), on="cik", how="left")
    ov["match_basis"] = "manual"
    L = pd.concat([L, ov[["ticker", "cik", "pred_cik", "pred_first_10k", "pred_last_10kq", "succ_first_10k", "match_basis", "pred_name"]]], ignore_index=True)
L.to_csv(out, index=False)
print(f"candidates {len(cand)}, mapped {int(L.pred_cik.notna().sum()) if len(L) else 0}, ambiguous {int((L.match_basis.str.startswith('ambiguous')).sum()) if len(L) else 0}")
if len(L):
    print(L[L.pred_cik.notna()][["ticker", "cik", "pred_cik", "pred_name", "succ_first_10k"]].head(15).to_string(index=False))
