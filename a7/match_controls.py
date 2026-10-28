"""A7 step 4b: rule-based matched controls (design note §4, pre-registration §1).
Usage: python -I match_controls.py universe.parquet winners.parquet out_dir

For each winner at its first qualifying snapshot, 3 controls from the SAME
snapshot that are eligible, not winners under the same tag at any snapshot,
and matched on, in order:
  1. sector: 2-digit SIC (fallback: 1-digit SIC if fewer than 3 candidates);
  2. size: same market-cap decile within the snapshot's eligible universe
     (fallback: adjacent deciles);
  3. Dickinson cash-flow pattern at the snapshot (sign of TTM CFO, investing
     proxy = -capex, financing proxy = sign of share-count change y/y), with
     only the CFO sign required when the full pattern yields < 3 candidates;
  4. trailing growth quintile (TTM revenue growth y/y, within snapshot),
     fallback: adjacent quintiles.
Tie-break, fixed: smallest |log(mcap) - log(winner mcap)|, then ticker A-Z.
Each control may serve at most 2 winners per snapshot.
All inputs are snapshot-date fields only; outcomes are never read here except
to exclude winners from the control pool (which the design requires).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

universe, winners, out = sys.argv[1:4]
out = Path(out); out.mkdir(parents=True, exist_ok=True)
U = pd.read_parquet(universe)
W = pd.read_parquet(winners)
U = U[U.eligible].copy()
U["sic2"] = U.sic.astype("string").str.zfill(4).str[:2]
U["sic1"] = U.sic2.str[:1]
U["logcap"] = np.log(U.mcap)
U["cap_decile"] = U.groupby("snapshot").logcap.transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
U["cfo_sign"] = np.sign(U.ttm_cfo.fillna(U.q_cfo))
U["inv_sign"] = -np.sign(U.ttm_capex.fillna(U.q_capex).fillna(0))
prev = U[["ticker", "snapshot", "shares_used", "ttm_revenue"]].assign(snapshot=lambda d: d.snapshot + 1) \
        .rename(columns={"shares_used": "shares_prev", "ttm_revenue": "rev_prev"})
U = U.merge(prev, on=["ticker", "snapshot"], how="left")
U["fin_sign"] = np.sign((U.shares_used / U.shares_prev - 1).fillna(0).round(3))
U["dickinson"] = U.cfo_sign.astype("Int64").astype(str) + U.inv_sign.astype("Int64").astype(str) + U.fin_sign.astype("Int64").astype(str)
U["rev_growth"] = U.ttm_revenue / U.rev_prev - 1
U["growth_q"] = U.groupby("snapshot").rev_growth.transform(
    lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) if s.notna().sum() >= 5 else np.nan)

pool_used = {}
rows = []
for tag in W.tag.unique():
    win_tickers = set(W[W.tag == tag].ticker)
    for w in W[W.tag == tag].itertuples():
        snap = U[(U.snapshot == w.snapshot) & (~U.ticker.isin(win_tickers)) & (U.ticker != w.ticker)]
        wi = U[(U.snapshot == w.snapshot) & (U.ticker == w.ticker)]
        if wi.empty:
            continue
        wi = wi.iloc[0]
        chosen = []
        # relaxation ladder, fixed order
        ladder = [
            dict(sic="sic2", dec=0, dick="full", gq=0), dict(sic="sic2", dec=0, dick="full", gq=1),
            dict(sic="sic2", dec=1, dick="full", gq=1), dict(sic="sic2", dec=1, dick="cfo", gq=1),
            dict(sic="sic1", dec=1, dick="cfo", gq=1), dict(sic="sic1", dec=2, dick="cfo", gq=2),
            dict(sic="sic1", dec=2, dick="none", gq=9), dict(sic="any", dec=2, dick="none", gq=9),
        ]
        for lv in ladder:
            c = snap
            if lv["sic"] != "any":
                c = c[c[lv["sic"]] == wi[lv["sic"]]]
            c = c[(c.cap_decile - wi.cap_decile).abs() <= lv["dec"]]
            if lv["dick"] == "full":
                c = c[c.dickinson == wi.dickinson]
            elif lv["dick"] == "cfo":
                c = c[c.cfo_sign == wi.cfo_sign]
            if lv["gq"] < 9 and pd.notna(wi.growth_q):
                c = c[(c.growth_q - wi.growth_q).abs() <= lv["gq"]]
            c = c[~c.ticker.isin([x[0] for x in chosen])]
            c = c[[pool_used.get((w.snapshot, t), 0) < 2 for t in c.ticker]] if len(c) else c
            if c.empty:
                continue
            c = c.assign(dist=(c.logcap - wi.logcap).abs()).sort_values(["dist", "ticker"])
            for r in c.itertuples():
                chosen.append((r.ticker, str(lv)))
                if len(chosen) == 3:
                    break
            if len(chosen) == 3:
                break
        for t, lvl in chosen:
            pool_used[(w.snapshot, t)] = pool_used.get((w.snapshot, t), 0) + 1
            rows.append(dict(tag=tag, winner=w.ticker, snapshot=w.snapshot, control=t, match_level=lvl))
C = pd.DataFrame(rows)
C.to_parquet(out / "controls.parquet", index=False)
U.to_parquet(out / "universe_features.parquet", index=False)
print("winners:", W.groupby("tag").ticker.nunique().to_dict())
print("controls per tag:", C.groupby("tag").size().to_dict())
print("winners with <3 controls:", (C.groupby(["tag", "winner"]).size() < 3).sum())
print(C.match_level.value_counts().head(8).to_string())
