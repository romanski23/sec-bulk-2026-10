"""A7 confirmatory analysis, part 1: events E1-E7 against buy windows (A5) and
against the universe (precision). Headline sample = complete windows only.
Usage: python -I analyze_events.py a7_dir out_md [setting]

Per event, on winners (headline tickers):
  share of firings labelled buy / too_early / too_late / unknown at the firing month;
  months from firing to the next run start (median, IQR) for too_early firings;
  annualised return lost by entering at a too-early firing instead of the run start.
Precision, on the whole eligible universe (headline snapshots, live table):
  P(a run start within 24 months | event fired) for winners versus the share of
  eligible company-months followed by a run start within 24 months. Non-winners by
  construction never have a run start, so precision = firings on winners that land
  within 24 months before a run start, divided by all firings on eligible companies
  (live table) in the same period. Dead companies (A1) cannot fire fundamental
  events after they stop filing; they are counted in the denominator only for
  E1-E6 firings that occurred while they filed. Printed with and without them.
Also: expected findings 6-9 graded; sharp vs steady split; sample sizes everywhere;
results with and without outcome_uncertain winners; match level not needed here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

a7, out_md = sys.argv[1], sys.argv[2]
SETTING = sys.argv[3] if len(sys.argv) > 3 else "h25_t25"
a7 = Path(a7)
U = pd.read_parquet(a7 / "universe.parquet")
W = pd.read_parquet(a7 / "winners.parquet")
E = pd.read_parquet(a7 / "events.parquet")
L = pd.read_parquet(a7 / "month_labels.parquet")
R = pd.read_parquet(a7 / "runs.parquet")
S = pd.read_parquet(a7 / "winner_shapes.parquet")
lines = []


def say(s=""):
    lines.append(s)


def pct(x):
    return "n/a" if pd.isna(x) else f"{x:.0%}"


# headline winners: complete windows
W["headline"] = np.where(W.tag == "win_10x10y", W.window_complete_10y, W.window_complete_5y)
WH = W[W.headline]
R = R[R.setting == SETTING]; S = S[S.setting == SETTING]
lab_col = f"label_{SETTING}"
L = L[["ticker", "month", lab_col]].rename(columns={lab_col: "label"})
L["ym"] = L.month.dt.to_period("M")
run_starts = R.groupby("ticker").start.apply(lambda s: sorted(pd.to_datetime(s)))
# run starts for EVERY eligible company (A5 labels applied to the whole eligible universe), for the precision denominator
RA = pd.read_parquet(a7 / "runs_all.parquet"); RA = RA[RA.setting == SETTING]
run_starts_all = RA.groupby("ticker").start.apply(lambda s: sorted(pd.to_datetime(s)))

say(f"# A7 confirmatory analysis 1: events against buy windows ({SETTING})\n")
say(f"Headline winners (complete windows): {WH.groupby('tag').ticker.nunique().to_dict()}; outcome_uncertain among them: {WH[WH.outcome_uncertain].groupby('tag').ticker.nunique().to_dict()}\n")

for excl_unc in (False, True):
    tick = set(WH.ticker) - (set(WH[WH.outcome_uncertain].ticker) if excl_unc else set())
    say(f"## Winner-side event scoring{' (excluding outcome_uncertain)' if excl_unc else ''}: {len(tick)} winner tickers\n")
    Ew = E[E.ticker.isin(tick)].copy()
    Ew["ym"] = Ew.event_date.dt.to_period("M")
    Ew = Ew.merge(L[["ticker", "ym", "label"]], on=["ticker", "ym"], how="left")
    # months to next run start
    def next_run(row):
        rs = run_starts.get(row.ticker, [])
        later = [r for r in rs if r >= row.event_date]
        return (later[0].to_period("M") - row.ym).n if later else np.nan
    Ew["months_to_run"] = Ew.apply(next_run, axis=1)
    say("| event | firings | buy | too early | too late | unknown | median months to next run (too-early) | IQR |")
    say("|---|---|---|---|---|---|---|---|")
    for e, g in Ew.groupby("event"):
        te = g[g.label == "too_early"].months_to_run
        q = te.quantile([.25, .5, .75]) if len(te) else pd.Series([np.nan] * 3, index=[.25, .5, .75])
        say(f"| {e} | {len(g)} | {pct((g.label == 'buy').mean())} | {pct((g.label == 'too_early').mean())} | {pct((g.label == 'too_late').mean())} | {pct((g.label == 'unknown').mean())} | {q[.5]:.0f} | {q[.25]:.0f}-{q[.75]:.0f} |")
    say()
    if excl_unc:
        break

# precision across the universe: firings on eligible companies in headline snapshot years; "hit" = a run start within 24 months after
say("## Precision across the eligible universe (live table; firings 2010-2021; hit = an A5 run start for that company within the next 24 months, winners or not)\n")
elig = U[U.eligible][["ticker", "snapshot"]]
elig_t = set(elig.ticker)
Ea = E[E.ticker.isin(elig_t) & (E.event_date >= "2010-01-01") & (E.event_date <= "2021-12-31")].copy()
def hit(row):
    rs = run_starts_all.get(row.ticker, [])
    return any((r >= row.event_date) and (r <= row.event_date + pd.DateOffset(months=24)) for r in rs)
Ea["hit"] = Ea.apply(hit, axis=1)
# base rate: eligible company-snapshots followed by a run start within 24 months
base_rows = elig.copy(); base_rows["snap_date"] = pd.to_datetime(base_rows.snapshot.astype(str) + "-12-31")
base_rows["hit"] = base_rows.apply(lambda r: any((x > r.snap_date) and (x <= r.snap_date + pd.DateOffset(months=24)) for x in run_starts_all.get(r.ticker, [])), axis=1)
base = base_rows.hit.mean()
say(f"Base rate: eligible company-year-ends followed by a run start within 24 months: {pct(base)} (n={len(base_rows):,})\n")
say("| event | firings on eligible companies | hit rate | lift vs base |")
say("|---|---|---|---|")
for e, g in Ea.groupby("event"):
    say(f"| {e} | {len(g):,} | {pct(g.hit.mean())} | {g.hit.mean() / base:.1f}x |")
# combinations: 2+ distinct events within 4 quarters (12 months) on the same ticker
Ea = Ea.sort_values(["ticker", "event_date"])
combo_rows = []
for t, g in Ea.groupby("ticker"):
    d = g.event_date.values; ev = g.event.values; h = g.hit.values
    for i in range(len(g)):
        win = (d >= d[i] - np.timedelta64(365, "D")) & (d <= d[i])
        n_distinct = len(set(ev[win]))
        fund = len(set(ev[win]) - {"E7"})
        combo_rows.append(dict(ticker=t, event_date=d[i], n_distinct=n_distinct, has_e7=("E7" in set(ev[win])), fund=fund, hit=h[i]))
C = pd.DataFrame(combo_rows)
say()
say("| condition at a firing | firings | hit rate | lift |")
say("|---|---|---|---|")
for name, mask in [("single event only (12m)", C.n_distinct == 1), ("2+ distinct events within 12m", C.n_distinct >= 2), ("3+ distinct within 12m", C.n_distinct >= 3),
                   ("E7 alone (no fundamental event in 12m)", (C.fund == 0) & C.has_e7), ("fundamental event(s), no E7", (C.fund >= 1) & ~C.has_e7), ("fundamental event(s) + E7", (C.fund >= 1) & C.has_e7)]:
    g = C[mask]
    say(f"| {name} | {len(g):,} | {pct(g.hit.mean()) if len(g) else 'n/a'} | {(g.hit.mean() / base) if len(g) else float('nan'):.1f}x |")

# expected findings 6, 7, 9
say("\n## Expected findings 6, 7, 9 (headline winners)\n")
Sh = S[S.ticker.isin(set(WH.ticker))]
Rh = R[R.ticker.isin(set(WH.ticker))]
first_runs = Rh.sort_values(["ticker", "run_no"]).drop_duplicates("ticker")
# EF6a: run starts with at least one of E1-E5 within 2 quarters before or after
E15 = E[E.event.isin(["E1", "E2", "E3", "E4", "E5"]) & E.ticker.isin(set(WH.ticker))]
def near(row):
    g = E15[E15.ticker == row.ticker]
    return ((g.event_date >= row.start - pd.DateOffset(months=6)) & (g.event_date <= row.start + pd.DateOffset(months=6))).any()
Rh = Rh.assign(near_event=Rh.apply(near, axis=1))
say(f"- EF6a: run starts with at least one of E1-E5 within +/-2 quarters: {pct(Rh.near_event.mean())} of {len(Rh)} runs (first runs only: {pct(Rh[Rh.run_no == 1].near_event.mean())}, n={int((Rh.run_no == 1).sum())}). Pre-registered expectation: most.")
late = Rh[Rh.start >= "2012-06-30"]
say(f"- EF6a (sensitivity, run starts from July 2012 when XBRL covers all filers): {pct(late.near_event.mean())} of {len(late)} runs; first runs starting then or later: {pct(late[late.run_no == 1].near_event.mean())} (n={int((late.run_no == 1).sum())}); any run that is the first after July 2012 per winner: {pct(late.sort_values(['ticker', 'start']).drop_duplicates('ticker').near_event.mean())} (n={late.ticker.nunique()}).")
# which fundamental event is nearest to run starts, and how far
def nearest_gap(row):
    g = E15[E15.ticker == row.ticker]
    if g.empty:
        return np.nan
    return ((g.event_date - row.start).dt.days / 30.4).abs().min()
late = late.assign(gap=late.apply(nearest_gap, axis=1))
say(f"- Distance from a run start (from July 2012) to the nearest E1-E5 firing, either side: median {late.gap.median():.0f} months, IQR {late.gap.quantile(.25):.0f}-{late.gap.quantile(.75):.0f}; no fundamental event at all in the data: {pct(late.gap.isna().mean())}.")
say(f"- EF6b: winners with a too-early stretch of 12+ months before the first run: {pct((first_runs.too_early_before >= 12).mean())} of {len(first_runs)}. Expectation: more than a third.")
say(f"- EF7a: median buy months per run: {Rh.buy_months.median():.0f} (first runs: {first_runs.buy_months.median():.0f}). Expectation: more than 24.")
say(f"- EF9: shape split: sharp {int((Sh['shape'] == 'sharp').sum())}, steady {int((Sh['shape'] == 'steady').sum())}, unknown {int((Sh['shape'] == 'unknown').sum())}.")
# event scoring by shape
Ew2 = E[E.ticker.isin(set(WH.ticker))].copy(); Ew2["ym"] = Ew2.event_date.dt.to_period("M")
Ew2 = Ew2.merge(L[["ticker", "ym", "label"]], on=["ticker", "ym"], how="left").merge(Sh[["ticker", "shape"]], on="ticker", how="left")
say("\n| shape | event | firings | buy | too early |")
say("|---|---|---|---|---|")
for (sh, e), g in Ew2.groupby(["shape", "event"]):
    say(f"| {sh} | {e} | {len(g)} | {pct((g.label == 'buy').mean())} | {pct((g.label == 'too_early').mean())} |")
# cost of being early
TE = pd.read_parquet(a7 / "too_early_months.parquet")
TE = TE[TE.ticker.isin(set(WH.ticker))]
say(f"\n## Cost of being early (headline winners, {len(TE):,} too-early months)\n")
say(f"- median months to the next run start: {TE.months_to_run.median():.0f}; median annualised return to it: {pct(TE.ann_to_run.median())}; median drawdown on the way: {pct(TE.dd_to_run.median())}; share of too-early months with a drawdown worse than -25% before the run: {pct((TE.dd_to_run < -0.25).mean())}")
Path(out_md).write_text("\n".join(lines) + "\n")
print("\n".join(lines))
