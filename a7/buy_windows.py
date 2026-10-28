"""A7 amendment A5: month labels (buy / too_early / too_late / unknown), runs,
and cost-of-early for every winner, on a fixed grid of hurdle h and drawdown
tolerance t. Supersedes run_starts.py (A2).
Usage: python -I buy_windows.py prices.parquet winners.parquet out_dir

Labels are a hindsight answer key only; nothing here is used to choose or tune
a signal. Outputs:
  month_labels.parquet : ticker, month, adj_close, fwd36_ann, fwd24_ann, dd12,
                         label_h{20,25,30}_t{25,40}
  runs.parquet         : ticker, setting, run_no, start, end, buy_months,
                         gain, peak_mult, too_early_before (months), dd_before
  winner_shapes.parquet: ticker, setting, n_runs, first_run_start,
                         longest_too_early, shape (sharp / steady / unknown)
  too_early_months.parquet (headline setting): per too-early month, months to the
                         next run start, annualised return to it, drawdown on the way
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

prices, winners, out = sys.argv[1:4]
out = Path(out); out.mkdir(parents=True, exist_ok=True)
HURDLES = (0.20, 0.25, 0.30)
TOLS = (0.25, 0.40)
HEADLINE = "h25_t25"

con = duckdb.connect()
tickers = sorted(set(pd.read_parquet(winners).ticker))
me = con.execute(f"""
    SELECT ticker, date, adj_close FROM (
        SELECT ticker, date, adj_close, ROW_NUMBER() OVER (PARTITION BY ticker, date_trunc('month', date) ORDER BY date DESC) rn
        FROM '{prices}' WHERE date >= '2008-01-01' AND ticker IN (SELECT UNNEST(?)))
    WHERE rn = 1 ORDER BY ticker, date
""", [tickers]).df()
me["date"] = pd.to_datetime(me.date)
me = me[me.date < me.date.max().to_period("M").to_timestamp()]   # drop the partial current month (audit F6.6)


def ann(mult, months):
    return mult ** (12.0 / months) - 1.0


label_rows, run_rows, shape_rows, early_rows = [], [], [], []
for t, g in me.groupby("ticker", sort=False):
    v = g.adj_close.values.astype(float); d = g.date.values; n = len(v)
    if n < 25:
        continue
    fwd36 = np.full(n, np.nan); fwd24 = np.full(n, np.nan); dd12 = np.full(n, np.nan)
    for i in range(n):
        if i + 24 < n:
            fwd24[i] = ann(v[i + 24] / v[i], 24)
        if i + 36 < n:
            fwd36[i] = ann(v[i + 36] / v[i], 36)
        if i + 1 < n:
            dd12[i] = v[i + 1:min(i + 13, n)].min() / v[i] - 1.0
    fwd = np.where(~np.isnan(fwd36), fwd36, fwd24)           # 36m where available, else 24m
    known = ~np.isnan(fwd)
    df = pd.DataFrame({"ticker": t, "month": d, "adj_close": v, "fwd36_ann": fwd36, "fwd24_ann": fwd24, "dd12": dd12})
    for h in HURDLES:
        for tol in TOLS:
            key = f"h{int(h * 100)}_t{int(tol * 100)}"
            buy = known & (fwd >= h) & (dd12 >= -tol)
            lab = np.where(~known, "unknown", np.where(buy, "buy", ""))
            # too_early if any buy month follows, else too_late
            later_buy = np.zeros(n, dtype=bool)
            seen = False
            for i in range(n - 1, -1, -1):
                later_buy[i] = seen
                if buy[i]:
                    seen = True
            lab = np.where((lab == ""), np.where(later_buy, "too_early", "too_late"), lab)
            df[f"label_{key}"] = lab
            # runs: maximal stretches of buy months with gaps <= 3
            idx = np.where(buy)[0]
            runs = []
            if len(idx):
                s = idx[0]; prev = idx[0]
                for i in idx[1:]:
                    if i - prev > 4:
                        runs.append((s, prev)); s = i
                    prev = i
                runs.append((s, prev))
            prev_end = -1
            longest_early = 0
            for k, (s, e) in enumerate(runs, 1):
                before = lab[prev_end + 1:s]
                early = int((before == "too_early").sum())
                longest_early = max(longest_early, early)
                # peak-to-trough drawdown between the previous run and this one (audit F6.5)
                seg = v[prev_end + 1:s + 1]
                dd_before = float((seg / np.maximum.accumulate(seg) - 1.0).min()) if len(seg) > 1 else 0.0
                # per too-early month: annualised return to this run's start and max drawdown on the way (A5 cost of being early)
                if key == HEADLINE:
                    for i in range(prev_end + 1, s):
                        if lab[i] == "too_early":
                            months = s - i
                            path = v[i:s + 1]
                            early_rows.append(dict(ticker=t, month=d[i], next_run_no=k, months_to_run=months,
                                                   ann_to_run=ann(v[s] / v[i], months), dd_to_run=float((path / np.maximum.accumulate(path) - 1.0).min())))
                run_end_i = min(e + 36, n - 1)  # the run's gain is measured 36m past its last buy month (what the last buyer earned)
                run_rows.append(dict(ticker=t, setting=key, run_no=k, start=d[s], end=d[e], buy_months=int(buy[s:e + 1].sum()),
                                     gain=v[run_end_i] / v[s] - 1.0, peak_mult=v[s:run_end_i + 1].max() / v[s],
                                     too_early_before=early, dd_before=dd_before))
                prev_end = e
            n_known = int(known.sum())
            if runs:
                first_s = runs[0][0]
                early_first = int((lab[:first_s] == "too_early").sum())
                shape = "sharp" if early_first >= 12 else "steady"
            else:
                shape = "unknown"
            shape_rows.append(dict(ticker=t, setting=key, n_runs=len(runs), first_run_start=d[runs[0][0]] if runs else pd.NaT,
                                   buy_share=float(buy.sum()) / n_known if n_known else np.nan,
                                   longest_too_early=longest_early, shape=shape))
    label_rows.append(df)

L = pd.concat(label_rows, ignore_index=True)
R = pd.DataFrame(run_rows)
S = pd.DataFrame(shape_rows)
L.to_parquet(out / "month_labels.parquet", index=False)
pd.DataFrame(early_rows).to_parquet(out / "too_early_months.parquet", index=False)
R.to_parquet(out / "runs.parquet", index=False)
S.to_parquet(out / "winner_shapes.parquet", index=False)

print(f"winners with price history: {L.ticker.nunique()}; month rows {len(L):,}")
for key in sorted(S.setting.unique()):
    s = S[S.setting == key]; r = R[R.setting == key]
    print(f"{key}: runs/winner median {s.n_runs.median():.0f} (max {s.n_runs.max()}), "
          f"winners with 2+ runs {(s.n_runs >= 2).mean():.0%}, shape sharp {(s['shape'] == 'sharp').mean():.0%} / steady {(s['shape'] == 'steady').mean():.0%}, "
          f"median buy months per run {r.buy_months.median():.0f}")
