"""Download daily price history for every ticker in prices/tickers.csv.

Primary source: Yahoo Finance via yfinance (auto_adjust=False so both the
split-adjusted close and the split+dividend-adjusted adj_close are kept, plus
dividends and splits). Fallback for tickers Yahoo will not serve: Stooq daily
CSV (split-adjusted close only, no dividends; flagged source='stooq').

Every run re-pulls full history. Yahoo restates the whole back-series on each
new dividend or split, so appending would mix adjustment bases.

Output: data/raw/<TICKER>.parquet with columns
  date, open, high, low, close, adj_close, volume, dividends, stock_splits, source
and data/fetch_log.csv with one row per ticker (status, rows, first, last).

Env: START (default 1990-01-01), BATCH (100), MAX_TICKERS (0 = all, for test runs),
     STOOQ_MAX (300, Stooq throttles after a few hundred requests a day).
"""
import csv
import os
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
RAW.mkdir(parents=True, exist_ok=True)
START = os.environ.get("START", "1990-01-01")
BATCH = int(os.environ.get("BATCH", "100"))
MAX_TICKERS = int(os.environ.get("MAX_TICKERS", "0"))
STOOQ_MAX = int(os.environ.get("STOOQ_MAX", "300"))
TODAY = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
COLS = ["date", "open", "high", "low", "close", "adj_close", "volume", "dividends", "stock_splits", "source"]


def tidy_yahoo(df, ticker):
    """One ticker's frame from yf.download(auto_adjust=False, actions=True)."""
    if df is None or df.empty or "Close" not in df:
        return None
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.copy()
    df.index = idx.normalize()
    df = df.dropna(subset=["Close"])
    df = df[df["Close"] > 0]
    df = df[df.index < TODAY]  # never keep a partial session
    if len(df) < 5:
        return None
    idx = df.index
    out = pd.DataFrame({
        "date": idx,
        "open": df["Open"].values, "high": df["High"].values, "low": df["Low"].values,
        "close": df["Close"].values, "adj_close": df["Adj Close"].values,
        "volume": df["Volume"].values,
        "dividends": df["Dividends"].fillna(0.0).values if "Dividends" in df else 0.0,
        "stock_splits": df["Stock Splits"].fillna(0.0).values if "Stock Splits" in df else 0.0,
        "source": "yahoo", "ticker": ticker,
    })
    return out


def yahoo_batch(tickers):
    """Returns {ticker: tidy frame or None}; raises on a whole-batch failure."""
    df = yf.download(tickers, start=START, actions=True, auto_adjust=False, group_by="ticker",
                     threads=True, progress=False, timeout=60)
    res = {}
    if df is None or df.empty:
        return {t: None for t in tickers}
    for t in tickers:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                sub = df[t] if t in df.columns.get_level_values(0) else df.xs(t, axis=1, level=1)
            else:
                sub = df
        except KeyError:
            res[t] = None
            continue
        res[t] = tidy_yahoo(sub, t)
    return res


def stooq(ticker):
    import io
    import urllib.request
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; sec-bulk-prices)"})
    txt = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
    if not txt.startswith("Date,"):
        return None
    df = pd.read_csv(io.StringIO(txt))
    if df.empty or "Close" not in df:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Date"] >= START) & (df["Date"] < TODAY) & (df["Close"] > 0)]
    if len(df) < 5:
        return None
    return pd.DataFrame({
        "date": df["Date"].values, "open": df["Open"].values, "high": df["High"].values,
        "low": df["Low"].values, "close": df["Close"].values, "adj_close": df["Close"].values,
        "volume": df.get("Volume", 0).values, "dividends": 0.0, "stock_splits": 0.0,
        "source": "stooq", "ticker": ticker,
    })


def save(frame):
    t = frame["ticker"].iloc[0]
    frame[COLS].to_parquet(RAW / f"{t}.parquet", index=False)
    return {"ticker": t, "status": "ok", "source": frame["source"].iloc[0], "rows": len(frame),
            "first": frame["date"].min().date(), "last": frame["date"].max().date()}


def main():
    tickers = [r["ticker"] for r in csv.DictReader(open(ROOT / "prices" / "tickers.csv"))]
    if MAX_TICKERS:
        tickers = tickers[:MAX_TICKERS]
    # SPY is the calendar and benchmark reference used by verify.py
    if "SPY" not in tickers:
        tickers.append("SPY")
    log = {}
    failed = []
    t0 = time.time()
    for i in range(0, len(tickers), BATCH):
        batch = tickers[i:i + BATCH]
        res = None
        for attempt in range(4):
            try:
                res = yahoo_batch(batch)
                break
            except Exception as e:  # rate limit, network
                wait = 30 * (attempt + 1)
                print(f"batch {i // BATCH} failed ({type(e).__name__}: {str(e)[:120]}), retry in {wait}s", flush=True)
                time.sleep(wait)
        if res is None:
            failed.extend(batch)
            continue
        for t in batch:
            if res.get(t) is not None:
                log[t] = save(res[t])
            else:
                failed.append(t)
        print(f"{i + len(batch)}/{len(tickers)} ok={len(log)} failed={len(failed)} {time.time() - t0:.0f}s", flush=True)
        time.sleep(1.0)

    # second pass: failed tickers one at a time (batch failures hide per-ticker data)
    still = []
    for t in failed:
        try:
            r = yahoo_batch([t]).get(t)
        except Exception:
            r = None
        if r is not None:
            log[t] = save(r)
        else:
            still.append(t)
        time.sleep(0.5)
    print(f"retry pass: recovered {len(failed) - len(still)}, still missing {len(still)}", flush=True)

    # third pass: Stooq for what Yahoo refused
    stooq_fail = []
    for n, t in enumerate(still):
        if n >= STOOQ_MAX:
            stooq_fail.append(t)
            continue
        try:
            r = stooq(t)
        except Exception as e:
            print(f"stooq {t}: {type(e).__name__}", flush=True)
            r = None
        if r is not None:
            log[t] = save(r)
        else:
            stooq_fail.append(t)
        time.sleep(1.5)
    for t in stooq_fail:
        log[t] = {"ticker": t, "status": "no_data", "source": "", "rows": 0, "first": "", "last": ""}

    with open(DATA / "fetch_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "status", "source", "rows", "first", "last"])
        w.writeheader()
        for t in tickers:
            w.writerow(log.get(t, {"ticker": t, "status": "no_data", "source": "", "rows": 0, "first": "", "last": ""}))
    ok = sum(1 for v in log.values() if v["status"] == "ok")
    print(f"done: {ok}/{len(tickers)} tickers with data, {len(stooq_fail)} without, {time.time() - t0:.0f}s")
    if ok < 0.8 * len(tickers):
        print("FAIL: fewer than 80% of tickers returned data; not publishing")
        sys.exit(1)


if __name__ == "__main__":
    main()
