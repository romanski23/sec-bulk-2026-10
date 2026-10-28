# sec-bulk-2026-10

Bulk data for the A7 bull-run study.

## Releases

| Release | Asset | Content |
|---|---|---|
| `v1` | `companyfacts.zip` | SEC EDGAR XBRL company facts, all filers (as-filed fundamentals) |
| `v1` | `submissions.zip` | SEC EDGAR filing histories per CIK (filing dates, tickers, exchanges, delistings) |
| `prices` | `prices.parquet` | Daily OHLCV, adjusted close, dividends and splits for every exchange-listed SEC filer (rebuilt weekly; see below) |
| `prices` | `coverage.csv`, `verify_report.txt`, `fetch_log.csv`, ... | Per-ticker span and gap counts; verification results |
| `prices-YYYY-MM` | same | Immutable monthly snapshot, for reproducible results |

## Price pipeline (`prices/`, `.github/workflows/prices.yml`)

- `prices/tickers.csv`: 7,691 tickers listed on Nasdaq, NYSE (incl. NYSE American), CBOE and TXSE, taken from the `tickers`/`exchanges` fields in `submissions.zip` (v1 release, Oct 2026), with the CIK of the filer. OTC tickers are excluded. Live tickers only: Yahoo has purged delisted companies, so dead firms are handled from the SEC filing data instead (study pre-registration, amendment A1).
- `prices/fetch.py`: full re-pull from Yahoo Finance every run (`auto_adjust=False`, so `close` is split-adjusted and `adj_close` is split+dividend adjusted), 100 tickers a batch; failures retried singly, then Stooq (close only, `source='stooq'`). History from 1990-01-01.
- `prices/build.py`: one `prices.parquet` (`ticker, date, open, high, low, close, adj_close, volume, dividends, stock_splits, source`) plus `coverage.csv`.
- `prices/verify.py`: adjustment sanity (factor changes only on dividend/split days, known splits), gaps against the SPY calendar, coverage against `tickers.csv`, and a daily-return cross-check against the CRSP-verified [sp500-data](https://github.com/Johnbrick123/sp500-data) release. A hard failure marks the run red but the report is still uploaded.

Run manually: Actions → "Daily price history" → Run workflow (set `max_tickers` for a test run). Query without downloading:

```sql
SELECT * FROM 'https://github.com/romanski23/sec-bulk-2026-10/releases/download/prices/prices.parquet' WHERE ticker='NVDA'
```
