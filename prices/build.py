"""Combine data/raw/*.parquet into data/prices.parquet (streamed through DuckDB,
so memory stays flat) and write data/coverage.csv: one row per ticker with
first/last date, row count, source, and the number of missing trading days
inside its span (measured against SPY's trading calendar)."""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"

con = duckdb.connect()
con.execute(f"""
    COPY (
        SELECT ticker, date, open, high, low, close, adj_close, volume, dividends, stock_splits, source
        FROM read_parquet('{RAW}/*.parquet')
        ORDER BY ticker, date
    ) TO '{DATA}/prices.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)
""")
con.execute(f"CREATE VIEW p AS SELECT * FROM '{DATA}/prices.parquet'")
con.execute("CREATE TABLE cal AS SELECT date FROM p WHERE ticker = 'SPY'")
con.execute(f"""
    COPY (
        WITH span AS (
            SELECT ticker, MIN(date) first, MAX(date) last, COUNT(*) rows, ANY_VALUE(source) source,
                   SUM(CASE WHEN dividends > 0 THEN 1 ELSE 0 END) dividend_days,
                   SUM(CASE WHEN stock_splits > 0 THEN 1 ELSE 0 END) split_days
            FROM p GROUP BY ticker
        ),
        expected AS (
            SELECT s.ticker, COUNT(c.date) expected_days
            FROM span s JOIN cal c ON c.date BETWEEN s.first AND s.last
            GROUP BY s.ticker
        )
        SELECT s.*, e.expected_days, e.expected_days - s.rows AS missing_days
        FROM span s LEFT JOIN expected e USING (ticker)
        ORDER BY ticker
    ) TO '{DATA}/coverage.csv' (HEADER)
""")
n, t, lo, hi = con.execute("SELECT COUNT(*), COUNT(DISTINCT ticker), MIN(date), MAX(date) FROM p").fetchone()
print(f"prices.parquet: {n:,} rows, {t:,} tickers, {lo.date()} -> {hi.date()}")
print(con.execute("SELECT source, COUNT(DISTINCT ticker) FROM p GROUP BY 1").fetchall())
