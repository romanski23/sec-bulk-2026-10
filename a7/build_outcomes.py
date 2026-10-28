"""Per-CIK outcome table from filings.parquet and companies.parquet.
Usage: python -I build_outcomes.py filings.parquet companies.parquet outcomes.parquet
Columns: company fields + first/last 10-K/10-Q date and count, first Form 15,
first Form 25/25-NSE, first 8-K with item 1.03 (bankruptcy), first S-1/424B4/10-12B,
number of 20-F/40-F filings (foreign private issuer flag).
"""
import sys

import duckdb

filings, companies, out = sys.argv[1:4]
duckdb.connect().execute(f"""
COPY (
WITH f AS (SELECT * FROM '{filings}'),
k AS (SELECT cik, MIN(filingDate) first_10k, MAX(filingDate) last_10kq, COUNT(*) n_10kq FROM f WHERE form IN ('10-K','10-Q','10-K405','10-KT','10-QT') GROUP BY cik),
f15 AS (SELECT cik, MIN(filingDate) form15 FROM f WHERE form LIKE '15%' GROUP BY cik),
f25 AS (SELECT cik, MIN(filingDate) form25 FROM f WHERE form IN ('25','25-NSE') GROUP BY cik),
bk AS (SELECT cik, MIN(filingDate) bankruptcy_8k FROM f WHERE form LIKE '8-K%' AND items LIKE '%1.03%' GROUP BY cik),
s1 AS (SELECT cik, MIN(filingDate) first_s1 FROM f WHERE form IN ('S-1','424B4','10-12B') GROUP BY cik),
ff AS (SELECT cik, COUNT(*) n_20f FROM f WHERE form IN ('20-F','40-F') GROUP BY cik),
c AS (SELECT cik, name, sic, sicDescription, entityType, stateOfIncorporation, fiscalYearEnd, tickers, exchanges, formerNames FROM '{companies}')
SELECT c.*, k.first_10k, k.last_10kq, k.n_10kq, f15.form15, f25.form25, bk.bankruptcy_8k, s1.first_s1, ff.n_20f
FROM c JOIN k USING (cik) LEFT JOIN f15 USING (cik) LEFT JOIN f25 USING (cik) LEFT JOIN bk USING (cik) LEFT JOIN s1 USING (cik) LEFT JOIN ff USING (cik)
) TO '{out}' (FORMAT PARQUET)
""")
print("ok")
