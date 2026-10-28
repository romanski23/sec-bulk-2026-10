"""From submissions.zip build (a) a company table and (b) a filing table restricted
to the forms that matter for the study: 10-K/10-Q (and amendments), 8-K with
items, 15, 25, 25-NSE, 10-K405, 20-F/40-F (to flag foreign filers), S-1/424B4
(IPO dating).
Usage: python -I extract_submissions.py submissions.zip companies.parquet filings.parquet
"""
import json
import sys
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

KEEP = {"10-K", "10-K/A", "10-Q", "10-Q/A", "10-K405", "10-KT", "10-QT", "8-K", "8-K/A", "15", "15-12B", "15-12G", "15-15D",
        "25", "25-NSE", "20-F", "40-F", "S-1", "424B4", "S-4", "DEFM14A", "SC 13E3", "10-12B",
        "SC 14D9", "SC TO-T", "SC TO-I", "SC 14D9/A", "SC TO-T/A", "PREM14A", "DEFM14C", "8-K12B", "15-12B/A"}
z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()
comp = {k: [] for k in ["cik", "name", "sic", "sicDescription", "entityType", "stateOfIncorporation", "fiscalYearEnd",
                        "tickers", "exchanges", "formerNames", "ein"]}
fschema = pa.schema([("cik", pa.int64()), ("accn", pa.string()), ("form", pa.string()), ("filingDate", pa.string()),
                     ("reportDate", pa.string()), ("acceptance", pa.string()), ("items", pa.string()),
                     ("primaryDocument", pa.string())])
fw = pq.ParquetWriter(sys.argv[3], fschema, compression="zstd")
fbuf = {k.name: [] for k in fschema}
extra = {}  # cik -> list of -submissions-NNN.json names
for n in names:
    if "-submissions-" in n:
        extra.setdefault(n.split("-submissions-")[0], []).append(n)


def add_filings(cik, rec):
    accs = rec.get("accessionNumber") or []
    forms = rec.get("form") or []
    fd = rec.get("filingDate") or []
    rd = rec.get("reportDate") or []
    acc = rec.get("acceptanceDateTime") or []
    items = rec.get("items") or []
    pdoc = rec.get("primaryDocument") or []
    for i, f in enumerate(forms):
        if f not in KEEP:
            continue
        fbuf["cik"].append(cik); fbuf["accn"].append(accs[i]); fbuf["form"].append(f)
        fbuf["filingDate"].append(fd[i] if i < len(fd) else None); fbuf["reportDate"].append(rd[i] if i < len(rd) else None)
        fbuf["acceptance"].append(acc[i] if i < len(acc) else None); fbuf["items"].append(items[i] if i < len(items) else None)
        fbuf["primaryDocument"].append(pdoc[i] if i < len(pdoc) else None)


k = 0
for n in names:
    if not n.endswith(".json") or "-submissions-" in n:
        continue
    d = json.loads(z.read(n))
    cik = int(d["cik"])
    comp["cik"].append(cik)
    for key in ["name", "sic", "sicDescription", "entityType", "stateOfIncorporation", "fiscalYearEnd", "ein"]:
        comp[key].append(d.get(key))
    comp["tickers"].append(json.dumps(d.get("tickers") or []))
    comp["exchanges"].append(json.dumps(d.get("exchanges") or []))
    comp["formerNames"].append(json.dumps(d.get("formerNames") or []))
    add_filings(cik, (d.get("filings") or {}).get("recent") or {})
    for en in extra.get(n[:-5], []):
        add_filings(cik, json.loads(z.read(en)))
    k += 1
    if len(fbuf["cik"]) > 1_000_000:
        fw.write_table(pa.table(fbuf, schema=fschema)); [v.clear() for v in fbuf.values()]
    if k % 50000 == 0:
        print(k, flush=True)
if fbuf["cik"]:
    fw.write_table(pa.table(fbuf, schema=fschema))
fw.close()
pq.write_table(pa.table(comp), sys.argv[2], compression="zstd")
print("done", k)
