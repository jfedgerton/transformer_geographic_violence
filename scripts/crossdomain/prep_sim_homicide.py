"""prep_sim_homicide.py (2026-10-23) -- Brazil SIM death records -> homicide event file (cross-domain test C3).
Reads the national DOBR{year}.dbc files only (complete for every year). Keeps assault deaths (ICD-10 X85-Y09) and
legal intervention (Y35). Writes data_intermediate/sim_homicide_events.parquet and outputs/crossdomain/sim_year_audit.csv.
Read-only on raw data; refuses to overwrite outputs. Run with venv_xd (pyreaddbc)."""
import os, glob, re
import numpy as np, pandas as pd
import tempfile
from pyreaddbc import dbc2dbf
from dbfread import DBF


def read_dbc_homicide(path):
    """dbc -> temporary dbf -> rows; keep only all-death count and the assault/legal-intervention rows (memory-light)."""
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "x.dbf")
        dbc2dbf(path, tmp)
        n_all, rows, cols = 0, [], None
        for rec in DBF(tmp, encoding="iso-8859-1", char_decode_errors="replace"):
            n_all += 1
            if cols is None: cols = list(rec.keys())
            c = str(rec.get("CAUSABAS", "") or "").strip().upper()[:3]
            if ("X85" <= c <= "Y09") or c == "Y35":
                rows.append(dict(rec))
    return n_all, pd.DataFrame(rows, columns=cols)
R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
OUT = f"{R}/data_intermediate/sim_homicide_events.parquet"
AUD = f"{R}/outputs/crossdomain/sim_year_audit.csv"
os.makedirs(f"{R}/outputs/crossdomain", exist_ok=True)
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
files = sorted(glob.glob(f"{R}/data_raw/sim_brazil/dores/DOBR*.dbc"))
years = [int(re.search(r"DOBR(\d{4})", f).group(1)) for f in files]
assert years == list(range(1996, 2025)), f"national files not complete: {years}"
KEEP = ["DTOBITO", "CODMUNOCOR", "CODMUNRES", "CAUSABAS", "SEXO", "IDADE", "RACACOR", "LOCOCOR", "CIRCOBITO"]
parts, aud = [], []
for f, y in zip(files, years):
    n_all, df = read_dbc_homicide(f)
    miss = [c for c in ["DTOBITO", "CAUSABAS", "CODMUNOCOR"] if c not in df.columns]
    assert not miss, f"{y}: missing columns {miss}; have {list(df.columns)[:40]}"
    cause = df["CAUSABAS"].astype(str).str.strip().str.upper().str[:3]
    is_assault = cause.between("X85", "Y09")
    is_police = cause.eq("Y35")
    sub = df.loc[is_assault | is_police, [c for c in KEEP if c in df.columns]].copy()
    sub["cause3"] = cause[is_assault | is_police].values
    sub["is_police"] = is_police[is_assault | is_police].values
    d8 = sub["DTOBITO"].astype(str).str.strip().str.zfill(8)
    sub["date"] = pd.to_datetime(d8, format="%d%m%Y", errors="coerce")
    sub["year_file"] = y
    bad_date = sub["date"].isna().mean(); wrong_year = (sub["date"].dt.year != y).mean()
    mun_ocor_missing = sub["CODMUNOCOR"].isna().mean() if len(sub) else np.nan
    aud.append(dict(year=y, all_deaths=n_all, kept=len(sub), assault=int(is_assault.sum()), police=int(is_police.sum()),
                    firearm=int(sub["cause3"].between("X93", "X95").sum()), bad_date_share=bad_date,
                    date_not_file_year_share=wrong_year, mun_ocor_missing_share=mun_ocor_missing))
    print(aud[-1], flush=True)
    parts.append(sub)
ev = pd.concat(parts, ignore_index=True)
a = pd.DataFrame(aud); a.to_csv(AUD, index=False)
assert (a["kept"] > 0).all(), "a year has no homicides"
assert a["bad_date_share"].max() < 0.01, "more than 1% unparseable dates in some year"
ev.to_parquet(OUT, index=False)
print(f"[done] {len(ev):,} events -> {OUT}", flush=True)
print(a.to_string(index=False), flush=True)
