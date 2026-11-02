"""prep_sim_traffic.py (2026-10-24) -- PLACEBO for the Brazil cross-domain test: SIM death records -> land-transport
(traffic) death event file. Identical to prep_sim_homicide.py except the cause filter: keeps ICD-10 V01-V89 (land
transport accidents) instead of X85-Y09 / Y35. Reads the national DOBR{year}.dbc files only.
Writes data_intermediate/sim_traffic_events.parquet and outputs/crossdomain/sim_traffic_year_audit.csv.
Columns match the homicide file (cause3, is_police = False) so build_brazil_traffic_panel.py can use the same code.
Read-only on raw data; refuses to overwrite outputs. Run with venv_xd (pyreaddbc)."""
import os, glob, re
import numpy as np, pandas as pd
import tempfile
from pyreaddbc import dbc2dbf
from dbfread import DBF


def is_traffic(c):
    return "V01" <= c <= "V89"


def read_dbc_traffic(path):
    """dbc -> temporary dbf -> rows; keep only the all-death count and the land-transport rows (memory-light)."""
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "x.dbf")
        dbc2dbf(path, tmp)
        n_all, rows, cols = 0, [], None
        for rec in DBF(tmp, encoding="iso-8859-1", char_decode_errors="replace"):
            n_all += 1
            if cols is None: cols = list(rec.keys())
            c = str(rec.get("CAUSABAS", "") or "").strip().upper()[:3]
            if is_traffic(c):
                rows.append(dict(rec))
    return n_all, pd.DataFrame(rows, columns=cols)


assert is_traffic("V01") and is_traffic("V89") and is_traffic("V43") and not is_traffic("V90") and not is_traffic("X95") \
    and not is_traffic("W01") and not is_traffic("U99"), "traffic code filter wrong"
R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
OUT = f"{R}/data_intermediate/sim_traffic_events.parquet"
AUD = f"{R}/outputs/crossdomain/sim_traffic_year_audit.csv"
os.makedirs(f"{R}/outputs/crossdomain", exist_ok=True)
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"
files = sorted(glob.glob(f"{R}/data_raw/sim_brazil/dores/DOBR*.dbc"))
years = [int(re.search(r"DOBR(\d{4})", f).group(1)) for f in files]
assert years == list(range(1996, 2025)), f"national files not complete: {years}"
KEEP = ["DTOBITO", "CODMUNOCOR", "CODMUNRES", "CAUSABAS", "SEXO", "IDADE", "RACACOR", "LOCOCOR", "CIRCOBITO"]
parts, aud = [], []
for f, y in zip(files, years):
    n_all, df = read_dbc_traffic(f)
    miss = [c for c in ["DTOBITO", "CAUSABAS", "CODMUNOCOR"] if c not in df.columns]
    assert not miss, f"{y}: missing columns {miss}; have {list(df.columns)[:40]}"
    cause = df["CAUSABAS"].astype(str).str.strip().str.upper().str[:3]
    keep = cause.map(is_traffic).astype(bool)
    assert keep.all(), f"{y}: reader kept non-traffic causes"
    sub = df.loc[keep, [c for c in KEEP if c in df.columns]].copy()
    sub["cause3"] = cause[keep].values
    sub["is_police"] = False
    d8 = sub["DTOBITO"].astype(str).str.strip().str.zfill(8)
    sub["date"] = pd.to_datetime(d8, format="%d%m%Y", errors="coerce")
    sub["year_file"] = y
    bad_date = sub["date"].isna().mean(); wrong_year = (sub["date"].dt.year != y).mean()
    mun_ocor_missing = sub["CODMUNOCOR"].isna().mean() if len(sub) else np.nan
    aud.append(dict(year=y, all_deaths=n_all, kept=len(sub), traffic=int(keep.sum()),
                    pedestrian_v01_v09=int(sub["cause3"].between("V01", "V09").sum()),
                    motorcycle_v20_v29=int(sub["cause3"].between("V20", "V29").sum()), bad_date_share=bad_date,
                    date_not_file_year_share=wrong_year, mun_ocor_missing_share=mun_ocor_missing))
    print(aud[-1], flush=True)
    parts.append(sub)
ev = pd.concat(parts, ignore_index=True)
a = pd.DataFrame(aud); a.to_csv(AUD, index=False)
assert (a["kept"] > 0).all(), "a year has no traffic deaths"
assert a["bad_date_share"].max() < 0.01, "more than 1% unparseable dates in some year"
ev.to_parquet(OUT, index=False)
print(f"[done] {len(ev):,} events -> {OUT}", flush=True)
print(a.to_string(index=False), flush=True)
