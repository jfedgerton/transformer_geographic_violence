"""prep_sim_placebos_v2.py (2026-10-24) -- v1 (fb2b4bb3) wrote suicide and drowning, then stopped on dengue: 1997 had
3 of 28 unparseable dates (outside the 2000+ panel). v2 changes only: (1) causes whose event file already exists are
skipped (no overwrite); (2) the date-quality assert applies to years >= 2000 only. Audit -> sim_placebos_year_audit_v2.csv.
v1 docstring follows.
prep_sim_placebos.py (2026-10-24) -- extra PLACEBOS for the Brazil cross-domain test, one pass over the national SIM
DOBR{year}.dbc files: suicide (ICD-10 X60-X84), drowning (W65-W74), dengue (A90, A91, A97). Same reader and columns as
prep_sim_traffic.py (cause3, is_police = False). Writes data_intermediate/sim_{cause}_events.parquet and
outputs/crossdomain/sim_placebos_year_audit.csv. Read-only on raw data; refuses to overwrite. Run with venv_xd."""
import os, glob, re, tempfile
import numpy as np, pandas as pd
from pyreaddbc import dbc2dbf
from dbfread import DBF

IN_CAUSE = {"suicide": lambda c: "X60" <= c <= "X84", "drowning": lambda c: "W65" <= c <= "W74",
            "dengue": lambda c: c in ("A90", "A91", "A97")}
for c_, yes, no in [("suicide", ["X60", "X70", "X84"], ["X85", "X59", "Y35"]), ("drowning", ["W65", "W69", "W74"], ["W64", "W75", "V90"]),
                    ("dengue", ["A90", "A91", "A97"], ["A92", "A95", "A96", "A98"])]:
    assert all(IN_CAUSE[c_](x) for x in yes) and not any(IN_CAUSE[c_](x) for x in no), f"{c_} code filter wrong"
R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
OUT = {c: f"{R}/data_intermediate/sim_{c}_events.parquet" for c in IN_CAUSE}
AUD = f"{R}/outputs/crossdomain/sim_placebos_year_audit_v2.csv"
IN_CAUSE = {c: fn for c, fn in IN_CAUSE.items() if not os.path.exists(OUT[c])}
print("[todo]", sorted(IN_CAUSE), flush=True)
assert IN_CAUSE, "all placebo event files already exist"
files = sorted(glob.glob(f"{R}/data_raw/sim_brazil/dores/DOBR*.dbc"))
years = [int(re.search(r"DOBR(\d{4})", f).group(1)) for f in files]
assert years == list(range(1996, 2025)), f"national files not complete: {years}"
KEEP = ["DTOBITO", "CODMUNOCOR", "CODMUNRES", "CAUSABAS", "SEXO", "IDADE", "RACACOR", "LOCOCOR", "CIRCOBITO"]
parts = {c: [] for c in IN_CAUSE}; aud = []
for f, y in zip(files, years):
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "x.dbf"); dbc2dbf(f, tmp)
        n_all, rows = 0, {c: [] for c in IN_CAUSE}
        for rec in DBF(tmp, encoding="iso-8859-1", char_decode_errors="replace"):
            n_all += 1
            c3 = str(rec.get("CAUSABAS", "") or "").strip().upper()[:3]
            for c, fn in IN_CAUSE.items():
                if fn(c3): rows[c].append(dict(rec)); break
    a = dict(year=y, all_deaths=n_all)
    for c in IN_CAUSE:
        df = pd.DataFrame(rows[c])
        if len(df) == 0: a[c] = 0; continue
        miss = [k for k in ["DTOBITO", "CAUSABAS", "CODMUNOCOR"] if k not in df.columns]
        assert not miss, f"{y} {c}: missing columns {miss}"
        cause = df["CAUSABAS"].astype(str).str.strip().str.upper().str[:3]
        assert cause.map(IN_CAUSE[c]).all(), f"{y} {c}: reader kept wrong causes"
        sub = df[[k for k in KEEP if k in df.columns]].copy()
        sub["cause3"] = cause.values; sub["is_police"] = False
        sub["date"] = pd.to_datetime(sub["DTOBITO"].astype(str).str.strip().str.zfill(8), format="%d%m%Y", errors="coerce")
        sub["year_file"] = y
        a[c] = len(sub); a[f"{c}_bad_date"] = float(sub["date"].isna().mean())
        parts[c].append(sub)
    aud.append(a); print(a, flush=True)
A = pd.DataFrame(aud); A.to_csv(AUD, index=False)
for c in IN_CAUSE:
    assert A.loc[A.year >= 2000, c].min() > 0, f"{c}: a year with zero deaths"
    assert A.loc[A.year >= 2000, f"{c}_bad_date"].max() < 0.01, f"{c}: >1% unparseable dates in 2000+"
    ev = pd.concat(parts[c], ignore_index=True); ev.to_parquet(OUT[c], index=False)
    print(f"[done] {c}: {len(ev):,} events -> {OUT[c]}", flush=True)
print(A.to_string(index=False))
