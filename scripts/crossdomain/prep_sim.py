"""prep_sim.py (consolidated 2026-11-03) -- Brazil SIM death records -> event file. XD_CAUSE selects the cause filter:
  homicide  was prep_sim_homicide.py (2026-10-23, cross-domain test C3): assault deaths (ICD-10 X85-Y09) and legal
            intervention (Y35). Writes data_intermediate/sim_homicide_events.parquet, outputs/crossdomain/sim_year_audit.csv
  traffic   was prep_sim_traffic.py (2026-10-24, PLACEBO): land-transport accidents (ICD-10 V01-V89).
            Writes data_intermediate/sim_traffic_events.parquet, outputs/crossdomain/sim_traffic_year_audit.csv.
            Columns match the homicide file (cause3, is_police = False) so brazil_panel.py uses the same code.
  placebos  was prep_sim_placebos_v2.py (2026-10-24): one pass for suicide (X60-X84), drowning (W65-W74) and dengue
            (A90, A91, A97); causes whose event file exists are skipped. Writes data_intermediate/sim_{cause}_events.parquet,
            outputs/crossdomain/sim_placebos_year_audit_v2.csv (its docstring is kept as a comment below)
Reads the national DOBR{year}.dbc files only (complete for every year).
Read-only on raw data; refuses to overwrite outputs. Run with venv_xd (pyreaddbc)."""
import os, glob, re
import numpy as np, pandas as pd
import tempfile
from pyreaddbc import dbc2dbf
from dbfread import DBF

CAUSE = os.environ.get("XD_CAUSE", "")
assert CAUSE in ("homicide", "traffic", "placebos"), f"set XD_CAUSE=homicide, traffic or placebos (got '{CAUSE}')"

if CAUSE == "placebos":
    # prep_sim_placebos_v2.py (2026-10-24) -- v1 (fb2b4bb3) wrote suicide and drowning, then stopped on dengue: 1997 had
    # 3 of 28 unparseable dates (outside the 2000+ panel). v2 changes only: (1) causes whose event file already exists are
    # skipped (no overwrite); (2) the date-quality assert applies to years >= 2000 only. Audit -> sim_placebos_year_audit_v2.csv.
    # v1 docstring follows.
    # prep_sim_placebos.py (2026-10-24) -- extra PLACEBOS for the Brazil cross-domain test, one pass over the national SIM
    # DOBR{year}.dbc files: suicide (ICD-10 X60-X84), drowning (W65-W74), dengue (A90, A91, A97). Same reader and columns as
    # prep_sim_traffic.py (cause3, is_police = False). Writes data_intermediate/sim_{cause}_events.parquet and
    # outputs/crossdomain/sim_placebos_year_audit.csv. Read-only on raw data; refuses to overwrite. Run with venv_xd.
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

else:


    def is_traffic(c):
        return "V01" <= c <= "V89"


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
    if CAUSE == "homicide":
        OUT = f"{R}/data_intermediate/sim_homicide_events.parquet"
        AUD = f"{R}/outputs/crossdomain/sim_year_audit.csv"
    else:
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
        if CAUSE == "homicide":
            n_all, df = read_dbc_homicide(f)
        else:
            n_all, df = read_dbc_traffic(f)
        miss = [c for c in ["DTOBITO", "CAUSABAS", "CODMUNOCOR"] if c not in df.columns]
        assert not miss, f"{y}: missing columns {miss}; have {list(df.columns)[:40]}"
        cause = df["CAUSABAS"].astype(str).str.strip().str.upper().str[:3]
        if CAUSE == "homicide":
            is_assault = cause.between("X85", "Y09")
            is_police = cause.eq("Y35")
            sub = df.loc[is_assault | is_police, [c for c in KEEP if c in df.columns]].copy()
            sub["cause3"] = cause[is_assault | is_police].values
            sub["is_police"] = is_police[is_assault | is_police].values
        else:
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
        if CAUSE == "homicide":
            aud.append(dict(year=y, all_deaths=n_all, kept=len(sub), assault=int(is_assault.sum()), police=int(is_police.sum()),
                            firearm=int(sub["cause3"].between("X93", "X95").sum()), bad_date_share=bad_date,
                            date_not_file_year_share=wrong_year, mun_ocor_missing_share=mun_ocor_missing))
        else:
            aud.append(dict(year=y, all_deaths=n_all, kept=len(sub), traffic=int(keep.sum()),
                            pedestrian_v01_v09=int(sub["cause3"].between("V01", "V09").sum()),
                            motorcycle_v20_v29=int(sub["cause3"].between("V20", "V29").sum()), bad_date_share=bad_date,
                            date_not_file_year_share=wrong_year, mun_ocor_missing_share=mun_ocor_missing))
        print(aud[-1], flush=True)
        parts.append(sub)
    ev = pd.concat(parts, ignore_index=True)
    a = pd.DataFrame(aud); a.to_csv(AUD, index=False)
    if CAUSE == "homicide":
        assert (a["kept"] > 0).all(), "a year has no homicides"
    else:
        assert (a["kept"] > 0).all(), "a year has no traffic deaths"
    assert a["bad_date_share"].max() < 0.01, "more than 1% unparseable dates in some year"
    ev.to_parquet(OUT, index=False)
    print(f"[done] {len(ev):,} events -> {OUT}", flush=True)
    print(a.to_string(index=False), flush=True)
