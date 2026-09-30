"""
01b_ingest_ucdp.py
==================
Read UCDP GED (Georeferenced Event Dataset) and harmonize to the ACLED-style
schema used downstream by 02_build_grid.py.

Priority B (2026-05-26): multi-source pre-training (ACLED + UCDP-GED).
This script is the first half of the merge: clean UCDP into a parquet that
matches the column conventions of acled_cleaned.parquet, then 01c_merge_sources.py
concatenates the two with a source_of_record column.

UCDP GED schema reference (v25.1):
    type_of_violence: 1 = state-based, 2 = non-state, 3 = one-sided
    side_a, side_b: actor names (text)
    date_start: ISO date
    best, low, high: fatality estimate triples
    latitude, longitude: location
    priogrid_gid: PRIO-GRID cell id (bonus — used in Stage B)

Mapping to ACLED-compatible 7-field event schema:
    event_type            = "Battles" (tov 1/2) or "Violence against civilians" (tov 3)
    sub_event_type        = ACLED-style sub-label (best-effort)
    actor1                = side_a
    actor2                = side_b (may be empty for one-sided)
    inter1                = ACLED-style string label inferred from tov + side
    inter2                = same
    fatalities            = best (UCDP's primary estimate)

Outputs:
    data_intermediate/ucdp_cleaned.parquet
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
master = import_module("00_master")
CONFIG = master.CONFIG
master.ensure_dirs()

np.random.seed(CONFIG["random_seed"])

# ---------------------------------------------------------------------------
# 1. Read raw UCDP GED CSV
# ---------------------------------------------------------------------------
ucdp_path = os.path.join(CONFIG["data_raw_dir"], "ucdp", "GEDEvent_v25_1.csv")
print(f"Reading UCDP GED from: {ucdp_path}")
if not os.path.exists(ucdp_path):
    raise FileNotFoundError(f"UCDP GED file not found: {ucdp_path}")

df = pd.read_csv(ucdp_path, low_memory=False)
print(f"Raw UCDP rows: {len(df):,}")
print(f"Columns ({len(df.columns)}): {list(df.columns)[:20]} ...")

# ---------------------------------------------------------------------------
# 2. Filter to region / countries (if configured)
# ---------------------------------------------------------------------------
# UCDP-GED's `region` column uses coarser bins than ACLED's. UCDP values:
# "Africa", "Asia", "Americas", "Europe", "Middle East".
# When CONFIG["region_filter"] is set, we translate the ACLED sub-regional
# bins to UCDP's single "Africa" / "Asia" / etc.
ACLED_REGION_TO_UCDP = {
    "Northern Africa": "Africa",
    "Western Africa": "Africa",
    "Middle Africa": "Africa",
    "Eastern Africa": "Africa",
    "Southern Africa": "Africa",
    "Western Asia": "Middle East",
    "Central Asia": "Asia",
    "Southern Asia": "Asia",
    "South-Eastern Asia": "Asia",
    "Eastern Asia": "Asia",
    "Caribbean": "Americas",
    "Central America": "Americas",
    "South America": "Americas",
    "Northern America": "Americas",
    "Eastern Europe": "Europe",
    "Northern Europe": "Europe",
    "Southern Europe": "Europe",
    "Western Europe": "Europe",
    "Oceania": "Oceania",
}

# UCDP country-name aliases for explicit overrides (pilot countries kept for back-compat)
ACLED_TO_UCDP_COUNTRY = {
    "Ukraine": ["Ukraine"],
    "Democratic Republic of Congo": ["DR Congo (Zaire)", "DR Congo", "Zaire"],
    "Myanmar": ["Myanmar (Burma)", "Myanmar", "Burma"],
}

region_filter = CONFIG.get("region_filter", None)
country_filter = CONFIG.get("country_filter", None)

if region_filter is not None and "region" in df.columns:
    ucdp_regions = sorted({ACLED_REGION_TO_UCDP.get(r, r) for r in region_filter})
    print(f"Region filter (UCDP names): {ucdp_regions}")
    before = len(df)
    df = df[df["region"].isin(ucdp_regions)].copy()
    print(f"  Rows before region filter: {before:,}, after: {len(df):,}")
    for r in ucdp_regions:
        cnt = (df["region"] == r).sum()
        print(f"  {r}: {cnt:,}")

if country_filter is not None and "country" in df.columns:
    ucdp_names = []
    for c in country_filter:
        if c in ACLED_TO_UCDP_COUNTRY:
            ucdp_names.extend(ACLED_TO_UCDP_COUNTRY[c])
        else:
            ucdp_names.append(c)
    print(f"Country filter (UCDP names): {ucdp_names}")
    before = len(df)
    df = df[df["country"].isin(ucdp_names)].copy()
    print(f"  Rows before country filter: {before:,}, after: {len(df):,}")
    for n in ucdp_names:
        cnt = (df["country"] == n).sum()
        print(f"  {n}: {cnt:,}")

# ---------------------------------------------------------------------------
# 3. Parse dates
# ---------------------------------------------------------------------------
df["event_date"] = pd.to_datetime(df["date_start"], errors="coerce")
n_bad = df["event_date"].isna().sum()
if n_bad > 0:
    print(f"WARNING: {n_bad} rows with unparseable dates — dropping")
    df = df.dropna(subset=["event_date"])

df["year_month"] = df["event_date"].dt.to_period("M").astype(str)
print(f"Date range: {df['event_date'].min().date()} to {df['event_date'].max().date()}")

# ---------------------------------------------------------------------------
# 4. Map UCDP fields to ACLED-style columns
# ---------------------------------------------------------------------------
# event_type from type_of_violence
TOV_TO_EVENT_TYPE = {
    1: "Battles",                          # state-based armed conflict
    2: "Battles",                          # non-state armed conflict
    3: "Violence against civilians",       # one-sided violence
}
df["event_type"] = df["type_of_violence"].map(TOV_TO_EVENT_TYPE).fillna("Battles")

# sub_event_type — UCDP doesn't have ACLED's granular sub-types.
# Use a coarse mapping that matches ACLED sub-event strings where possible.
TOV_TO_SUB_EVENT_TYPE = {
    1: "Armed clash",
    2: "Armed clash",
    3: "Attack",
}
df["sub_event_type"] = df["type_of_violence"].map(TOV_TO_SUB_EVENT_TYPE).fillna("Armed clash")

# Actors: side_a and side_b are the ACLED actor1/actor2 equivalents.
df["actor1"] = df["side_a"].fillna("Unknown").astype(str)
df["actor2"] = df["side_b"].fillna("").astype(str)
# UCDP one-sided events have side_b=Civilians implicitly; make it explicit
df.loc[(df["type_of_violence"] == 3) & (df["actor2"] == ""), "actor2"] = "Civilians"

# inter1 / inter2 — best-effort heuristic mapping to ACLED's string labels.
# This is approximate; later (full multi-source paper version) we'd link to
# the UCDP Actor Dataset for proper org-type lookup.
def infer_inter(side_name, tov, is_side_b=False):
    s = str(side_name).lower()
    if tov == 3 and is_side_b:
        return "Civilians"
    if "government of" in s or "armed forces" in s or "military" in s:
        return "State forces"
    if tov == 1:
        return "Rebel group" if is_side_b else "State forces"
    if tov == 2:
        return "Rebel group"
    if tov == 3 and not is_side_b:
        return "Rebel group" if "government" not in s else "State forces"
    return ""

df["inter1"] = df.apply(lambda r: infer_inter(r["actor1"], r["type_of_violence"], False), axis=1)
df["inter2"] = df.apply(lambda r: infer_inter(r["actor2"], r["type_of_violence"], True), axis=1)

# ---------------------------------------------------------------------------
# 5. Coordinates
# ---------------------------------------------------------------------------
for col in ["latitude", "longitude"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["latitude", "longitude"])
df = df[(df["latitude"].abs() <= 90) & (df["longitude"].abs() <= 180)]

# ---------------------------------------------------------------------------
# 6. Fatalities — UCDP's "best" is the primary point estimate
# ---------------------------------------------------------------------------
df["fatalities"] = pd.to_numeric(df["best"], errors="coerce").fillna(0).astype(int)
df["deaths_civilians"] = pd.to_numeric(df["deaths_civilians"], errors="coerce").fillna(0).astype(int)

# ---------------------------------------------------------------------------
# 7. Admin levels
# ---------------------------------------------------------------------------
df["admin1"] = df["adm_1"].fillna("Unknown") if "adm_1" in df.columns else "Unknown"
df["admin2"] = df["adm_2"].fillna("Unknown") if "adm_2" in df.columns else "Unknown"

# Normalize country names back to ACLED-style so downstream filter / aggregation
# uses a single consistent string per country.
UCDP_TO_ACLED_COUNTRY = {v_name: k_acled
                        for k_acled, v_list in ACLED_TO_UCDP_COUNTRY.items()
                        for v_name in v_list}
df["country"] = df["country"].map(lambda c: UCDP_TO_ACLED_COUNTRY.get(c, c))

# ---------------------------------------------------------------------------
# 8. disorder_type — mark all UCDP events as "Political violence"
# ---------------------------------------------------------------------------
df["disorder_type"] = "Political violence"

# ---------------------------------------------------------------------------
# 9. Source count
# ---------------------------------------------------------------------------
df["source_count"] = df["number_of_sources"].fillna(1).astype(int) if "number_of_sources" in df.columns else 1
df.loc[df["source_count"] < 1, "source_count"] = 1

# ---------------------------------------------------------------------------
# 10. Subset to the columns the downstream pipeline expects
# ---------------------------------------------------------------------------
KEEP_COLS = [
    "event_date", "year_month",
    "event_type", "sub_event_type",
    "actor1", "actor2", "inter1", "inter2",
    "disorder_type",
    "country", "admin1", "admin2",
    "latitude", "longitude",
    "fatalities", "deaths_civilians", "source_count", "type_of_violence",
]
# Optionally keep priogrid_gid for Stage B
if "priogrid_gid" in df.columns:
    KEEP_COLS.append("priogrid_gid")

df = df[KEEP_COLS].copy()

# ---------------------------------------------------------------------------
# 11. Diagnostics + save
# ---------------------------------------------------------------------------
print(f"\n--- UCDP HARMONIZED ---")
print(f"Rows: {len(df):,}")
print(f"Country breakdown:")
print(df["country"].value_counts().to_string())
print(f"\nEvent type breakdown:")
print(df["event_type"].value_counts().to_string())
print(f"\ninter1 distribution: {df['inter1'].value_counts().to_dict()}")
print(f"inter2 distribution: {df['inter2'].value_counts().to_dict()}")

out_path = os.path.join(CONFIG["data_intermediate_dir"], "ucdp_cleaned.parquet")
df.to_parquet(out_path, index=False)
print(f"\nSaved cleaned UCDP to: {out_path}")
print(f"Shape: {df.shape}")
