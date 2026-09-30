"""
00b_subset_acled.py
===================
Take the full ACLED pull (data_raw/acled_full.csv) and write a filtered subset
to data_raw/acled_export.csv (the file the rest of the pipeline reads).

Pull ACLED ONCE into acled_full.csv, then re-run this script with different
filter settings to produce different use cases without re-downloading.

Usage:
    1. Pull full ACLED to data_raw/acled_full.csv (see Data section in README
       or use the API).
    2. Edit the FILTER block below to match the use case.
    3. Run: python scripts/00b_subset_acled.py
    4. Run the rest of the pipeline as usual.

Inputs:
    data_raw/acled_full.csv

Outputs:
    data_raw/acled_export.csv
    logs/subset_summary.txt
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

# ==========================================================================
# FILTER BLOCK — edit these for each use case.
# Set a value to None to disable that filter.
# ==========================================================================

# Country filter (list of country names, EXACT match against ACLED's `country`)
COUNTRY_FILTER = [
    "Ukraine",
    "Democratic Republic of Congo",
    "Myanmar",
]

# Region filter (list of region names, e.g. "Eastern Africa", "South Asia")
# Useful when you want a region's worth of countries rather than naming each.
REGION_FILTER = None

# Date range (YYYY-MM-DD strings, both inclusive). None = no bound.
START_DATE = "2018-01-01"
END_DATE   = None  # None means "up through whatever's in the file"

# Event type filter — ACLED uses these exact strings:
#   "Battles", "Explosions/Remote violence", "Violence against civilians",
#   "Protests", "Riots", "Strategic developments"
# Set to None to keep all event types.
EVENT_TYPE_FILTER = [
    "Battles",
    "Explosions/Remote violence",
    "Violence against civilians",
]

# Disorder type filter (ACLED 2022+ schema).
# Common values: "Political violence", "Demonstrations", "Strategic developments"
DISORDER_TYPE_FILTER = None

# Optional minimum fatalities threshold (None to disable)
MIN_FATALITIES = None

# Drop events with missing lat/lon
DROP_MISSING_GEO = True

# ==========================================================================
# Read full pull
# ==========================================================================
full_path = os.path.join(CONFIG["data_raw_dir"], "acled_full.csv")
if not os.path.exists(full_path):
    raise FileNotFoundError(
        f"Expected full ACLED pull at {full_path}.\n"
        "Pull ACLED first (see README — manual Data Export Tool or API).\n"
        "Save it as 'acled_full.csv' in data_raw/."
    )

print(f"Reading {full_path} ...")
# ACLED's CSV is well-behaved but large; chunked read keeps memory bounded.
full = pd.read_csv(full_path, low_memory=False)
print(f"  full rows: {len(full):,}")
print(f"  columns: {list(full.columns)[:10]}{' ...' if len(full.columns) > 10 else ''}")

n_before = len(full)
df = full.copy()

# ==========================================================================
# Apply filters one at a time, log row count after each
# ==========================================================================
report_lines = []
report_lines.append(f"Starting rows: {len(df):,}")

if COUNTRY_FILTER is not None:
    df = df[df["country"].isin(COUNTRY_FILTER)]
    report_lines.append(f"After country filter ({len(COUNTRY_FILTER)} countries): {len(df):,}")

if REGION_FILTER is not None:
    if "region" not in df.columns:
        raise KeyError("REGION_FILTER set but no 'region' column in ACLED file.")
    df = df[df["region"].isin(REGION_FILTER)]
    report_lines.append(f"After region filter: {len(df):,}")

# Date column — ACLED uses 'event_date' (YYYY-MM-DD)
if "event_date" in df.columns:
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    if START_DATE is not None:
        df = df[df["event_date"] >= pd.Timestamp(START_DATE)]
        report_lines.append(f"After START_DATE >= {START_DATE}: {len(df):,}")
    if END_DATE is not None:
        df = df[df["event_date"] <= pd.Timestamp(END_DATE)]
        report_lines.append(f"After END_DATE <= {END_DATE}: {len(df):,}")
    # Restore the column to a string form that 01_ingest_acled.py expects
    df["event_date"] = df["event_date"].dt.strftime("%Y-%m-%d")

if EVENT_TYPE_FILTER is not None:
    df = df[df["event_type"].isin(EVENT_TYPE_FILTER)]
    report_lines.append(f"After event_type filter: {len(df):,}")

if DISORDER_TYPE_FILTER is not None and "disorder_type" in df.columns:
    df = df[df["disorder_type"].isin(DISORDER_TYPE_FILTER)]
    report_lines.append(f"After disorder_type filter: {len(df):,}")

if MIN_FATALITIES is not None and "fatalities" in df.columns:
    df = df[df["fatalities"].fillna(0) >= MIN_FATALITIES]
    report_lines.append(f"After fatalities >= {MIN_FATALITIES}: {len(df):,}")

if DROP_MISSING_GEO:
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))]
    report_lines.append(f"After dropping missing/invalid lat-lon: {len(df):,}  "
                        f"(removed {before - len(df):,})")

# ==========================================================================
# Save
# ==========================================================================
out_path = CONFIG["acled_input_file"]   # data_raw/acled_export.csv
df.to_csv(out_path, index=False)
report_lines.append(f"\nSaved subset: {out_path}")
report_lines.append(f"Final rows: {len(df):,}  (kept {len(df)/n_before*100:.1f}% of full pull)")

# Country breakdown
if "country" in df.columns:
    cc = df["country"].value_counts()
    report_lines.append(f"\nCountry breakdown (top 20):")
    for c, n in cc.head(20).items():
        report_lines.append(f"  {c:30s}  {n:>10,}")

# Year breakdown
if "year" in df.columns:
    yc = df["year"].value_counts().sort_index()
    report_lines.append(f"\nYear breakdown:")
    for y, n in yc.items():
        report_lines.append(f"  {int(y)}  {n:>10,}")

# Event type breakdown
if "event_type" in df.columns:
    ec = df["event_type"].value_counts()
    report_lines.append(f"\nEvent type breakdown:")
    for et, n in ec.items():
        report_lines.append(f"  {et:35s}  {n:>10,}")

# Print and persist the report
report = "\n".join(report_lines)
print("\n" + report)

log_path = os.path.join(CONFIG["logs_dir"], "subset_summary.txt")
with open(log_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"\nWrote subset summary: {log_path}")
