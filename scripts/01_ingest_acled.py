"""
01_ingest_acled.py
==================
Read raw ACLED CSV, validate columns, clean dates, apply minimal filters,
and save a cleaned intermediate file.

Outputs:
    data_intermediate/acled_cleaned.parquet

Diagnostics printed:
    - row counts before/after cleaning
    - date range
    - event type distribution
    - top countries
    - missing values per column
"""

import os
import sys
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Import config
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
master = import_module("00_master")
CONFIG = master.CONFIG
master.ensure_dirs()

np.random.seed(CONFIG["random_seed"])

# ---------------------------------------------------------------------------
# 1. Read raw CSV
# ---------------------------------------------------------------------------
input_path = CONFIG["acled_input_file"]
print(f"Reading ACLED data from: {input_path}")

df = pd.read_csv(input_path, low_memory=False)
print(f"Raw rows: {len(df):,}")
print(f"Raw columns: {list(df.columns)}")

# ---------------------------------------------------------------------------
# 2. Validate expected columns exist
# ---------------------------------------------------------------------------
col_map = {
    "event_date": CONFIG["col_event_date"],
    "latitude": CONFIG["col_latitude"],
    "longitude": CONFIG["col_longitude"],
    "event_type": CONFIG["col_event_type"],
    "sub_event_type": CONFIG["col_sub_event_type"],
    "actor1": CONFIG["col_actor1"],
    "actor2": CONFIG["col_actor2"],
    "disorder_type": CONFIG["col_disorder_type"],
    "country": CONFIG["col_country"],
    "admin1": CONFIG["col_admin1"],
    "admin2": CONFIG["col_admin2"],
    "fatalities": CONFIG["col_fatalities"],
}

missing_cols = [v for v in col_map.values() if v not in df.columns]
if missing_cols:
    print(f"WARNING: Missing columns in CSV: {missing_cols}")
    print(f"Available columns: {list(df.columns)}")
    # Try to continue with what we have

# Rename to internal standard names for downstream consistency
rename_map = {v: k for k, v in col_map.items() if v in df.columns}
df = df.rename(columns=rename_map)
print(f"Columns after renaming: {list(df.columns)}")

# ---------------------------------------------------------------------------
# 3. Parse dates
# ---------------------------------------------------------------------------
df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
n_bad_dates = df["event_date"].isna().sum()
if n_bad_dates > 0:
    print(f"WARNING: {n_bad_dates} rows with unparseable dates — dropping them")
    df = df.dropna(subset=["event_date"])

print(f"Date range: {df['event_date'].min()} to {df['event_date'].max()}")

# ---------------------------------------------------------------------------
# 3b. Filter to selected countries / regions (if configured)
# ---------------------------------------------------------------------------
country_filter = CONFIG.get("country_filter", None)
region_filter = CONFIG.get("region_filter", None)

if region_filter is not None and "region" in df.columns:
    before = len(df)
    df = df[df["region"].isin(region_filter)].copy()
    print(f"Region filter: {region_filter}")
    print(f"  Rows before filter: {before:,}, after: {len(df):,}")
    if len(df) == 0:
        print("WARNING: Zero rows after region filter! Check region names.")
    region_breakdown = df["region"].value_counts()
    print(f"  Region breakdown:")
    for r, n in region_breakdown.items():
        print(f"    {r}: {n:,}")

if country_filter is not None and "country" in df.columns:
    before = len(df)
    df = df[df["country"].isin(country_filter)].copy()
    print(f"Country filter: {country_filter}")
    print(f"  Rows before filter: {before:,}, after: {len(df):,}")
    if len(df) == 0:
        print("WARNING: Zero rows after country filter! Check country names.")
    for c in country_filter:
        n = (df["country"] == c).sum()
        print(f"  {c}: {n:,} events")

# ---------------------------------------------------------------------------
# 4. Parse coordinates
# ---------------------------------------------------------------------------
for col in ["latitude", "longitude"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

n_bad_coords = df[["latitude", "longitude"]].isna().any(axis=1).sum()
if n_bad_coords > 0:
    print(f"WARNING: {n_bad_coords} rows with missing/bad coordinates — dropping them")
    df = df.dropna(subset=["latitude", "longitude"])

# Basic coordinate sanity
bad_lat = (df["latitude"].abs() > 90).sum()
bad_lon = (df["longitude"].abs() > 180).sum()
if bad_lat > 0 or bad_lon > 0:
    print(f"WARNING: {bad_lat} bad latitudes, {bad_lon} bad longitudes — dropping")
    df = df[(df["latitude"].abs() <= 90) & (df["longitude"].abs() <= 180)]

# ---------------------------------------------------------------------------
# 5. Parse fatalities
# ---------------------------------------------------------------------------
if "fatalities" in df.columns:
    df["fatalities"] = pd.to_numeric(df["fatalities"], errors="coerce").fillna(0).astype(int)
    df["deaths_civilians"] = np.where(df["event_type"] == "Violence against civilians", df["fatalities"], 0).astype(int)

# ---------------------------------------------------------------------------
# 6. Fill missing string columns
# ---------------------------------------------------------------------------
for col in ["event_type", "sub_event_type", "actor1", "actor2",
            "disorder_type", "country", "admin1", "admin2"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# ---------------------------------------------------------------------------
# 6b. Source count (observability proxy)
# ---------------------------------------------------------------------------
# ACLED separates multiple sources with semicolons
if "source" in df.columns:
    df["source_count"] = df["source"].fillna("").str.split(";").apply(
        lambda x: len([s.strip() for s in x if s.strip()])
    )
    # Ensure minimum of 1 for any non-null source
    df.loc[df["source_count"] == 0, "source_count"] = 1
else:
    # If source column doesn't exist, assume all events have 1 source
    df["source_count"] = 1

print(f"\n--- SOURCE COUNT DIAGNOSTICS ---")
print(f"Mean source count per event: {df['source_count'].mean():.2f}")
print(f"Median source count: {df['source_count'].median():.1f}")
print(f"Max source count: {df['source_count'].max():.0f}")
print(f"Source count distribution:")
print(df["source_count"].value_counts().sort_index().to_string())

# ---------------------------------------------------------------------------
# 7. Create year_month column
# ---------------------------------------------------------------------------
df["year_month"] = df["event_date"].dt.to_period("M").astype(str)
# e.g. "2020-01"

# ---------------------------------------------------------------------------
# 8. Diagnostics
# ---------------------------------------------------------------------------
print(f"\n--- CLEANING DIAGNOSTICS ---")
print(f"Cleaned rows: {len(df):,}")
print(f"Date range: {df['event_date'].min().date()} to {df['event_date'].max().date()}")
print(f"Unique year-months: {df['year_month'].nunique()}")

if "event_type" in df.columns:
    print(f"\nEvent type distribution:")
    print(df["event_type"].value_counts().to_string())

if "country" in df.columns:
    print(f"\nTop 20 countries by event count:")
    print(df["country"].value_counts().head(20).to_string())

print(f"\nMissing values per column:")
print(df.isnull().sum().to_string())

# ---------------------------------------------------------------------------
# 9. Save
# ---------------------------------------------------------------------------
out_path = os.path.join(CONFIG["data_intermediate_dir"], "acled_cleaned.parquet")
df.to_parquet(out_path, index=False)
print(f"\nSaved cleaned ACLED data to: {out_path}")
print(f"Shape: {df.shape}")
