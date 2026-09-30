"""
01c_merge_sources.py
====================
Concatenate cleaned ACLED + cleaned UCDP events into a single events file
that the rest of the pipeline reads as if it were ACLED.

Priority B (2026-05-26): multi-source pre-training.
Strategy: drop the unified events back into `acled_cleaned.parquet` so
02_build_grid.py reads the merged corpus with zero downstream changes. The
source-of-origin is preserved as a `source_of_record` column for any future
analysis.

Inputs:
    data_intermediate/acled_cleaned.parquet   (from 01_ingest_acled.py)
    data_intermediate/ucdp_cleaned.parquet    (from 01b_ingest_ucdp.py)

Outputs:
    data_intermediate/events_unified.parquet  (full merged copy, persisted)
    data_intermediate/acled_cleaned.parquet   (OVERWRITTEN — downstream reads this)
    data_intermediate/acled_only_backup.parquet (backup of ACLED-only events)
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

acled_path = os.path.join(CONFIG["data_intermediate_dir"], "acled_cleaned.parquet")
ucdp_path = os.path.join(CONFIG["data_intermediate_dir"], "ucdp_cleaned.parquet")
backup_path = os.path.join(CONFIG["data_intermediate_dir"], "acled_only_backup.parquet")
unified_path = os.path.join(CONFIG["data_intermediate_dir"], "events_unified.parquet")

# ---------------------------------------------------------------------------
# 1. Load both sources
# ---------------------------------------------------------------------------
print(f"Reading ACLED: {acled_path}")
acled = pd.read_parquet(acled_path)
print(f"  ACLED rows: {len(acled):,}")
print(f"  ACLED columns: {list(acled.columns)}")

print(f"\nReading UCDP: {ucdp_path}")
ucdp = pd.read_parquet(ucdp_path)
print(f"  UCDP rows: {len(ucdp):,}")
print(f"  UCDP columns: {list(ucdp.columns)}")

# ---------------------------------------------------------------------------
# 2. Backup ACLED-only file if not already present
# ---------------------------------------------------------------------------
if not os.path.exists(backup_path):
    acled.to_parquet(backup_path, index=False)
    print(f"\nBacked up ACLED-only events to: {backup_path}")
else:
    print(f"\nACLED-only backup already exists at: {backup_path} (not overwriting)")

# ---------------------------------------------------------------------------
# 3. Stamp source_of_record
# ---------------------------------------------------------------------------
acled = acled.copy()
ucdp = ucdp.copy()
acled["source_of_record"] = "ACLED"
ucdp["source_of_record"] = "UCDP"

# ---------------------------------------------------------------------------
# 4. Align columns
# ---------------------------------------------------------------------------
# Drop columns that exist in ACLED but not UCDP, and add empty columns where
# UCDP needs them but ACLED has them too.
common_cols = sorted(set(acled.columns) & set(ucdp.columns))
print(f"\nCommon columns ({len(common_cols)}): {common_cols}")

# For columns ACLED has that UCDP doesn't, drop them from ACLED (keeps schema tight)
acled_only = set(acled.columns) - set(ucdp.columns)
ucdp_only = set(ucdp.columns) - set(acled.columns)
print(f"ACLED-only columns (dropped from merge): {sorted(acled_only)}")
print(f"UCDP-only columns (dropped from merge): {sorted(ucdp_only)}")

acled_merge = acled[common_cols].copy()
ucdp_merge = ucdp[common_cols].copy()

# ---------------------------------------------------------------------------
# 5. Concatenate
# ---------------------------------------------------------------------------
merged = pd.concat([acled_merge, ucdp_merge], ignore_index=True)
print(f"\nMerged rows: {len(merged):,}  ({len(acled_merge):,} ACLED + {len(ucdp_merge):,} UCDP)")

# Re-sort by event_date (chronological) for any downstream code that cares
if "event_date" in merged.columns:
    merged["event_date"] = pd.to_datetime(merged["event_date"], errors="coerce")
    merged = merged.sort_values("event_date").reset_index(drop=True)

# ---------------------------------------------------------------------------
# 6. Diagnostics
# ---------------------------------------------------------------------------
print(f"\n--- MERGED EVENTS DIAGNOSTICS ---")
print(f"Source breakdown:")
print(merged["source_of_record"].value_counts().to_string())
print(f"\nCountry breakdown:")
print(merged["country"].value_counts().to_string())
print(f"\nDate range: {merged['event_date'].min()} to {merged['event_date'].max()}")
print(f"\nEvent type breakdown:")
print(merged["event_type"].value_counts().to_string())
print(f"\ninter1 distribution (top 10): {merged['inter1'].value_counts().head(10).to_dict()}")
print(f"inter2 distribution (top 10): {merged['inter2'].value_counts().head(10).to_dict()}")

# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------
# Persistent unified copy
merged.to_parquet(unified_path, index=False)
print(f"\nSaved unified events to: {unified_path}")

# Overwrite the file the rest of the pipeline reads
merged.to_parquet(acled_path, index=False)
print(f"Overwrote {acled_path} with merged events (so 02_build_grid.py reads it).")
print(f"Shape: {merged.shape}")
