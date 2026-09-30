"""
03_aggregate_location_month.py
==============================
Build a balanced grid-cell × month panel with aggregated event counts
and the main binary outcome variable.

Key steps:
    1. Aggregate events to grid-month level
    2. Build complete balanced panel (all grid-cells × all months)
    3. Fill missing grid-months with zeros
    4. Create outcome: violence_next_month
    5. Create alternative outcomes (counts, type-specific)

Outputs:
    data_intermediate/grid_month_panel.parquet

NOTE ON OBSERVABILITY:
    ACLED coverage reflects both actual violence and reporting/monitoring capacity.
    Zero-event months may reflect true peace OR reporting gaps. This pipeline
    predicts *observed* event patterns, not latent "true" conflict. Country and
    spatial metadata are preserved to enable future observability controls.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
master = import_module("00_master")
CONFIG = master.CONFIG
master.ensure_dirs()

np.random.seed(CONFIG["random_seed"])

# ---------------------------------------------------------------------------
# 1. Load gridded events
# ---------------------------------------------------------------------------
in_path = os.path.join(CONFIG["data_intermediate_dir"], "acled_with_grid.parquet")
print(f"Loading gridded ACLED from: {in_path}")
df = pd.read_parquet(in_path)
print(f"Event rows: {len(df):,}")

# ---------------------------------------------------------------------------
# 2. Flag violent events
# ---------------------------------------------------------------------------
violent_types = CONFIG["violent_event_types"]
df["is_violent"] = df["event_type"].isin(violent_types).astype(int)
print(f"\nViolent event types: {violent_types}")
print(f"Violent events: {df['is_violent'].sum():,} / {len(df):,} "
      f"({df['is_violent'].mean()*100:.1f}%)")

# ---------------------------------------------------------------------------
# 3. Aggregate to grid-month
# ---------------------------------------------------------------------------
# Count events by grid-month
agg = df.groupby(["grid_id", "year_month"]).agg(
    count_all_events=("event_type", "size"),
    count_violent_events=("is_violent", "sum"),
    count_fatalities=("fatalities", "sum") if "fatalities" in df.columns else ("is_violent", "size"),
    grid_lat_center=("grid_lat_center", "first"),
    grid_lon_center=("grid_lon_center", "first"),
    country=("country", "first"),  # modal country for this cell
).reset_index()

# Also count by event type for the baseline features later
event_type_counts = df.groupby(["grid_id", "year_month", "event_type"]).size().unstack(
    fill_value=0
).reset_index()
event_type_counts.columns.name = None
# Prefix event type columns
rename_et = {c: f"etype_{c.lower().replace(' ', '_').replace('/', '_')}"
             for c in event_type_counts.columns if c not in ["grid_id", "year_month"]}
event_type_counts = event_type_counts.rename(columns=rename_et)

agg = agg.merge(event_type_counts, on=["grid_id", "year_month"], how="left")

print(f"Grid-months with events: {len(agg):,}")

# ---------------------------------------------------------------------------
# 4. Build balanced panel
# ---------------------------------------------------------------------------
# Get all unique grid cells and all unique months
all_grids = agg["grid_id"].unique()
all_months = sorted(df["year_month"].unique())

print(f"Unique grid cells: {len(all_grids):,}")
print(f"Unique months: {len(all_months)}")
print(f"Month range: {all_months[0]} to {all_months[-1]}")
print(f"Full panel size: {len(all_grids) * len(all_months):,}")

# Build complete index
panel_index = pd.MultiIndex.from_product(
    [all_grids, all_months], names=["grid_id", "year_month"]
)
panel = pd.DataFrame(index=panel_index).reset_index()

# Merge in aggregated data
panel = panel.merge(agg, on=["grid_id", "year_month"], how="left")

# Fill missing counts with 0
count_cols = [c for c in panel.columns if c.startswith("count_") or c.startswith("etype_")]
panel[count_cols] = panel[count_cols].fillna(0).astype(int)

# Fill grid center coordinates and country from any non-null row for that grid
grid_meta = agg.groupby("grid_id").agg(
    grid_lat_center=("grid_lat_center", "first"),
    grid_lon_center=("grid_lon_center", "first"),
    country=("country", "first"),
).reset_index()

# Drop the partial fill from the merge, re-merge from grid_meta
panel = panel.drop(columns=["grid_lat_center", "grid_lon_center", "country"], errors="ignore")
panel = panel.merge(grid_meta, on="grid_id", how="left")

# Binary violence indicator for this month
panel["has_violence"] = (panel["count_violent_events"] > 0).astype(int)

# ---------------------------------------------------------------------------
# 5. Create forward-shifted target: violence_next_month
# ---------------------------------------------------------------------------
# Sort by grid and time
panel = panel.sort_values(["grid_id", "year_month"]).reset_index(drop=True)

# Shift within each grid cell
panel["violence_next_month"] = (
    panel.groupby("grid_id")["has_violence"]
    .shift(-1)
)

# Alternative targets
panel["count_violent_next_month"] = (
    panel.groupby("grid_id")["count_violent_events"]
    .shift(-1)
)
panel["count_all_next_month"] = (
    panel.groupby("grid_id")["count_all_events"]
    .shift(-1)
)

# The last month for each grid will have NaN target — that's correct.
# Those rows will be excluded from training/evaluation.
n_missing_target = panel["violence_next_month"].isna().sum()
print(f"\nRows with missing target (last month per cell): {n_missing_target:,}")

# ---------------------------------------------------------------------------
# 6. Diagnostics
# ---------------------------------------------------------------------------
valid = panel.dropna(subset=["violence_next_month"])
print(f"\n--- PANEL DIAGNOSTICS ---")
print(f"Total panel rows: {len(panel):,}")
print(f"Rows with valid target: {len(valid):,}")
print(f"Target class balance:")
print(f"  violence_next_month=0: {(valid['violence_next_month']==0).sum():,} "
      f"({(valid['violence_next_month']==0).mean()*100:.1f}%)")
print(f"  violence_next_month=1: {(valid['violence_next_month']==1).sum():,} "
      f"({(valid['violence_next_month']==1).mean()*100:.1f}%)")
print(f"\nFraction of grid-months with zero events: "
      f"{(panel['count_all_events']==0).mean()*100:.1f}%")

# Class balance plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

counts = valid["violence_next_month"].value_counts().sort_index()
axes[0].bar(["No violence (0)", "Violence (1)"], counts.values,
            color=["steelblue", "firebrick"])
axes[0].set_ylabel("Count")
axes[0].set_title("Target class balance: violence_next_month")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + len(valid)*0.01, f"{v:,}", ha="center", fontsize=9)

# Events per grid-month distribution
axes[1].hist(panel["count_all_events"].values, bins=50,
             edgecolor="black", alpha=0.7)
axes[1].set_xlabel("Events per grid-month")
axes[1].set_ylabel("Count (grid-months)")
axes[1].set_title("Distribution of events per grid-month")
axes[1].set_yscale("log")

plt.tight_layout()
fig_path = os.path.join(CONFIG["outputs_dir"], "panel_diagnostics.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"Saved panel diagnostics plot: {fig_path}")

# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------
out_path = os.path.join(CONFIG["data_intermediate_dir"], "grid_month_panel.parquet")
panel.to_parquet(out_path, index=False)
print(f"\nSaved grid-month panel to: {out_path}")
print(f"Shape: {panel.shape}")
print(f"Columns: {list(panel.columns)}")
