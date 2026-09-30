"""
02_build_grid.py
================
Assign each cleaned event to a grid cell.

Two grid schemes supported via CONFIG["grid_scheme"]:

  "priogrid" (Stage B+ default):
      Use PRIO-GRID 2.0 native cell IDs (gid). Cells are 0.5° x 0.5°.
      gid = row * 720 + col + 1
      where row = floor((lat + 90) * 2)  in [0, 360)
            col = floor((lon + 180) * 2) in [0, 720)
      No shapefile needed; the gid is a pure arithmetic function of lat/lon
      because PRIO-GRID is a regular global lat/lon grid.
      Comparability with ViEWS and the rest of the PRIO-GRID covariate stack.

  "synthetic" (pilot / Stages A and Priority A/B):
      Floor lat/lon to nearest grid_size_deg increment; grid_id is the
      string "{lat_center:.2f}_{lon_center:.2f}". Legacy.

Outputs:
    data_intermediate/acled_with_grid.parquet
        events with grid_id (int gid OR string lat_lon), grid_lat_center,
        grid_lon_center. Use grid_id as the unique key downstream.

Diagnostics:
    - number of unique grid cells
    - events per grid cell distribution
    - top 20 most active grid cells
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
# 1. Load cleaned data
# ---------------------------------------------------------------------------
in_path = os.path.join(CONFIG["data_intermediate_dir"], "acled_cleaned.parquet")
print(f"Loading cleaned ACLED from: {in_path}")
df = pd.read_parquet(in_path)
print(f"Rows: {len(df):,}")

# ---------------------------------------------------------------------------
# 2. Assign grid cells
# ---------------------------------------------------------------------------
grid_size = CONFIG["grid_size_deg"]
grid_scheme = CONFIG.get("grid_scheme", "synthetic")
print(f"Grid size: {grid_size}°  scheme={grid_scheme}")

# Floor to nearest grid increment (used for centroids in both schemes)
df["grid_lat_floor"] = np.floor(df["latitude"] / grid_size) * grid_size
df["grid_lon_floor"] = np.floor(df["longitude"] / grid_size) * grid_size
df["grid_lat_center"] = df["grid_lat_floor"] + grid_size / 2
df["grid_lon_center"] = df["grid_lon_floor"] + grid_size / 2

if grid_scheme == "priogrid":
    # PRIO-GRID 2.0 native cell numbering.
    # Requires grid_size == 0.5 to match PRIO-GRID's 720 x 360 layout.
    assert abs(grid_size - 0.5) < 1e-9, (
        f"grid_scheme='priogrid' requires grid_size_deg=0.5, got {grid_size}"
    )
    # Use the floor (lower-left corner) to determine the cell, not the centroid,
    # to avoid edge-case rounding at exact half-degree boundaries.
    col_idx = np.floor((df["longitude"].values + 180.0) * 2).astype(np.int64)
    row_idx = np.floor((df["latitude"].values + 90.0) * 2).astype(np.int64)
    # Clamp to valid range (in case of any out-of-range coords)
    col_idx = np.clip(col_idx, 0, 719)
    row_idx = np.clip(row_idx, 0, 359)
    df["grid_id"] = (row_idx * 720 + col_idx + 1).astype(np.int64)
    print(f"  Assigned PRIO-GRID gid (range: {df['grid_id'].min()} - {df['grid_id'].max()})")
elif grid_scheme == "synthetic":
    # Legacy string grid_id "{lat_center:.2f}_{lon_center:.2f}"
    df["grid_id"] = (
        df["grid_lat_center"].map(lambda x: f"{x:.2f}")
        + "_"
        + df["grid_lon_center"].map(lambda x: f"{x:.2f}")
    )
else:
    raise ValueError(f"Unknown grid_scheme: {grid_scheme!r}")

# Drop intermediate floor columns
df = df.drop(columns=["grid_lat_floor", "grid_lon_floor"])

# ---------------------------------------------------------------------------
# 3. Diagnostics
# ---------------------------------------------------------------------------
n_cells = df["grid_id"].nunique()
events_per_cell = df.groupby("grid_id").size()

print(f"\n--- GRID DIAGNOSTICS ---")
print(f"Unique grid cells: {n_cells:,}")
print(f"Events per cell — min: {events_per_cell.min()}, "
      f"median: {events_per_cell.median():.0f}, "
      f"mean: {events_per_cell.mean():.1f}, "
      f"max: {events_per_cell.max()}")

print(f"\nTop 20 most active grid cells:")
top20 = events_per_cell.nlargest(20)
for cell_id, count in top20.items():
    # Get country for context
    country = df.loc[df["grid_id"] == cell_id, "country"].mode()
    country_str = country.iloc[0] if len(country) > 0 else "?"
    print(f"  {cell_id}  ({country_str}): {count:,} events")

# Distribution plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram of events per cell (log scale)
axes[0].hist(events_per_cell.values, bins=50, edgecolor="black", alpha=0.7)
axes[0].set_xlabel("Events per grid cell")
axes[0].set_ylabel("Number of grid cells")
axes[0].set_title("Distribution of events per grid cell")
axes[0].set_yscale("log")

# Top 30 cells bar chart
top30 = events_per_cell.nlargest(30)
axes[1].barh(range(len(top30)), top30.values)
axes[1].set_yticks(range(len(top30)))
axes[1].set_yticklabels(top30.index, fontsize=6)
axes[1].set_xlabel("Event count")
axes[1].set_title("Top 30 most active grid cells")
axes[1].invert_yaxis()

plt.tight_layout()
fig_path = os.path.join(CONFIG["outputs_dir"], "grid_activity_distribution.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"Saved activity distribution plot: {fig_path}")

# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
out_path = os.path.join(CONFIG["data_intermediate_dir"], "acled_with_grid.parquet")
df.to_parquet(out_path, index=False)
print(f"\nSaved gridded data to: {out_path}")
print(f"Shape: {df.shape}")
