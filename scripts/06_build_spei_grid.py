"""
06_build_spei_grid.py

Build a SPEI drought-onset grid-month panel, matching the real schema of
data_intermediate/grid_month_panel.parquet (per scripts/02_build_grid.py and
scripts/03_aggregate_location_month.py), for use as a transfer-learning
target domain with the geographic transformer.

Schema match, confirmed against the actual pipeline on 2026-10-01:
  - grid_id: PRIO-GRID 2.0 native arithmetic cell id, NOT a shapefile lookup.
    col = floor((lon + 180) * 2), clipped to [0, 719]
    row = floor((lat + 90) * 2), clipped to [0, 359]
    grid_id = row * 720 + col + 1
  - year_month: string "YYYY-MM" (pandas Period("M") cast to str)
  - Output: parquet, not csv, matching data_intermediate/grid_month_panel.parquet
  - Target naming convention: "<var>_next_month" (forward-shifted by one month,
    last month per cell is NaN by construction, dropped before eval)

Run sequentially, no helper functions, per Jared's coding preference.
Fails loudly (assertions) rather than silently producing a malformed panel.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr

# ---- Hardcoded config -------------------------------------------------

SEED = 123
np.random.seed(SEED)

SPEI_NETCDF_PATH = "data_raw/spei/spei01.nc"  # SPEIbase v2.9/2.10, 1-month SPEI
OUT_PATH = "data_intermediate/spei_grid_month_panel.parquet"

# SPEI accumulation window. 1-month SPEI matches the conflict panel's
# month-to-month granularity. Use spei03/spei06 var names instead for
# accumulated-deficit framing.
SPEI_VAR_NAME = "spei"

# Severe-drought threshold (standard SPEI convention)
ONSET_THRESHOLD = -1.5

# Restrict the first run to a bounding box, not the full panel, to keep
# this cheap to validate. Horn of Africa bounding box (lon_min, lon_max,
# lat_min, lat_max) -- adjust to match your actual LORO region definitions
# before treating results as final.
REGION_NAME = "horn_of_africa"
LON_MIN, LON_MAX = 32.0, 51.0
LAT_MIN, LAT_MAX = -5.0, 15.0

# Date range -- must overlap your actual conflict panel's year_month range.
# Confirm the real range with:
#   python -c "import pandas as pd; p = pd.read_parquet('data_intermediate/grid_month_panel.parquet'); print(p['year_month'].min(), p['year_month'].max())"
# before trusting this placeholder.
START_YEAR_MONTH = "1990-01"
END_YEAR_MONTH = "2023-12"

# ---- Step 1: load SPEI NetCDF ------------------------------------------

assert os.path.exists(SPEI_NETCDF_PATH), (
    f"SPEI NetCDF not found at {SPEI_NETCDF_PATH}. Download SPEIbase v2.9/2.10 "
    "1-month SPEI (global, 0.5deg, monthly) from https://spei.csic.es/database.html "
    "before running this script."
)
spei_ds = xr.open_dataset(SPEI_NETCDF_PATH)
assert SPEI_VAR_NAME in spei_ds.data_vars, (
    f"'{SPEI_VAR_NAME}' not in SPEI NetCDF. Available vars: "
    f"{list(spei_ds.data_vars)} -- fix SPEI_VAR_NAME"
)
lat_dim = "lat" if "lat" in spei_ds.dims else "latitude"
lon_dim = "lon" if "lon" in spei_ds.dims else "longitude"
assert lat_dim in spei_ds.dims and lon_dim in spei_ds.dims, (
    f"Could not find lat/lon dims in SPEI NetCDF -- actual dims: {list(spei_ds.dims)}"
)
print(f"Loaded SPEI NetCDF: dims={dict(spei_ds.dims)}, lat_dim={lat_dim}, lon_dim={lon_dim}")

# ---- Step 2: subset to region bounding box and date range -------------

spei_ds = spei_ds.sel(
    {lon_dim: slice(LON_MIN, LON_MAX), lat_dim: slice(LAT_MIN, LAT_MAX)}
)
spei_ds = spei_ds.sel(time=slice(f"{START_YEAR_MONTH}-01", f"{END_YEAR_MONTH}-28"))
n_lat = spei_ds.dims[lat_dim]
n_lon = spei_ds.dims[lon_dim]
n_time = spei_ds.dims.get("time", 0)
assert n_lat > 0 and n_lon > 0, (
    f"Region bounding box ({LON_MIN},{LAT_MIN})-({LON_MAX},{LAT_MAX}) returned "
    "an empty grid -- check the box against the NetCDF's actual coordinate range"
)
assert n_time > 0, (
    f"No SPEI data in range {START_YEAR_MONTH} to {END_YEAR_MONTH} -- check "
    "the NetCDF's actual time coverage with spei_ds.time.values"
)
print(f"Subset to {REGION_NAME}: {n_lat} lat x {n_lon} lon x {n_time} months")

# ---- Step 3: flatten to a long dataframe, compute grid_id exactly as ---
# ---- scripts/02_build_grid.py does (arithmetic PRIO-GRID 2.0 index) ----

df = spei_ds[SPEI_VAR_NAME].to_dataframe().reset_index()
df = df.rename(columns={lat_dim: "latitude", lon_dim: "longitude", SPEI_VAR_NAME: "spei"})
df = df.dropna(subset=["spei"])
assert len(df) > 0, "All SPEI values are NaN in this region/date range -- check source data"

col_idx = np.floor((df["longitude"].values + 180.0) * 2).astype(np.int64)
row_idx = np.floor((df["latitude"].values + 90.0) * 2).astype(np.int64)
col_idx = np.clip(col_idx, 0, 719)
row_idx = np.clip(row_idx, 0, 359)
df["grid_id"] = (row_idx * 720 + col_idx + 1).astype(np.int64)
df["year_month"] = df["time"].dt.to_period("M").astype(str)
print(f"Assigned grid_id (range: {df['grid_id'].min()} - {df['grid_id'].max()})")

# SPEIbase is already 0.5deg, same resolution as PRIO-GRID, so multiple
# SPEI pixels should not map to the same grid_id within one bounding box --
# verify that assumption explicitly rather than silently averaging duplicates.
dupe_check = df.groupby(["grid_id", "year_month"]).size()
n_dupes = (dupe_check > 1).sum()
if n_dupes > 0:
    print(
        f"WARNING: {n_dupes} grid_id-year_month pairs have >1 SPEI pixel -- "
        "taking the mean. This is unexpected if SPEIbase is truly 0.5deg; "
        "check SPEI_NETCDF_PATH's actual resolution before trusting this."
    )
    df = df.groupby(["grid_id", "year_month"], as_index=False)["spei"].mean()
else:
    df = df[["grid_id", "year_month", "spei"]]

# ---- Step 4: build complete balanced panel (all grid-cells x all months) -
# Mirrors scripts/03_aggregate_location_month.py section 2.

all_grids = sorted(df["grid_id"].unique())
all_months = sorted(df["year_month"].unique())
print(f"Unique grid cells: {len(all_grids)}, unique months: {len(all_months)}")
print(f"Full panel size: {len(all_grids) * len(all_months):,}")

panel_index = pd.MultiIndex.from_product(
    [all_grids, all_months], names=["grid_id", "year_month"]
)
panel = pd.DataFrame(index=panel_index).reset_index()
panel = panel.merge(df, on=["grid_id", "year_month"], how="left")
# SPEIbase should have complete coverage within the region/date box -- any
# NaNs here mean the bounding box clips the edge of the NetCDF's grid.
n_missing = panel["spei"].isna().sum()
if n_missing > 0:
    print(
        f"WARNING: {n_missing} / {len(panel)} grid-months missing SPEI value "
        "-- likely bounding-box edge effect, not a true gap. Inspect before dropping."
    )

# ---- Step 5: binary drought onset, forward-shifted to match the --------
# "<var>_next_month" convention used for violence_next_month -------------

panel = panel.sort_values(["grid_id", "year_month"]).reset_index(drop=True)
panel["in_drought"] = panel["spei"] < ONSET_THRESHOLD
panel["drought_onset_this_month"] = (
    panel["in_drought"] & ~panel.groupby("grid_id")["in_drought"].shift(1).fillna(False)
).astype(int)

# Forward-shift by one month per grid_id, same direction as violence_next_month
panel["drought_onset_next_month"] = panel.groupby("grid_id")[
    "drought_onset_this_month"
].shift(-1)

n_missing_target = panel["drought_onset_next_month"].isna().sum()
print(f"Rows with missing target (last month per cell): {n_missing_target:,}")

valid = panel.dropna(subset=["drought_onset_next_month"])
onset_rate = valid["drought_onset_next_month"].mean()
print(f"drought_onset_next_month=1 rate: {onset_rate:.4%} ({len(valid):,} valid rows)")
assert 0.0 < onset_rate < 0.5, (
    f"Onset rate {onset_rate:.4%} is outside a plausible range -- check "
    "ONSET_THRESHOLD and the onset logic above before trusting this panel"
)

# ---- Step 6: write out, parquet to match the existing pipeline's format

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
panel.to_parquet(OUT_PATH, index=False)
print(f"Wrote {OUT_PATH}: {panel.shape}")
print(panel.head())
