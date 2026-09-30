"""
05_build_baseline_features.py
=============================
Construct a ViEWS-style baseline feature set for the downstream forecasting task.

Features (all using only past data — no leakage):
    1. Lagged violence indicators: has_violence at t-1, t-3, t-6
    2. Rolling violent event counts: sum over past 1, 3, 6 months
    3. Rolling all-event counts: sum over past 1, 3, 6 months
    4. Rolling fatality counts: sum over past 1, 3, 6 months
    5. Event-type counts for prior month (one column per type)
    6. Spatial lag: mean violence indicator of cardinal neighbors at t-1
    7. Country indicators (one-hot, capped at top N countries)

All features are constructed from information available at time t,
to predict violence_next_month (at t+1).

Outputs:
    data_intermediate/baseline_features.parquet
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

# Optional suffix to build a feature-set variant (e.g. REPR_SEQ_SUFFIX=_super
# reads sequences_super.parquet -> baseline_features_super.parquet). Default "".
SUF = os.environ.get("REPR_SEQ_SUFFIX", "")

# ---------------------------------------------------------------------------
# 1. Load panel with sequences (has split assignments and targets)
# ---------------------------------------------------------------------------
in_path = os.path.join(CONFIG["data_intermediate_dir"], f"sequences{SUF}.parquet")
print(f"Loading panel from: {in_path}")
panel = pd.read_parquet(in_path)
print(f"Rows: {len(panel):,}")

# Sort by grid and time — critical for lag operations
panel = panel.sort_values(["grid_id", "year_month"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. Temporal lags and rolling features (within each grid cell)
# ---------------------------------------------------------------------------
print("Building temporal lag features...")

# Group by grid for shift operations
grp = panel.groupby("grid_id")

# Lagged violence indicators
for lag in [1, 3, 6]:
    panel[f"violence_lag{lag}"] = grp["has_violence"].shift(lag)

# Rolling sums — use shift(1) first so we don't include current month
# Then rolling over the shifted series
for window in [1, 3, 6]:
    # Violent event counts: sum of past `window` months (not including current)
    panel[f"violent_count_roll{window}"] = (
        grp["count_violent_events"]
        .apply(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
        .reset_index(level=0, drop=True)
    )

    # All event counts
    panel[f"all_count_roll{window}"] = (
        grp["count_all_events"]
        .apply(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
        .reset_index(level=0, drop=True)
    )

    # Fatality counts
    if "count_fatalities" in panel.columns:
        panel[f"fatalities_roll{window}"] = (
            grp["count_fatalities"]
            .apply(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
            .reset_index(level=0, drop=True)
        )

print("  Done with temporal lags and rolling features.")

# ---------------------------------------------------------------------------
# 3. Event-type counts from prior month
# ---------------------------------------------------------------------------
etype_cols = [c for c in panel.columns if c.startswith("etype_")]
print(f"Event-type columns for prior-month counts: {etype_cols}")

for col in etype_cols:
    panel[f"{col}_lag1"] = grp[col].shift(1)

# ---------------------------------------------------------------------------
# 3b. Observability proxy features
# ---------------------------------------------------------------------------
# These features capture reporting coverage and recency, helping distinguish
# true absence of violence from reporting gaps. Grid-months with no recent
# events may reflect lack of media/NGO coverage rather than actual peace.
# See also source_count in raw ACLED data for a more direct observability proxy.

print("Building observability proxy features...")

# log event density: log-transform of all-event count to smooth scale
# Shifted by 1 month so it only uses past info
panel["log_event_density_lag1"] = np.log1p(
    grp["count_all_events"].shift(1).fillna(0)
)

# Binary indicator: any event at all in prior month
panel["any_event_lag1"] = (grp["count_all_events"].shift(1) > 0).astype(int)

# Months since last event (consecutive zero-event months, capped at 12)
# For each grid, count backwards from current month until we hit a nonzero month
def months_since_event(series):
    """Count consecutive zero months backward from current index."""
    result = []
    for i in range(len(series)):
        if series.iloc[i] > 0:
            result.append(0)
        else:
            # Count backward
            count = 0
            for j in range(i - 1, -1, -1):
                if series.iloc[j] > 0:
                    break
                count += 1
            result.append(min(count, 12))  # cap at 12 months
    return pd.Series(result, index=series.index)

panel["months_since_last_event"] = (
    grp["count_all_events"]
    .apply(months_since_event)
    .reset_index(level=0, drop=True)
)

print("  Done with observability proxy features.")

# ---------------------------------------------------------------------------
# 3c. Event-type transition features
# ---------------------------------------------------------------------------
print("Building event-type transition features...")

# 1. Escalation: protests/riots in t-2 AND battles/explosions/violence-against-civilians in t-1
protest_types = [c for c in etype_cols if any(x in c.lower() for x in ["protest", "riot"])]
violence_types = [c for c in etype_cols if any(x in c.lower() for x in ["battle", "explosion", "violence"])]

print(f"  Protest/riot types identified: {protest_types}")
print(f"  Battle/explosion/violence types identified: {violence_types}")

if protest_types and violence_types:
    # Has protests/riots in t-2
    had_protest_2mo_ago = (grp[protest_types].shift(2) > 0).any(axis=1).astype(int)
    # Has violence in t-1
    had_violence_1mo_ago = (grp[violence_types].shift(1) > 0).any(axis=1).astype(int)
    panel["escalation_lag1"] = had_protest_2mo_ago * had_violence_1mo_ago
else:
    print("  Warning: Could not identify sufficient protest or violence event types. Setting escalation_lag1 to 0.")
    panel["escalation_lag1"] = 0

# 2. New violence onset: has_violence=1 at t-1 AND has_violence=0 at t-2
violence_t1 = grp["has_violence"].shift(1)
violence_t2 = grp["has_violence"].shift(2)
panel["new_violence_onset"] = ((violence_t1 == 1) & (violence_t2 == 0)).astype(int)

# 3. Violence cessation: has_violence=0 at t-1 AND has_violence=1 at t-2
panel["violence_cessation"] = ((violence_t1 == 0) & (violence_t2 == 1)).astype(int)

# 4. Event diversity in t-1: count of distinct event types with nonzero count
# For each row, count how many etype_ columns had count > 0 in t-1
panel["event_diversity_lag1"] = grp[etype_cols].apply(
    lambda x: (x.shift(1) > 0).sum(axis=1)
).reset_index(level=0, drop=True)

# 5. Fatality spike: fatalities in t-1 exceed cell's rolling 6-month mean by > 2 std
if "count_fatalities" in panel.columns:
    # Rolling mean and std of fatalities (using shifted data for no leakage)
    rolling_mean_6m = grp["count_fatalities"].apply(
        lambda x: x.shift(1).rolling(6, min_periods=1).mean()
    ).reset_index(level=0, drop=True)
    rolling_std_6m = grp["count_fatalities"].apply(
        lambda x: x.shift(1).rolling(6, min_periods=1).std()
    ).reset_index(level=0, drop=True)

    # Fatalities at t-1 (already shifted by 1)
    fatalities_t1 = grp["count_fatalities"].shift(1)

    # Spike: t-1 value > mean + 2*std (use fillna(0) for std in early months)
    panel["fatality_spike_lag1"] = (
        (fatalities_t1 > rolling_mean_6m + 2 * rolling_std_6m.fillna(0))
    ).astype(int)
    panel["fatality_spike_lag1"] = panel["fatality_spike_lag1"].fillna(0)
    print("  Fatality spike feature created.")
else:
    print("  Note: count_fatalities column not found; skipping fatality_spike_lag1.")

# Fill any remaining NaNs in transition features
transition_cols = ["escalation_lag1", "new_violence_onset", "violence_cessation",
                   "event_diversity_lag1"]
if "fatality_spike_lag1" in panel.columns:
    transition_cols.append("fatality_spike_lag1")

for col in transition_cols:
    if col in panel.columns:
        panel[col] = panel[col].fillna(0)

print("  Done with event-type transition features.")

# ---------------------------------------------------------------------------
# 4. Spatial lag: mean violence of cardinal neighbors at t-1
# ---------------------------------------------------------------------------
print("Building spatial lag features...")

grid_size = CONFIG["grid_size_deg"]

# Parse grid centers
panel["_glat"] = panel["grid_lat_center"]
panel["_glon"] = panel["grid_lon_center"]

# Build a (rounded lat, rounded lon) -> grid_id lookup. Works for both
# integer PRIO-GRID gids and legacy synthetic string grid_ids — we use the
# panel's own grid_id values rather than constructing them.
unique_cells = panel[["grid_id", "_glat", "_glon"]].drop_duplicates()

# Round to avoid floating-point key collisions
def _rkey(lat, lon):
    return (round(float(lat), 4), round(float(lon), 4))

cell_lookup = {_rkey(r["_glat"], r["_glon"]): r["grid_id"] for _, r in unique_cells.iterrows()}

neighbor_pairs = []
for _, row in unique_cells.iterrows():
    lat, lon = row["_glat"], row["_glon"]
    gid = row["grid_id"]
    # 4 cardinal neighbors. If a neighbor cell isn't in the panel (edge
    # of analysis region, ocean, etc.) we just skip it.
    for dlat, dlon in [(grid_size, 0), (-grid_size, 0), (0, grid_size), (0, -grid_size)]:
        nbr_key = _rkey(lat + dlat, lon + dlon)
        if nbr_key in cell_lookup:
            neighbor_pairs.append({"grid_id": gid, "neighbor_id": cell_lookup[nbr_key]})

neighbors_df = pd.DataFrame(neighbor_pairs)
print(f"  Neighbor pairs: {len(neighbors_df):,}")

# Get lagged violence by grid-month
lag_violence = panel[["grid_id", "year_month", "has_violence"]].copy()
lag_violence = lag_violence.rename(columns={"grid_id": "neighbor_id",
                                             "has_violence": "nbr_violence"})

# Merge neighbors with their lagged violence, then average per (grid_id, year_month)
# This is the spatial lag at time t (not shifted — we'll shift after)
nbr_merged = neighbors_df.merge(lag_violence, on="neighbor_id", how="inner")
spatial_lag = nbr_merged.groupby(["grid_id", "year_month"])["nbr_violence"].mean().reset_index()
spatial_lag = spatial_lag.rename(columns={"nbr_violence": "spatial_lag_violence"})

panel = panel.merge(spatial_lag, on=["grid_id", "year_month"], how="left")
panel["spatial_lag_violence"] = panel["spatial_lag_violence"].fillna(0)

# Shift spatial lag by 1 month so it uses only past info.
# Rebuild the groupby because the original grp (line ~52) predates the merge
# of spatial_lag_violence onto panel.
panel["spatial_lag_violence"] = panel.groupby("grid_id")["spatial_lag_violence"].shift(1)

# Clean up temp columns
panel = panel.drop(columns=["_glat", "_glon"], errors="ignore")

print("  Done with spatial lag.")

# ---------------------------------------------------------------------------
# 5. Country indicators
# ---------------------------------------------------------------------------
print("Building country indicators...")

# Cap at top N countries by event frequency to avoid huge dummy matrix
top_countries = panel.loc[panel["count_all_events"] > 0, "country"].value_counts().head(50).index.tolist()
panel["country_cat"] = panel["country"].where(panel["country"].isin(top_countries), "OTHER")
country_dummies = pd.get_dummies(panel["country_cat"], prefix="country", dtype=int)
panel = pd.concat([panel, country_dummies], axis=1)

country_cols = [c for c in country_dummies.columns]
print(f"  Country indicator columns: {len(country_cols)}")

# ---------------------------------------------------------------------------
# 6. Collect feature column names
# ---------------------------------------------------------------------------
baseline_feature_cols = []

# Temporal lags
baseline_feature_cols += [f"violence_lag{lag}" for lag in [1, 3, 6]]

# Rolling counts
for window in [1, 3, 6]:
    baseline_feature_cols.append(f"violent_count_roll{window}")
    baseline_feature_cols.append(f"all_count_roll{window}")
    if f"fatalities_roll{window}" in panel.columns:
        baseline_feature_cols.append(f"fatalities_roll{window}")

# Event type lags
baseline_feature_cols += [f"{c}_lag1" for c in etype_cols]

# Observability proxy features
baseline_feature_cols.extend([
    "log_event_density_lag1",
    "any_event_lag1",
    "months_since_last_event"
])

# Event-type transition features
transition_feature_cols = [
    "escalation_lag1",
    "new_violence_onset",
    "violence_cessation",
    "event_diversity_lag1"
]
if "fatality_spike_lag1" in panel.columns:
    transition_feature_cols.append("fatality_spike_lag1")
baseline_feature_cols.extend(transition_feature_cols)

# Spatial lag
baseline_feature_cols.append("spatial_lag_violence")

# Country dummies
baseline_feature_cols += country_cols

# Verify all columns exist
missing = [c for c in baseline_feature_cols if c not in panel.columns]
if missing:
    print(f"WARNING: Missing feature columns: {missing}")
    baseline_feature_cols = [c for c in baseline_feature_cols if c in panel.columns]

print(f"\nTotal baseline features: {len(baseline_feature_cols)}")

# Save feature column list for downstream scripts
import json
feat_list_path = os.path.join(CONFIG["data_intermediate_dir"], f"baseline_feature_cols{SUF}.json")
with open(feat_list_path, "w") as f:
    json.dump(baseline_feature_cols, f, indent=2)
print(f"Saved feature column list to: {feat_list_path}")

# ---------------------------------------------------------------------------
# 7. Fill NaN features with 0
# ---------------------------------------------------------------------------
# NaN arises from lags at the start of the series — those months are excluded
# from train/val/test anyway, but we fill for safety
panel[baseline_feature_cols] = panel[baseline_feature_cols].fillna(0)

# ---------------------------------------------------------------------------
# 8. Diagnostics
# ---------------------------------------------------------------------------
print(f"\n--- BASELINE FEATURE DIAGNOSTICS ---")
for col in baseline_feature_cols[:20]:  # show first 20
    print(f"  {col}: mean={panel[col].mean():.4f}, "
          f"std={panel[col].std():.4f}, "
          f"min={panel[col].min():.1f}, max={panel[col].max():.1f}")
if len(baseline_feature_cols) > 20:
    print(f"  ... and {len(baseline_feature_cols) - 20} more features")

# Show diagnostics for the new event-type transition features
print(f"\n--- EVENT-TYPE TRANSITION FEATURE DIAGNOSTICS ---")
for col in transition_feature_cols:
    if col in panel.columns:
        print(f"  {col}: mean={panel[col].mean():.4f}, "
              f"std={panel[col].std():.4f}, "
              f"min={panel[col].min():.1f}, max={panel[col].max():.1f}")

# Verify no leakage: for train rows, all feature values should use only past data
# This is guaranteed by construction (shift operations), but let's double-check
# that violence_lag1 for month t equals has_violence for month t-1
sample_grid = panel["grid_id"].iloc[0]
sample = panel[panel["grid_id"] == sample_grid].head(15)[
    ["year_month", "has_violence", "violence_lag1", "violence_next_month"]
]
print(f"\nSanity check — lag alignment for grid {sample_grid}:")
print(sample.to_string(index=False))

# ---------------------------------------------------------------------------
# 9. Save
# ---------------------------------------------------------------------------
out_path = os.path.join(CONFIG["data_intermediate_dir"], f"baseline_features{SUF}.parquet")
panel.to_parquet(out_path, index=False)
print(f"\nSaved baseline features to: {out_path}")
print(f"Shape: {panel.shape}")
