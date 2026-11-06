"""build_panel.py (consolidated 2026-11-05) -- stage 01: the PRIO-GRID cell-month panel, in one file.
Replaces eleven scripts (tag v5-as-run): 00_master.py (its configuration, verbatim below; its old full-pipeline
__main__ block is dropped) and the ten steps
  acled      was scripts/01_ingest_acled.py
  ucdp       was scripts/01b_ingest_ucdp.py
  merge      was scripts/01c_merge_sources.py
  grid       was scripts/02_build_grid.py
  aggregate  was scripts/03_aggregate_location_month.py
  sequences  was scripts/04_build_sequences.py
  baseline   was scripts/05_build_baseline_features.py
  super      was scripts/build_sequences_super.py
  prio       was scripts/build_priogrid_static.py
  pgm        was scripts/build_pgm_fatalities_target.py
Run one step per call, from the project root:  python scripts/build_panel.py <step>
Each call is a fresh Python process that runs exactly one step, as the old one-script-per-step calls did. Every step block
is the old script unchanged, indented under its `if`, with two edits only: its docstring is kept as a comment, and its
config import reads this file (see IMPORT lines). pipeline/tasks.sbatch (rebuild, featbuild) and pipeline/run.sh
(data_prio, data_pgm) call the steps in the old order with the old environment (PHASE_MODE, REPR_SEQ_SUFFIX).
Parity: pipeline/tests/parity_phase2.py panel_build."""
# ======================= 00_master.py (configuration; verbatim apart from the dropped __main__ block) =======================
"""
00_master.py
============
Central configuration and orchestration for the conflict forecasting pipeline.

All configurable parameters live here in one place.
Run this script to execute the full pipeline sequentially,
or import CONFIG from here into individual scripts.

Usage:
    python scripts/00_master.py          # run full pipeline
    python scripts/01_ingest_acled.py    # run individual step (imports CONFIG from here)
"""

import os
import sys
import subprocess
import time

# ==========================================================================
# PROJECT ROOT — everything is relative to this
# ==========================================================================
# Assumes scripts/ is one level below project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ==========================================================================
# CENTRAL CONFIGURATION
# ==========================================================================
CONFIG = {
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    "project_root": PROJECT_ROOT,
    "data_raw_dir": os.path.join(PROJECT_ROOT, "data_raw"),
    "data_intermediate_dir": os.path.join(PROJECT_ROOT, "data_intermediate"),
    "data_model_dir": os.path.join(PROJECT_ROOT, "data_model"),
    "outputs_dir": os.path.join(PROJECT_ROOT, "outputs"),
    "logs_dir": os.path.join(PROJECT_ROOT, "logs"),
    "scripts_dir": SCRIPT_DIR,

    # Input file — put your ACLED CSV export here
    "acled_input_file": os.path.join(PROJECT_ROOT, "data_raw", "acled_full.csv"),

    # ------------------------------------------------------------------
    # ACLED column name mapping
    # Map from our internal names to the actual column names in your CSV.
    # Edit these if your export uses different headers.
    # ------------------------------------------------------------------
    "col_event_date": "event_date",
    "col_latitude": "latitude",
    "col_longitude": "longitude",
    "col_event_type": "event_type",
    "col_sub_event_type": "sub_event_type",
    "col_actor1": "actor1",
    "col_actor2": "actor2",
    "col_disorder_type": "disorder_type",
    "col_country": "country",
    "col_admin1": "admin1",
    "col_admin2": "admin2",
    "col_fatalities": "fatalities",
    "col_notes": "notes",
    "col_source": "source",

    # ------------------------------------------------------------------
    # Grid parameters
    # ------------------------------------------------------------------
    # Grid cell size in degrees.
    # 0.5 = PRIO-GRID standard (~55km). 1.0 = coarser (~111km), fewer cells.
    # Stage B uses 0.5° + native PRIO-GRID `gid` integers for grid IDs
    # (computed as row*720 + col + 1, matching PRIO-GRID 2.0 convention).
    "grid_size_deg": 0.5,
    "grid_scheme": "priogrid",   # "priogrid" -> use PRIO-GRID gid; "synthetic" -> lat_lon str (legacy)

    # ------------------------------------------------------------------
    # Country / region filter
    # country_filter: list of country names, or None to keep all.
    # region_filter: list of ACLED region strings, or None to keep all.
    # ------------------------------------------------------------------
    "country_filter": None,
    "region_filter": [
        "Northern Africa",
        "Western Africa",
        "Middle Africa",
        "Eastern Africa",
        "Southern Africa",
    ],

    # ------------------------------------------------------------------
    # Time parameters
    # ------------------------------------------------------------------
    "time_unit": "month",  # only month supported for now

    # ------------------------------------------------------------------
    # History window for event sequences (in months)
    # ------------------------------------------------------------------
    "history_window_months": 12,

    # ------------------------------------------------------------------
    # Train / Validation / Test cutoffs (YYYY-MM format, inclusive boundaries)
    # train: everything up through train_end
    # val:   train_end+1 through val_end
    # test:  val_end+1 through end of data
    # ------------------------------------------------------------------
    "train_end": "2020-12",
    "val_end": "2021-12",
    # test = everything after val_end

    # ------------------------------------------------------------------
    # Violence category rules
    # These ACLED event_type values count as "violent" for the main outcome.
    # ------------------------------------------------------------------
    "violent_event_types": [
        "Battles",
        "Explosions/Remote violence",
        "Violence against civilians",
    ],

    # ------------------------------------------------------------------
    # Actor vocabulary
    # Cap actor names at top N; everything else becomes "OTHER"
    # ------------------------------------------------------------------
    "actor_vocab_size": 500,

    # ------------------------------------------------------------------
    # Representation model hyperparameters
    # ------------------------------------------------------------------
    "repr_embed_dim": 64,          # embedding dimension for each field
    "repr_n_heads": 4,             # transformer attention heads
    "repr_n_layers": 2,            # transformer encoder layers
    "repr_max_seq_len": 200,       # max events in a sequence (truncate if more)
    "repr_dropout": 0.1,
    "repr_batch_size": 256,
    "repr_learning_rate": 1e-3,
    "repr_epochs": 20,
    "repr_patience": 5,            # early stopping patience

    # ------------------------------------------------------------------
    # Downstream model
    # ------------------------------------------------------------------
    "downstream_batch_size": 512,

    # ------------------------------------------------------------------
    # Random seed — used everywhere
    # ------------------------------------------------------------------
    "random_seed": 123,

    # ------------------------------------------------------------------
    # Spatial lag: how many grid-cell steps for neighbors (1 = immediate neighbors)
    # ------------------------------------------------------------------
    "spatial_lag_steps": 1,

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    "verbose": True,
}



# ------------------------------------------------------------------------
# Phase 3 mode toggle: PHASE_MODE=global removes region/country filters
# ------------------------------------------------------------------------
PHASE_MODE = os.environ.get("PHASE_MODE", "africa")
if PHASE_MODE == "global":
    CONFIG["region_filter"] = None
    CONFIG["country_filter"] = None
    print(f"[00_master] PHASE_MODE=global -> region_filter=None, country_filter=None")
elif PHASE_MODE != "africa":
    print(f"[00_master] WARNING: PHASE_MODE={PHASE_MODE!r} unrecognized; using Africa default")

def ensure_dirs():
    """Create all output directories if they don't exist."""
    for key in ["data_raw_dir", "data_intermediate_dir", "data_model_dir",
                "outputs_dir", "logs_dir"]:
        os.makedirs(CONFIG[key], exist_ok=True)


def run_script(script_name):
    """Run a pipeline script as a subprocess."""
    script_path = os.path.join(CONFIG["scripts_dir"], script_name)
    print(f"\n{'='*60}")
    print(f"  RUNNING: {script_name}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_ROOT,
        capture_output=False,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  FAILED: {script_name} (exit code {result.returncode})")
        sys.exit(1)
    print(f"  DONE: {script_name} ({elapsed:.1f}s)")

# ======================= step selector =======================
STEPS = ["acled", "ucdp", "merge", "grid", "aggregate", "sequences", "baseline", "super", "prio", "pgm"]
if len(sys.argv) != 2 or sys.argv[1] not in STEPS:
    sys.exit("usage: python scripts/build_panel.py <step>, step one of: " + " ".join(STEPS))
STEP = sys.argv[1]

# ======================= step acled: was scripts/01_ingest_acled.py =======================
# 01_ingest_acled.py
# ==================
# Read raw ACLED CSV, validate columns, clean dates, apply minimal filters,
# and save a cleaned intermediate file.
#
# Outputs:
#     data_intermediate/acled_cleaned.parquet
#
# Diagnostics printed:
#     - row counts before/after cleaning
#     - date range
#     - event type distribution
#     - top countries
#     - missing values per column
if STEP == "acled":

    import os
    import sys
    import pandas as pd
    import numpy as np

    # ---------------------------------------------------------------------------
    # Import config
    # ---------------------------------------------------------------------------
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    master = sys.modules[__name__]   # was: master = import_module("00_master") (its code is at the top of this file)
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

# ======================= step ucdp: was scripts/01b_ingest_ucdp.py =======================
# 01b_ingest_ucdp.py
# ==================
# Read UCDP GED (Georeferenced Event Dataset) and harmonize to the ACLED-style
# schema used downstream by 02_build_grid.py.
#
# Priority B (2026-05-26): multi-source pre-training (ACLED + UCDP-GED).
# This script is the first half of the merge: clean UCDP into a parquet that
# matches the column conventions of acled_cleaned.parquet, then 01c_merge_sources.py
# concatenates the two with a source_of_record column.
#
# UCDP GED schema reference (v25.1):
#     type_of_violence: 1 = state-based, 2 = non-state, 3 = one-sided
#     side_a, side_b: actor names (text)
#     date_start: ISO date
#     best, low, high: fatality estimate triples
#     latitude, longitude: location
#     priogrid_gid: PRIO-GRID cell id (bonus — used in Stage B)
#
# Mapping to ACLED-compatible 7-field event schema:
#     event_type            = "Battles" (tov 1/2) or "Violence against civilians" (tov 3)
#     sub_event_type        = ACLED-style sub-label (best-effort)
#     actor1                = side_a
#     actor2                = side_b (may be empty for one-sided)
#     inter1                = ACLED-style string label inferred from tov + side
#     inter2                = same
#     fatalities            = best (UCDP's primary estimate)
#
# Outputs:
#     data_intermediate/ucdp_cleaned.parquet
if STEP == "ucdp":

    import os
    import sys
    import pandas as pd
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    master = sys.modules[__name__]   # was: master = import_module("00_master") (its code is at the top of this file)
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

# ======================= step merge: was scripts/01c_merge_sources.py =======================
# 01c_merge_sources.py
# ====================
# Concatenate cleaned ACLED + cleaned UCDP events into a single events file
# that the rest of the pipeline reads as if it were ACLED.
#
# Priority B (2026-05-26): multi-source pre-training.
# Strategy: drop the unified events back into `acled_cleaned.parquet` so
# 02_build_grid.py reads the merged corpus with zero downstream changes. The
# source-of-origin is preserved as a `source_of_record` column for any future
# analysis.
#
# Inputs:
#     data_intermediate/acled_cleaned.parquet   (from 01_ingest_acled.py)
#     data_intermediate/ucdp_cleaned.parquet    (from 01b_ingest_ucdp.py)
#
# Outputs:
#     data_intermediate/events_unified.parquet  (full merged copy, persisted)
#     data_intermediate/acled_cleaned.parquet   (OVERWRITTEN — downstream reads this)
#     data_intermediate/acled_only_backup.parquet (backup of ACLED-only events)
if STEP == "merge":

    import os
    import sys
    import pandas as pd
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    master = sys.modules[__name__]   # was: master = import_module("00_master") (its code is at the top of this file)
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

# ======================= step grid: was scripts/02_build_grid.py =======================
# 02_build_grid.py
# ================
# Assign each cleaned event to a grid cell.
#
# Two grid schemes supported via CONFIG["grid_scheme"]:
#
#   "priogrid" (Stage B+ default):
#       Use PRIO-GRID 2.0 native cell IDs (gid). Cells are 0.5° x 0.5°.
#       gid = row * 720 + col + 1
#       where row = floor((lat + 90) * 2)  in [0, 360)
#             col = floor((lon + 180) * 2) in [0, 720)
#       No shapefile needed; the gid is a pure arithmetic function of lat/lon
#       because PRIO-GRID is a regular global lat/lon grid.
#       Comparability with ViEWS and the rest of the PRIO-GRID covariate stack.
#
#   "synthetic" (pilot / Stages A and Priority A/B):
#       Floor lat/lon to nearest grid_size_deg increment; grid_id is the
#       string "{lat_center:.2f}_{lon_center:.2f}". Legacy.
#
# Outputs:
#     data_intermediate/acled_with_grid.parquet
#         events with grid_id (int gid OR string lat_lon), grid_lat_center,
#         grid_lon_center. Use grid_id as the unique key downstream.
#
# Diagnostics:
#     - number of unique grid cells
#     - events per grid cell distribution
#     - top 20 most active grid cells
if STEP == "grid":

    import os
    import sys
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    master = sys.modules[__name__]   # was: master = import_module("00_master") (its code is at the top of this file)
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

# ======================= step aggregate: was scripts/03_aggregate_location_month.py =======================
# 03_aggregate_location_month.py
# ==============================
# Build a balanced grid-cell × month panel with aggregated event counts
# and the main binary outcome variable.
#
# Key steps:
#     1. Aggregate events to grid-month level
#     2. Build complete balanced panel (all grid-cells × all months)
#     3. Fill missing grid-months with zeros
#     4. Create outcome: violence_next_month
#     5. Create alternative outcomes (counts, type-specific)
#
# Outputs:
#     data_intermediate/grid_month_panel.parquet
#
# NOTE ON OBSERVABILITY:
#     ACLED coverage reflects both actual violence and reporting/monitoring capacity.
#     Zero-event months may reflect true peace OR reporting gaps. This pipeline
#     predicts *observed* event patterns, not latent "true" conflict. Country and
#     spatial metadata are preserved to enable future observability controls.
if STEP == "aggregate":

    import os
    import sys
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    master = sys.modules[__name__]   # was: master = import_module("00_master") (its code is at the top of this file)
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

# ======================= step sequences: was scripts/04_build_sequences.py =======================
# 04_build_sequences.py
# =====================
# Build event-history sequences for the representation model — STREAMING VERSION.
#
# PRIORITY A baseline (2026-05-26): 7-field hierarchical actor schema.
# STAGE B refactor (2026-05-26): chunked-by-grid_id streaming, writes per-grid
# sequence shards instead of materializing the full events_by_grid dict in RAM.
#
# Memory: O(largest single grid_id's events) instead of O(all events).
# At Africa-scale (~10K cells, ~10M events), the old approach would OOM at
# roughly 20-30 GB resident. This version uses ~1-2 GB regardless of corpus
# size because we never hold more than one cell's events in memory at a time.
#
# For each grid-month (l, t), we gather all events from the previous H months
# (t-H through t-1) in that grid cell. Each event is a 7-tuple:
#     (event_type_idx, sub_event_type_idx,
#      actor1_idx, actor2_idx,
#      actor1_type_idx, actor2_type_idx,
#      month_offset)
#
# actor1_type / actor2_type come from ACLED's inter1 / inter2 string labels.
#
# Outputs:
#     data_intermediate/vocab.json           — vocabulary mappings (7 fields)
#     data_intermediate/sequences.parquet    — one row per grid-month
#     data_intermediate/splits.json          — train/val/test grid-month IDs
if STEP == "sequences":

    import os
    import sys
    import json
    import time
    import pandas as pd
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    master = sys.modules[__name__]   # was: master = import_module("00_master") (its code is at the top of this file)
    CONFIG = master.CONFIG
    master.ensure_dirs()

    np.random.seed(CONFIG["random_seed"])

    # ---------------------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------------------
    events_path = os.path.join(CONFIG["data_intermediate_dir"], "acled_with_grid.parquet")
    panel_path = os.path.join(CONFIG["data_intermediate_dir"], "grid_month_panel.parquet")

    print("Loading events and panel...")
    events = pd.read_parquet(events_path)
    panel = pd.read_parquet(panel_path)
    print(f"Events: {len(events):,} rows")
    print(f"Panel: {len(panel):,} rows")

    # Verify the inter1/inter2 columns are present
    for col in ["inter1", "inter2"]:
        if col not in events.columns:
            raise KeyError(
                f"Column '{col}' missing from acled_with_grid.parquet. "
                "Priority A requires `inter1` and `inter2` to be preserved through "
                "the ingest."
            )

    # Post-2022 ACLED stores inter1/inter2 as STRINGS
    INTER_STRING_TO_TYPE = {
        "": "no_actor",
        "nan": "no_actor",
        "State forces": "state",
        "Rebel group": "rebel",
        "Rebel groups": "rebel",
        "Political militia": "political_militia",
        "Political militias": "political_militia",
        "Identity militia": "identity_militia",
        "Identity militias": "identity_militia",
        "Rioters": "rioters",
        "Protesters": "protesters",
        "Civilians": "civilians",
        "External/Other forces": "external_other",
        "External forces": "external_other",
        "Other forces": "external_other",
    }
    for col in ["inter1", "inter2"]:
        events[col] = events[col].fillna("").astype(str).str.strip()

    print(f"  inter1 raw value distribution (top 20): {events['inter1'].value_counts().head(20).to_dict()}")
    print(f"  inter2 raw value distribution (top 20): {events['inter2'].value_counts().head(20).to_dict()}")

    # ---------------------------------------------------------------------------
    # 2. Build vocabularies
    # ---------------------------------------------------------------------------
    def build_vocab(series, max_size=None):
        counts = series.value_counts()
        if max_size is not None:
            counts = counts.head(max_size)
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for i, val in enumerate(counts.index):
            vocab[str(val)] = i + 2
        return vocab

    event_type_vocab = build_vocab(events["event_type"], max_size=None)
    sub_event_type_vocab = build_vocab(events["sub_event_type"], max_size=None)
    actor_vocab_size = CONFIG["actor_vocab_size"]
    actor1_vocab = build_vocab(events["actor1"], max_size=actor_vocab_size)
    actor2_vocab = build_vocab(events["actor2"], max_size=actor_vocab_size)

    CANONICAL_TYPES = ["no_actor", "state", "rebel", "political_militia",
                       "identity_militia", "rioters", "protesters", "civilians",
                       "external_other"]
    actor1_type_vocab = {"<PAD>": 0, "<UNK>": 1}
    actor2_type_vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, label in enumerate(CANONICAL_TYPES):
        actor1_type_vocab[label] = i + 2
        actor2_type_vocab[label] = i + 2

    H = CONFIG["history_window_months"]
    month_offset_vocab = {"<PAD>": 0}
    for m in range(1, H + 1):
        month_offset_vocab[str(m)] = m

    vocabs = {
        "event_type": event_type_vocab,
        "sub_event_type": sub_event_type_vocab,
        "actor1": actor1_vocab,
        "actor2": actor2_vocab,
        "actor1_type": actor1_type_vocab,
        "actor2_type": actor2_type_vocab,
        "month_offset": month_offset_vocab,
    }

    print(f"Vocab sizes: { {k: len(v) for k,v in vocabs.items()} }")

    vocab_path = os.path.join(CONFIG["data_intermediate_dir"], "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocabs, f, indent=2)
    print(f"Saved vocabularies to: {vocab_path}")

    # ---------------------------------------------------------------------------
    # 3. Encode events with vocabulary indices
    # ---------------------------------------------------------------------------
    def encode_field(series, vocab):
        return series.astype(str).map(lambda x: vocab.get(x, 1)).astype(np.int32)

    events["event_type_idx"] = encode_field(events["event_type"], event_type_vocab)
    events["sub_event_type_idx"] = encode_field(events["sub_event_type"], sub_event_type_vocab)
    events["actor1_idx"] = encode_field(events["actor1"], actor1_vocab)
    events["actor2_idx"] = encode_field(events["actor2"], actor2_vocab)

    events["actor1_type_label"] = events["inter1"].map(INTER_STRING_TO_TYPE).fillna("no_actor")
    events["actor2_type_label"] = events["inter2"].map(INTER_STRING_TO_TYPE).fillna("no_actor")
    events["actor1_type_idx"] = encode_field(events["actor1_type_label"], actor1_type_vocab)
    events["actor2_type_idx"] = encode_field(events["actor2_type_label"], actor2_type_vocab)

    print(f"  actor1_type label distribution: {events['actor1_type_label'].value_counts().to_dict()}")
    print(f"  actor2_type label distribution: {events['actor2_type_label'].value_counts().to_dict()}")

    # ---------------------------------------------------------------------------
    # 4. Map year_month -> month_idx
    # ---------------------------------------------------------------------------
    all_months = sorted(panel["year_month"].unique())
    month_to_idx = {m: i for i, m in enumerate(all_months)}
    events["month_idx"] = events["year_month"].map(month_to_idx)
    panel["month_idx"] = panel["year_month"].map(month_to_idx)

    # ---------------------------------------------------------------------------
    # 5. STREAMING sequence build — chunked by grid_id
    # ---------------------------------------------------------------------------
    max_seq_len = CONFIG["repr_max_seq_len"]

    print(f"\nStreaming sequence build (H={H}, max_seq_len={max_seq_len})...")

    # Keep only columns we need from events for memory efficiency
    events_slim = events[[
        "grid_id", "month_idx",
        "event_type_idx", "sub_event_type_idx",
        "actor1_idx", "actor2_idx",
        "actor1_type_idx", "actor2_type_idx",
    ]].copy()
    del events  # release memory; we have what we need

    # Sort once, then iterate by grid_id
    events_slim = events_slim.sort_values(["grid_id", "month_idx"])
    panel_sorted = panel.sort_values(["grid_id", "month_idx"]).reset_index(drop=True)

    # Index events by grid_id offsets for fast slicing
    grid_ids_events = events_slim["grid_id"].values
    unique_grids_events, first_idx = np.unique(grid_ids_events, return_index=True)
    last_idx = np.append(first_idx[1:], len(grid_ids_events))
    grid_event_offsets = dict(zip(unique_grids_events, zip(first_idx, last_idx)))

    event_array_full = events_slim[[
        "month_idx", "event_type_idx", "sub_event_type_idx",
        "actor1_idx", "actor2_idx",
        "actor1_type_idx", "actor2_type_idx",
    ]].values.astype(np.int32)

    # Group panel rows by grid_id once
    panel_groups = panel_sorted.groupby("grid_id", sort=False)

    n_total = len(panel_sorted)
    n_processed = 0
    t_start = time.time()

    # Collect output rows in a list of dicts (small enough at Africa scale ~3M rows;
    # for global scale we'd switch to ParquetWriter shards).
    out_rows = []

    for grid_id, panel_subgroup in panel_groups:
        if grid_id in grid_event_offsets:
            lo, hi = grid_event_offsets[grid_id]
            grid_events = event_array_full[lo:hi]   # (n_events_in_grid, 7)
        else:
            grid_events = None

        for row in panel_subgroup.itertuples():
            t = row.month_idx
            seq_json = "[]"
            seq_len = 0

            if grid_events is not None and t - H >= 0:
                mask = (grid_events[:, 0] >= t - H) & (grid_events[:, 0] <= t - 1)
                window = grid_events[mask]
                if len(window) > 0:
                    order = np.argsort(window[:, 0])
                    window = window[order]
                    if len(window) > max_seq_len:
                        window = window[-max_seq_len:]
                    month_offsets = (t - window[:, 0]).astype(int)
                    seq = [
                        [int(window[j, 1]), int(window[j, 2]),
                         int(window[j, 3]), int(window[j, 4]),
                         int(window[j, 5]), int(window[j, 6]),
                         int(month_offsets[j])]
                        for j in range(len(window))
                    ]
                    seq_json = json.dumps(seq)
                    seq_len = len(seq)

            out_rows.append({
                "grid_id": grid_id,
                "year_month": row.year_month,
                "month_idx": t,
                "violence_next_month": row.violence_next_month,
                "event_sequence": seq_json,
                "seq_length": seq_len,
            })

            n_processed += 1
            if n_processed % 100000 == 0:
                elapsed = time.time() - t_start
                rate = n_processed / max(elapsed, 1e-6)
                eta = (n_total - n_processed) / rate
                print(f"  {n_processed:,} / {n_total:,}  "
                      f"({100*n_processed/n_total:.1f}%)  "
                      f"rate={rate:.0f}/s  eta={eta/60:.1f} min")

    print(f"\nSequence build complete in {time.time()-t_start:.1f}s")
    print(f"Processed {n_processed:,} grid-months")

    # ---------------------------------------------------------------------------
    # 6. Assemble into a panel-shaped DataFrame, merge other panel columns, save
    # ---------------------------------------------------------------------------
    seq_df = pd.DataFrame(out_rows)
    del out_rows

    # Bring back the rest of panel columns (everything except violence_next_month
    # and month_idx, which are already in seq_df)
    panel_extra = panel_sorted.drop(columns=["violence_next_month", "month_idx"])
    panel_merged = seq_df.merge(panel_extra, on=["grid_id", "year_month"], how="left")

    # Sequence length diagnostic
    sl = np.array(panel_merged["seq_length"].values)
    print(f"\nSequence length: min={sl.min()}, median={np.median(sl):.0f}, "
          f"mean={sl.mean():.1f}, max={sl.max()}, zero-len={(sl==0).sum():,} ({(sl==0).mean()*100:.1f}%)")

    # ---------------------------------------------------------------------------
    # 7. Splits
    # ---------------------------------------------------------------------------
    train_end = CONFIG["train_end"]
    val_end = CONFIG["val_end"]
    valid_mask = panel_merged["violence_next_month"].notna()
    history_mask = panel_merged["month_idx"] >= H
    usable = valid_mask & history_mask

    panel_merged["split"] = "exclude"
    panel_merged.loc[usable & (panel_merged["year_month"] <= train_end), "split"] = "train"
    panel_merged.loc[usable & (panel_merged["year_month"] > train_end) & (panel_merged["year_month"] <= val_end), "split"] = "val"
    panel_merged.loc[usable & (panel_merged["year_month"] > val_end), "split"] = "test"

    print(f"\n--- SPLIT DIAGNOSTICS ---")
    for s in ["train", "val", "test", "exclude"]:
        n = (panel_merged["split"] == s).sum()
        print(f"  {s}: {n:,}")
    for s in ["train", "val", "test"]:
        sub = panel_merged[panel_merged["split"] == s]
        pos = sub["violence_next_month"].sum()
        total = len(sub)
        if total > 0:
            print(f"  {s} positive rate: {pos/total*100:.2f}% ({int(pos):,}/{total:,})")

    splits_info = {
        "train_end": train_end,
        "val_end": val_end,
        "history_window": H,
        "train_months": sorted(panel_merged.loc[panel_merged["split"]=="train", "year_month"].unique().tolist()),
        "val_months": sorted(panel_merged.loc[panel_merged["split"]=="val", "year_month"].unique().tolist()),
        "test_months": sorted(panel_merged.loc[panel_merged["split"]=="test", "year_month"].unique().tolist()),
    }
    splits_path = os.path.join(CONFIG["data_intermediate_dir"], "splits.json")
    with open(splits_path, "w") as f:
        json.dump(splits_info, f, indent=2)
    print(f"Saved split info to: {splits_path}")

    # ---------------------------------------------------------------------------
    # 8. Save
    # ---------------------------------------------------------------------------
    out_path = os.path.join(CONFIG["data_intermediate_dir"], "sequences.parquet")
    panel_merged.to_parquet(out_path, index=False)
    print(f"\nSaved sequences to: {out_path}")
    print(f"Shape: {panel_merged.shape}")

# ======================= step baseline: was scripts/05_build_baseline_features.py =======================
# 05_build_baseline_features.py
# =============================
# Construct a ViEWS-style baseline feature set for the downstream forecasting task.
#
# Features (all using only past data — no leakage):
#     1. Lagged violence indicators: has_violence at t-1, t-3, t-6
#     2. Rolling violent event counts: sum over past 1, 3, 6 months
#     3. Rolling all-event counts: sum over past 1, 3, 6 months
#     4. Rolling fatality counts: sum over past 1, 3, 6 months
#     5. Event-type counts for prior month (one column per type)
#     6. Spatial lag: mean violence indicator of cardinal neighbors at t-1
#     7. Country indicators (one-hot, capped at top N countries)
#
# All features are constructed from information available at time t,
# to predict violence_next_month (at t+1).
#
# Outputs:
#     data_intermediate/baseline_features.parquet
if STEP == "baseline":

    import os
    import sys
    import pandas as pd
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    master = sys.modules[__name__]   # was: master = import_module("00_master") (its code is at the top of this file)
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

# ======================= step super: was scripts/build_sequences_super.py =======================
# build_sequences_super.py
# =============================
# Superset of 04_build_sequences.py: builds the SAME 7-field event sequences PLUS
# two extra per-event binned fields appended at the end:
#     field 7 = fatalities_bin   (from ACLED fatalities / UCDP best)
#     field 8 = civdeath_bin     (from harmonized deaths_civilians)
# so one build serves feature sets a/b/c/e/f via field selection in the trainer.
#
# Each event tuple becomes a 9-list:
#     [event_type, sub_event_type, actor1, actor2, actor1_type, actor2_type,
#      month_offset, fatalities_bin, civdeath_bin]
#
# Bins (both): 0 -> 1, 1-2 -> 2, 3-5 -> 3, 6-15 -> 4, 16-50 -> 5, 50+ -> 6
# (index 0 reserved for PAD, matching the other fields' padding_idx=0.)
#
# Outputs:
#     data_intermediate/vocab_super.json
#     data_intermediate/sequences_super.parquet
# (then run 05 with REPR_SEQ_SUFFIX=_super to produce baseline_features_super.parquet)
if STEP == "super":
    import os, sys, json, time
    import pandas as pd
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    master = sys.modules[__name__]   # was: master = import_module("00_master") (its code is at the top of this file)
    CONFIG = master.CONFIG
    master.ensure_dirs()
    np.random.seed(CONFIG["random_seed"])

    events_path = os.path.join(CONFIG["data_intermediate_dir"], "acled_with_grid.parquet")
    panel_path = os.path.join(CONFIG["data_intermediate_dir"], "grid_month_panel.parquet")
    print("Loading events and panel...")
    events = pd.read_parquet(events_path)
    panel = pd.read_parquet(panel_path)
    print(f"Events: {len(events):,}  Panel: {len(panel):,}")
    for col in ["inter1", "inter2", "fatalities", "deaths_civilians"]:
        if col not in events.columns:
            raise KeyError(f"Column '{col}' missing from acled_with_grid.parquet")

    INTER_STRING_TO_TYPE = {
        "": "no_actor", "nan": "no_actor", "State forces": "state",
        "Rebel group": "rebel", "Rebel groups": "rebel",
        "Political militia": "political_militia", "Political militias": "political_militia",
        "Identity militia": "identity_militia", "Identity militias": "identity_militia",
        "Rioters": "rioters", "Protesters": "protesters", "Civilians": "civilians",
        "External/Other forces": "external_other", "External forces": "external_other",
        "Other forces": "external_other",
    }
    for col in ["inter1", "inter2"]:
        events[col] = events[col].fillna("").astype(str).str.strip()


    def build_vocab(series, max_size=None):
        counts = series.value_counts()
        if max_size is not None:
            counts = counts.head(max_size)
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for i, val in enumerate(counts.index):
            vocab[str(val)] = i + 2
        return vocab


    event_type_vocab = build_vocab(events["event_type"])
    sub_event_type_vocab = build_vocab(events["sub_event_type"])
    actor_vocab_size = CONFIG["actor_vocab_size"]
    actor1_vocab = build_vocab(events["actor1"], max_size=actor_vocab_size)
    actor2_vocab = build_vocab(events["actor2"], max_size=actor_vocab_size)
    CANONICAL_TYPES = ["no_actor", "state", "rebel", "political_militia",
                       "identity_militia", "rioters", "protesters", "civilians", "external_other"]
    actor1_type_vocab = {"<PAD>": 0, "<UNK>": 1}
    actor2_type_vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, label in enumerate(CANONICAL_TYPES):
        actor1_type_vocab[label] = i + 2
        actor2_type_vocab[label] = i + 2
    H = CONFIG["history_window_months"]
    month_offset_vocab = {"<PAD>": 0}
    for m in range(1, H + 1):
        month_offset_vocab[str(m)] = m

    # binned death fields: 0 reserved for PAD, bins 1..6
    DEATH_BIN_VOCAB = {"<PAD>": 0, "0": 1, "1-2": 2, "3-5": 3, "6-15": 4, "16-50": 5, "50+": 6}


    def bin_deaths(arr):
        # arr int >=0 ; returns bin index 1..6
        a = np.asarray(arr, dtype=np.float64)
        # edges: (-inf,0]->1, (0,2]->2, (2,5]->3, (5,15]->4, (15,50]->5, (50,inf)->6
        idx = np.ones(len(a), dtype=np.int32)
        idx = np.where(a >= 1, 2, idx)
        idx = np.where(a >= 3, 3, idx)
        idx = np.where(a >= 6, 4, idx)
        idx = np.where(a >= 16, 5, idx)
        idx = np.where(a >= 51, 6, idx)
        return idx


    vocabs = {
        "event_type": event_type_vocab, "sub_event_type": sub_event_type_vocab,
        "actor1": actor1_vocab, "actor2": actor2_vocab,
        "actor1_type": actor1_type_vocab, "actor2_type": actor2_type_vocab,
        "month_offset": month_offset_vocab,
        "fatalities": DEATH_BIN_VOCAB, "deaths_civilians": DEATH_BIN_VOCAB,
    }
    print(f"Vocab sizes: { {k: len(v) for k,v in vocabs.items()} }")
    with open(os.path.join(CONFIG["data_intermediate_dir"], "vocab_super.json"), "w") as f:
        json.dump(vocabs, f, indent=2)


    def encode_field(series, vocab):
        return series.astype(str).map(lambda x: vocab.get(x, 1)).astype(np.int32)


    events["event_type_idx"] = encode_field(events["event_type"], event_type_vocab)
    events["sub_event_type_idx"] = encode_field(events["sub_event_type"], sub_event_type_vocab)
    events["actor1_idx"] = encode_field(events["actor1"], actor1_vocab)
    events["actor2_idx"] = encode_field(events["actor2"], actor2_vocab)
    events["actor1_type_label"] = events["inter1"].map(INTER_STRING_TO_TYPE).fillna("no_actor")
    events["actor2_type_label"] = events["inter2"].map(INTER_STRING_TO_TYPE).fillna("no_actor")
    events["actor1_type_idx"] = encode_field(events["actor1_type_label"], actor1_type_vocab)
    events["actor2_type_idx"] = encode_field(events["actor2_type_label"], actor2_type_vocab)
    events["fat_bin"] = bin_deaths(events["fatalities"].fillna(0).values)
    events["civ_bin"] = bin_deaths(events["deaths_civilians"].fillna(0).values)

    all_months = sorted(panel["year_month"].unique())
    month_to_idx = {m: i for i, m in enumerate(all_months)}
    events["month_idx"] = events["year_month"].map(month_to_idx)
    panel["month_idx"] = panel["year_month"].map(month_to_idx)

    max_seq_len = CONFIG["repr_max_seq_len"]
    print(f"Streaming build (H={H}, max_seq_len={max_seq_len})...")
    events_slim = events[[
        "grid_id", "month_idx", "event_type_idx", "sub_event_type_idx",
        "actor1_idx", "actor2_idx", "actor1_type_idx", "actor2_type_idx",
        "fat_bin", "civ_bin",
    ]].copy()
    del events
    events_slim = events_slim.sort_values(["grid_id", "month_idx"])
    panel_sorted = panel.sort_values(["grid_id", "month_idx"]).reset_index(drop=True)
    grid_ids_events = events_slim["grid_id"].values
    unique_grids_events, first_idx = np.unique(grid_ids_events, return_index=True)
    last_idx = np.append(first_idx[1:], len(grid_ids_events))
    grid_event_offsets = dict(zip(unique_grids_events, zip(first_idx, last_idx)))
    # columns: 0 month_idx,1 et,2 set,3 a1,4 a2,5 a1t,6 a2t,7 fat,8 civ
    event_array_full = events_slim[[
        "month_idx", "event_type_idx", "sub_event_type_idx", "actor1_idx", "actor2_idx",
        "actor1_type_idx", "actor2_type_idx", "fat_bin", "civ_bin",
    ]].values.astype(np.int32)
    panel_groups = panel_sorted.groupby("grid_id", sort=False)
    n_total = len(panel_sorted)
    n_processed = 0
    t_start = time.time()
    out_rows = []
    for grid_id, panel_subgroup in panel_groups:
        if grid_id in grid_event_offsets:
            lo, hi = grid_event_offsets[grid_id]
            grid_events = event_array_full[lo:hi]
        else:
            grid_events = None
        for row in panel_subgroup.itertuples():
            t = row.month_idx
            seq_json = "[]"
            seq_len = 0
            if grid_events is not None and t - H >= 0:
                mask = (grid_events[:, 0] >= t - H) & (grid_events[:, 0] <= t - 1)
                window = grid_events[mask]
                if len(window) > 0:
                    window = window[np.argsort(window[:, 0])]
                    if len(window) > max_seq_len:
                        window = window[-max_seq_len:]
                    month_offsets = (t - window[:, 0]).astype(int)
                    seq = [
                        [int(window[j, 1]), int(window[j, 2]), int(window[j, 3]),
                         int(window[j, 4]), int(window[j, 5]), int(window[j, 6]),
                         int(month_offsets[j]), int(window[j, 7]), int(window[j, 8])]
                        for j in range(len(window))
                    ]
                    seq_json = json.dumps(seq)
                    seq_len = len(seq)
            out_rows.append({
                "grid_id": grid_id, "year_month": row.year_month, "month_idx": t,
                "violence_next_month": row.violence_next_month,
                "event_sequence": seq_json, "seq_length": seq_len,
            })
            n_processed += 1
            if n_processed % 500000 == 0:
                el = time.time() - t_start
                print(f"  {n_processed:,}/{n_total:,} ({100*n_processed/n_total:.0f}%) {el/60:.1f}min", flush=True)
    print(f"Sequence build complete in {time.time()-t_start:.1f}s")

    seq_df = pd.DataFrame(out_rows)
    del out_rows
    panel_extra = panel_sorted.drop(columns=["violence_next_month", "month_idx"])
    panel_merged = seq_df.merge(panel_extra, on=["grid_id", "year_month"], how="left")

    train_end = CONFIG["train_end"]
    val_end = CONFIG["val_end"]
    usable = panel_merged["violence_next_month"].notna() & (panel_merged["month_idx"] >= H)
    panel_merged["split"] = "exclude"
    panel_merged.loc[usable & (panel_merged["year_month"] <= train_end), "split"] = "train"
    panel_merged.loc[usable & (panel_merged["year_month"] > train_end) & (panel_merged["year_month"] <= val_end), "split"] = "val"
    panel_merged.loc[usable & (panel_merged["year_month"] > val_end), "split"] = "test"
    for s in ["train", "val", "test", "exclude"]:
        print(f"  {s}: {(panel_merged['split']==s).sum():,}")

    out_path = os.path.join(CONFIG["data_intermediate_dir"], "sequences_super.parquet")
    panel_merged.to_parquet(out_path, index=False)
    print(f"Saved {out_path}  shape={panel_merged.shape}")

# ======================= step prio: was scripts/build_priogrid_static.py =======================
# build_priogrid_static.py
# ================================
# Build data_intermediate/priogrid_static.parquet: the 34 PRIO-GRID static
# covariates keyed by grid_id (== PRIO-GRID gid, since grid_scheme="priogrid").
# Each value is z-scored across cells. Used by feature sets (e) +PRIO-static and
# (f) all, injected into the encoder as a per-cell CLS context vector.
#
# Input : data_raw/priogrid/static/<name>.json  (each: {"cells":[{gid,value,year}]})
# Output: data_intermediate/priogrid_static.parquet  (grid_id + 34 z-scored cols)
if STEP == "prio":
    import os
    import sys
    import glob
    import json
    import numpy as np
    import pandas as pd

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    master = sys.modules[__name__]   # was: master = import_module("00_master") (its code is at the top of this file)
    CONFIG = master.CONFIG
    master.ensure_dirs()

    RAW = os.path.join(CONFIG["data_raw_dir"], "priogrid", "static")
    OUT = os.path.join(CONFIG["data_intermediate_dir"], "priogrid_static.parquet")

    files = sorted(glob.glob(os.path.join(RAW, "*.json")))
    assert files, f"no static json found in {RAW}"
    print(f"[priogrid_static] {len(files)} static vars")

    cols = {}
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        cells = json.load(open(f))["cells"]
        cols[name] = pd.Series(
            {int(c["gid"]): float(c["value"]) for c in cells if c.get("value") is not None},
            dtype="float64",
        )

    df = pd.DataFrame(cols)
    df.index.name = "grid_id"

    # z-score each covariate; missing -> column mean (0 after standardizing)
    for c in df.columns:
        mu = df[c].mean()
        sd = df[c].std()
        df[c] = (df[c] - mu) / (sd if sd and sd > 0 else 1.0)
    df = df.fillna(0.0).reset_index()
    df["grid_id"] = df["grid_id"].astype("int64")

    df.to_parquet(OUT, index=False)
    print(f"[priogrid_static] wrote {OUT} shape={df.shape}")
    print(f"[priogrid_static] vars: {list(df.columns)[1:]}")

# ======================= step pgm: was scripts/build_pgm_fatalities_target.py =======================
if STEP == "pgm":
    import pandas as pd, numpy as np
    R="/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
    u=pd.read_parquet(f"{R}/data_intermediate/ucdp_cleaned.parquet",columns=["priogrid_gid","year_month","type_of_violence","fatalities"])
    u=u[u.type_of_violence==1]
    agg=u.groupby(["priogrid_gid","year_month"],as_index=False)["fatalities"].sum().rename(columns={"priogrid_gid":"grid_id","fatalities":"sb_fatalities"})
    print("UCDP sb total fatalities:",int(agg.sb_fatalities.sum()),"nonzero cell-months:",len(agg))
    p=pd.read_parquet(f"{R}/data_intermediate/grid_month_panel.parquet",columns=["grid_id","year_month","grid_lat_center","grid_lon_center"])
    p=p[(p.grid_lat_center>=-36)&(p.grid_lat_center<=43)&(p.grid_lon_center>=-19)&(p.grid_lon_center<=63)].copy()
    print("AfricaME cell-months:",len(p),"cells:",p.grid_id.nunique())
    m=p.merge(agg,on=["grid_id","year_month"],how="left")
    m["sb_fatalities"]=m["sb_fatalities"].fillna(0).astype(int)
    print("JOIN nonzero:",int((m.sb_fatalities>0).sum()),"fatalities landed:",int(m.sb_fatalities.sum()))
    m=m.sort_values(["grid_id","year_month"]).reset_index(drop=True)
    m["sb_fat_next"]=m.groupby("grid_id")["sb_fatalities"].shift(-1)
    m=m.dropna(subset=["sb_fat_next"]); m["sb_fat_next"]=m["sb_fat_next"].astype(int)
    m.to_parquet(f"{R}/data_intermediate/pgm_fatalities_target.parquet",index=False)
    print("SAVED rows",len(m),"zero_rate %.4f"%(m.sb_fat_next==0).mean())
    print("describe",{k:round(float(v),3) for k,v in m.sb_fat_next.describe().items()})
