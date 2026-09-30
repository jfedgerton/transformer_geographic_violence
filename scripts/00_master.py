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


# ==========================================================================
# MAIN: run full pipeline
# ==========================================================================
if __name__ == "__main__":
    ensure_dirs()

    pipeline_steps = [
        "01_ingest_acled.py",
        "02_build_grid.py",
        "03_aggregate_location_month.py",
        "04_build_sequences.py",
        "05_build_baseline_features.py",
        "06_train_representation_model.py",
        "07_train_downstream_model.py",
        "08_evaluate_models.py",
        "09_visualize_embeddings.py",
        # "10_rolling_origin.py",  # disabled 2026-05-09 — uses CONFIG.data_root attribute access (CONFIG is a dict), references nonexistent file paths (processed/embeddings.parquet vs. actual data_model/embeddings_all.parquet), wrong column names (country_id vs. grid_id, conflict_next_month vs. violence_next_month). See scripts/10_rolling_origin_fixed.py for proposed rewrite.
        # "11_ablations.py",  # disabled — references nonexistent files / wrong CONFIG keys; needs rewrite
    ]

    print("=" * 60)
    print("  CONFLICT FORECASTING PIPELINE")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Input file:   {CONFIG['acled_input_file']}")
    print(f"  Grid size:    {CONFIG['grid_size_deg']}°")
    print(f"  History:      {CONFIG['history_window_months']} months")
    print(f"  Train end:    {CONFIG['train_end']}")
    print(f"  Val end:      {CONFIG['val_end']}")
    print(f"  Seed:         {CONFIG['random_seed']}")
    print("=" * 60)

    t_total = time.time()
    for step in pipeline_steps:
        run_script(step)

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE ({time.time() - t_total:.1f}s total)")
    print(f"{'='*60}")
