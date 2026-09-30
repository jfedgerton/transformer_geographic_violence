"""build_priogrid_static.py
================================
Build data_intermediate/priogrid_static.parquet: the 34 PRIO-GRID static
covariates keyed by grid_id (== PRIO-GRID gid, since grid_scheme="priogrid").
Each value is z-scored across cells. Used by feature sets (e) +PRIO-static and
(f) all, injected into the encoder as a per-cell CLS context vector.

Input : data_raw/priogrid/static/<name>.json  (each: {"cells":[{gid,value,year}]})
Output: data_intermediate/priogrid_static.parquet  (grid_id + 34 z-scored cols)
"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
master = import_module("00_master")
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
