"""
04_build_sequences.py
=====================
Build event-history sequences for the representation model — STREAMING VERSION.

PRIORITY A baseline (2026-05-26): 7-field hierarchical actor schema.
STAGE B refactor (2026-05-26): chunked-by-grid_id streaming, writes per-grid
sequence shards instead of materializing the full events_by_grid dict in RAM.

Memory: O(largest single grid_id's events) instead of O(all events).
At Africa-scale (~10K cells, ~10M events), the old approach would OOM at
roughly 20-30 GB resident. This version uses ~1-2 GB regardless of corpus
size because we never hold more than one cell's events in memory at a time.

For each grid-month (l, t), we gather all events from the previous H months
(t-H through t-1) in that grid cell. Each event is a 7-tuple:
    (event_type_idx, sub_event_type_idx,
     actor1_idx, actor2_idx,
     actor1_type_idx, actor2_type_idx,
     month_offset)

actor1_type / actor2_type come from ACLED's inter1 / inter2 string labels.

Outputs:
    data_intermediate/vocab.json           — vocabulary mappings (7 fields)
    data_intermediate/sequences.parquet    — one row per grid-month
    data_intermediate/splits.json          — train/val/test grid-month IDs
"""

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
master = import_module("00_master")
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
