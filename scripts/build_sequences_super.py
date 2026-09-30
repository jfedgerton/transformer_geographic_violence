"""build_sequences_super.py
=============================
Superset of 04_build_sequences.py: builds the SAME 7-field event sequences PLUS
two extra per-event binned fields appended at the end:
    field 7 = fatalities_bin   (from ACLED fatalities / UCDP best)
    field 8 = civdeath_bin     (from harmonized deaths_civilians)
so one build serves feature sets a/b/c/e/f via field selection in the trainer.

Each event tuple becomes a 9-list:
    [event_type, sub_event_type, actor1, actor2, actor1_type, actor2_type,
     month_offset, fatalities_bin, civdeath_bin]

Bins (both): 0 -> 1, 1-2 -> 2, 3-5 -> 3, 6-15 -> 4, 16-50 -> 5, 50+ -> 6
(index 0 reserved for PAD, matching the other fields' padding_idx=0.)

Outputs:
    data_intermediate/vocab_super.json
    data_intermediate/sequences_super.parquet
(then run 05 with REPR_SEQ_SUFFIX=_super to produce baseline_features_super.parquet)
"""
import os, sys, json, time
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
master = import_module("00_master")
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
