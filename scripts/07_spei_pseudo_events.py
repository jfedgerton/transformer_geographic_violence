"""
07_spei_pseudo_events.py

Build a pseudo-event-sequence tokenization of the SPEI drought panel so it can be
fed through the same Encoder/vocab pipeline as crime_ingest.py and the conflict panel.

Design decisions (flagged explicitly, not silent defaults -- confirm with Jared
before treating transfer-eval results on this as final):
  - "Event" definition: any grid-month with SPEI < EVENT_THRESHOLD (-0.5, the
    standard McKee et al. 1993 SPI/SPEI "mild drought or worse" cutoff). Grid-months
    at or above -0.5 are NOT tokenized as events, mirroring crime_ingest.py treating
    only actual incidents (not quiet months) as events.
  - Categorical field: severity_bucket, using the standard 4-class McKee convention
    (mild/moderate/severe/extreme) -- the categorical analog to crime's
    offense_type/offense_group/offense_against. Only ONE field, not three like
    crime, because SPEI genuinely only carries one piece of categorical
    information; inventing extra fields would not reflect anything in the data.
  - event_sequence per grid-cell-month: up to MAXLEN=200 most recent drought
    events strictly before/at the target month, each token = [severity_bucket_id,
    month_offset], month_offset = t - event_t capped at 199 -- mirrors
    crime_ingest.py's [a,b,c,min(t-d,199)] token structure exactly.
  - Target: reuses drought_onset_next_month already built in
    data_intermediate/spei_grid_month_panel.parquet (severe/extreme onset,
    SPEI < -1.5) -- NOT redefined here.

Run sequentially, no helper functions, per Jared's coding preference.
Fails loudly (assertions) rather than silently producing a malformed panel.
"""

import os
import json
import bisect
import numpy as np
import pandas as pd

SEED = 123
np.random.seed(SEED)

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
IN_PATH = f"{R}/data_intermediate/spei_grid_month_panel.parquet"
TAG = "spei_horn_of_africa"
OUT_PARQUET = f"{R}/data_intermediate/spei_events_{TAG}.parquet"
OUT_VOCAB = f"{R}/data_intermediate/spei_vocab_{TAG}.json"

EVENT_THRESHOLD = -0.5  # McKee et al. 1993: SPEI < -0.5 = mild drought or worse
MAXLEN = 200            # same cap as crime_ingest.py

# ---- Step 1: load the balanced SPEI panel built by 06_build_spei_grid.py -------

assert os.path.exists(IN_PATH), f"{IN_PATH} not found -- run scripts/06_build_spei_grid.py first"
df = pd.read_parquet(IN_PATH)
for col in ["grid_id", "year_month", "spei", "drought_onset_next_month"]:
    assert col in df.columns, f"expected column '{col}' missing from {IN_PATH}"

df["t"] = df["year_month"].str[:4].astype(int) * 12 + df["year_month"].str[5:7].astype(int)

# ---- Step 2: severity_bucket, standard McKee 1993 4-class convention -----------

conditions = [
    (df["spei"] < -2.0),
    (df["spei"] >= -2.0) & (df["spei"] < -1.5),
    (df["spei"] >= -1.5) & (df["spei"] < -1.0),
    (df["spei"] >= -1.0) & (df["spei"] < EVENT_THRESHOLD),
]
choices = ["extreme", "severe", "moderate", "mild"]
df["severity_bucket"] = np.select(conditions, choices, default="none")

is_event = df["severity_bucket"] != "none"
print(f"[cfg] rows={len(df):,} event_rows={int(is_event.sum()):,} ({is_event.mean():.4%})", flush=True)
assert is_event.sum() > 0, "No rows meet EVENT_THRESHOLD -- check SPEI sign convention and threshold"

# ---- Step 3: vocab, 1-indexed, matching crime_ingest.py's {c:i+1} convention ---

vocab = {}
cats = sorted(df.loc[is_event, "severity_bucket"].unique())
vocab["severity_bucket"] = {c: i + 1 for i, c in enumerate(cats)}
df["severity_bucket_id"] = df["severity_bucket"].map(vocab["severity_bucket"]).fillna(0).astype(int)
os.makedirs(os.path.dirname(OUT_VOCAB), exist_ok=True)
json.dump(vocab, open(OUT_VOCAB, "w"))
print(f"[vocab] {vocab}", flush=True)

# ---- Step 4: per-cell event token lists (only actual drought events) ----------

events = df[is_event].sort_values(["grid_id", "t"]).reset_index(drop=True)
evtok = {}
evtt = {}
for cell, sub in events.groupby("grid_id"):
    tk = list(zip(sub.severity_bucket_id.astype(int), sub.t.astype(int)))
    evtok[cell] = tk
    evtt[cell] = [e[1] for e in tk]

# ---- Step 5: build event_sequence + target for every row of the balanced panel -

df = df.sort_values(["grid_id", "t"]).reset_index(drop=True)
rows = []
for cell, sub in df.groupby("grid_id"):
    tk = evtok.get(cell, [])
    tt = evtt.get(cell, [])
    for _, r in sub.iterrows():
        t = int(r["t"])
        hi = bisect.bisect_right(tt, t)
        seq = tk[max(0, hi - MAXLEN):hi]
        seq = [[a, min(t - d, 199)] for (a, d) in seq]
        rows.append((cell, t, r["year_month"], r["drought_onset_next_month"], json.dumps(seq)))

out = pd.DataFrame(
    rows,
    columns=["grid_id", "t", "year_month", "drought_onset_next_month", "event_sequence"],
)
assert len(out) == len(df), f"row count mismatch: built {len(out):,} vs panel {len(df):,}"

# sanity: sequence length distribution, should mostly be short given drought is rare
seq_lens = out["event_sequence"].map(lambda s: len(json.loads(s)))
print(f"[seq] mean_len={seq_lens.mean():.2f} max_len={seq_lens.max()} pct_empty={(seq_lens==0).mean():.4%}", flush=True)

valid = out.dropna(subset=["drought_onset_next_month"])
print(
    f"[out] rows={len(out):,} cells={out.grid_id.nunique()} "
    f"pos_rate={valid.drought_onset_next_month.mean():.4%} -> {OUT_PARQUET}",
    flush=True,
)

out.to_parquet(OUT_PARQUET, index=False)
print("[done]", flush=True)
