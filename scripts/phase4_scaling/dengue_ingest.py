"""Dengue (Brazil) ingest -> municipality-by-week panel, mirrors crime_ingest.py structure.
Target = outbreak_next_period (z-score severity bucket > 0 next week), same convention as
crime's crime_next_month (any-event-next-period, not strict onset).
Spatial lag uses k=5 nearest municipalities by centroid haversine distance (grid-adjacency
analog is not available at this spatial unit).
Out: data_intermediate/dengue_{TAG}.parquet + dengue_vocab_{TAG}.json
"""
import os, json, bisect
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
TAG = "br_muni_v1"
MAXLEN = 200
K_NEIGH = 5

print(f"[cfg] TAG={TAG} MAXLEN={MAXLEN} K_NEIGH={K_NEIGH}", flush=True)

df = pd.read_csv(f"{R}/data_intermediate/dengue_panel_raw.csv.gz")
df = df.sort_values(["MUNICIP", "t"]).reset_index(drop=True)
wmin, wmax = int(df["t"].min()), int(df["t"].max())
print(f"[data] rows={len(df):,} municipalities={df.MUNICIP.nunique()} t range=[{wmin},{wmax}]", flush=True)

vocab = {"severity": {"mild": 1, "moderate": 2, "severe": 3, "extreme": 4}}
json.dump(vocab, open(f"{R}/data_intermediate/dengue_vocab_{TAG}.json", "w"))

meta = df.groupby("MUNICIP").agg(lat=("lat", "mean"), lon=("lon", "mean"), pop=("pop", "mean")).reset_index()

# k nearest neighbors by haversine distance (centroid lat/lon)
rad = np.radians(meta[["lat", "lon"]].to_numpy())
tree = BallTree(rad, metric="haversine")
dist, idx = tree.query(rad, k=K_NEIGH + 1)  # includes self at k=0
muni_arr = meta["MUNICIP"].to_numpy()
neighbors = {muni_arr[i]: [muni_arr[j] for j in idx[i][1:]] for i in range(len(muni_arr))}
print(f"[neighbors] built kNN for {len(neighbors)} municipalities", flush=True)

cellmon = {}   # MUNICIP -> {t: pos}
evtok = {}     # MUNICIP -> [(severity_id, t), ...] sorted by t (positive-severity weeks only)
evtt = {}      # MUNICIP -> [t, ...] parallel to evtok
for muni, sub in df.groupby("MUNICIP"):
    cellmon[muni] = dict(zip(sub.t.astype(int), sub.pos.astype(int)))
    pos_sub = sub[sub.severity > 0]
    tk = list(zip(pos_sub.severity.astype(int), pos_sub.t.astype(int)))
    evtok[muni] = tk
    evtt[muni] = [e[1] for e in tk]

rows = []
for muni in meta.MUNICIP.values:
    cm = cellmon[muni]; tk = evtok[muni]; tt = evtt[muni]
    nb = neighbors.get(muni, [])
    present_t = sorted(cm.keys())
    first_t = min(present_t)
    for t in range(max(first_t, wmin), wmax):
        if t + 1 not in cm:
            continue
        hi = bisect.bisect_right(tt, t)
        seq = tk[max(0, hi - MAXLEN):hi]
        seq = [[a, min(t - d, 199)] for (a, d) in seq]
        y = cm.get(t + 1, 0)
        lag1 = 1 if cm.get(t, 0) > 0 else 0
        lag3 = 1 if sum(cm.get(t - k, 0) for k in range(0, 3)) > 0 else 0
        lag6 = 1 if sum(cm.get(t - k, 0) for k in range(0, 6)) > 0 else 0
        roll3 = sum(cm.get(t - k, 0) for k in range(0, 3))
        pastm = [m for m in cm if m <= t and cm[m] > 0]
        msl = (t - max(pastm)) if pastm else 99
        slag = sum(cellmon.get(x, {}).get(t, 0) for x in nb) if nb else 0
        rows.append((muni, t, y, lag1, lag3, lag6, roll3, msl, slag, json.dumps(seq)))

out = pd.DataFrame(rows, columns=["city", "t", "dengue_next_period", "lag1", "lag3", "lag6",
                                   "roll3", "months_since_last", "spatial_lag", "event_sequence"])
out = out.merge(meta, left_on="city", right_on="MUNICIP", how="left").drop(columns=["MUNICIP"])
out["year_month"] = out["t"]
op = f"{R}/data_intermediate/dengue_{TAG}.parquet"
out.to_parquet(op)
print(f"[out] rows={len(out)} municipalities={out.city.nunique()} pos_rate={out.dengue_next_period.mean():.4f} -> {op}", flush=True)
print("[done]", flush=True)
