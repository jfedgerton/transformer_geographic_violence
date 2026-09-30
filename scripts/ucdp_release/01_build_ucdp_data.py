"""01_build_ucdp_data.py -- UCDP GED -> cell-month event-sequence panel (open-data HF arm).
Looks for an existing event-level GED csv in data_raw (env UCDP_CSV overrides); downloads
GED 25.1 (CC BY 4.0) only if none found. Outputs:
  data_intermediate/ucdp_panel.parquet   (grid_id, year_month, lat/lon, split,
                                          event_sequence, violence_next_month, violence_lag1)
  data_intermediate/ucdp_vocab.json
Fields per event: type_of_violence, side_a, side_b, deaths_bin, month_offset (ids start at 1; 0=pad).
Split: train<=2020-12, val<=2021-12, test>=2022-01 (matches ACLED pipeline).
"""
import os, glob, json, zipfile, urllib.request
import numpy as np
import pandas as pd

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
RAW = f"{R}/data_raw/ucdp_ged"
OUT_PANEL = f"{R}/data_intermediate/ucdp_panel.parquet"
OUT_VOCAB = f"{R}/data_intermediate/ucdp_vocab.json"
URL = os.environ.get("UCDP_URL", "https://ucdp.uu.se/downloads/ged/ged251-csv.zip")
TOP_ACTORS = 300
WINDOW = 12
MAX_SEQ_LEN = 200
os.makedirs(RAW, exist_ok=True)

# ---- locate or download event-level GED ----
csv_path = os.environ.get("UCDP_CSV", "")
if not csv_path:
    cands = []
    for pat in ("*.csv", "**/*.csv"):
        cands += glob.glob(os.path.join(f"{R}/data_raw", pat), recursive=True)
    cands = [c for c in cands if ("ged" in os.path.basename(c).lower()
                                  or "ucdp" in os.path.basename(c).lower())]
    if cands:
        csv_path = max(cands, key=os.path.getsize)
if not csv_path:
    zp = os.path.join(RAW, os.path.basename(URL))
    print(f"[dl] no GED csv found locally; downloading {URL}", flush=True)
    urllib.request.urlretrieve(URL, zp)
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        assert names, "no csv inside GED zip"
        z.extract(names[0], RAW)
        csv_path = os.path.join(RAW, names[0])
assert "acled" not in csv_path.lower(), f"Refusing to build UCDP-only release panel from path containing acled: {csv_path}"
print(f"[data] using GED file: {csv_path}", flush=True)

need = ["type_of_violence", "side_a", "side_b", "best", "date_start",
        "priogrid_gid", "year"]
ged = pd.read_csv(csv_path, usecols=lambda c: c in set(need), low_memory=False)
miss = [c for c in need if c not in ged.columns]
assert not miss, f"GED file missing columns: {miss} -- is this event-level GED?"
ged = ged.dropna(subset=["priogrid_gid", "date_start", "type_of_violence"])
ged["priogrid_gid"] = ged["priogrid_gid"].astype(np.int64)
ged = ged[ged["priogrid_gid"] > 0]
ym = ged["date_start"].astype(str).str.slice(0, 7)
yy = ym.str.slice(0, 4).astype(int)
mm = ym.str.slice(5, 7).astype(int)
ged["midx"] = yy * 12 + (mm - 1)
print(f"[data] events: {len(ged):,}  span {ym.min()}..{ym.max()}", flush=True)

# ---- vocabs (ids start at 1; 0 = pad) ----
def actor_map(series):
    top = series.value_counts().head(TOP_ACTORS).index.tolist()
    m = {a: i + 1 for i, a in enumerate(top)}
    m["OTHER"] = len(m) + 1
    return m

map_a = actor_map(ged["side_a"].astype(str))
map_b = actor_map(ged["side_b"].astype(str))
map_tov = {"1": 1, "2": 2, "3": 3}
map_db = {"0": 1, "1-5": 2, "6-25": 3, "26-100": 4, "101+": 5}
map_off = {str(k): k + 1 for k in range(WINDOW)}
vocab = {"type_of_violence": map_tov, "side_a": map_a, "side_b": map_b,
         "deaths_bin": map_db, "month_offset": map_off}
json.dump(vocab, open(OUT_VOCAB, "w"))
print(f"[vocab] sizes: tov={len(map_tov)} side_a={len(map_a)} side_b={len(map_b)} "
      f"deaths={len(map_db)} offset={len(map_off)} -> {OUT_VOCAB}", flush=True)

def dbin(x):
    x = 0 if pd.isna(x) else float(x)
    if x <= 0: return 1
    if x <= 5: return 2
    if x <= 25: return 3
    if x <= 100: return 4
    return 5

ged["c_tov"] = ged["type_of_violence"].astype(int).clip(1, 3)
ged["c_a"] = ged["side_a"].astype(str).map(lambda a: map_a.get(a, map_a["OTHER"]))
ged["c_b"] = ged["side_b"].astype(str).map(lambda b: map_b.get(b, map_b["OTHER"]))
ged["c_db"] = ged["best"].map(dbin)
ged = ged.sort_values(["priogrid_gid", "midx"]).reset_index(drop=True)

# ---- build cell-month panel ----
evd = {}
for gid, grp in ged.groupby("priogrid_gid"):
    evd[gid] = (grp["midx"].to_numpy(),
                grp[["c_tov", "c_a", "c_b", "c_db"]].to_numpy())
max_midx = int(ged["midx"].max())
rows = []
for gid, (ems, evs) in evd.items():
    grow = (gid - 1) // 720
    gcol = (gid - 1) % 720
    glat = -90.0 + 0.5 * grow + 0.25
    glon = -180.0 + 0.5 * gcol + 0.25
    emset = set(ems.tolist())
    lo, hi = int(ems.min()), int(ems.max())
    for m in range(lo, min(hi + WINDOW, max_midx) + 1):
        wmask = (ems > m - WINDOW) & (ems <= m)
        if not wmask.any():
            continue
        wm = ems[wmask]
        we = evs[wmask]
        seq = [[int(we[k, 0]), int(we[k, 1]), int(we[k, 2]), int(we[k, 3]),
                int(m - wm[k]) + 1] for k in range(len(wm))]
        seq = seq[-MAX_SEQ_LEN:]
        nxt = np.nan if m >= max_midx else float((m + 1) in emset)
        lag1 = float(m in emset)
        yy2, mm2 = divmod(m, 12)
        yms = f"{yy2:04d}-{mm2+1:02d}"
        split = "train" if yms <= "2020-12" else ("val" if yms <= "2021-12" else "test")
        rows.append((int(gid), yms, glat, glon, split, json.dumps(seq), nxt, lag1))

panel = pd.DataFrame(rows, columns=["grid_id", "year_month", "grid_lat_center",
                                    "grid_lon_center", "split", "event_sequence",
                                    "violence_next_month", "violence_lag1"])
panel.to_parquet(OUT_PANEL, index=False)
print(f"[panel] rows={len(panel):,} cells={panel.grid_id.nunique():,}", flush=True)
print(panel["split"].value_counts().to_string(), flush=True)
print(f"[panel] pos rate (next-month): {panel.violence_next_month.mean():.4f}", flush=True)
print(f"[done] -> {OUT_PANEL}", flush=True)
