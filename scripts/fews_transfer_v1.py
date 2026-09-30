"""
fews_transfer_v1.py  —  Frozen-embedding transfer test.

Question: do the conflict-learned embeddings predict FOOD-INSECURITY onset
(FEWS NET IPC phase >= 3, "Crisis or worse") at African grid cells?

This is "mechanism A" transfer (frozen embeddings, no fine-tuning) — valid here because
African cells DO have rich conflict-event histories, so the encoder produces informative
embeddings for them. We test whether those embeddings carry food-insecurity signal beyond
a conflict-history baseline and beyond naive persistence.

Design / first-pass choices (flagged for review):
  - admin(fnid) -> grid-cell mapping: NEAREST CENTROID. geopandas/shapely aren't installed,
    so admin centroids are computed as the vertex-mean of each polygon and each cell is
    assigned to the nearest admin centroid. Approximate (boundary cells can misassign);
    exact point-in-polygon is the v2 refinement.
  - Target: next-period IPC phase >= 3 (binary onset), from the FEWS current-situation (CS)
    classifications we pulled. IPC is published ~3x/year, so "period" is the IPC cycle, not month.
  - Features: frozen 64-dim seed-123 conflict embeddings vs conflict baseline_features vs combined.
  - Persistence baseline = current period's phase >= 3. Temporal split: train <= 2019, test 2020+.
  - Metrics: ROC-AUC and PR-AUC (the honest one for a rare-ish onset target).

Run (SLURM):
  sbatch --job-name=tgv_fews --account=jfe4_cr_default --partition=standard --gres=gpu:a100:1 \
    --cpus-per-task=4 --mem=64G --time=01:00:00 --output=logs/fews_%j.out --error=logs/fews_%j.err \
    --wrap="module load anaconda && source venv/bin/activate && python scripts/fews_transfer_v1.py"
"""
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
SEED = 123
META = {"grid_id", "year_month", "split", "country", "grid_lat_center", "grid_lon_center"}

# ---- 1. admin polygons -> vertex-mean centroids (fnid, lon, lat) ----
gj = json.load(open(f"{R}/data_raw/fews/fews_fsc_admin.geojson"))
def coord_pairs(geom):
    def rec(x):
        if isinstance(x, (list, tuple)):
            if len(x) == 2 and all(isinstance(v, (int, float)) for v in x):
                yield (x[0], x[1])
            else:
                for y in x:
                    yield from rec(y)
    yield from rec(geom["coordinates"])
rows = []
for f in gj["features"]:
    pts = np.array(list(coord_pairs(f["geometry"])))
    if len(pts):
        rows.append((f["properties"]["fnid"], pts[:, 0].mean(), pts[:, 1].mean()))
adm = pd.DataFrame(rows, columns=["fnid", "lon", "lat"])
print(f"admin units with centroid: {len(adm)}")

# ---- 2. African grid cells -> nearest admin centroid (cell -> fnid) ----
pan = pd.read_parquet(f"{R}/data_intermediate/grid_month_panel.parquet",
                      columns=["grid_id", "year_month", "grid_lat_center", "grid_lon_center"])
cells = pan.drop_duplicates("grid_id")
cells = cells[(cells.grid_lat_center >= -36) & (cells.grid_lat_center <= 38) &
              (cells.grid_lon_center >= -19) & (cells.grid_lon_center <= 52)].copy()
_, idx = cKDTree(adm[["lon", "lat"]].to_numpy()).query(cells[["grid_lon_center", "grid_lat_center"]].to_numpy())
cells["fnid"] = adm.fnid.to_numpy()[idx]
print(f"african cells mapped: {len(cells)}")

# ---- 3. IPC current-situation phase per (fnid, period); keep valid phases 1-5 only ----
ipc = pd.read_csv(f"{R}/data_raw/fews/ipcphase_africa_cs.csv",
                  usecols=["fnid", "projection_start", "value"])
ipc = ipc.dropna(subset=["projection_start", "value"])
ipc = ipc[(ipc.value >= 1) & (ipc.value <= 5)]            # drop 66/88/99 (water/parks/missing)
ipc["period"] = ipc.projection_start.str.slice(0, 7)
ipc = ipc.groupby(["fnid", "period"], as_index=False)["value"].max()
ipc["ge3"] = (ipc.value >= 3).astype(int)
print(f"ipc fnid-periods: {len(ipc)}  crisis(>=3) rate: {ipc.ge3.mean():.4f}")

# ---- 4. cell x period target: next-period onset of phase>=3 ----
cp = cells[["grid_id", "fnid"]].merge(ipc, on="fnid", how="inner")
cp = cp.sort_values(["grid_id", "period"]).reset_index(drop=True)
cp["ge3_next"] = cp.groupby("grid_id")["ge3"].shift(-1)
cp = cp.dropna(subset=["ge3_next"]); cp["ge3_next"] = cp["ge3_next"].astype(int)
cp["year_month"] = cp["period"]
print(f"cell-period rows: {len(cp)}  next-period crisis rate: {cp.ge3_next.mean():.4f}")

# ---- 5. join frozen conflict embeddings + conflict baseline features ----
emb = pd.read_parquet(f"{R}/data_model/embeddings_all_seed{SEED}_d64.parquet")
ec = [c for c in emb.columns if c not in META]
bf = pd.read_parquet(f"{R}/data_intermediate/baseline_features.parquet")
bcols = [c for c in bf.columns if c not in META and c != "split"
         and not c.endswith("_next_month") and pd.api.types.is_numeric_dtype(bf[c])]
d = (cp.merge(emb[["grid_id", "year_month"] + ec], on=["grid_id", "year_month"], how="inner")
       .merge(bf[["grid_id", "year_month"] + bcols], on=["grid_id", "year_month"], how="inner"))
print(f"joined rows (emb+baseline): {len(d)}  embed cols {len(ec)}  baseline cols {len(bcols)}")
print(f"next-period crisis rate (joined): {d.ge3_next.mean():.4f}")

# ---- 6. temporal split + xgb models ----
d = d.sort_values("year_month").reset_index(drop=True)
CUT = "2019-12"
tr = d[d.year_month <= CUT]; te = d[d.year_month > CUT]
ytr, yte = tr.ge3_next.to_numpy(), te.ge3_next.to_numpy()
print(f"train {len(tr)}  test {len(te)}  test crisis rate {yte.mean():.4f}")

def fit_eval(cols, name):
    m = xgb.train({"objective": "binary:logistic", "max_depth": 6, "eta": 0.1,
                   "subsample": 0.8, "colsample_bytree": 0.8, "seed": SEED},
                  xgb.DMatrix(tr[cols].to_numpy(), label=ytr), num_boost_round=300)
    p = m.predict(xgb.DMatrix(te[cols].to_numpy()))
    return name, roc_auc_score(yte, p), average_precision_score(yte, p)

res = [fit_eval(ec, "embedding"), fit_eval(bcols, "baseline"), fit_eval(ec + bcols, "combined"),
       ("persistence", roc_auc_score(yte, te.ge3.to_numpy()), average_precision_score(yte, te.ge3.to_numpy()))]
print("=" * 60)
print(f"FEWS transfer  test {CUT}+  n={len(te)}  crisis_rate={yte.mean():.3f}")
print("=" * 60)
for n, r, a in res:
    print(f"{n:14s} ROC {r:.4f}  PR-AUC {a:.4f}")
print("=" * 60)
pd.DataFrame(res, columns=["model", "roc_auc", "pr_auc"]).to_csv(f"{R}/outputs/fews_transfer_v1.csv", index=False)
print("saved outputs/fews_transfer_v1.csv")
