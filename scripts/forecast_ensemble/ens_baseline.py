"""ens_baseline.py -- forecasting ensemble, comparison models for one target (CPU only).

Same rows, targets and windows as ens_finetune.py (the panel/target block is copied verbatim; ens_combine.py also checks
labels row by row). Two comparison forecasts for the stack (2021-07..2021-12) and test (>= 2022-01) windows:
  persistence  incidence / onset: violence in month t (onset rows have none, so this is constant 0);
               civilian: count of violence-against-civilians events in month t; fatal25: fatalities in month t
  base         XGBoost on the hand-built features + PRIO-GRID static features (the main paper's base model),
               trained on the SAME balanced train sample as ens_finetune.py, fixed settings, seed 123
Output (never overwritten): outputs/ens/preds_baseline_{target}.npz  grid_id, year_month, region, window, y,
  p_persistence, p_base
"""
import os, sys
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

R = os.environ.get("ENS_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
E = os.environ.get
TARGET = E("ENS_TARGET", "incidence"); SMOKE = E("SMOKE", "0") == "1"
REGION_ORDER = ["africa", "mena", "europe", "americas", "south_asia", "asia_pacific"]
TARGETS = ["incidence", "civilian", "onset", "fatal25"]
assert TARGET in TARGETS, TARGET
TRAIN_START, TRAIN_END = "1990-01", "2020-12"
STOP_START, STOP_END = "2021-01", "2021-06"
STACK_START, STACK_END = "2021-07", "2021-12"
TEST_START = "2022-01"
POS_CAP = int(E("ENS_POS_CAP", "100000")); NEG_PER_POS = int(E("ENS_NEG_PER_POS", "4"))
SAMPLE_SEED, XGB_SEED = 123, 123
if SMOKE: POS_CAP = int(E("ENS_POS_CAP", "200"))
SUF = "_smoke" if SMOKE else ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regions import REGION_OF
OD = f"{R}/outputs/ens"; os.makedirs(OD, exist_ok=True)
OUT = f"{OD}/preds_baseline_{TARGET}{SUF}.npz"
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"

base_cols = ["violence_lag1", "violence_lag3", "violence_lag6", "spatial_lag_violence",
             "violent_count_roll1", "violent_count_roll3", "all_count_roll1", "any_event_lag1",
             "log_event_density_lag1", "months_since_last_event"]
# ---- panel and targets: copied from ens_finetune.py (plus base_cols) ----
panel = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet",
                        columns=["grid_id", "year_month", "violence_next_month", "has_violence",
                                 "count_fatalities", "etype_violence_against_civilians"] + base_cols)
assert not panel.duplicated(["grid_id", "year_month"]).any(), "duplicate grid-months"
srt = panel[["grid_id", "year_month", "has_violence", "count_fatalities", "etype_violence_against_civilians"]].sort_values(
    ["grid_id", "year_month"], kind="mergesort")
g_ = srt["grid_id"].to_numpy(); per = pd.PeriodIndex(srt["year_month"].astype(str), freq="M"); mi = (per.year * 12 + per.month).to_numpy()
assert ((g_[1:] != g_[:-1]) | (mi[1:] - mi[:-1] == 1)).all(), "month gap within a cell; next-month labels would misalign"
nxt = srt.groupby("grid_id")[["has_violence", "count_fatalities", "etype_violence_against_civilians"]].shift(-1)
srt["has_next"] = nxt["has_violence"].notna()
srt["fat_next"] = nxt["count_fatalities"]; srt["civ_next"] = nxt["etype_violence_against_civilians"].fillna(0.0)
for c in ("has_next", "fat_next", "civ_next"): panel[c] = srt[c]
del srt, nxt, g_, per, mi
cm = pd.read_parquet(f"{R}/data_intermediate/baseline_features.parquet", columns=["grid_id", "year_month", "country"])
panel = panel.merge(cm, on=["grid_id", "year_month"], how="left"); del cm
panel["region"] = panel["country"].map(REGION_OF).fillna("other")
d = panel[panel["region"].isin(REGION_ORDER) & panel["violence_next_month"].notna() & panel["has_next"]].reset_index(drop=True)
del panel
if TARGET == "onset": d = d[d["has_violence"].fillna(0).astype(int) == 0].reset_index(drop=True)
if TARGET == "incidence": y = d["violence_next_month"].to_numpy().astype(int)
elif TARGET == "civilian": y = (d["civ_next"].to_numpy() > 0).astype(int)
elif TARGET == "onset": y = d["violence_next_month"].to_numpy().astype(int)
else:
    assert d["fat_next"].notna().all(), "missing next-month fatalities"
    y = (d["fat_next"].to_numpy() >= 25).astype(int)
ym = d["year_month"].astype(str).to_numpy()
w_tr = (ym >= TRAIN_START) & (ym <= TRAIN_END); w_stack = (ym >= STACK_START) & (ym <= STACK_END); w_te = ym >= TEST_START
# ---- the same train sample as ens_finetune.py (same generator, same draw order) ----
rng = np.random.default_rng([SAMPLE_SEED, TARGETS.index(TARGET)])
pos = np.where(w_tr & (y == 1))[0]; neg = np.where(w_tr & (y == 0))[0]
if len(pos) > POS_CAP: pos = rng.choice(pos, POS_CAP, replace=False)
neg = rng.choice(neg, min(len(neg), NEG_PER_POS * len(pos)), replace=False)
tr_idx = np.sort(np.concatenate([pos, neg]))
out_idx = np.where(w_stack | w_te)[0]
# ---- provenance: same hash as ens_finetune.py (Codex 12, finding 4) ----
import hashlib
def data_version():
    out = []
    for f in ("baseline_features_super.parquet", "baseline_features.parquet"):
        st_ = os.stat(f"{R}/data_intermediate/{f}"); out.append(f"{f}:{st_.st_size}:{int(st_.st_mtime)}")
    return "|".join(out)
def sample_hash(keys_g, keys_ym):
    h = hashlib.sha256(); h.update(np.ascontiguousarray(keys_g.astype(np.int64)).tobytes()); h.update("|".join(keys_ym).encode())
    h.update(f"{TARGET}|{POS_CAP}|{NEG_PER_POS}|{SAMPLE_SEED}|{TRAIN_START}|{TRAIN_END}|{data_version()}".encode())
    return h.hexdigest()
SAMPLE_HASH = sample_hash(d["grid_id"].to_numpy()[tr_idx], ym[tr_idx].astype(str))
print(f"[provenance] train-sample hash {SAMPLE_HASH[:16]}", flush=True)
ps = pd.read_parquet(f"{R}/data_intermediate/priogrid_static.parquet")
assert ps["grid_id"].is_unique
n0 = len(d); d = d.merge(ps, on="grid_id", how="left"); assert len(d) == n0, "prio merge changed row count"
FEATS = base_cols + [c for c in ps.columns if c != "grid_id"]
X = d[FEATS].fillna(0.0).to_numpy(np.float32)
b = xgb.train({"objective": "binary:logistic", "max_depth": 6, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
               "seed": XGB_SEED}, xgb.DMatrix(X[tr_idx], label=y[tr_idx]), 200)
p_base = b.predict(xgb.DMatrix(X[out_idx]))
if TARGET in ("incidence", "onset"): pers = d["has_violence"].fillna(0).to_numpy().astype(float)
elif TARGET == "civilian": pers = d["etype_violence_against_civilians"].fillna(0).to_numpy().astype(float)
else: pers = d["count_fatalities"].fillna(0).to_numpy().astype(float)
win = np.where(w_te[out_idx], "test", "stack")
for w in ("stack", "test"):
    s = win == w
    print(f"[{TARGET} {w}] n={s.sum():,} pos={int(y[out_idx][s].sum()):,} PR persistence={average_precision_score(y[out_idx][s], pers[out_idx][s]):.4f} "
          f"base={average_precision_score(y[out_idx][s], p_base[s]):.4f}", flush=True)
np.savez_compressed(OUT + ".part.npz", grid_id=d["grid_id"].to_numpy().astype(np.int64)[out_idx], year_month=ym[out_idx].astype("U7"),
                    region=d["region"].to_numpy()[out_idx].astype("U12"), window=win.astype("U5"), y=y[out_idx].astype(np.int8),
                    p_persistence=pers[out_idx].astype(np.float32), p_base=p_base.astype(np.float32),
                    sample_hash=np.array(SAMPLE_HASH), data_version=np.array(data_version()), pos_cap=POS_CAP, neg_per_pos=NEG_PER_POS)
os.replace(OUT + ".part.npz", OUT)
print(f"[done] {OUT}", flush=True)
