"""ens_baseline_mt.py -- comparison models for the four-head forecasting ensemble (CPU). One job writes all four targets.

The panel, targets, windows and SHARED TRAINING SAMPLE are built by the block between PANEL-BLOCK-START/END, copied
verbatim from ens_multitask.py (ens_combine.py checks the sample hash, so any drift is caught).
Per target, on the stack (2021-07..12) and test (>= 2022-01) rows (onset: rows with has_violence == 0 only):
  persistence  incidence / onset: violence in month t (constant 0 on onset rows); civilian: month-t count of
               violence-against-civilians events; fatal25: month-t fatalities
  base         XGBoost on the hand-built features + PRIO-GRID static features, trained on the shared sample's rows
               where the target is defined, fixed settings, seed 123
Outputs (never overwritten): outputs/ens/preds_baseline_mt_{target}.npz, then outputs/ens/done_baseline_mt.json (marker).
"""
import os, sys, json, hashlib
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

R = os.environ.get("ENS_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
E = os.environ.get
SMOKE = E("SMOKE", "0") == "1"; SUF = "_smoke" if SMOKE else ""
REGION_ORDER = ["africa", "mena", "europe", "americas", "south_asia", "asia_pacific"]
TARGETS = ["incidence", "civilian", "onset", "fatal25"]
TRAIN_START, TRAIN_END = "1990-01", "2020-12"
STOP_START, STOP_END = "2021-01", "2021-06"
STACK_START, STACK_END = "2021-07", "2021-12"
TEST_START = "2022-01"
POS_CAP = int(E("ENS_POS_CAP", "150000")); NEG_PER_POS = int(E("ENS_NEG_PER_POS", "4"))
STOP_SHARE = 0.25; SAMPLE_SEED, XGB_SEED = 123, 123
if SMOKE: POS_CAP = int(E("ENS_POS_CAP", "300"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regions import REGION_OF
OD = f"{R}/outputs/ens"; os.makedirs(OD, exist_ok=True)
OUT_DONE = f"{OD}/done_baseline_mt{SUF}.json"
OUT = {t: f"{OD}/preds_baseline_mt_{t}{SUF}.npz" for t in TARGETS}
import fcntl
def _job_lock(path):
    """Exclusive per-job lock (Codex 12c): taken before the completion-marker check and held until exit, so an
    overlapping duplicate job fails fast instead of racing through publication."""
    fh = open(path, "a")
    try: fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit(f"[lock] {path} is held by another running job; exiting without writing")
    return fh
_LOCK = _job_lock(f"{OD}/lock_baseline_mt{SUF}.lock")
assert not os.path.exists(OUT_DONE), f"refusing to overwrite {OUT_DONE}"

# PANEL-BLOCK-START (identical in ens_baseline_mt.py)
panel = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet",
                        columns=["grid_id", "year_month", "violence_next_month", "has_violence",
                                 "count_fatalities", "etype_violence_against_civilians"])
assert not panel.duplicated(["grid_id", "year_month"]).any(), "duplicate grid-months"
srt = panel[["grid_id", "year_month", "has_violence", "count_fatalities", "etype_violence_against_civilians"]].sort_values(
    ["grid_id", "year_month"], kind="mergesort")
g_ = srt["grid_id"].to_numpy(); per = pd.PeriodIndex(srt["year_month"].astype(str), freq="M"); mi = (per.year * 12 + per.month).to_numpy()
assert ((g_[1:] != g_[:-1]) | (mi[1:] - mi[:-1] == 1)).all(), "month gap within a cell; next-month labels would misalign"
nxt = srt.groupby("grid_id")[["has_violence", "count_fatalities", "etype_violence_against_civilians"]].shift(-1)
srt["has_next"] = nxt["has_violence"].notna()
srt["fat_next"] = nxt["count_fatalities"]; srt["civ_next"] = nxt["etype_violence_against_civilians"].fillna(0.0)
srt["hv_next"] = nxt["has_violence"]
for c in ("has_next", "fat_next", "civ_next", "hv_next"): panel[c] = srt[c]
del srt, nxt, g_, per, mi
ok = panel["has_next"] & panel["violence_next_month"].notna()
assert (panel.loc[ok, "hv_next"].astype(int) == panel.loc[ok, "violence_next_month"].astype(int)).all(), "has_violence(t+1) != violence_next_month(t)"
cm = pd.read_parquet(f"{R}/data_intermediate/baseline_features.parquet", columns=["grid_id", "year_month", "country"])
panel = panel.merge(cm, on=["grid_id", "year_month"], how="left"); del cm
panel["region"] = panel["country"].map(REGION_OF).fillna("other")
d = panel[panel["region"].isin(REGION_ORDER) & panel["violence_next_month"].notna() & panel["has_next"]].reset_index(drop=True)
del panel
assert d["fat_next"].notna().all(), "missing next-month fatalities"
quiet = d["has_violence"].fillna(0).astype(int).to_numpy() == 0
Y = np.full((len(d), 4), np.nan, np.float32)
Y[:, 0] = d["violence_next_month"].to_numpy().astype(int)
Y[:, 1] = (d["civ_next"].to_numpy() > 0).astype(int)
Y[quiet, 2] = Y[quiet, 0]                                   # onset: defined only where month t is quiet
Y[:, 3] = (d["fat_next"].to_numpy() >= 25).astype(int)
ym = d["year_month"].astype(str).to_numpy(); gids = d["grid_id"].to_numpy().astype(np.int64); reg = d["region"].to_numpy()
w_tr = (ym >= TRAIN_START) & (ym <= TRAIN_END); w_stop = (ym >= STOP_START) & (ym <= STOP_END)
w_stack = (ym >= STACK_START) & (ym <= STACK_END); w_te = ym >= TEST_START
assert not (w_tr & w_stop).any() and not (w_stop & w_stack).any() and not (w_stack & w_te).any()
for k, t in enumerate(TARGETS):
    for nm, w in (("train", w_tr), ("stop", w_stop), ("stack", w_stack), ("test", w_te)):
        yy = Y[w, k]; yy = yy[~np.isnan(yy)]
        assert len(yy) and yy.sum() > 0, f"{t} {nm}: no positives"
        print(f"[split] {t:9s} {nm:5s} rows={len(yy):,} positives={int(yy.sum()):,} rate={yy.mean():.4f}", flush=True)
# ---- shared training sample: rare heads never subsampled ----
rng = np.random.default_rng([SAMPLE_SEED, 99])
rare_pos = w_tr & ((Y[:, 1] == 1) | (Y[:, 2] == 1) | (Y[:, 3] == 1))
inc_only = w_tr & (Y[:, 0] == 1) & ~rare_pos
all_neg = w_tr & (Y[:, 0] == 0) & (Y[:, 1] == 0) & (Y[:, 3] == 0)          # onset is 0 or masked on these rows
p_rare = np.where(rare_pos)[0]; p_inc = np.where(inc_only)[0]
n_inc = max(0, min(len(p_inc), POS_CAP - len(p_rare)))
pos = np.concatenate([p_rare, rng.choice(p_inc, n_inc, replace=False)])
neg = rng.choice(np.where(all_neg)[0], min(int(all_neg.sum()), NEG_PER_POS * len(pos)), replace=False)
tr_idx = np.sort(np.concatenate([pos, neg]))
any_pos_stop = w_stop & ((Y[:, 0] == 1) | (Y[:, 1] == 1) | (Y[:, 3] == 1))
neg_stop = np.where(w_stop & ~any_pos_stop)[0]
st_idx = np.sort(np.concatenate([np.where(any_pos_stop)[0], rng.choice(neg_stop, int(round(STOP_SHARE * len(neg_stop))), replace=False)]))
assert len(np.intersect1d(tr_idx, st_idx)) == 0
def data_version():
    out = []
    for f in ("baseline_features_super.parquet", "baseline_features.parquet"):
        st_ = os.stat(f"{R}/data_intermediate/{f}"); out.append(f"{f}:{st_.st_size}:{int(st_.st_mtime)}")
    return "|".join(out)
h_ = hashlib.sha256(); h_.update(np.ascontiguousarray(gids[tr_idx]).tobytes()); h_.update("|".join(ym[tr_idx]).encode())
h_.update(f"multitask|{POS_CAP}|{NEG_PER_POS}|{SAMPLE_SEED}|{TRAIN_START}|{TRAIN_END}|{data_version()}".encode())
SAMPLE_HASH = h_.hexdigest()
# PANEL-BLOCK-END
print(f"[provenance] train-sample hash {SAMPLE_HASH[:16]} ({data_version()}); train={len(tr_idx):,}", flush=True)
base_cols = ["violence_lag1", "violence_lag3", "violence_lag6", "spatial_lag_violence",
             "violent_count_roll1", "violent_count_roll3", "all_count_roll1", "any_event_lag1",
             "log_event_density_lag1", "months_since_last_event"]
bf = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet", columns=["grid_id", "year_month"] + base_cols)
n0 = len(d)
d = d.merge(bf, on=["grid_id", "year_month"], how="left", validate="one_to_one"); del bf
ps = pd.read_parquet(f"{R}/data_intermediate/priogrid_static.parquet"); assert ps["grid_id"].is_unique
d = d.merge(ps, on="grid_id", how="left"); assert len(d) == n0, "feature merges changed row count"
assert np.array_equal(d["grid_id"].to_numpy().astype(np.int64), gids), "row order changed by merges"
FEATS = base_cols + [c for c in ps.columns if c != "grid_id"]
X = d[FEATS].fillna(0.0).to_numpy(np.float32)
pers_all = {"incidence": d["has_violence"].fillna(0).to_numpy().astype(float),
            "onset": d["has_violence"].fillna(0).to_numpy().astype(float),
            "civilian": d["etype_violence_against_civilians"].fillna(0).to_numpy().astype(float),
            "fatal25": d["count_fatalities"].fillna(0).to_numpy().astype(float)}
out_all = np.where(w_stack | w_te)[0]
for k, t in enumerate(TARGETS):
    yk_tr = Y[tr_idx, k]; s_tr = ~np.isnan(yk_tr)
    b = xgb.train({"objective": "binary:logistic", "max_depth": 6, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
                   "seed": XGB_SEED}, xgb.DMatrix(X[tr_idx][s_tr], label=yk_tr[s_tr]), 200)
    yk = Y[out_all, k]; s = ~np.isnan(yk); rows = out_all[s]
    p_base = b.predict(xgb.DMatrix(X[rows])); pers = pers_all[t][rows]
    win = np.where(w_te[rows], "test", "stack")
    for w in ("stack", "test"):
        q = win == w
        print(f"[{t} {w}] n={q.sum():,} pos={int(yk[s][q].sum()):,} PR persistence={average_precision_score(yk[s][q], pers[q]):.4f} "
              f"base={average_precision_score(yk[s][q], p_base[q]):.4f}", flush=True)
    np.savez_compressed(OUT[t] + ".part.npz", grid_id=gids[rows], year_month=ym[rows].astype("U7"), region=reg[rows].astype("U12"),
                        window=win.astype("U5"), y=yk[s].astype(np.int8), p_persistence=pers.astype(np.float32),
                        p_base=p_base.astype(np.float32), sample_hash=np.array(SAMPLE_HASH), data_version=np.array(data_version()),
                        pos_cap=POS_CAP, neg_per_pos=NEG_PER_POS)
for t in TARGETS: os.replace(OUT[t] + ".part.npz", OUT[t])
json.dump(dict(sample_hash=SAMPLE_HASH, files=OUT), open(OUT_DONE + ".part", "w"), indent=1); os.replace(OUT_DONE + ".part", OUT_DONE)
print(f"[done] {OUT_DONE}", flush=True)
