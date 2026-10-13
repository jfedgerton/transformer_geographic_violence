"""gbm_region_frac.py -- XGBoost on hand-crafted features at the SAME train fractions and the SAME
subsampled rows as transfer_eval_region_ft.py (same rng call sequence, seed 123, Mode B split).
Features and XGBoost settings copied from transfer_eval_region.py (base_cols + PRIO-GRID statics;
depth 6, eta 0.1, subsample 0.8, colsample 0.8, 200 rounds).
Out: outputs/gbm_region_frac_{RG}_s123.csv
"""
import os, sys
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
RG = os.environ["EVAL_REGION"]
SEED = 123
FRACS = [0.01, 0.05, 0.10, 0.25, 1.0]
FT_MAX_ROWS = 150000
TRAIN_END, TEST_START = "2020-12", "2022-01"
OUT = f"{R}/outputs/gbm_region_frac_{RG}_s{SEED}.csv"
REF = f"{R}/outputs/ft_region_L_contrastive_e_loro_{RG}_s{SEED}.csv"
sys.path.insert(0, f"{R}/scripts/phase4_scaling")
from regions import REGION_OF, REGIONS
assert RG in REGIONS, f"bad EVAL_REGION {RG}"
assert os.path.exists(REF), f"missing reference {REF}"
REGION_COUNTRIES = set(c for c, r in REGION_OF.items() if r == RG)
rng = np.random.default_rng(SEED)

base_cols = ["violence_lag1", "violence_lag3", "violence_lag6", "spatial_lag_violence",
             "violent_count_roll1", "violent_count_roll3", "all_count_roll1", "any_event_lag1",
             "log_event_density_lag1", "months_since_last_event"]
panel = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet",
                        columns=["grid_id", "year_month", "violence_next_month"] + base_cols)
cm = pd.read_parquet(f"{R}/data_intermediate/baseline_features.parquet", columns=["grid_id", "year_month", "country"])
panel = panel.merge(cm, on=["grid_id", "year_month"], how="left")
d = panel[panel["country"].isin(REGION_COUNTRIES) & panel["violence_next_month"].notna()].sort_values("year_month").reset_index(drop=True)
n0 = len(d)
pst = pd.read_parquet(f"{R}/data_intermediate/priogrid_static.parquet")
prio_cols = [c for c in pst.columns if c != "grid_id"]
assert pst["grid_id"].is_unique, "priogrid_static has duplicate grid_id"
d = d.merge(pst, on="grid_id", how="left")
assert len(d) == n0, "prio merge changed row count"
FEATS = base_cols + prio_cols
d[FEATS] = d[FEATS].fillna(0.0)
X = d[FEATS].to_numpy(np.float32)

ym = d["year_month"].to_numpy(); trm = ym <= TRAIN_END; tem = ym >= TEST_START
y = d["violence_next_month"].to_numpy().astype(int); ytr, yte = y[trm], y[tem]
print(f"[region {RG}] rows={len(d):,} train={trm.sum():,} test={tem.sum():,} test_pos={yte.mean():.4f} feats={len(FEATS)}", flush=True)

# ---- must be the same panel the fine-tune run saw ----
ref = pd.read_csv(REF)
assert int(ref["n_train"].iloc[0]) == int(trm.sum()), (ref["n_train"].iloc[0], trm.sum())
assert int(ref["n_test"].iloc[0]) == int(tem.sum()), (ref["n_test"].iloc[0], tem.sum())
assert abs(float(ref["test_pos"].iloc[0]) - float(yte.mean())) < 1e-12
persist = np.nan_to_num(d["violence_lag1"].to_numpy())[tem]
pr_p = average_precision_score(yte, persist)
ref_p = float(ref.loc[ref["model"] == "persistence", "pr"].iloc[0])
assert abs(pr_p - ref_p) < 1e-9, f"persistence PR mismatch {pr_p} vs {ref_p}"
print("[check] panel matches fine-tune reference (n_train, n_test, test_pos, persistence PR)", flush=True)

def fit_eval(idx):
    clf = xgb.train({"objective": "binary:logistic", "max_depth": 6, "eta": 0.1, "subsample": 0.8,
                     "colsample_bytree": 0.8, "seed": SEED}, xgb.DMatrix(X[idx], label=y[idx]), 200)
    p = clf.predict(xgb.DMatrix(X[tem]))
    return roc_auc_score(yte, p), average_precision_score(yte, p)

rows = []
base = dict(region=RG, seed=SEED, n_train=int(trm.sum()), n_test=int(tem.sum()), test_pos=float(yte.mean()))
rows.append(dict(base, model="persistence", frac=1.0, roc=roc_auc_score(yte, persist), pr=pr_p))
tr_idx = np.where(trm)[0]; pos_idx = tr_idx[y[tr_idx] == 1]; neg_idx = tr_idx[y[tr_idx] == 0]
for frac in FRACS:
    npos = int(round(len(pos_idx) * frac)); nneg = int(round(len(neg_idx) * frac))
    if npos < 5:
        rows.append(dict(base, model="too_few_pos", frac=frac, roc=np.nan, pr=np.nan)); continue
    idx = np.concatenate([rng.choice(pos_idx, size=npos, replace=False), rng.choice(neg_idx, size=nneg, replace=False)])
    if len(idx) > FT_MAX_ROWS: idx = rng.choice(idx, size=FT_MAX_ROWS, replace=False)
    rng.shuffle(idx)
    n_ref = int(ref.loc[(ref["model"] == "ft_pretrained") & (np.isclose(ref["frac"], frac)), "n_ft"].iloc[0])
    assert len(idx) == n_ref, f"frac {frac}: n {len(idx)} vs fine-tune {n_ref}"
    roc, pr = fit_eval(idx)
    rows.append(dict(base, model="gbm_base", frac=frac, n_ft=int(len(idx)), roc=roc, pr=pr))
    print(f"  gbm_base frac={frac} n={len(idx)} ROC {roc:.4f} PR {pr:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
# validation row: every train row, no 150k cap (should sit near the Table 1 'base' arm)
roc, pr = fit_eval(tr_idx)
rows.append(dict(base, model="gbm_base_alltrain", frac=1.0, n_ft=int(len(tr_idx)), roc=roc, pr=pr))
print(f"  gbm_base_alltrain n={len(tr_idx)} ROC {roc:.4f} PR {pr:.4f}", flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"[done] wrote {OUT} ({len(rows)} rows)", flush=True)
