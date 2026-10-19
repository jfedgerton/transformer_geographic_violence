"""cell_alerts_persistence.py -- cell-level version of alert_counts.py part A1 (base + embedding vs persistence). CPU only.

Same rows, forecasts and alert rule as alert_counts.py (A1), so the cell sums reproduce outputs/alert_counts_persistence.csv:
  persistence alerts = cell-months with violence in month t (binary); k = number of persistence alerts
  base + embedding flags its top-k cell-months (stable sort on saved row order)
Per PRIO-GRID cell and region x seed: true and false alerts for each model. Cell values are averaged over the 3 seeds.
Checks: (1) every forecast re-scores to the eval CSV PR-AUC; (2) month-t violence built exactly as in alert_counts.py
(two derivations must agree); (3) per seed, summed cell gains equal extra_hits_same_alerts in alert_counts_persistence.csv;
(4) all 18 region x seed files present; every cell in exactly one region and all 3 seeds.
Output: outputs/cell_alerts_persistence.csv.gz (never overwritten).
"""
import os
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
O = f"{R}/outputs"
REGIONS = ["africa", "americas", "asia_pacific", "europe", "mena", "south_asia"]
SEEDS = [123, 456, 789]
TOL_EVAL = 1e-6
FO = f"{O}/cell_alerts_persistence.csv.gz"
assert not os.path.exists(FO), f"refusing to overwrite {FO}"


def topk(p, k):
    m = np.zeros(len(p), bool)
    if k > 0: m[np.argsort(-p, kind="stable")[:k]] = True
    return m


# ---- month-t violence: copied from alert_counts.py ----
pan = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet",
                      columns=["grid_id", "year_month", "violence_next_month", "violence_lag1"])
pan = pan.sort_values(["grid_id", "year_month"]).reset_index(drop=True)
assert not pan.duplicated(["grid_id", "year_month"]).any()
per = pd.PeriodIndex(pan["year_month"].astype(str), freq="M"); mi = pd.Series(per.year * 12 + per.month, index=pan.index)
consec = pan["grid_id"].eq(pan["grid_id"].shift(1)) & (mi - mi.shift(1)).eq(1)
pan["violence_t"] = np.where(consec, pan.groupby("grid_id")["violence_next_month"].shift(1), np.nan)
v2 = pan.groupby("grid_id")["violence_lag1"].shift(-1); both = pan["violence_t"].notna() & consec.shift(-1, fill_value=False) & v2.notna()
assert (pan.loc[both, "violence_t"] == v2[both]).all(), "month-t violence derivations disagree"
vt = pd.Series(np.nan_to_num(pan["violence_t"].to_numpy()), index=pd.MultiIndex.from_arrays(
    [pan["grid_id"].astype(np.int64).to_numpy(), pan["year_month"].astype(str).to_numpy()]))
del pan
print(f"[panel] month-t violence ready for {len(vt):,} grid-months", flush=True)

ref = pd.read_csv(f"{O}/alert_counts_persistence.csv").set_index(["region", "seed"])
per_seed = []
for rg in REGIONS:
    for sd in SEEDS:
        tag = f"L_contrastive_e_loro_{rg}_s{sd}"
        z = np.load(f"{O}/preds_region_{tag}.npz", allow_pickle=True)
        g = z["grid_id"].astype(np.int64); ym = z["year_month"].astype(str); y = z["y"].astype(int); peb = z["emb_base"].astype(float)
        assert set(np.unique(y)) <= {0, 1} and np.isfinite(peb).all()
        ev = pd.read_csv(f"{O}/eval_region_{tag}.csv")
        ev = ev[(ev.target == "incidence") & (ev["mode"] == "B_temporal")].set_index("model")["score"]
        s = average_precision_score(y, peb)
        assert abs(s - ev["emb_base"]) < TOL_EVAL, f"{tag}: preds {s:.8f} vs eval {ev['emb_base']:.8f}"
        pt = vt.reindex(pd.MultiIndex.from_arrays([g, ym])).to_numpy()
        assert not np.isnan(pt).any(), f"{tag}: month-t violence missing"
        ap = pt.astype(bool); k = int(ap.sum()); ae = topk(peb, k)
        gain = int(y[ae].sum()) - int(y[ap].sum())
        assert gain == int(ref.loc[(rg, sd), "extra_hits_same_alerts"]), f"{tag}: {gain} != alert_counts_persistence"
        df = pd.DataFrame({"gid": g, "y": y, "tp_pers": ap & (y == 1), "fp_pers": ap & (y == 0),
                           "tp_emb": ae & (y == 1), "fp_emb": ae & (y == 0)})
        c = df.groupby("gid").agg(months=("y", "size"), events=("y", "sum"), tp_pers=("tp_pers", "sum"), fp_pers=("fp_pers", "sum"),
                                  tp_emb=("tp_emb", "sum"), fp_emb=("fp_emb", "sum")).reset_index()
        c["region"], c["seed"] = rg, sd
        per_seed.append(c)
        print(f"[{tag}] k={k:,} extra hits {gain:+d} (matches alert_counts_persistence)", flush=True)

assert len(per_seed) == 18
allc = pd.concat(per_seed, ignore_index=True)
assert allc.groupby("gid").region.nunique().max() == 1, "cell in more than one region"
nseed = allc.groupby("gid").seed.nunique()
assert (nseed == 3).all(), f"{int((nseed < 3).sum())} cells missing a seed"
num = ["months", "events", "tp_pers", "fp_pers", "tp_emb", "fp_emb"]
out = allc.groupby(["gid", "region"])[num].mean().reset_index()
out["d_tp"] = out.tp_emb - out.tp_pers; out["d_fp"] = out.fp_emb - out.fp_pers
row = (out.gid - 1) // 720; col = (out.gid - 1) % 720
out["lat"], out["lon"] = -90 + 0.5 * row + 0.25, -180 + 0.5 * col + 0.25
out.round(5).to_csv(FO + ".part.gz", index=False, compression="gzip"); os.replace(FO + ".part.gz", FO)
print(f"[cell] wrote {FO}: {len(out):,} cells; extra correct alerts {out.d_tp.sum():+.1f}, extra false alerts {out.d_fp.sum():+.1f}")
