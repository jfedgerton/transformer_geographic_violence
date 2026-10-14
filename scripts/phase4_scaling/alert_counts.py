"""alert_counts.py -- equal-alert-budget counts for the Results text (Sections 4.1, 4.2, 4.4). CPU only.

All counts come from saved test-period forecasts; no model is refit. Every input forecast is first re-scored and
checked against the reported PR-AUC, so the counts are guaranteed to come from the same forecasts as the tables.

A. vs persistence (4.1). Persistence = violence in month t (binary). Temporal test, remap, 3 seeds.
   A1 same number of alerts as persistence: k = number of persistence alerts; base+emb flags its top-k cell-months.
      extra_hits = TP(base+emb, top-k) - TP(persistence).
   A2 same detection as persistence: smallest k' at which base+emb reaches persistence's TP.
      fewer_false_alarms = FP(persistence) - FP(base+emb at k').
B. nested (4.2). Budget k = number of violent test cell-months. extra_hits = TP(base+emb) - TP(base).
   With equal budgets fewer false alarms = extra hits by construction.
C. share grid (4.4). Same as B for every share x draw (nested arm, preds_nested_{region}_s{seed}.npz).
Ties in scores are broken by a stable sort on the saved row order (deterministic).

Inputs : outputs/preds_region_L_contrastive_e_loro_{region}_s{seed}.npz, outputs/eval_region_*.csv,
         outputs/share_grid/preds_nested_{region}_s{seed}.npz + metrics_nested_*.csv,
         data_intermediate/baseline_features_super.parquet (month-t violence)
Outputs: outputs/alert_counts_persistence.csv, alert_counts_nested.csv, alert_counts_share.csv,
         alert_counts_monthly.csv (per test month, budget-B alerts; appendix footnote)
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
O = f"{R}/outputs"
REGIONS = ["africa", "americas", "asia_pacific", "europe", "mena", "south_asia"]
SEEDS = [123, 456, 789]
TOL_EVAL, TOL_GRID = 1e-6, 1e-5          # grid forecasts are stored as float32


def topk(p, k):
    """Boolean mask of the k highest scores; ties broken by saved row order (stable)."""
    m = np.zeros(len(p), bool)
    if k > 0: m[np.argsort(-p, kind="stable")[:k]] = True
    return m


def k_to_match(p, y, tp_target):
    """Smallest k such that the top-k by p contains at least tp_target true positives."""
    hits = np.cumsum(y[np.argsort(-p, kind="stable")])
    idx = np.searchsorted(hits, tp_target)            # first position where cumulative hits >= tp_target
    assert idx < len(hits), "base+emb never reaches persistence's detections"
    return int(idx + 1)


# ---- month-t violence (same construction and checks as persistence_month_t.py) ----
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
print(f"[panel] month-t violence ready for {len(vt):,} grid-months", flush=True)

rowsA, rowsB, rowsM = [], [], []
for rg in REGIONS:
    for sd in SEEDS:
        tag = f"L_contrastive_e_loro_{rg}_s{sd}"
        z = np.load(f"{O}/preds_region_{tag}.npz", allow_pickle=True)
        y = z["y"].astype(int); pb = z["base"]; peb = z["emb_base"]
        ev = pd.read_csv(f"{O}/eval_region_{tag}.csv")
        ev = ev[(ev.target == "incidence") & (ev["mode"] == "B_temporal")].set_index("model")["score"]
        for name, p in [("base", pb), ("emb_base", peb)]:
            s = average_precision_score(y, p)
            assert abs(s - ev[name]) < TOL_EVAL, f"{tag} {name}: preds give {s:.8f}, eval file {ev[name]:.8f}"
        key = pd.MultiIndex.from_arrays([z["grid_id"].astype(np.int64), z["year_month"].astype(str)])
        pt = vt.reindex(key).to_numpy()
        assert not np.isnan(pt).any(), f"{tag}: month-t violence missing for some test rows"
        ym = z["year_month"].astype(str)
        # A. vs persistence
        k = int(pt.sum()); tp_p = int((pt * y).sum()); fp_p = k - tp_p
        tp_eb = int(y[topk(peb, k)].sum())
        k2 = k_to_match(peb, y, tp_p); fp_eb2 = k2 - int(y[topk(peb, k2)].sum())
        rowsA.append(dict(region=rg, seed=sd, n_test=len(y), events=int(y.sum()), persistence_alerts=k,
                          tp_persistence=tp_p, fp_persistence=fp_p, tp_embbase_same_alerts=tp_eb,
                          extra_hits_same_alerts=tp_eb - tp_p, alerts_embbase_same_detection=k2,
                          fp_embbase_same_detection=fp_eb2, fewer_false_alarms_same_detection=fp_p - fp_eb2))
        # B. nested, budget = number of events
        kb = int(y.sum()); mb, meb = topk(pb, kb), topk(peb, kb)
        rowsB.append(dict(region=rg, seed=sd, events=kb, tp_base=int(y[mb].sum()), tp_embbase=int(y[meb].sum()),
                          extra_hits=int(y[meb].sum() - y[mb].sum())))
        mp = pt.astype(bool)
        for mth in np.unique(ym):
            w = ym == mth
            rowsM.append(dict(region=rg, seed=sd, year_month=mth, events=int(y[w].sum()),
                              tp_persistence=int((mp & w & (y == 1)).sum()), fp_persistence=int((mp & w & (y == 0)).sum()),
                              tp_base=int((mb & w & (y == 1)).sum()), fp_base=int((mb & w & (y == 0)).sum()),
                              tp_embbase=int((meb & w & (y == 1)).sum()), fp_embbase=int((meb & w & (y == 0)).sum())))
        print(f"[{tag}] A extra hits {tp_eb - tp_p:+d}, fewer FA {fp_p - fp_eb2:+d}; B extra hits {rowsB[-1]['extra_hits']:+d}", flush=True)

A = pd.DataFrame(rowsA); B = pd.DataFrame(rowsB); M = pd.DataFrame(rowsM)
assert len(A) == 18 and len(B) == 18

# C. share grid
rowsC = []
for rg in REGIONS:
    for sd in SEEDS:
        z = np.load(f"{O}/share_grid/preds_nested_{rg}_s{sd}.npz", allow_pickle=True)
        y = z["__y"].astype(int); kb = int(y.sum())
        met = pd.read_csv(f"{O}/share_grid/metrics_nested_{rg}_s{sd}.csv")
        met = met[met.model.isin(["base", "emb_base"])]
        for r in met.itertuples(index=False):
            p = z[f"{r.model}|{r.share:g}|{r.draw}"]
            assert abs(average_precision_score(y, p) - r.pr) < TOL_GRID, f"grid {rg} s{sd} {r.model} {r.share} {r.draw}"
        for (sh, dr), g in met.groupby(["share", "draw"]):
            if set(g.model) != {"base", "emb_base"}: continue
            pb, peb = z[f"base|{sh:g}|{dr}"], z[f"emb_base|{sh:g}|{dr}"]
            tb, te = int(y[topk(pb, kb)].sum()), int(y[topk(peb, kb)].sum())
            rowsC.append(dict(region=rg, seed=sd, share=sh, draw=int(dr), events=kb, tp_base=tb, tp_embbase=te, extra_hits=te - tb))
C = pd.DataFrame(rowsC)
assert set(C.region) == set(REGIONS) and set(C.seed) == set(SEEDS)

A.to_csv(f"{O}/alert_counts_persistence.csv", index=False); B.to_csv(f"{O}/alert_counts_nested.csv", index=False)
C.to_csv(f"{O}/alert_counts_share.csv", index=False); M.to_csv(f"{O}/alert_counts_monthly.csv", index=False)

# ---- summary: seed means per region, then sums over regions ----
def summ(df, cols):
    r = df.groupby("region")[cols].mean()
    return r, r.sum()
ra, ta = summ(A, ["extra_hits_same_alerts", "fewer_false_alarms_same_detection"])
rb, tb = summ(B, ["extra_hits"])
c = C.groupby(["region", "seed", "share"]).extra_hits.mean().groupby(["region", "share"]).mean().unstack("share")
pd.set_option("display.width", 160)
print("\n[A] vs persistence, seed means by region\n", ra.round(1).to_string(), "\nTOTAL", ta.round(1).to_dict())
print("\n[B] nested, extra hits (= fewer false alarms), seed means\n", rb.round(1).to_string(), "\nTOTAL", tb.round(1).to_dict())
print("\n[C] share grid, extra hits by region x share (mean over seeds and draws)\n", c.round(1).to_string())
print("\n[C] four gaining regions at 1%:", round(c.loc[["africa", "americas", "mena", "south_asia"], 0.01].sum(), 1))
print("[done] alert_counts_{persistence,nested,share,monthly}.csv", flush=True)
