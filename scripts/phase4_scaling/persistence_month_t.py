"""persistence_month_t.py -- incidence persistence using the most recent month (t) instead of t-1.

Why: the target at row t is violence in month t+1 (03_aggregate_location_month.py: shift(-1)). The evaluator's
incidence persistence uses violence_lag1 (month t-1), a two-month-old signal, while the base model already sees
month-t information through months_since_last_event and civilian persistence uses month t. This script rescores
incidence persistence with has_violence in month t on EXACTLY the evaluator's rows and folds. No model is refit.

Row/fold identity with transfer_eval_region_fullpreds.py (md5 98414c81be44) is enforced, not assumed: the script
also recomputes the old t-1 persistence on the same rows and asserts it equals the persistence score in each
outputs/eval_region_{TAG}.csv (|diff| < 1e-9) for every region x seed x mode.

Inputs : data_intermediate/baseline_features_super.parquet, baseline_features.parquet (country),
         outputs/eval_region_L_contrastive_e_loro_{region}_s{seed}.csv (remap main spec)
Output : outputs/persistence_month_t.csv (one row per region x seed x mode)
Run    : CPU only, ~10 min. Same venv as the evaluator (row order of the pooled folds depends on pandas sort).
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
sys.path.insert(0, os.path.join(R, "scripts", "phase4_scaling"))
from regions import REGION_OF, REGIONS

TRAIN_END, TEST_START = "2020-12", "2022-01"      # same constants as the evaluator
SEEDS = [123, 456, 789]
OUT = f"{R}/outputs/persistence_month_t.csv"
TOL = 1e-9
assert sorted(REGIONS) == sorted(["africa", "americas", "asia_pacific", "europe", "mena", "south_asia"]), REGIONS

# ---- load: same file, same merge, same sort as the evaluator ----
cols = ["grid_id", "year_month", "violence_next_month", "violence_lag1"]
panel = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet", columns=cols)
cm = pd.read_parquet(f"{R}/data_intermediate/baseline_features.parquet", columns=["grid_id", "year_month", "country"])
panel = panel.merge(cm, on=["grid_id", "year_month"], how="left")
panel = panel.sort_values(["grid_id", "year_month"]).reset_index(drop=True)
assert not panel.duplicated(["grid_id", "year_month"]).any(), "duplicate grid-months"
print(f"[load] {len(panel):,} rows, {panel.grid_id.nunique():,} cells", flush=True)

# ---- month-t violence: the previous month's target, valid only when the previous row is exactly one month earlier ----
per = pd.PeriodIndex(panel["year_month"].astype(str), freq="M")
mi = per.year * 12 + per.month
same_cell = panel["grid_id"].eq(panel["grid_id"].shift(1))
consec = same_cell & (pd.Series(mi, index=panel.index) - pd.Series(mi, index=panel.index).shift(1)).eq(1)
v_t = panel.groupby("grid_id")["violence_next_month"].shift(1)        # row t-1's target = violence in month t
panel["violence_t"] = np.where(consec, v_t, np.nan)
# cross-check against the independent derivation: row t+1's violence_lag1 is also violence in month t
nxt_consec = consec.shift(-1, fill_value=False)
v_t2 = panel.groupby("grid_id")["violence_lag1"].shift(-1)
both = panel["violence_t"].notna() & nxt_consec & v_t2.notna()
mism = (panel.loc[both, "violence_t"] != v_t2[both]).sum()
print(f"[check] month-t violence from two derivations: compared {both.sum():,} rows, mismatches {mism}", flush=True)
assert mism == 0, "the two derivations of month-t violence disagree -> panel/target construction differs from assumptions"
print(f"[check] month gaps within cells: {int((same_cell & ~consec).sum()):,} rows have a non-consecutive predecessor", flush=True)

rows = []
for rg in REGIONS:
    rc = set(c for c, r in REGION_OF.items() if r == rg)
    d = panel[panel["country"].isin(rc) & panel["violence_next_month"].notna()].sort_values("year_month").reset_index(drop=True)
    y = d["violence_next_month"].to_numpy().astype(int)
    lag1 = np.nan_to_num(d["violence_lag1"].to_numpy())
    vt_raw = d["violence_t"].to_numpy()
    n_missing = int(np.isnan(vt_raw).sum())
    vt = np.nan_to_num(vt_raw)          # first month of a cell / after a gap: no month-t value -> 0, same rule as lag1
    ym = d["year_month"].astype(str).to_numpy()
    te_b = ym >= TEST_START
    print(f"[{rg}] n={len(d):,} test_B={te_b.sum():,} month-t missing={n_missing:,} (in test_B {int(np.isnan(vt_raw[te_b]).sum()):,})", flush=True)
    for sd in SEEDS:
        tag = f"L_contrastive_e_loro_{rg}_s{sd}"
        ev = pd.read_csv(f"{R}/outputs/eval_region_{tag}.csv")
        ev = ev[(ev.target == "incidence") & (ev.model == "persistence")].set_index("mode")
        masks = [("B_temporal", te_b)]
        kf = KFold(n_splits=5, shuffle=True, random_state=sd)
        for k, (_, tek) in enumerate(kf.split(np.arange(len(d)))):
            m = np.zeros(len(d), bool); m[tek] = True
            masks.append((f"A_pooled_f{k}", m))
        for mode, te in masks:
            old = average_precision_score(y[te], lag1[te])
            new = average_precision_score(y[te], vt[te])
            rep = float(ev.loc[mode, "score"]); n_rep = int(ev.loc[mode, "n"])
            assert int(te.sum()) == n_rep, f"{tag} {mode}: n {int(te.sum())} != reported {n_rep}"
            assert abs(old - rep) < TOL, f"{tag} {mode}: recomputed t-1 persistence {old:.10f} != reported {rep:.10f}"
            rows.append(dict(region=rg, seed=sd, mode=mode, n=int(te.sum()), pos_rate=float(y[te].mean()),
                             persistence_tm1_reported=rep, persistence_tm1_recomputed=old, persistence_t=new,
                             month_t_missing_in_test=int(np.isnan(vt_raw[te]).sum())))

out = pd.DataFrame(rows)
assert len(out) == 6 * 3 * 6, len(out)
out.to_csv(OUT, index=False)
print(f"[done] wrote {OUT} ({len(out)} rows); all {len(out)} t-1 scores match the eval files within {TOL}", flush=True)
out["m"] = np.where(out["mode"] == "B_temporal", "temporal", "pooled")
s = out.groupby(["m", "region", "seed"])[["persistence_tm1_reported", "persistence_t"]].mean().groupby(["m", "region"]).mean()
pd.set_option("display.width", 160)
print(s.round(4).to_string())
