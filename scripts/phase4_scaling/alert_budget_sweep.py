"""alert_budget_sweep.py -- hits at a range of alert budgets (appendix figure behind the footnote in Section 4.1).
CPU only. No model is refit: every count comes from the saved temporal-test forecasts, and every forecast is first
re-scored and checked against the reported PR-AUC (same inputs and checks as alert_counts.py).

Budgets. k = round(q x number of test cell-months) for q in 1, 2, 5, 10 percent, per region x encoder seed. Each
model flags its k highest-scored cell-months over the whole test period (a retrospective ranking diagnostic, as in
alert_counts.py). Because every model issues k alerts, hits and false alarms always sum to k.
Models.
  emb_base, base   top-k by forecast score; ties broken by a stable sort on the saved row order (deterministic)
  persistence      violence in month t (binary, the month-t construction of persistence_month_t.py). A binary
                   score cannot rank within its two groups, so persistence hits are the EXPECTED value under random
                   tie-breaking: k <= A (A = persistence alerts): k x TP_A / A; k > A: TP_A + (k - A) x (P - TP_A) / (N - A)
                   with P violent cell-months and N test cell-months. Not rounded.
Outputs: outputs/alert_budget_sweep.csv (one row per region x seed x budget) and a printed summary of seed means
summed over regions.
"""
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
import os

R = os.environ.get("SWEEP_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
O = f"{R}/outputs"
REGIONS = ["africa", "americas", "asia_pacific", "europe", "mena", "south_asia"]
SEEDS = [123, 456, 789]
BUDGETS = [0.01, 0.02, 0.05, 0.10]
TOL_EVAL = 1e-6
OUT = f"{O}/alert_budget_sweep.csv"
assert not os.path.exists(OUT), f"refusing to overwrite {OUT}"


def topk_hits(p, y, k):
    return int(y[np.argsort(-p, kind="stable")[:k]].sum())


# ---- month-t violence (same construction and checks as alert_counts.py) ----
pan = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet",
                      columns=["grid_id", "year_month", "violence_next_month", "violence_lag1"])
pan = pan.sort_values(["grid_id", "year_month"]).reset_index(drop=True)
assert not pan.duplicated(["grid_id", "year_month"]).any()
per = pd.PeriodIndex(pan["year_month"].astype(str), freq="M"); mi = pd.Series(per.year * 12 + per.month, index=pan.index)
consec = pan["grid_id"].eq(pan["grid_id"].shift(1)) & (mi - mi.shift(1)).eq(1)
pan["violence_t"] = np.where(consec, pan.groupby("grid_id")["violence_next_month"].shift(1), np.nan)
v2 = pan.groupby("grid_id")["violence_lag1"].shift(-1); both = pan["violence_t"].notna() & consec.shift(-1, fill_value=False) & v2.notna()
assert (pan.loc[both, "violence_t"] == v2[both]).all(), "month-t violence derivations disagree"
vt = pd.Series(pan["violence_t"].to_numpy(), index=pd.MultiIndex.from_arrays(       # Codex 11, S01: unknown stays NaN
    [pan["grid_id"].astype(np.int64).to_numpy(), pan["year_month"].astype(str).to_numpy()]))
del pan, v2
print(f"[panel] month-t violence ready for {len(vt):,} grid-months", flush=True)

rows = []
for rg in REGIONS:
    for sd in SEEDS:
        tag = f"L_contrastive_e_loro_{rg}_s{sd}"
        z = np.load(f"{O}/preds_region_{tag}.npz", allow_pickle=True)
        y = z["y"].astype(int); pb = z["base"]; peb = z["emb_base"]
        assert set(np.unique(y)) <= {0, 1} and np.isfinite(pb).all() and np.isfinite(peb).all(), tag
        ev = pd.read_csv(f"{O}/eval_region_{tag}.csv")
        ev = ev[(ev.target == "incidence") & (ev["mode"] == "B_temporal")].set_index("model")["score"]
        for name, p in [("base", pb), ("emb_base", peb)]:
            s = average_precision_score(y, p)
            assert abs(s - ev[name]) < TOL_EVAL, f"{tag} {name}: preds give {s:.8f}, eval file {ev[name]:.8f}"
        key = pd.MultiIndex.from_arrays([z["grid_id"].astype(np.int64), np.asarray(z["year_month"]).astype(str)])
        assert key.is_unique, f"{tag}: duplicate test keys"
        pt = vt.reindex(key).to_numpy()
        assert not np.isnan(pt).any(), f"{tag}: month-t violence missing for some test rows"
        assert set(np.unique(pt)) <= {0.0, 1.0}, f"{tag}: month-t violence is not binary"
        N, P = len(y), int(y.sum()); A = int(pt.sum()); TPA = int((pt * y).sum())
        for q in BUDGETS:
            k = int(round(q * N)); assert 0 < k < N
            exp_p = k * TPA / A if k <= A else TPA + (k - A) * (P - TPA) / (N - A)
            hb, he = topk_hits(pb, y, k), topk_hits(peb, y, k)
            rows.append(dict(region=rg, seed=sd, budget=q, n_test=N, events=P, alerts=k, persistence_alerts=A,
                             hits_persistence_expected=exp_p, hits_base=hb, hits_embbase=he,
                             extra_hits_vs_persistence=he - exp_p, extra_hits_vs_base=he - hb))
        print(f"[{tag}] N={N:,} P={P:,} persistence alerts={A:,} ({A / N:.3f})", flush=True)

S = pd.DataFrame(rows)
assert len(S) == len(REGIONS) * len(SEEDS) * len(BUDGETS)
S.to_csv(OUT + ".part", index=False); os.replace(OUT + ".part", OUT)
cols = ["hits_persistence_expected", "hits_base", "hits_embbase", "extra_hits_vs_persistence", "extra_hits_vs_base"]
reg = S.groupby(["budget", "region"])[cols].mean()            # seed means
tot = reg.groupby("budget").sum()
npos = S.groupby(["budget", "region"])["extra_hits_vs_base"].apply(lambda x: int((x > 0).sum())).groupby("budget").sum()
pd.set_option("display.width", 160)
print("\n[by region] seed means\n", reg.round(1).to_string())
print("\n[all regions] sums of seed means\n", tot.round(1).to_string())
print("\n[region x seed cells with extra_hits_vs_base > 0, of 18]\n", npos.to_string())
print(f"[done] {OUT}", flush=True)
