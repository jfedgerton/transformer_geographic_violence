"""cell_month_decompose.py -- what the embedding's extra correct alerts (base + embedding vs base) are, at the cell-month level. CPU only.

Same forecasts, budget and tie rule as cell_alerts_agg.py (incidence): k = number of event cell-months in the region's test window,
each model flags its top-k cell-months (stable sort on saved row order).
Each event cell-month (y = 1) is classed as caught by both models, embedding only (gained), base only (lost), or neither.
For every cell-month we attach
  * fatalities in the target month (count_fatalities at month t+1; the forecast row is month t)
  * the cell's history: violent months among the 12 months t-11..t  ->  new (0), sporadic (1-3), recurrent (4-11), chronic (12)
Outputs (never overwritten), outputs/cell_month_decompose/:
  by_outcome_seed.csv   per region x seed x outcome: event cell-months and fatalities
  by_history_seed.csv   per region x seed x history class: events, hits and false alerts for each model
  gained_lost_rows.csv.gz  every gained or lost event cell-month (grid, month, region, seed, fatalities, history)
Checks: forecasts re-score to eval CSV PR-AUC; y equals violence_next_month in the panel; target-month fatalities > 0 implies
violence (warn count only); per region the seed-mean gain equals the d_tp sum in outputs/cell_alerts_incidence.csv.gz.
"""
import os
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
O = f"{R}/outputs"
OD = f"{O}/cell_month_decompose"
REGIONS = ["africa", "americas", "asia_pacific", "europe", "mena", "south_asia"]
SEEDS = [123, 456, 789]
TOL_EVAL = 1e-6
assert not os.path.exists(OD), f"refusing to overwrite {OD}"
os.makedirs(OD + "_part")


def topk(p, k):
    m = np.zeros(len(p), bool)
    if k > 0: m[np.argsort(-p, kind="stable")[:k]] = True
    return m


# ---- panel: target-month fatalities and 12-month history, by grid-month ----
pan = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet",
                      columns=["grid_id", "year_month", "has_violence", "count_fatalities", "violence_next_month"])
pan = pan.sort_values(["grid_id", "year_month"]).reset_index(drop=True)
assert not pan.duplicated(["grid_id", "year_month"]).any()
assert pan["count_fatalities"].notna().all() and (pan["count_fatalities"] >= 0).all()
per = pd.PeriodIndex(pan["year_month"].astype(str), freq="M"); mi = pd.Series(per.year * 12 + per.month, index=pan.index)
same_next = pan["grid_id"].eq(pan["grid_id"].shift(-1)) & (mi.shift(-1) - mi).eq(1)
pan["fat_next"] = np.where(same_next, pan.groupby("grid_id")["count_fatalities"].shift(-1), np.nan)
hv_next = np.where(same_next, pan.groupby("grid_id")["has_violence"].shift(-1), np.nan)
ok = ~np.isnan(hv_next)
assert (hv_next[ok].astype(int) == pan.loc[ok, "violence_next_month"].astype(int)).all(), "has_violence(t+1) != violence_next_month(t)"
# 12-month history t-11..t; require the 12 months to be consecutive rows of the same grid
cs = pan.groupby("grid_id")["has_violence"].cumsum().astype(float)
prev = pan.groupby("grid_id")["has_violence"].cumsum().astype(float).groupby(pan["grid_id"]).shift(12)
first12 = pan.groupby("grid_id").cumcount() == 11          # exactly 12 rows so far: window is rows 0..11
pan["hist12"] = np.where(first12, cs, cs - prev)
span_ok = pan["grid_id"].eq(pan["grid_id"].shift(11)) & (mi - mi.shift(11)).eq(11)
pan.loc[~span_ok, "hist12"] = np.nan
idx = pd.MultiIndex.from_arrays([pan["grid_id"].astype(np.int64).to_numpy(), pan["year_month"].astype(str).to_numpy()])
P = pd.DataFrame({"fat_next": pan["fat_next"].to_numpy(), "hist12": pan["hist12"].to_numpy(),
                  "vnext": pan["violence_next_month"].astype(float).to_numpy()}, index=idx)
del pan
print(f"[panel] {len(P):,} grid-months", flush=True)

HIST = ["new (0 of 12)", "sporadic (1-3)", "recurrent (4-11)", "chronic (12 of 12)"]
def hclass(h):
    return np.select([h == 0, h <= 3, h <= 11], HIST[:3], HIST[3])

by_out, by_hist, rows = [], [], []
for rg in REGIONS:
    for sd in SEEDS:
        tag = f"L_contrastive_e_loro_{rg}_s{sd}"
        z = np.load(f"{O}/preds_region_{tag}.npz", allow_pickle=True)
        g = z["grid_id"].astype(np.int64); ym = z["year_month"].astype(str); y = z["y"].astype(int)
        pb = z["base"].astype(float); pe = z["emb_base"].astype(float)
        ev = pd.read_csv(f"{O}/eval_region_{tag}.csv")
        ev = ev[(ev.target == "incidence") & (ev["mode"] == "B_temporal")].set_index("model")["score"]
        for mdl, p in [("base", pb), ("emb_base", pe)]:
            assert abs(average_precision_score(y, p) - ev[mdl]) < TOL_EVAL, f"{tag}: {mdl} PR-AUC mismatch"
        m = P.reindex(pd.MultiIndex.from_arrays([g, ym]))
        assert not m["fat_next"].isna().any() and not m["hist12"].isna().any(), f"{tag}: panel values missing"
        assert (m["vnext"].to_numpy() == y).all(), f"{tag}: y != violence_next_month"
        fat = m["fat_next"].to_numpy(); hc = hclass(m["hist12"].to_numpy())
        nz = int(((fat > 0) & (y == 0)).sum())
        if nz: print(f"[warn] {tag}: {nz} cell-months with fatalities but no violent event", flush=True)
        k = int(y.sum()); ab, ae = topk(pb, k), topk(pe, k)
        out = np.select([ab & ae, ae & ~ab, ab & ~ae], ["both", "gained", "lost"], "neither")
        e = y == 1
        for o in ["both", "gained", "lost", "neither"]:
            s = e & (out == o)
            by_out.append(dict(region=rg, seed=sd, outcome=o, events=int(s.sum()), fatalities=float(fat[s].sum()),
                               fat_ge25=int((fat[s] >= 25).sum()), fat_ge100=int((fat[s] >= 100).sum())))
        for h in HIST:
            s = hc == h
            by_hist.append(dict(region=rg, seed=sd, history=h, rows=int(s.sum()), events=int((s & e).sum()),
                                hits_base=int((s & e & ab).sum()), hits_emb=int((s & e & ae).sum()),
                                false_base=int((s & ~e & ab).sum()), false_emb=int((s & ~e & ae).sum()),
                                fat_events=float(fat[s & e].sum()), fat_hit_base=float(fat[s & e & ab].sum()),
                                fat_hit_emb=float(fat[s & e & ae].sum())))
        gl = e & ((out == "gained") | (out == "lost"))
        rows.append(pd.DataFrame({"grid_id": g[gl], "year_month": ym[gl], "region": rg, "seed": sd, "outcome": out[gl],
                                  "fat_next": fat[gl], "history": hc[gl]}))
        gn, ls = int((e & (out == "gained")).sum()), int((e & (out == "lost")).sum())
        print(f"[{tag}] k={k:,} gained {gn} lost {ls} net {gn - ls:+d}; fatalities gained {fat[e & (out == 'gained')].sum():,.0f} "
              f"lost {fat[e & (out == 'lost')].sum():,.0f}", flush=True)

bo = pd.DataFrame(by_out); bh = pd.DataFrame(by_hist)
assert len(bo) == 18 * 4 and len(bh) == 18 * 4
# tie back to the published cell file (seed-averaged d_tp by region)
net = bo.pivot_table(index=["region", "seed"], columns="outcome", values="events").eval("gained - lost").groupby("region").mean()
cell = pd.read_csv(f"{O}/cell_alerts_incidence.csv.gz").groupby("region").d_tp.sum()
gap = (net - cell.reindex(net.index)).abs().max()
assert gap < 0.01, f"net gain differs from cell_alerts_incidence by {gap}"
print(f"[check] seed-mean net gain by region matches cell_alerts_incidence.csv.gz (total {net.sum():+.1f})")
bo.to_csv(f"{OD}_part/by_outcome_seed.csv", index=False)
bh.to_csv(f"{OD}_part/by_history_seed.csv", index=False)
pd.concat(rows, ignore_index=True).to_csv(f"{OD}_part/gained_lost_rows.csv.gz", index=False, compression="gzip")
os.replace(f"{OD}_part", OD)

# ---- printed summary (seed means summed over regions) ----
s = bo.groupby(["outcome", "seed"])[["events", "fatalities", "fat_ge25", "fat_ge100"]].sum().groupby("outcome").mean()
s["fat_per_event"] = s.fatalities / s.events
print("\n== event cell-months by outcome (seed means, all regions) ==\n" + s.round(1).to_string())
h = bh.groupby(["history", "seed"])[[c for c in bh.columns if c not in ("region", "seed", "history")]].sum().groupby("history").mean()
print("\n== by 12-month history (seed means, all regions) ==\n" + h.reindex(HIST).round(1).to_string())
print(f"\n[done] wrote {OD}")
