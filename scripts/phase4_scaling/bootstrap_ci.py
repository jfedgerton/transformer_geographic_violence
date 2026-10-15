"""bootstrap_ci.py -- paired cell-cluster bootstrap 95% intervals for the Results comparisons (temporal test). CPU only.

Resampling unit: the PRIO-GRID cell. Each replicate draws the held-out region's test cells with replacement and keeps
every test month of a drawn cell together (so within-cell temporal dependence is preserved). The SAME cell draw is
applied to every model, encoder seed and share-grid draw in that region and replicate, so all differences are paired.
Regions are resampled independently. Spatial dependence between neighbouring cells is not modelled (limitation).

Resampling unit (env BOOT_UNIT): "cell" = one PRIO-GRID cell; "block5" = 5x5-degree blocks of cells (about 550 km),
which keeps neighbouring cells together and so allows for spatial dependence. Outputs carry the unit in their name.

Comparisons (PR-AUC differences, means over encoder seeds 123/456/789):
  persistence  base+embedding - persistence (violence in month t)            preds_region_{TAG}.npz
  nested       base+embedding - base                                          preds_region_{TAG}.npz
  random       base+pretrained embedding - base+random-weight embedding       preds_region_{TAG}.npz, _rand.npz
  share_{s}    base+embedding - base at each training share s (0.25% .. 10%),  share_grid/preds_nested_{rg}_s{seed}.npz
               also averaged over the five share draws
"ALL" = equal-weight mean of the six region estimates (as in the tables).
Point estimates use the full sample and are checked against the reported evaluation files before any resampling.
Outputs: outputs/bootstrap_ci_{unit}.csv (estimate, 2.5/97.5 percentiles, share of replicates <= 0) and
         outputs/bootstrap_ci_{unit}_draws.npz (every replicate statistic).
Env: BOOT_B (default 1000), BOOT_UNIT (cell | block5).
"""
import os
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

R = os.environ.get("BOOT_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
O = f"{R}/outputs"
REGIONS = ["africa", "americas", "asia_pacific", "europe", "mena", "south_asia"]
SEEDS = [123, 456, 789]
DRAWS = [0, 1, 2, 3, 4]
B = int(os.environ.get("BOOT_B", "1000"))
BOOT_SEED = 123
SHARES = [0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.10]
UNIT = os.environ.get("BOOT_UNIT", "cell"); assert UNIT in ("cell", "block5"), UNIT
COMPS = ["persistence", "nested", "random"] + [f"share_{s:g}" for s in SHARES]
OUT_CSV, OUT_NPZ = f"{O}/bootstrap_ci_{UNIT}.csv", f"{O}/bootstrap_ci_{UNIT}_draws.npz"
assert not os.path.exists(OUT_CSV), f"refusing to overwrite {OUT_CSV}"


def prep(p, y):
    """Sort once per forecast vector; keep the order, sorted labels and the last index of each tied-score group."""
    p = np.asarray(p, np.float64)
    order = np.argsort(-p, kind="stable")
    s = p[order]
    ends = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    return order.astype(np.int64), y[order].astype(np.float64), ends


def ap_w(pr, w):
    """Weighted average precision with sklearn's step definition (ties grouped by threshold)."""
    order, ys, ends = pr
    ws = w[order]
    tp = np.cumsum(ws * ys)[ends]; fp = np.cumsum(ws * (1.0 - ys))[ends]
    den = tp + fp
    prec = np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)
    if tp[-1] <= 0:                                   # Codex 7, B01: a resample with no positives scores 0, never NaN
        return 0.0
    rec = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


# ---- month-t violence (same construction and checks as alert_counts.py / persistence_month_t.py) ----
pan = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet",
                      columns=["grid_id", "year_month", "violence_next_month", "violence_lag1"])
pan = pan.sort_values(["grid_id", "year_month"]).reset_index(drop=True)
assert not pan.duplicated(["grid_id", "year_month"]).any()
per = pd.PeriodIndex(pan["year_month"].astype(str), freq="M"); mi = pd.Series(per.year * 12 + per.month, index=pan.index)
consec = pan["grid_id"].eq(pan["grid_id"].shift(1)) & (mi - mi.shift(1)).eq(1)
pan["violence_t"] = np.where(consec, pan.groupby("grid_id")["violence_next_month"].shift(1), np.nan)
v2 = pan.groupby("grid_id")["violence_lag1"].shift(-1); both = pan["violence_t"].notna() & consec.shift(-1, fill_value=False) & v2.notna()
assert (pan.loc[both, "violence_t"] == v2[both]).all(), "month-t violence derivations disagree"
vt = pd.Series(pan["violence_t"].to_numpy(), index=pd.MultiIndex.from_arrays(
    [pan["grid_id"].astype(np.int64).to_numpy(), pan["year_month"].astype(str).str[:7].to_numpy()]))
del pan, v2
print(f"[panel] month-t violence ready for {len(vt):,} grid-months", flush=True)

# ---- load forecasts, align rows, check point estimates against the reported files ----
data = {}
for rg in REGIONS:
    ref = None; arrays = {}
    for sd in SEEDS:
        tag = f"L_contrastive_e_loro_{rg}_s{sd}"
        z = np.load(f"{O}/preds_region_{tag}.npz", allow_pickle=True)
        zr = np.load(f"{O}/preds_region_{tag}_rand.npz", allow_pickle=True)
        gid = z["grid_id"].astype(np.int64); ym = np.asarray(z["year_month"]).astype(str)
        y = z["y"].astype(int)
        assert not pd.MultiIndex.from_arrays([gid, ym]).duplicated().any(), f"{tag}: duplicate keys"   # Codex 7, B04
        assert set(np.unique(z["y"]).tolist()) <= {0, 1}, f"{tag}: labels not binary"                     # Codex 7, B03
        if ref is None: ref = (gid, ym, y)
        assert np.array_equal(gid, ref[0]) and np.array_equal(ym, ref[1]) and np.array_equal(y, ref[2]), f"{tag}: rows differ across seeds"
        assert np.array_equal(zr["grid_id"].astype(np.int64), gid) and np.array_equal(np.asarray(zr["year_month"]).astype(str), ym), f"{tag}: random-encoder rows differ"
        assert np.array_equal(zr["y"].astype(int), y)
        ev = pd.read_csv(f"{O}/eval_region_{tag}.csv")
        ev = ev[(ev.target == "incidence") & (ev["mode"] == "B_temporal")].set_index("model")["score"]
        for name in ("base", "emb_base"):
            s = average_precision_score(y, z[name]); assert abs(s - ev[name]) < 1e-6, f"{tag} {name}: {s} vs {ev[name]}"
        arrays[("emb_base", sd)] = z["emb_base"]; arrays[("base", sd)] = z["base"]; arrays[("rand_emb_base", sd)] = zr["emb_base"]
        g = np.load(f"{O}/share_grid/preds_nested_{rg}_s{sd}.npz")
        key = pd.Series(np.arange(len(g["__y"])), index=pd.MultiIndex.from_arrays([g["__grid_id"].astype(np.int64), g["__year_month"].astype(str)]))
        assert key.index.is_unique, f"{rg} s{sd}: duplicate (grid_id, year_month) keys in the share-grid forecasts"   # Codex 7, B04
        pos = key.reindex(pd.MultiIndex.from_arrays([gid, ym])).to_numpy()
        assert not np.isnan(pos).any() and len(g["__y"]) == len(y), f"{rg} s{sd}: share-grid test rows differ from the main evaluation"
        pos = pos.astype(int); assert np.array_equal(g["__y"].astype(int)[pos], y)
        met = pd.read_csv(f"{O}/share_grid/metrics_nested_{rg}_s{sd}.csv")
        for sh in SHARES:
            for k in DRAWS:
                for m in ("base", "emb_base"):
                    kk = f"{m}|{sh:g}|{k}"
                    if kk not in g.files: continue
                    pr_rep = met[(met.model == m) & (np.isclose(met.share, sh)) & (met.draw == k)]["pr"]
                    assert len(pr_rep) == 1 and abs(average_precision_score(y, g[kk][pos]) - pr_rep.iloc[0]) < 1e-5, f"{rg} s{sd} {kk}"
                    arrays[(f"share_{m}_{sh:g}_{k}", sd)] = g[kk][pos]
    pt = vt.reindex(pd.MultiIndex.from_arrays([ref[0], ref[1]])).to_numpy()
    assert not np.isnan(pt).any(), f"{rg}: month-t violence missing for some test rows"
    arrays[("persistence_t", 0)] = pt
    if UNIT == "cell":
        unit_id = ref[0]
    else:                                            # PRIO-GRID gid -> 0.5-degree row/col -> 5-degree block
        row = (ref[0] - 1) // 720; col = (ref[0] - 1) % 720
        assert (row >= 0).all() and (row < 360).all()
        unit_id = (row // 10) * 1000 + (col // 10)
    for k, v in arrays.items():                       # Codex 7, B03: every forecast vector finite and row-aligned
        assert len(v) == len(ref[2]) and np.isfinite(np.asarray(v, np.float64)).all(), f"{rg} {k}: non-finite or misaligned forecasts"
    need = [(f"share_{m}_{shr:g}_{k}", s) for m in ("base", "emb_base") for shr in SHARES for k in DRAWS for s in SEEDS]
    miss = [k for k in need if k not in arrays]       # Codex 7, B02: all seed x draw pairs present for every share
    assert not miss, f"{rg}: {len(miss)} share forecasts missing, e.g. {miss[:3]}"
    cells, cidx = np.unique(unit_id, return_inverse=True)
    y = ref[2]
    data[rg] = dict(y=y, cidx=cidx, ncell=len(cells), pr={k: prep(v, y) for k, v in arrays.items()})
    print(f"[{rg}] test rows={len(y):,} resampling units ({UNIT})={len(cells):,} arrays={len(arrays)}", flush=True)


def stats(rg, w):
    d = data[rg]; ap = {k: ap_w(v, w) for k, v in d["pr"].items()}
    out = {}
    out["persistence"] = np.mean([ap[("emb_base", s)] for s in SEEDS]) - ap[("persistence_t", 0)]
    out["nested"] = np.mean([ap[("emb_base", s)] - ap[("base", s)] for s in SEEDS])
    out["random"] = np.mean([ap[("emb_base", s)] - ap[("rand_emb_base", s)] for s in SEEDS])
    for shr in SHARES:
        sh = [ap[(f"share_emb_base_{shr:g}_{k}", s)] - ap[(f"share_base_{shr:g}_{k}", s)] for s in SEEDS for k in DRAWS]
        out[f"share_{shr:g}"] = float(np.mean(sh))
    return out


# full-sample estimates (weights all one) and a check of the weighted AP against sklearn
est = {}
for rg in REGIONS:
    d = data[rg]; w1 = np.ones(len(d["y"]))
    k0 = ("emb_base", 123); o, ys, e = d["pr"][k0]
    z = np.load(f"{O}/preds_region_L_contrastive_e_loro_{rg}_s123.npz", allow_pickle=True)
    assert abs(ap_w(d["pr"][k0], w1) - average_precision_score(d["y"], z["emb_base"])) < 1e-9, "fast AP disagrees with sklearn"
    wt = np.random.default_rng(0).integers(0, 3, len(d["y"])).astype(float)
    assert abs(ap_w(d["pr"][k0], wt) - average_precision_score(d["y"], z["emb_base"], sample_weight=wt)) < 1e-9, "weighted AP disagrees"
    est[rg] = stats(rg, w1)
    print(f"[est] {rg} " + " ".join(f"{c}={est[rg][c]:+.4f}" for c in COMPS), flush=True)
    assert all(np.isfinite(est[rg][c]) for c in COMPS), f"{rg}: a comparison has no forecasts"

# ---- bootstrap ----
draws = {rg: {c: np.empty(B) for c in COMPS} for rg in REGIONS}
for i, rg in enumerate(REGIONS):
    d = data[rg]; rng = np.random.default_rng([BOOT_SEED, i])
    for b in range(B):
        wc = np.bincount(rng.integers(0, d["ncell"], d["ncell"]), minlength=d["ncell"]).astype(np.float64)
        st = stats(rg, wc[d["cidx"]])
        for c in COMPS: draws[rg][c][b] = st[c]
        assert all(np.isfinite(st[c]) for c in COMPS), f"{rg} replicate {b}: non-finite statistic"
        if (b + 1) % 100 == 0: print(f"  [{rg}] {b + 1}/{B}", flush=True)

rows = []
for c in COMPS:
    for rg in REGIONS + ["ALL"]:
        if rg == "ALL":
            e = float(np.mean([est[r][c] for r in REGIONS])); x = np.mean([draws[r][c] for r in REGIONS], axis=0)
        else:
            e = float(est[rg][c]); x = draws[rg][c]
        rows.append(dict(comparison=c, region=rg, estimate=e, lo95=float(np.percentile(x, 2.5)),
                         hi95=float(np.percentile(x, 97.5)), share_le0=float(np.mean(x <= 0)), B=B))
res = pd.DataFrame(rows)
np.savez_compressed(OUT_NPZ, **{f"{rg}|{c}": draws[rg][c] for rg in REGIONS for c in COMPS})
res.to_csv(OUT_CSV, index=False)
pd.set_option("display.width", 160)
print("\n" + res.round(4).to_string(index=False), flush=True)
print(f"[done] {OUT_CSV}", flush=True)
