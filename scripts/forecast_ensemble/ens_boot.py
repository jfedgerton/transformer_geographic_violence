"""ens_boot.py -- forecasting ensemble: regional PR-AUC and paired cluster-bootstrap intervals (CPU).
(Forecasting with a pretrained encoding; not the LORO transfer design.)

Input: outputs/ens/preds_ens_mt_{target}{RTAG}.npz written by ens_combine.py (default RTAG = 3-seed roster
_pilot_s123-456-789). Test window only (window == "test", 2022-01 on).
Models scored: ens_mean, ens_stack, base (XGBoost), persistence, and each single encoder present.
Comparisons: ens_mean - base, ens_mean - persistence, ens_stack - base.
Bootstrap: paired, stratified by region, resampling clusters with replacement within each region; the same draw
weights are applied to every model (paired). Two cluster units, as in the paper's bootstrap: "cell" (PRIO-GRID cell)
and "block5" (5 x 5 degree blocks, 10 x 10 cells; conservative). B = 1000, seed 123. ALL = pooled over regions
(one model scored on all rows; unlike the paper's equal-weight mean of LORO regions).
Intervals condition on the fitted models (no retraining or seed variation).
PR-AUC = average precision with ties grouped by score, identical to sklearn's average_precision_score
(asserted on the point estimates).
Outputs (never overwritten): outputs/ens/boot_mt_{RTAG}.csv (estimates + 95% intervals) and
outputs/ens/boot_mt_{RTAG}_draws.npz. Written as .part then renamed; CSV last.
"""
import os
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

R = os.environ.get("ENS_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
OD = f"{R}/outputs/ens"
RTAG = os.environ.get("ENS_RTAG", "_pilot_s123-456-789")
B = int(os.environ.get("ENS_BOOT_B", "1000"))
SEED = 123
TARGETS = ["incidence", "civilian", "onset", "fatal25"]
UNITS = ["cell", "block5"]
OUT_CSV, OUT_NPZ = f"{OD}/boot_mt{RTAG}.csv", f"{OD}/boot_mt{RTAG}_draws.npz"
assert not os.path.exists(OUT_CSV), f"refusing to overwrite {OUT_CSV}"
assert B >= 1


def ap_sorted(y_sorted, w_sorted, last_of_group):
    """Weighted average precision for rows already sorted by descending score.
    last_of_group: indices (in sorted order) of the last row of each tied-score group."""
    tp = np.cumsum(w_sorted * y_sorted)[last_of_group]
    fp = np.cumsum(w_sorted * (1 - y_sorted))[last_of_group]
    P = tp[-1]
    if P <= 0:
        return np.nan
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / P
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def prep(p, y):
    o = np.argsort(-p, kind="stable")
    ps = p[o]
    last = np.flatnonzero(np.r_[ps[1:] != ps[:-1], True])
    return o, last


rows, draws = [], {}
for t in TARGETS:
    f = f"{OD}/preds_ens_mt_{t}{RTAG}.npz"
    assert os.path.exists(f), f"missing {f}"
    z = np.load(f)
    te = z["window"].astype(str) == "test"
    assert te.sum() > 0
    y = z["y"][te].astype(np.int64)
    gid = z["grid_id"][te].astype(np.int64)
    region = z["region"][te].astype(str)
    assert set(np.unique(y)) <= {0, 1} and y.sum() > 0
    models = {k[2:]: z[k][te].astype(np.float64) for k in z.files if k.startswith("p_")}
    for need in ("ens_mean", "ens_stack", "base", "persistence"):
        assert need in models, f"{t}: {need} missing"
    for k, p in models.items():
        assert np.isfinite(p).all(), f"{t} {k}: non-finite"
    # PRIO-GRID gid -> row/col (gid = row*720 + col + 1, row 0 at -90 lat)
    assert (gid >= 1).all() and (gid <= 259200).all(), "grid_id outside PRIO-GRID range"
    rowi, coli = (gid - 1) // 720, (gid - 1) % 720
    cl_of = {"cell": gid, "block5": (rowi // 10) * 72 + (coli // 10)}
    regs = sorted(set(region))
    subsets = {"ALL": np.ones(len(y), bool)} | {rg: region == rg for rg in regs}
    # point estimates; check fast AP against sklearn
    pre = {}
    for k, p in models.items():
        for s_name, s in subsets.items():
            if y[s].sum() == 0:
                continue
            o, last = prep(p[s], y[s])
            pre[(k, s_name)] = (o, last, y[s][o].astype(float), np.flatnonzero(s)[o])
            est = ap_sorted(y[s][o].astype(float), np.ones(s.sum()), last)
            ref = average_precision_score(y[s], p[s])
            assert abs(est - ref) < 1e-9, f"{t} {k} {s_name}: fast AP {est} != sklearn {ref}"
    comps = [("ens_mean", "base"), ("ens_mean", "persistence"), ("ens_stack", "base")]
    for unit in UNITS:
        cl = cl_of[unit]
        rng = np.random.default_rng([SEED, TARGETS.index(t), UNITS.index(unit)])
        # stratified by region: clusters are nested in regions (a block5 spanning two regions is split by region)
        keys = pd.Series(list(zip(region, cl)))
        codes, uniq = pd.factorize(keys)
        reg_of_code = np.array([u[0] for u in uniq])
        codes_by_reg = {rg: np.flatnonzero(reg_of_code == rg) for rg in regs}
        D = {}
        for b in range(B):
            cnt = np.zeros(len(uniq), np.float64)
            for rg in regs:
                c = codes_by_reg[rg]
                cnt += np.bincount(rng.choice(c, size=len(c), replace=True), minlength=len(uniq))
            w = cnt[codes]
            for (k, s_name), (o, last, ys, idx) in pre.items():
                D.setdefault((k, s_name), np.empty(B))[b] = ap_sorted(ys, w[idx], last)
        for s_name in subsets:
            if (("base", s_name)) not in pre:
                continue
            s = subsets[s_name]
            n, ev = int(s.sum()), int(y[s].sum())
            for k in models:
                pt = average_precision_score(y[s], models[k][s])
                d = D[(k, s_name)]
                rows.append(dict(target=t, unit=unit, region=s_name, n=n, events=ev, quantity=k, estimate=pt,
                                 lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5)))
            for a, bb in comps:
                pt = average_precision_score(y[s], models[a][s]) - average_precision_score(y[s], models[bb][s])
                d = D[(a, s_name)] - D[(bb, s_name)]
                draws[f"{t}|{unit}|{s_name}|{a}-{bb}"] = d
                rows.append(dict(target=t, unit=unit, region=s_name, n=n, events=ev, quantity=f"{a} - {bb}",
                                 estimate=pt, lo=np.nanpercentile(d, 2.5), hi=np.nanpercentile(d, 97.5),
                                 share_draws_gt0=float(np.mean(d > 0))))
        print(f"[{t}] {unit} done", flush=True)

out = pd.DataFrame(rows)
# point estimates must not depend on the cluster unit
chk = out.pivot_table(index=["target", "region", "quantity"], columns="unit", values="estimate")
assert np.allclose(chk["cell"], chk["block5"]), "point estimates differ by unit"
np.savez_compressed(OUT_NPZ + ".part.npz", **draws)
os.replace(OUT_NPZ + ".part.npz", OUT_NPZ)
out.to_csv(OUT_CSV + ".part", index=False)
os.replace(OUT_CSV + ".part", OUT_CSV)
show = out[(out.region == "ALL") & out.quantity.str.contains(" - ")]
print(show[["target", "unit", "quantity", "estimate", "lo", "hi"]].round(4).to_string(index=False))
print(f"[done] {OUT_CSV}")
