"""bootstrap_ft.py -- paired cluster bootstrap 95% intervals for fine-tuning vs training from scratch (step 7b).
CPU only; uses saved test forecasts, no refitting. Same design as bootstrap_ci.py (reviewed, Codex 7).

Comparison at each training share s (0.25% .. 10%):
  ft_{s} = mean over encoder seeds 123/456/789 and the five share draws of
           PR-AUC(ft_pretrained) - PR-AUC(ft_scratch)
  inputs  outputs/share_grid/preds_finetune_{region}_s{seed}_f{share}{TSUF}.npz  (keys "{model}|{share}|{draw}")
          outputs/share_grid/metrics_finetune_{region}_s{seed}_f{share}{TSUF}.csv (reported PR, checked first)
Resampling: each replicate draws the held-out region's test units with replacement (unit = PRIO-GRID cell, or a
5x5-degree block with BOOT_UNIT=block5), keeps every test month of a drawn unit together, and applies the SAME draw to
both arms, every seed, draw and share in that region and replicate, so all differences are paired. Regions are
resampled independently; "ALL" = equal-weight mean of the six region estimates. Intervals condition on the fitted
networks (no retraining).
Every (region, seed, share, draw) pair must be present for both arms; a missing run stops the script. The one
exception (Codex 9, point 4): a region x share that the job's availability file (avail_*.csv) marks as not estimable for
ALL draws and ALL seeds (too few positives; deterministic, so all-or-none) is reported as not estimable, and "ALL" for
that share is not computed. Mixed availability, or a missing availability file next to missing forecasts, stops the
script. Runs written before availability files existed (the incidence grid) must be complete.
Env: BOOT_UNIT (cell | block5), BOOT_B (default 1000), BOOT_TARGET (incidence | civilian; civilian reads the _civ files)
Outputs: outputs/bootstrap_ft_{target}_{unit}.csv and _draws.npz (never overwritten).
"""
import os
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

R = os.environ.get("BOOT_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
O = f"{R}/outputs"
REGIONS = ["africa", "americas", "asia_pacific", "europe", "mena", "south_asia"]
SEEDS = [123, 456, 789]
DRAWS = [0, 1, 2, 3, 4]
SHARES = [0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.10]
B = int(os.environ.get("BOOT_B", "1000"))
BOOT_SEED = 123
UNIT = os.environ.get("BOOT_UNIT", "cell"); assert UNIT in ("cell", "block5"), UNIT
TARGET = os.environ.get("BOOT_TARGET", "incidence"); assert TARGET in ("incidence", "civilian"), TARGET
TSUF = "" if TARGET == "incidence" else "_civ"
COMPS = [f"ft_{s:g}" for s in SHARES]
OUT_CSV, OUT_NPZ = f"{O}/bootstrap_ft_{TARGET}_{UNIT}.csv", f"{O}/bootstrap_ft_{TARGET}_{UNIT}_draws.npz"
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
    if tp[-1] <= 0:                                   # a resample with no positives scores 0, never NaN
        return 0.0
    den = tp + fp
    prec = np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)
    rec = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


# ---- load forecasts, check rows and reported scores ----
data = {}; NE = set()                                  # NE = (region, share) not estimable
for rg in REGIONS:
    ref = None; arrays = {}
    for sh in SHARES:                                  # availability first: all-or-none across seeds and draws
        st = []
        for sd in SEEDS:
            fa = f"{O}/share_grid/avail_finetune_{rg}_s{sd}_f{sh:g}{TSUF}.csv"
            if os.path.exists(fa):
                a = pd.read_csv(fa); assert sorted(a.draw.tolist()) == DRAWS, f"{fa}: draws {a.draw.tolist()}"
                st += a.status.tolist()
            else:
                st += ["ok"] * len(DRAWS)              # no availability file: the run must be complete
        if all(x == "ok" for x in st): continue
        assert not any(x == "ok" for x in st), f"{rg} {sh:g}: mixed availability {sorted(set(st))}"
        NE.add((rg, sh)); print(f"[not estimable] {rg} share={sh:g}: {sorted(set(st))}", flush=True)
    for sd in SEEDS:
        for sh in SHARES:
            if (rg, sh) in NE: continue
            tag = f"finetune_{rg}_s{sd}_f{sh:g}{TSUF}"
            fp, fm = f"{O}/share_grid/preds_{tag}.npz", f"{O}/share_grid/metrics_{tag}.csv"
            assert os.path.exists(fp) and os.path.exists(fm), f"missing run {tag}"
            z = np.load(fp)
            gid = z["__grid_id"].astype(np.int64); ym = z["__year_month"].astype(str); y = z["__y"].astype(int)
            assert set(np.unique(y).tolist()) <= {0, 1}, f"{tag}: labels not binary"
            assert not pd.MultiIndex.from_arrays([gid, ym]).duplicated().any(), f"{tag}: duplicate keys"
            if ref is None: ref = (gid, ym, y)
            assert np.array_equal(gid, ref[0]) and np.array_equal(ym, ref[1]) and np.array_equal(y, ref[2]), \
                f"{tag}: test rows differ from the region's other runs"
            met = pd.read_csv(fm)
            for k in DRAWS:
                for m in ("ft_pretrained", "ft_scratch"):
                    key = f"{m}|{sh:g}|{k}"
                    assert key in z.files, f"{tag}: missing {key}"
                    p = z[key].astype(np.float64)
                    assert len(p) == len(y) and np.isfinite(p).all(), f"{tag} {key}: non-finite or misaligned"
                    rep = met[(met.model == m) & np.isclose(met.share, sh) & (met.draw == k)]["pr"]
                    assert len(rep) == 1 and abs(average_precision_score(y, p) - rep.iloc[0]) < 1e-5, f"{tag} {key}"
                    arrays[(m, sh, k, sd)] = p
    if UNIT == "cell":
        unit_id = ref[0]
    else:                                            # PRIO-GRID gid -> 0.5-degree row/col -> 5-degree block
        row = (ref[0] - 1) // 720; col = (ref[0] - 1) % 720
        assert (row >= 0).all() and (row < 360).all()
        unit_id = (row // 10) * 1000 + (col // 10)
    assert ref is not None, f"{rg}: no estimable share"
    units, uidx = np.unique(unit_id, return_inverse=True)
    y = ref[2]
    assert 0 < y.sum() < len(y)
    data[rg] = dict(y=y, uidx=uidx, nunit=len(units), pr={k: prep(v, y) for k, v in arrays.items()})
    print(f"[{rg}] test rows={len(y):,} units ({UNIT})={len(units):,} forecasts={len(arrays)}", flush=True)


def stats(rg, w):
    pr = data[rg]["pr"]
    out = {}
    for sh in SHARES:
        if (rg, sh) in NE: out[f"ft_{sh:g}"] = np.nan; continue
        d = [ap_w(pr[("ft_pretrained", sh, k, sd)], w) - ap_w(pr[("ft_scratch", sh, k, sd)], w) for sd in SEEDS for k in DRAWS]
        out[f"ft_{sh:g}"] = float(np.mean(d))
    return out


est = {}
for rg in REGIONS:
    w1 = np.ones(len(data[rg]["y"]))
    sh0 = [sh for sh in SHARES if (rg, sh) not in NE][0]
    k0 = ("ft_pretrained", sh0, 0, SEEDS[0])
    z = np.load(f"{O}/share_grid/preds_finetune_{rg}_s{SEEDS[0]}_f{sh0:g}{TSUF}.npz")
    p0 = z[f"ft_pretrained|{sh0:g}|0"].astype(np.float64)
    assert abs(ap_w(data[rg]["pr"][k0], w1) - average_precision_score(data[rg]["y"], p0)) < 1e-9, "fast AP disagrees with sklearn"
    est[rg] = stats(rg, w1)
    print(f"[est] {rg} " + " ".join(f"{c}={est[rg][c]:+.4f}" for c in COMPS), flush=True)
    assert all(np.isfinite(est[rg][f"ft_{sh:g}"]) for sh in SHARES if (rg, sh) not in NE)

draws = {rg: {c: np.empty(B) for c in COMPS} for rg in REGIONS}
for i, rg in enumerate(REGIONS):
    d = data[rg]; rng = np.random.default_rng([BOOT_SEED, i])
    for b in range(B):
        wc = np.bincount(rng.integers(0, d["nunit"], d["nunit"]), minlength=d["nunit"]).astype(np.float64)
        st = stats(rg, wc[d["uidx"]])
        for c in COMPS: draws[rg][c][b] = st[c]
        assert all(np.isfinite(st[f"ft_{sh:g}"]) for sh in SHARES if (rg, sh) not in NE), f"{rg} replicate {b}: non-finite"
        if (b + 1) % 100 == 0: print(f"  [{rg}] {b + 1}/{B}", flush=True)

rows = []
for c in COMPS:
    for rg in REGIONS + ["ALL"]:
        sh = SHARES[COMPS.index(c)]
        miss = [r for r in REGIONS if (r, sh) in NE]
        if (rg == "ALL" and miss) or (rg, sh) in NE:      # ALL is never an average over fewer regions
            rows.append(dict(target=TARGET, comparison=c, region=rg, estimate=np.nan, lo95=np.nan, hi95=np.nan,
                             share_le0=np.nan, B=B, status="not_estimable" + ("" if rg != "ALL" else f" ({','.join(miss)})")))
            continue
        if rg == "ALL":
            e = float(np.mean([est[r][c] for r in REGIONS])); x = np.mean([draws[r][c] for r in REGIONS], axis=0)
        else:
            e = float(est[rg][c]); x = draws[rg][c]
        rows.append(dict(target=TARGET, comparison=c, region=rg, estimate=e, lo95=float(np.percentile(x, 2.5)),
                         hi95=float(np.percentile(x, 97.5)), share_le0=float(np.mean(x <= 0)), B=B, status="ok"))
res = pd.DataFrame(rows)
np.savez_compressed(OUT_NPZ, **{f"{rg}|{c}": draws[rg][c] for rg in REGIONS for c in COMPS})
res.to_csv(OUT_CSV + ".part", index=False); os.replace(OUT_CSV + ".part", OUT_CSV)
pd.set_option("display.width", 160)
print("\n" + res.round(4).to_string(index=False), flush=True)
print(f"[done] {OUT_CSV}", flush=True)
