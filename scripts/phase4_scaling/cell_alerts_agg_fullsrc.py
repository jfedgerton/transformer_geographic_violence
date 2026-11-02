# cell_alerts_agg_fullsrc.py -- written 2026-10-27 (Claude, approved plan step 3).
# Copies: scripts/phase4_scaling/cell_alerts_agg.py (md5 0f41335920a93112cb339d49a1bfde43); everything not listed below is byte-identical.
# Changes vs the original:
#   * env SIZE (S|M|L, required): forecasts preds_region_{SIZE}_contrastive_e_fullsrc_loro_{rg}_s{seed}.npz and eval_region_{SIZE}_contrastive_e_fullsrc_loro_{rg}_s{seed}.csv (original: L_contrastive_e_loro_...)
#   * every output name carries _{SIZE}_fullsrc{OUTSFX} (OUTSFX, e.g. _smoke, is new)
#   * incidence only: the target argument is optional and must be 'incidence' (civilian forecasts are not saved by the evaluator)
#   * a missing region x seed file is a hard error (original: skipped and listed)
#   * output outputs/cell_alerts_incidence_{SIZE}_fullsrc{OUTSFX}.csv.gz: refused if it exists (the original overwrote), written as .part and renamed
#   * env FS_ROOT overrides the project root (synthetic tests only)
"""cell_alerts_agg.py -- cell-level summary of where the pretrained embedding changes forecasts (2026-10-13).
For each held-out region x encoder seed (main spec, vocab remap), reads the saved Mode B (temporal) forecasts and:
  1. re-scores PR-AUC from the saved predictions and checks it against the eval CSV (fails loudly if they disagree)
  2. sets an alert budget k = number of cell-months with an event in that region's test window (same k for both
     models) and flags the top-k cell-months by predicted probability, separately for base and base + embedding
  3. per PRIO-GRID cell, counts true and false alerts for each model and averages predicted probabilities
Cell values are then averaged over the available seeds.
Usage: python cell_alerts_agg.py incidence|civilian
  incidence -> preds_region_{TAG}.npz,     scores checked against target "incidence" in eval_region_{TAG}.csv
  civilian  -> preds_region_{TAG}_civ.npz, scores checked against target "civilian"  in eval_region_{TAG}.csv
Output: outputs/cell_alerts_{target}.csv.gz (one row per cell) and printed region totals.
Ties at the alert cutoff are broken by row order (stable sort); documented, not randomized.
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

R = os.environ.get("FS_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
SIZE = os.environ.get("SIZE", ""); assert SIZE in ("S", "M", "L"), f"set SIZE=S|M|L, got {SIZE!r}"
OUTSFX = os.environ.get("OUTSFX", "")  # e.g. _smoke
T = f"_{SIZE}_fullsrc{OUTSFX}"  # tag carried by every output and every chained input
TARGET = sys.argv[1] if len(sys.argv) > 1 else "incidence"
assert TARGET == "incidence", "full-source copy: incidence only"
FO = f"{R}/outputs/cell_alerts_{TARGET}{T}.csv.gz"
assert not os.path.exists(FO), f"refusing to overwrite {FO}"
REGIONS = ["africa", "mena", "europe", "americas", "south_asia", "asia_pacific"]
SEEDS = [123, 456, 789]
TOL = 1e-6
SUF = "" if TARGET == "incidence" else "_civ"

def alerts(score, k):
    a = np.zeros(len(score), bool)
    if k > 0: a[np.argsort(-score, kind="stable")[:k]] = True
    return a

per_seed, totals, missing = [], [], []
for rg in REGIONS:
    for sd in SEEDS:
        tag = f"{SIZE}_contrastive_e_fullsrc_loro_{rg}_s{sd}"
        fp, fe = f"{R}/outputs/preds_region_{tag}{SUF}.npz", f"{R}/outputs/eval_region_{tag}.csv"
        assert os.path.exists(fp) and os.path.exists(fe), f"missing {fp} or {fe}"
        z = np.load(fp, allow_pickle=True)
        g, ym, y = z["grid_id"].astype(np.int64), z["year_month"].astype(str), z["y"].astype(int)
        pb, pe = z["base"].astype(float), z["emb_base"].astype(float)
        n = len(y)
        assert n > 0 and all(len(a) == n for a in (g, ym, pb, pe)), f"{tag}: array lengths differ"
        assert set(np.unique(y)) <= {0, 1} and y.sum() > 0, f"{tag}: bad labels"
        assert np.isfinite(pb).all() and np.isfinite(pe).all(), f"{tag}: non-finite predictions"
        assert not pd.DataFrame({"g": g, "m": ym}).duplicated().any(), f"{tag}: duplicate cell-months"
        ev = pd.read_csv(fe)
        ev = ev[(ev["target"] == TARGET) & (ev["mode"] == "B_temporal")].set_index("model")["score"]
        for mdl, p in [("base", pb), ("emb_base", pe)]:
            gap = abs(average_precision_score(y, p) - ev[mdl])
            assert gap <= TOL, f"{tag}: re-scored {mdl} PR-AUC differs from eval CSV by {gap:.3g}"
        k = int(y.sum())
        ab, ae = alerts(pb, k), alerts(pe, k)
        df = pd.DataFrame({"gid": g, "y": y, "tp_base": ab & (y == 1), "fp_base": ab & (y == 0),
                           "tp_emb": ae & (y == 1), "fp_emb": ae & (y == 0), "p_base": pb, "p_emb": pe})
        c = df.groupby("gid").agg(months=("y", "size"), events=("y", "sum"), tp_base=("tp_base", "sum"),
                                  fp_base=("fp_base", "sum"), tp_emb=("tp_emb", "sum"), fp_emb=("fp_emb", "sum"),
                                  p_base=("p_base", "mean"), p_emb=("p_emb", "mean")).reset_index()
        c["region"], c["seed"] = rg, sd
        per_seed.append(c)
        totals.append(dict(region=rg, seed=sd, k=k, tp_base=int(ab[y == 1].sum()), tp_emb=int(ae[y == 1].sum()),
                           cells=c.shape[0]))

print(f"[cell] {TARGET}: {len(per_seed)} of 18 region x seed files; missing: {missing}")
assert len(per_seed) == 18, f"{len(per_seed)} of 18 region x seed files"
t = pd.DataFrame(totals); t["extra_tp"] = t.tp_emb - t.tp_base
print(t.to_string(index=False))
allc = pd.concat(per_seed, ignore_index=True)
assert allc.groupby("gid").region.nunique().max() == 1, "a cell appears in more than one held-out region"
num = ["months", "events", "tp_base", "fp_base", "tp_emb", "fp_emb", "p_base", "p_emb"]
out = allc.groupby(["gid", "region"])[num].mean().reset_index()
out["n_seeds"] = allc.groupby(["gid", "region"]).seed.nunique().values
out["d_tp"] = out.tp_emb - out.tp_base
out["d_fp"] = out.fp_emb - out.fp_base
out["d_p"] = out.p_emb - out.p_base
row = (out.gid - 1) // 720; col = (out.gid - 1) % 720
out["lat"], out["lon"] = -90 + 0.5 * row + 0.25, -180 + 0.5 * col + 0.25
fo = FO
out.round(5).to_csv(fo + ".part.gz", index=False, compression="gzip"); os.replace(fo + ".part.gz", fo)
print(f"[cell] wrote {fo}: {len(out):,} cells; extra true alerts {out.d_tp.sum():+.1f}, "
      f"extra false alerts {out.d_fp.sum():+.1f} (seed-averaged)")
