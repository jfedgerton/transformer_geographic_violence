"""loss_vs_downstream.py -- APPENDIX: how much is a difference in encoder validation loss worth downstream?

Question (Jared, 2026-10-20): if M and L differ by ~0.01 in contrastive validation loss on a fold,
what does that translate to in out-of-sample forecasting?

Design (descriptive, CPU only, no new model fits):
  unit      = one full-source LORO encoder (sizes S, M, L x 6 regions x 3 seeds = 54)
  predictor = best 2021 validation loss from that encoder's training log (the checkpoint that is kept)
  outcome   = 2022+ test PR-AUC in the held-out region, from the region evaluator
              (outputs/eval_region_{TAG}.csv, mode B_temporal), for
              incidence x {emb, emb_base}   (emb = embedding only; emb_base = paper's nested arm)
  model     = OLS of outcome on loss with region fixed effects (each fold has its own 2021
              validation set, so raw loss levels are not comparable across folds), HC1 SEs.
  reported  = slope, SE, and the implied PR-AUC change for a 0.01 drop in loss;
              plus the within-region Spearman correlation as a rank check.

Inputs : logs/fullsrc_*.out (training logs), outputs/eval_region_{TAG}.csv
Output : outputs/appendix/loss_vs_downstream_encoders.csv (one row per encoder x arm)
         outputs/appendix/loss_vs_downstream_fit.csv      (one row per arm)
Never overwrites; fails loudly if any encoder log or eval file is missing or duplicated.
"""
import os, re, glob
import numpy as np, pandas as pd

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
OD = f"{R}/outputs/appendix"
OUT_ENC = f"{OD}/loss_vs_downstream_encoders.csv"
OUT_FIT = f"{OD}/loss_vs_downstream_fit.csv"
os.makedirs(OD, exist_ok=True)
assert not os.path.exists(OUT_ENC), f"refusing to overwrite {OUT_ENC}"
assert not os.path.exists(OUT_FIT), f"refusing to overwrite {OUT_FIT}"

SIZES = ["S", "M", "L"]
REGIONS = ["africa", "mena", "europe", "americas", "south_asia", "asia_pacific"]
SEEDS = [123, 456, 789]
ARMS = ["emb", "emb_base"]
TARGET = "incidence"
MODE = "B_temporal"

# ---- 1. best validation loss per encoder, from training logs ----
best = {}
for f in sorted(glob.glob(f"{R}/logs/fullsrc_*.out")):
    txt = open(f).read()
    m = re.search(r"TAG=(\S+)", txt)
    if not m or "_fullsrc_loro_" not in m.group(1):
        continue  # all-region encoders and dry runs are not part of this check
    tag = m.group(1)
    if f"FULLSRC_DONE_{tag}" not in txt:
        continue  # unfinished or failed run; the completeness assert below catches gaps
    vals = [float(v) for v in re.findall(r"^\[epoch \d+\] train=\S+ val=([0-9.]+)", txt, flags=re.M)]
    assert vals, f"no epoch lines in finished log {f}"
    assert tag not in best, f"two finished logs for {tag}: {best[tag]['log']} and {f}"
    best[tag] = dict(log=os.path.basename(f), best_val_loss=min(vals), n_epochs=len(vals),
                     best_epoch=int(np.argmin(vals)) + 1)

expected = [f"{c}_contrastive_e_fullsrc_loro_{r}_s{s}" for c in SIZES for r in REGIONS for s in SEEDS]
missing = [t for t in expected if t not in best]
assert not missing, f"{len(missing)} encoders have no finished training log, e.g. {missing[:3]}"
print(f"[loss] {len(expected)} finished encoders", flush=True)

# ---- 2. downstream test PR-AUC per encoder ----
rows = []
for c in SIZES:
    for r in REGIONS:
        for s in SEEDS:
            tag = f"{c}_contrastive_e_fullsrc_loro_{r}_s{s}"
            ef = f"{R}/outputs/eval_region_{tag}.csv"
            assert os.path.exists(ef), f"missing eval file {ef}"
            ev = pd.read_csv(ef)
            for arm in ARMS:
                sel = ev[(ev["mode"] == MODE) & (ev["target"] == TARGET) & (ev["model"] == arm)]
                assert len(sel) == 1, f"{tag} {arm}: expected 1 row, got {len(sel)}"
                score = float(sel["score"].iloc[0])
                assert np.isfinite(score), f"{tag} {arm}: non-finite score"
                rows.append(dict(tag=tag, size=c, region=r, seed=s, arm=arm, pr_auc=score, **best[tag]))
enc = pd.DataFrame(rows)
assert len(enc) == len(expected) * len(ARMS)

# ---- 3. OLS with region fixed effects, HC1 standard errors (hard-coded, numpy only) ----
fits = []
for arm in ARMS:
    d = enc[enc["arm"] == arm].reset_index(drop=True)
    y = d["pr_auc"].to_numpy()
    X = np.column_stack([d["best_val_loss"].to_numpy()] +
                        [(d["region"] == r).astype(float).to_numpy() for r in REGIONS])  # 6 region intercepts, no global constant
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    V = XtX_inv @ (X.T * e**2) @ X @ XtX_inv * n / (n - k)  # HC1
    slope, se = b[0], np.sqrt(V[0, 0])
    # within-region Spearman: rank both variables inside each region, then correlate
    rk = d.groupby("region")[["best_val_loss", "pr_auc"]].rank()
    rho = np.corrcoef(rk["best_val_loss"], rk["pr_auc"])[0, 1]
    fits.append(dict(arm=arm, target=TARGET, mode=MODE, n_encoders=n,
                     slope_prauc_per_unit_loss=slope, se_hc1=se,
                     prauc_change_per_0p01_lower_loss=-0.01 * slope,
                     se_per_0p01=0.01 * se,
                     within_region_spearman=rho,
                     loss_sd_within_region=d.groupby("region")["best_val_loss"].std().mean(),
                     prauc_sd_within_region=d.groupby("region")["pr_auc"].std().mean()))
    print(f"[fit] {arm}: 0.01 lower loss -> {(-0.01*slope):+.4f} PR-AUC (SE {0.01*se:.4f}); "
          f"within-region Spearman {rho:+.2f}", flush=True)
fit = pd.DataFrame(fits)

enc.to_csv(OUT_ENC + ".part", index=False); os.replace(OUT_ENC + ".part", OUT_ENC)
fit.to_csv(OUT_FIT + ".part", index=False); os.replace(OUT_FIT + ".part", OUT_FIT)
print(f"[done] {OUT_ENC}\n[done] {OUT_FIT}", flush=True)
