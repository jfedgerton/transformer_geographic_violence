#!/usr/bin/env python
"""
#13b: aggregate the 60 transfer-eval CSVs and produce heatmaps + a winners table.

Reads  outputs/eval_full_*.csv  (long: tag,obj,feat,unit,mode,target,model,score,n)
Writes outputs/eval_aggregated.csv   (per loss x feat x mode x target x contrast: mean delta + bootstrap CI)
       outputs/eval_winners.csv      (best loss x feat per mode x target x contrast)
       outputs/figures/heatmap_<contrast>_<feat>.png  (10 figures)

Two contrasts (positive = embedding is BETTER, CRPS sign-flipped):
  embbase_vs_base : emb_base - base       (does the transformer add over the strong conventional baseline)
  emb_vs_pers     : emb     - persistence (can the embedding alone beat naive persistence in an unseen country)

Hardcoded, sequential. Seed 123.
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

np.random.seed(123)

# ---------------------------------------------------------------- load
FILES = sorted(glob.glob("outputs/eval_full_*.csv"))
assert len(FILES) == 60, f"expected 60 eval CSVs, found {len(FILES)}"
df = pd.concat([pd.read_csv(f) for f in FILES], ignore_index=True)
print(f"[load] {len(FILES)} files, {len(df)} rows", flush=True)

# seed from tag  M_{obj}_{feat}_s{seed}
df["seed"] = df["tag"].str.extract(r"s(\d+)$").astype(int)
# collapse the 5 spatial CV folds vs the temporal mode
df["mode2"] = np.where(df["mode"].str.startswith("A_pooled"), "A_spatial", "B_forecast")

# ---------------------------------------------------------------- fixed orders
LOSSES  = ["mlm", "autoregressive", "contrastive", "tversky"]
FEATS   = ["a", "b", "c", "e", "f"]
FEATLAB = {"a": "a: no-fat", "b": "b: +fatal", "c": "c: +civ",
           "e": "e: +PRIO", "f": "f: +fat+civ+PRIO"}
TARGETS = ["incidence", "onset", "noncontig_onset", "rare_event_f1", "fatalities", "civilian", "severity_crps"]
MODES   = ["A_spatial", "B_forecast"]
ISO3 = {"Sudan":"SDN","Somalia":"SOM","Uganda":"UGA","Kenya":"KEN","Ethiopia":"ETH",
        "South Africa":"ZAF","Iraq":"IRQ","Tunisia":"TUN","Iran":"IRN","Algeria":"DZA",
        "Morocco":"MAR","Pakistan":"PAK","Bangladesh":"BGD","Nepal":"NPL","Sri Lanka":"LKA",
        "Myanmar":"MMR","Philippines":"PHL","Thailand":"THA","South Korea":"KOR","Ukraine":"UKR",
        "France":"FRA","Germany":"DEU","Mexico":"MEX","Venezuela":"VEN","Colombia":"COL","Chile":"CHL"}
UNITS = list(ISO3.keys())  # fixed 26, grouped by region

# ---------------------------------------------------------------- mean over folds+seeds, pivot models
# per (obj,feat,unit,mode2,target,seed): average the A folds within a seed
gs = (df.groupby(["obj","feat","unit","mode2","target","seed","model"], as_index=False)["score"].mean())
# pivot models to columns
ps = gs.pivot_table(index=["obj","feat","unit","mode2","target","seed"],
                    columns="model", values="score").reset_index()
hib = ps["target"] != "severity_crps"   # higher-is-better for all but CRPS
ps["d_embbase_base"] = np.where(hib, ps["emb_base"]-ps["base"], ps["base"]-ps["emb_base"])
ps["d_emb_pers"]     = np.where(hib, ps["emb"]-ps["persistence"], ps["persistence"]-ps["emb"])

# per-cell mean over the 3 seeds, and seed sign-consistency flag
cell = (ps.groupby(["obj","feat","unit","mode2","target"])
          .agg(d_embbase_base=("d_embbase_base","mean"),
               d_emb_pers=("d_emb_pers","mean"),
               sc_embbase=("d_embbase_base", lambda s: (s>0).all() or (s<0).all()),
               sc_emb=("d_emb_pers", lambda s: (s>0).all() or (s<0).all()))
          .reset_index())

CONTRASTS = [("embbase_vs_base", "d_embbase_base", "sc_embbase", "emb_base − base"),
             ("emb_vs_pers",     "d_emb_pers",     "sc_emb",     "emb − persistence")]

# ---------------------------------------------------------------- bootstrap over units -> aggregated table + winners
def boot_ci(vals, B=2000):
    vals = np.asarray(vals, float); vals = vals[~np.isnan(vals)]
    if len(vals) == 0: return (np.nan, np.nan, np.nan)
    idx = np.random.randint(0, len(vals), size=(B, len(vals)))
    means = vals[idx].mean(axis=1)
    return vals.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

agg_rows = []
for cname, dcol, scol, _ in CONTRASTS:
    for lo in LOSSES:
        for ft in FEATS:
            for md in MODES:
                for tg in TARGETS:
                    sub = cell[(cell.obj==lo)&(cell.feat==ft)&(cell.mode2==md)&(cell.target==tg)]
                    m, lo_ci, hi_ci = boot_ci(sub[dcol].values)
                    sig = (lo_ci > 0) or (hi_ci < 0)
                    agg_rows.append(dict(contrast=cname, loss=lo, feat=ft, mode=md, target=tg,
                                         mean_delta=m, ci_lo=lo_ci, ci_hi=hi_ci, sig=sig,
                                         n_units=int(sub[dcol].notna().sum())))
agg = pd.DataFrame(agg_rows)
agg.to_csv("outputs/eval_aggregated.csv", index=False)
print("[write] outputs/eval_aggregated.csv", flush=True)

# winners: best loss x feat per (contrast, mode, target) by mean unit-averaged delta
win = (agg.sort_values("mean_delta", ascending=False)
          .groupby(["contrast","mode","target"], as_index=False).first()
          [["contrast","mode","target","loss","feat","mean_delta","ci_lo","ci_hi","sig"]])
win.to_csv("outputs/eval_winners.csv", index=False)
print("[write] outputs/eval_winners.csv", flush=True)
print("\n===== WINNERS (best loss x feat per mode x target) =====", flush=True)
with pd.option_context("display.width", 200, "display.max_rows", 100):
    print(win.to_string(index=False), flush=True)

# ---------------------------------------------------------------- heatmaps
os.makedirs("outputs/figures", exist_ok=True)
for cname, dcol, scol, clab in CONTRASTS:
    # symmetric color scale from robust quantile of this contrast
    scale = np.nanpercentile(np.abs(cell[dcol].values), 95)
    scale = max(scale, 1e-3)
    norm = TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)
    for ft in FEATS:
        fig, axes = plt.subplots(len(TARGETS), len(MODES),
                                 figsize=(0.34*len(UNITS)*len(MODES)+2, 1.5*len(TARGETS)+1.2),
                                 squeeze=False)
        for ti, tg in enumerate(TARGETS):
            for mi, md in enumerate(MODES):
                ax = axes[ti][mi]
                sub = cell[(cell.feat==ft)&(cell.target==tg)&(cell.mode2==md)]
                M = np.full((len(LOSSES), len(UNITS)), np.nan)
                S = np.zeros((len(LOSSES), len(UNITS)), bool)
                for li, lo in enumerate(LOSSES):
                    for ui, u in enumerate(UNITS):
                        r = sub[(sub.obj==lo)&(sub.unit==u)]
                        if len(r):
                            M[li, ui] = r[dcol].values[0]
                            S[li, ui] = bool(r[scol].values[0])
                ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
                # seed-consistent cells: small dot
                ys, xs = np.where(S & ~np.isnan(M))
                ax.scatter(xs, ys, s=4, c="k", marker=".")
                ax.set_yticks(range(len(LOSSES)))
                ax.set_yticks(range(len(LOSSES)))
                ax.set_yticklabels([l[:4] for l in LOSSES], fontsize=7)
                ax.set_xticks(range(len(UNITS)))
                if ti == len(TARGETS)-1:
                    ax.set_xticklabels([ISO3[u] for u in UNITS], fontsize=6, rotation=0)
                else:
                    ax.set_xticklabels([])
                if mi == 0:
                    ax.set_ylabel(tg, fontsize=8)
                if ti == 0:
                    ax.set_title({"A_spatial":"Mode A (spatial)","B_forecast":"Mode B (forecast)"}[md],
                                 fontsize=9)
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
        cb = fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.01)
        cb.set_label(f"{clab}  (>0 = embedding better; CRPS sign-flipped)", fontsize=8)
        fig.suptitle(f"Transfer skill  |  feature set {FEATLAB[ft]}  |  {clab}\n"
                     f"rows=loss, cols=held-out country (ISO3); dot=sign-consistent across 3 seeds",
                     fontsize=10)
        out = f"outputs/figures/heatmap_{cname}_{ft}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig] {out}", flush=True)

print("\n[done]", flush=True)
