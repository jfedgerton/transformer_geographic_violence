"""aggregate_region_eval.py — summarize the LORO zero-shot region transfer eval.

Reads outputs/eval_region_L_contrastive_e_loro_{region}_s123.csv for each region
(produced by transfer_eval_region.py). For each held-out region, reports whether the
encoder that NEVER saw that region predicts it — vs persistence and vs the strong
featurized baseline — on incidence PR-AUC (Mode B temporal forecast is the headline;
Mode A pooled-CV reported alongside).

Output: outputs/loro_region_summary.csv + a printed table.
Run: python scripts/phase4_scaling/aggregate_region_eval.py
"""
import os, glob
import numpy as np, pandas as pd

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
REGIONS = ["africa", "mena", "europe", "americas", "south_asia", "asia_pacific"]
TARGET = os.environ.get("AGG_TARGET", "incidence")  # headline target (PR-AUC); override via AGG_TARGET=civilian

rows = []
for rg in REGIONS:
    f = f"{R}/outputs/eval_region_L_contrastive_e_loro_{rg}_s123.csv"
    if not os.path.exists(f):
        print(f"[skip] missing {f}", flush=True)
        continue
    d = pd.read_csv(f)
    d = d[d["target"] == TARGET]
    for mode_label, mode_mask in [("B_temporal", d["mode"] == "B_temporal"),
                                  ("A_pooled", d["mode"].str.startswith("A_pooled"))]:
        sub = d[mode_mask]
        if sub.empty:
            continue
        # average across A folds; B has one row per model
        piv = sub.groupby("model")["score"].mean()
        persistence = piv.get("persistence", np.nan)
        base = piv.get("base", np.nan)
        emb = piv.get("emb", np.nan)
        emb_base = piv.get("emb_base", np.nan)
        n = int(sub["n"].max())
        rows.append(dict(
            region=rg, mode=mode_label, n=n,
            persistence=round(persistence, 4), base=round(base, 4),
            emb=round(emb, 4), emb_base=round(emb_base, 4),
            d_emb_vs_persist=round(emb - persistence, 4),
            d_emb_vs_base=round(emb - base, 4),
            d_embbase_vs_base=round(emb_base - base, 4),
        ))

out = pd.DataFrame(rows)
op = f"{R}/outputs/loro_region_summary.csv" if TARGET == "incidence" else f"{R}/outputs/loro_region_summary_{TARGET}.csv"
out.to_csv(op, index=False)

pd.set_option("display.width", 200, "display.max_columns", 30)
print("\n=== LORO zero-shot region transfer (incidence PR-AUC) ===", flush=True)
print("emb = model that NEVER saw this region. d_* > 0 means the learned embedding helps.\n", flush=True)
for mode in ["B_temporal", "A_pooled"]:
    m = out[out["mode"] == mode]
    if m.empty:
        continue
    print(f"--- Mode {mode} ---", flush=True)
    print(m[["region", "n", "persistence", "base", "emb", "emb_base",
             "d_emb_vs_persist", "d_emb_vs_base"]].to_string(index=False), flush=True)
    print("", flush=True)

# verdict helper
b = out[out["mode"] == "B_temporal"]
if not b.empty:
    beats_persist = (b["d_emb_vs_persist"] > 0).sum()
    beats_base = (b["d_emb_vs_base"] > 0).sum()
    print(f"[verdict] Mode B: emb beats persistence in {beats_persist}/{len(b)} regions; "
          f"beats strong base in {beats_base}/{len(b)} regions.", flush=True)
    print(f"[verdict] SYMMETRIC transfer (supports region-invariant claim) requires emb>>persistence "
          f"in ALL regions incl. africa. Only-africa-source would show africa-as-source elsewhere but "
          f"africa-as-target failing.", flush=True)
print(f"\n[done] wrote {op}", flush=True)
