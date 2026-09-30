"""Aggregate the capacity-curve transfer eval: S/M/L/XL x {a,e}, Mode-B forecast.
Reads the 24 eval CSVs, collapses folds+seeds, averages over the 26 held-out units,
prints per-capacity transfer skill on the core targets. Does the embedding transfer
BETTER as capacity grows, or saturate?"""
import os, numpy as np, pandas as pd
R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
caps, feats, seeds = ["S","M","L","XL"], ["a","e"], ["123","456","789"]
files = [f"{R}/outputs/eval_full_{c}_contrastive_{f}_s{s}.csv"
         for c in caps for f in feats for s in seeds]
files = [x for x in files if os.path.exists(x)]
print(f"[load] {len(files)} of 24 CSVs", flush=True)
df = pd.concat([pd.read_csv(x) for x in files], ignore_index=True)
df["cap"] = df["tag"].str.split("_").str[0]
df["mode2"] = np.where(df["mode"].str.startswith("A_pooled"), "A_spatial", "B_forecast")
g = df.groupby(["cap","feat","unit","mode2","target","model"], as_index=False)["score"].mean()
p = g.pivot_table(index=["cap","feat","unit","mode2","target"], columns="model", values="score").reset_index()
hib = p["target"] != "severity_crps"
p["d_eb"] = np.where(hib, p["emb_base"]-p["base"], p["base"]-p["emb_base"])
p["d_ep"] = np.where(hib, p["emb"]-p["persistence"], p["persistence"]-p["emb"])
agg = p.groupby(["cap","feat","mode2","target"], as_index=False)[["d_eb","d_ep","emb_base","base","emb","persistence"]].mean()
order = {"S":0,"M":1,"L":2,"XL":3}
B = agg[agg["mode2"] == "B_forecast"].copy()
for tgt in ["incidence","onset","noncontig_onset","rare_event_f1"]:
    print(f"\n===== {tgt}  (Mode B / forecast) =====")
    print(f"  {'cap':3s} {'ft':2s} | emb_base  base   |  Δ(emb_base-base)  Δ(emb-persistence)")
    sub = B[B["target"] == tgt].copy(); sub["o"] = sub["cap"].map(order)
    sub = sub.sort_values(["feat","o"])
    for _, r in sub.iterrows():
        eb = r["emb_base"]; ba = r["base"]
        ebs = f"{eb:.3f}" if pd.notna(eb) else "  -  "
        bas = f"{ba:.3f}" if pd.notna(ba) else "  -  "
        print(f"  {r['cap']:3s} {r['feat']:2s} |  {ebs}  {bas}  |   {r['d_eb']:+.3f}            {r['d_ep']:+.3f}")
print("\n[done]", flush=True)
