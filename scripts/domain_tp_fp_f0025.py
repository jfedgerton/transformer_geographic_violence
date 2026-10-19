# TP/FP counts behind the Discussion sentence (fine-tuned vs scratch, 0.25% share), from saved predictions. 2026-10-17
import glob, re, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as ap
rows = []
for d in ["ces", "crime", "dengue"]:
    fs = sorted(glob.glob(f"outputs/share_grid/preds_{d}_finetune_*_s*_f0.0025.npz"))
    assert fs, d
    for f in fs:
        unit, seed = re.search(rf"preds_{d}_finetune_(.+)_s(\d+)_f0.0025.npz", f).groups()
        z = np.load(f); y = z["__y"].astype(int); k = int(y.sum())
        for dr in range(5):
            r = dict(domain=d, unit=unit, seed=int(seed), draw=dr, n=len(y), events=k)
            for arm in ["ft_pretrained", "ft_scratch"]:
                p = z[f"{arm}|0.0025|{dr}"].astype(float); top = np.argsort(-p, kind="stable")[:k]
                r[f"tpk_{arm}"] = int(y[top].sum()); r[f"tp50_{arm}"] = int(((p >= .5) & (y == 1)).sum()); r[f"fp50_{arm}"] = int(((p >= .5) & (y == 0)).sum()); r[f"pr_{arm}"] = ap(y, p)
            rows.append(r)
df = pd.DataFrame(rows); df.to_csv("outputs/share_grid/tp_fp_f0025.csv", index=False)
g = df.groupby(["domain", "seed", "draw"]).sum(numeric_only=True).groupby("domain").mean()
out = pd.DataFrame({"units": df.groupby("domain").unit.nunique(), "files": df.groupby("domain").size() // 5,
    "pr_gain": df.assign(g=df.pr_ft_pretrained - df.pr_ft_scratch).groupby("domain").g.mean(),
    "events": g.events, "topk_TP_gain": g.tpk_ft_pretrained - g.tpk_ft_scratch,
    "p50_TP_gain": g.tp50_ft_pretrained - g.tp50_ft_scratch, "p50_FP_drop": g.fp50_ft_scratch - g.fp50_ft_pretrained})
print(out.round(3).to_string())
