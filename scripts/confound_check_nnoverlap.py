import numpy as np, pandas as pd
from scipy import stats
df = pd.read_csv("outputs/embedding_nn_overlap.csv")
df = df.dropna(subset=["n_violent","nn_overlap"])
n = len(df)
y = df["nn_overlap"].astype(float).values
x = df["n_violent"].astype(float).values
lx = np.log1p(x)
pear_r, pear_p = stats.pearsonr(x, y)
spr_r, spr_p = stats.spearmanr(x, y)
lpear_r, lpear_p = stats.pearsonr(lx, y)
# OLS nn_overlap ~ log1p(n_violent)
sl, ic, r, p, se = stats.linregress(lx, y)
print(f"N cells = {n}")
print(f"nn_overlap: mean={y.mean():.3f} sd={y.std():.3f} min={y.min():.0f} max={y.max():.0f}")
print(f"n_violent : mean={x.mean():.1f} median={np.median(x):.0f} max={x.max():.0f}")
print(f"Pearson(n_violent, nn_overlap)        r={pear_r:+.3f} p={pear_p:.2e}")
print(f"Spearman(n_violent, nn_overlap)       rho={spr_r:+.3f} p={spr_p:.2e}")
print(f"Pearson(log1p n_violent, nn_overlap)  r={lpear_r:+.3f} p={lpear_p:.2e}")
print(f"OLS nn_overlap ~ log1p(n_violent): slope={sl:+.3f} R2={r**2:.3f} p={p:.2e}")
# low-overlap cells: are they the high-activity ones?
lo = df[df["nn_overlap"]<=3]; hi = df[df["nn_overlap"]>=7]
print(f"low-overlap(<=3) cells: n={len(lo)} median n_violent={lo['n_violent'].median():.0f}")
print(f"high-overlap(>=7) cells: n={len(hi)} median n_violent={hi['n_violent'].median():.0f}")
