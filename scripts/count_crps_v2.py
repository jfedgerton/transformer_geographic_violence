"""
count_crps_v2.py  —  Hurdle (zero-inflated) embedding model vs conflictology OWN-12 AND
NEIGHBORS-12 (the competition's real pgm benchmark), on the 128-dim 5-seed ensemble.
pgm Africa+ME, s=1, test 2018-2023. CRPS + MIS, overall and on truth-positive cells.

What changed from v1 (which lost on y>0 to own-12):
  (1) HURDLE head: an xgb gate P(y_next>0) + an xgb-Poisson magnitude model fit only on
      positive cases, replacing v1's single global NB. This separates "will there be any
      violence" (where the embeddings looked strong on AUC) from "how much" — the natural
      fix for a 98%-zero target.
  (2) NEIGHBORS-12 benchmark: the 3x3 spatial stencil x past 12 months empirical distribution
      = the actual ViEWS pgm conflictology benchmark (own-12 was a weaker proxy).
  (3) 128-dim embeddings, 5-seed MEAN-PROBABILITY ensemble (matches the 08b ensemble
      convention: average gate prob and magnitude mean across seeds).
Still leakage-optimistic (full-data encoder); IGN/log-score + leakage-clean encoder = v3.

Run (SLURM, needs memory for the neighbour gather + 5 xgb fits):
  sbatch --job-name=tgv_crps2 --account=jfe4_cr_default --partition=standard --gres=gpu:a100:1 \
    --cpus-per-task=4 --mem=96G --time=03:00:00 --output=logs/crps2_%j.out --error=logs/crps2_%j.err \
    --wrap="module load anaconda && source venv/bin/activate && python scripts/count_crps_v2.py"
"""
import numpy as np
import pandas as pd
from scipy import stats
import xgboost as xgb

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
SEEDS = [123, 456, 789, 1024, 2025]
S = 250            # Monte-Carlo draws for the hurdle model
CHUNK = 100_000
META = {"grid_id", "year_month", "split", "country", "grid_lat_center", "grid_lon_center"}
np.random.seed(123)

# ---- 1. target panel + pivot matrix (cell x month) of sb_fatalities, with a sentinel zero row ----
tgt = pd.read_parquet(f"{R}/data_intermediate/pgm_fatalities_target.parquet")
tgt = tgt.sort_values(["grid_id", "year_month"]).reset_index(drop=True)
months = sorted(tgt.year_month.unique()); mi = {m: i for i, m in enumerate(months)}
cells = sorted(tgt.grid_id.unique()); ci = {c: i for i, c in enumerate(cells)}
nC, nM = len(cells), len(months)
M = np.zeros((nC + 1, nM))                       # row nC = sentinel zero row (missing neighbours)
M[tgt.grid_id.map(ci).to_numpy(), tgt.year_month.map(mi).to_numpy()] = tgt.sb_fatalities.to_numpy()
tgt["crow"] = tgt.grid_id.map(ci).to_numpy(); tgt["mcol"] = tgt.year_month.map(mi).to_numpy()
print(f"cells {nC}  months {nM} ({months[0]}..{months[-1]})")

# ---- 2. 3x3 neighbour stencil per cell (lat/lon at 0.5 deg); missing -> sentinel zero row ----
ll = tgt.drop_duplicates("grid_id").set_index("grid_id")[["grid_lat_center", "grid_lon_center"]]
key2row = {(round(la * 2), round(lo * 2)): ci[c] for c, (la, lo) in ll.iterrows()}
neigh = np.full((nC, 9), nC, dtype=np.int64)
for c, (la, lo) in ll.iterrows():
    r = ci[c]; k = 0
    for dla in (-1, 0, 1):
        for dlo in (-1, 0, 1):
            neigh[r, k] = key2row.get((round(la * 2) + dla, round(lo * 2) + dlo), nC); k += 1
print("neighbour stencil built")

# ---- 3. canonical row set (>=12 months history) and train/test masks ----
base = tgt[tgt.mcol >= 11].reset_index(drop=True)
tr_m = base.year_month <= "2017-12"
te_m = (base.year_month >= "2018-01") & (base.year_month <= "2023-12")
yte = base.loc[te_m, "sb_fat_next"].to_numpy(float)
ytr_pos = base.loc[tr_m, "sb_fat_next"].to_numpy(float) > 0
print(f"train {int(tr_m.sum())}  test {int(te_m.sum())}  test pos_rate {(yte>0).mean():.4f}")

# ---- 4. hurdle, 5-seed mean-probability ensemble (gate + magnitude), aligned to `base` order ----
keys = base[["grid_id", "year_month"]]
gate = np.zeros(int(te_m.sum())); mupos = np.zeros(int(te_m.sum()))
ylab = base["sb_fat_next"].to_numpy(float)
for s in SEEDS:
    e = pd.read_parquet(f"{R}/data_model/embeddings_all_seed{s}.parquet")
    ec = [c for c in e.columns if c not in META]
    X = keys.merge(e[["grid_id", "year_month"] + ec], on=["grid_id", "year_month"], how="left")[ec].to_numpy()
    X = np.nan_to_num(X)
    Xtr, Xte = X[tr_m.to_numpy()], X[te_m.to_numpy()]
    ytr = ylab[tr_m.to_numpy()]
    # gate: P(y_next > 0)
    g = xgb.train({"objective": "binary:logistic", "max_depth": 6, "eta": 0.1,
                   "subsample": 0.8, "colsample_bytree": 0.8, "seed": s},
                  xgb.DMatrix(Xtr, label=(ytr > 0).astype(int)), num_boost_round=300)
    gate += g.predict(xgb.DMatrix(Xte))
    # magnitude: E[y_next | y_next > 0], fit on positive train rows only
    pos = ytr > 0
    mm = xgb.train({"objective": "count:poisson", "max_depth": 6, "eta": 0.1, "subsample": 0.8, "seed": s},
                   xgb.DMatrix(Xtr[pos], label=ytr[pos]), num_boost_round=300)
    mupos += np.clip(mm.predict(xgb.DMatrix(Xte)), 1.0, None)
    print(f"  seed {s} done")
gate /= len(SEEDS); mupos /= len(SEEDS)
ytr_all = ylab[tr_m.to_numpy()]; yp = ytr_all[ytr_all > 0]
mbp, vbp = yp.mean(), yp.var(); rpos = max(mbp * mbp / max(vbp - mbp, 1e-6), 1e-3)
print(f"gate mean {gate.mean():.4f}  mupos mean {mupos.mean():.3f}  rpos {rpos:.4f}")

# ---- 5. scoring rules ----
def crps_ens(draws, y):
    n, Sx = draws.shape
    t1 = np.abs(draws - y[:, None]).mean(1)
    xs = np.sort(draws, 1); k = np.arange(1, Sx + 1)
    EdXX = (2.0 / Sx**2) * ((2 * k - Sx - 1) * xs).sum(1)
    return t1 - 0.5 * EdXX

def mis90(draws, y):
    lo = np.quantile(draws, 0.05, 1); hi = np.quantile(draws, 0.95, 1); a = 0.10
    return (hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi)

# ---- 6. hurdle draws (chunked): Bernoulli(gate) * zero-truncated NB(mupos, rpos) ----
nte = len(yte); cr_h = np.empty(nte); mi_h = np.empty(nte)
for i in range(0, nte, CHUNK):
    g = gate[i:i+CHUNK]; mu = mupos[i:i+CHUNK]; n = len(g)
    fire = np.random.random((n, S)) < g[:, None]
    p = rpos / (rpos + mu[:, None])
    cnt = np.maximum(stats.nbinom.rvs(rpos, p, size=(n, S), random_state=123), 1)
    d = (fire * cnt).astype(float)
    cr_h[i:i+CHUNK] = crps_ens(d, yte[i:i+CHUNK]); mi_h[i:i+CHUNK] = mis90(d, yte[i:i+CHUNK])

# ---- 7. benchmarks via pivot matrix: own-12, neighbours-12, no-change ----
crow = base.loc[te_m, "crow"].to_numpy(); mcol = base.loc[te_m, "mcol"].to_numpy()
wcols = mcol[:, None] - np.arange(11, -1, -1)[None, :]      # (n,12) window cols j-11..j
own = M[crow[:, None], wcols]                               # (n,12)
nbr = M[neigh[crow][:, :, None], wcols[:, None, :]].reshape(nte, -1)  # (n,108)
cr_own, mi_own = crps_ens(own, yte), mis90(own, yte)
cr_nbr, mi_nbr = crps_ens(nbr, yte), mis90(nbr, yte)
cr_nc = np.abs(M[crow, mcol] - yte)

# ---- 8. report ----
nz = yte > 0
def line(name, cr, mi=None):
    s = f"{name:26s} CRPS {cr.mean():.4f}  CRPS|y>0 {cr[nz].mean():.3f}"
    if mi is not None: s += f"  MIS {mi.mean():.3f}"
    print(s)
print("=" * 74)
print(f"pgm Africa+ME  s=1  test 2018-2023  n={nte}  pos={int(nz.sum())}  (128-dim, 5-seed)")
print("=" * 74)
line("hurdle_emb (128,5seed)", cr_h, mi_h)
line("conflictology_own12", cr_own, mi_own)
line("conflictology_neighbors12", cr_nbr, mi_nbr)
line("no_change", cr_nc)
print("=" * 74)
pd.DataFrame({
    "model": ["hurdle_emb", "conflictology_own12", "conflictology_neighbors12", "no_change"],
    "crps": [cr_h.mean(), cr_own.mean(), cr_nbr.mean(), cr_nc.mean()],
    "crps_pos": [cr_h[nz].mean(), cr_own[nz].mean(), cr_nbr[nz].mean(), cr_nc[nz].mean()],
    "mis": [mi_h.mean(), mi_own.mean(), mi_nbr.mean(), np.nan],
}).to_csv(f"{R}/outputs/crps_v2_pgm_africame.csv", index=False)
print("saved outputs/crps_v2_pgm_africame.csv")
