"""
count_crps_v3.py  —  INCREMENTAL-VALIDITY test (the honest point-2 claim).

Instead of "embedding hurdle vs neighbors-12" (v2, where the embedding LOST), this asks:
does ADDING the embedding ON TOP of a rich neighbor-12 feature set improve fit?

  base      = hurdle on neighbor-window features only      (featurized neighbors-12)
  treatment = hurdle on neighbor-window features + EMBEDDING

If treatment beats base (CRPS/MIS, overall and on y>0) with a CI that excludes 0, the
embedding carries information ORTHOGONAL to the strongest empirical benchmark — i.e. it
would improve a ViEWS-style ensemble. We ALSO report the raw nonparametric neighbors-12 /
own-12 / no-change for honesty (so the "baseline" isn't a strawman).

pgm Africa+ME, s=1, test 2018-2023, 128-dim 5-seed embeddings. Same machinery as v2.
Significance: cell-block bootstrap on the paired per-row CRPS difference.

Run (SLURM, A100):
  sbatch --job-name=tgv_crps3 --account=jfe4_cr_default --partition=standard --gres=gpu:a100:1 \
    --cpus-per-task=4 --mem=96G --time=03:00:00 --output=logs/crps3_%j.out --error=logs/crps3_%j.err \
    --wrap="module load anaconda && source venv/bin/activate && python scripts/count_crps_v3.py"
"""
import numpy as np
import pandas as pd
from scipy import stats
import xgboost as xgb

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
SEEDS = [123, 456, 789, 1024, 2025]
S = 250
CHUNK = 100_000
META = {"grid_id", "year_month", "split", "country", "grid_lat_center", "grid_lon_center"}
np.random.seed(123)

# ---- 1. target + pivot matrix M (cell x month), sentinel zero row at index nC ----
tgt = pd.read_parquet(f"{R}/data_intermediate/pgm_fatalities_target.parquet")
tgt = tgt.sort_values(["grid_id", "year_month"]).reset_index(drop=True)
months = sorted(tgt.year_month.unique()); mi = {m: i for i, m in enumerate(months)}
cells = sorted(tgt.grid_id.unique()); ci = {c: i for i, c in enumerate(cells)}
nC, nM = len(cells), len(months)
M = np.zeros((nC + 1, nM))
M[tgt.grid_id.map(ci).to_numpy(), tgt.year_month.map(mi).to_numpy()] = tgt.sb_fatalities.to_numpy()
tgt["crow"] = tgt.grid_id.map(ci).to_numpy(); tgt["mcol"] = tgt.year_month.map(mi).to_numpy()
print(f"cells {nC}  months {nM} ({months[0]}..{months[-1]})", flush=True)

# ---- 2. 3x3 neighbour stencil ----
ll = tgt.drop_duplicates("grid_id").set_index("grid_id")[["grid_lat_center", "grid_lon_center"]]
key2row = {(round(la * 2), round(lo * 2)): ci[c] for c, (la, lo) in ll.iterrows()}
neigh = np.full((nC, 9), nC, dtype=np.int64)
for c, (la, lo) in ll.iterrows():
    r = ci[c]; k = 0
    for dla in (-1, 0, 1):
        for dlo in (-1, 0, 1):
            neigh[r, k] = key2row.get((round(la * 2) + dla, round(lo * 2) + dlo), nC); k += 1
print("neighbour stencil built", flush=True)

# ---- 3. canonical rows + masks ----
base = tgt[tgt.mcol >= 11].reset_index(drop=True)
tr_m = (base.year_month <= "2017-12").to_numpy()
te_m = ((base.year_month >= "2018-01") & (base.year_month <= "2023-12")).to_numpy()
ylab = base["sb_fat_next"].to_numpy(float)
ytr = ylab[tr_m]; yte = ylab[te_m]
print(f"train {tr_m.sum()}  test {te_m.sum()}  test pos_rate {(yte>0).mean():.4f}", flush=True)

# ---- 4. neighbour-window FEATURES for ALL base rows (featurized neighbors-12) ----
crow_all = base["crow"].to_numpy(); mcol_all = base["mcol"].to_numpy()
wcols = mcol_all[:, None] - np.arange(11, -1, -1)[None, :]            # (N,12)
own = M[crow_all[:, None], wcols]                                     # (N,12) own past 12 mo
nbr = M[neigh[crow_all][:, :, None], wcols[:, None, :]].reshape(len(base), -1)  # (N,108)
nbr_recent = M[neigh[crow_all], mcol_all[:, None]]                    # (N,9) neighbours this month
NF = np.column_stack([
    own[:, -1], own.mean(1), own.max(1), (own > 0).mean(1),          # own: last, mean, max, freq
    nbr.mean(1), nbr.max(1), nbr.sum(1), (nbr > 0).mean(1),          # nbr 9x12: mean,max,sum,freq
    np.quantile(nbr, 0.9, axis=1), nbr_recent.mean(1),               # nbr q90, neighbours-now mean
]).astype(np.float32)
del own, nbr, nbr_recent
NF_tr, NF_te = NF[tr_m], NF[te_m]
print(f"neighbour features: {NF.shape[1]} cols", flush=True)

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

# ---- 6. hurdle fit/predict (gate + zero-truncated-NB magnitude), 5-seed ensembled ----
def fit_one(Xtr, Xte, seed):
    g = xgb.train({"objective": "binary:logistic", "max_depth": 6, "eta": 0.1,
                   "subsample": 0.8, "colsample_bytree": 0.8, "seed": seed},
                  xgb.DMatrix(Xtr, label=(ytr > 0).astype(int)), num_boost_round=300)
    gate = g.predict(xgb.DMatrix(Xte))
    pos = ytr > 0
    mm = xgb.train({"objective": "count:poisson", "max_depth": 6, "eta": 0.1, "subsample": 0.8, "seed": seed},
                   xgb.DMatrix(Xtr[pos], label=ytr[pos]), num_boost_round=300)
    mu = np.clip(mm.predict(xgb.DMatrix(Xte)), 1.0, None)
    return gate, mu

yp = ytr[ytr > 0]
rpos = max(yp.mean() ** 2 / max(yp.var() - yp.mean(), 1e-6), 1e-3)

def ensemble(use_embedding):
    nte = te_m.sum(); gate = np.zeros(nte); mupos = np.zeros(nte)
    if not use_embedding:
        for s in SEEDS:
            g, mu = fit_one(NF_tr, NF_te, s); gate += g; mupos += mu
            print(f"  base seed {s} done", flush=True)
    else:
        keys = base[["grid_id", "year_month"]]
        for s in SEEDS:
            e = pd.read_parquet(f"{R}/data_model/embeddings_all_seed{s}.parquet")
            ec = [c for c in e.columns if c not in META]
            emb = np.nan_to_num(keys.merge(e[["grid_id", "year_month"] + ec],
                                           on=["grid_id", "year_month"], how="left")[ec].to_numpy())
            Xtr = np.hstack([NF_tr, emb[tr_m]]); Xte = np.hstack([NF_te, emb[te_m]])
            g, mu = fit_one(Xtr, Xte, s); gate += g; mupos += mu
            print(f"  treat seed {s} done", flush=True)
    return gate / len(SEEDS), mupos / len(SEEDS)

def hurdle_score(gate, mupos):
    nte = len(yte); cr = np.empty(nte); miv = np.empty(nte)
    for i in range(0, nte, CHUNK):
        g = gate[i:i+CHUNK]; mu = mupos[i:i+CHUNK]; n = len(g)
        fire = np.random.random((n, S)) < g[:, None]
        p = rpos / (rpos + mu[:, None])
        cnt = np.maximum(stats.nbinom.rvs(rpos, p, size=(n, S), random_state=123), 1)
        d = (fire * cnt).astype(float)
        cr[i:i+CHUNK] = crps_ens(d, yte[i:i+CHUNK]); miv[i:i+CHUNK] = mis90(d, yte[i:i+CHUNK])
    return cr, miv

print("fitting BASE (neighbour features only)...", flush=True)
gb, mub = ensemble(False); cr_base, mi_base = hurdle_score(gb, mub)
print("fitting TREATMENT (neighbour features + embedding)...", flush=True)
gt, mut = ensemble(True);  cr_treat, mi_treat = hurdle_score(gt, mut)

# ---- 7. raw nonparametric references (own-12, neighbours-12, no-change) ----
crow = base["crow"].to_numpy()[te_m]; mcol = base["mcol"].to_numpy()[te_m]
wte = mcol[:, None] - np.arange(11, -1, -1)[None, :]
own_te = M[crow[:, None], wte]
nbr_te = M[neigh[crow][:, :, None], wte[:, None, :]].reshape(len(yte), -1)
cr_own = crps_ens(own_te, yte); cr_nbr = crps_ens(nbr_te, yte); cr_nc = np.abs(M[crow, mcol] - yte)

# ---- 8. paired significance: cell-block bootstrap on (base - treatment) CRPS ----
diff = cr_base - cr_treat                      # >0  => treatment (with embedding) better
uniq, inv = np.unique(crow, return_inverse=True)
csum = np.zeros(len(uniq)); ccnt = np.zeros(len(uniq))
np.add.at(csum, inv, diff); np.add.at(ccnt, inv, 1.0)
rng = np.random.default_rng(123)
boot = np.array([(lambda ix: csum[ix].sum() / ccnt[ix].sum())(rng.integers(0, len(uniq), len(uniq)))
                 for _ in range(1000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
nz = yte > 0
diff_pos = cr_base[nz].mean() - cr_treat[nz].mean()

# ---- 9. report ----
print("=" * 78)
print(f"INCREMENTAL TEST  pgm Africa+ME  test 2018-2023  n={len(yte)}  pos={int(nz.sum())}")
print("=" * 78)
print(f"base   (nbr feats)        CRPS {cr_base.mean():.4f}  CRPS|y>0 {cr_base[nz].mean():.3f}  MIS {mi_base.mean():.3f}")
print(f"treat  (nbr feats + emb)  CRPS {cr_treat.mean():.4f}  CRPS|y>0 {cr_treat[nz].mean():.3f}  MIS {mi_treat.mean():.3f}")
print(f"  Delta CRPS (base-treat) = {diff.mean():.5f}   95% CI [{lo:.5f}, {hi:.5f}]   (>0 => embedding helps)")
print(f"  Delta CRPS|y>0          = {diff_pos:.4f}")
print("-" * 78)
print(f"[ref] nonparametric neighbors12  CRPS {cr_nbr.mean():.4f}  CRPS|y>0 {cr_nbr[nz].mean():.3f}")
print(f"[ref] nonparametric own12        CRPS {cr_own.mean():.4f}  CRPS|y>0 {cr_own[nz].mean():.3f}")
print(f"[ref] no_change                  CRPS {cr_nc.mean():.4f}  CRPS|y>0 {cr_nc[nz].mean():.3f}")
print("=" * 78)
sig = "YES" if lo > 0 else ("worse" if hi < 0 else "n.s.")
print(f"VERDICT: embedding adds incremental value over featurized neighbors-12? {sig}")

pd.DataFrame({
    "model": ["base_nbrfeats", "treat_nbrfeats_emb", "ref_np_neighbors12", "ref_np_own12", "ref_no_change"],
    "crps": [cr_base.mean(), cr_treat.mean(), cr_nbr.mean(), cr_own.mean(), cr_nc.mean()],
    "crps_pos": [cr_base[nz].mean(), cr_treat[nz].mean(), cr_nbr[nz].mean(), cr_own[nz].mean(), cr_nc[nz].mean()],
    "mis": [mi_base.mean(), mi_treat.mean(), np.nan, np.nan, np.nan],
}).to_csv(f"{R}/outputs/crps_v3_incremental.csv", index=False)
pd.DataFrame({"delta_crps_mean": [diff.mean()], "ci_lo": [lo], "ci_hi": [hi],
              "delta_crps_pos": [diff_pos], "significant": [sig]}).to_csv(
    f"{R}/outputs/crps_v3_delta_ci.csv", index=False)
print("saved outputs/crps_v3_incremental.csv + crps_v3_delta_ci.csv")
