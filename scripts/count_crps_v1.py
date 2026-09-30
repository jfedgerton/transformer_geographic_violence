"""
count_crps_v1.py  —  First-pass distributional fatality forecast vs ViEWS-style benchmarks.

Goal: a current (2023/24 VIEWS Prediction Challenge) one-to-one at pgm, Africa+ME.
The challenge target is the NUMBER of UCDP state-based fatalities per cell-month, predicted
as a probability distribution (Monte-Carlo draws), scored with proper scoring rules.

This v1 answers the first question: does an embedding-conditioned count model beat the
conflictology and no-change benchmarks on CRPS? (The real bar is conflictology-neighbors-12;
v1 uses conflictology-OWN-12 as a slightly weaker proxy — neighbors-12 is the v2 refinement.)

First-pass modeling choices (deliberately simple, flagged for review):
  - Encoder: full-data (1989-2025) 64-dim seed-123 embeddings  -> leakage-OPTIMISTIC (Fork-2 default).
  - Mean model: XGBoost Poisson regression on the 64 embedding dims.
  - Predictive distribution: Negative Binomial(mu_i, r) with a single global dispersion r
    (method-of-moments on train). NB is the standard overdispersed-count choice; given the
    99% zero rate a zero-inflated / hurdle NB is the obvious v2 upgrade.
  - Horizon: s = 1 (next month).  Windows: train <= 2017-12, test 2018-01..2023-12 (the
    challenge backtest span). True-future Jul2024-Jun2025 is a v2 add.
  - Metrics: CRPS (energy estimator) and MIS (90% interval). IGN/log-score is a v2 add
    (needs the analytic PMF for NB and a smoothed PMF for the empirical benchmarks).

Run on Roar as a SLURM job (needs memory for the xgboost fit on ~2.7M train rows), e.g.:
  sbatch --job-name=tgv_crps --account=jfe4_cr_default --partition=standard \
         --gres=gpu:a100:1 --cpus-per-task=4 --mem=64G --time=02:00:00 \
         --output=logs/crps_%j.out --error=logs/crps_%j.err \
         --wrap="module load anaconda && source venv/bin/activate && python scripts/count_crps_v1.py"
"""
import numpy as np
import pandas as pd
from scipy import stats
import xgboost as xgb

R = "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast"
SEED = 123
S = 200            # Monte-Carlo draws for the NB model
CHUNK = 100_000    # rows per batch when sampling NB draws (bounds memory)
np.random.seed(SEED)

# ---- 1. load target + 64-dim seed-123 embeddings (the _d64 backup), join on cell-month ----
tgt = pd.read_parquet(f"{R}/data_intermediate/pgm_fatalities_target.parquet")
emb = pd.read_parquet(f"{R}/data_model/embeddings_all_seed{SEED}_d64.parquet")
META = {"grid_id", "year_month", "split", "country", "grid_lat_center", "grid_lon_center"}
ec = [c for c in emb.columns if c not in META]          # the 64 embedding columns
print(f"embedding columns: {len(ec)}")
df = (tgt.merge(emb[["grid_id", "year_month"] + ec], on=["grid_id", "year_month"], how="inner")
         .sort_values(["grid_id", "year_month"]).reset_index(drop=True))
print(f"joined rows: {len(df)}")

# ---- 2. conflictology own-12: forecast draws = the cell's own fatalities in the prior 12 months ----
# predicting y_{t+1}=sb_fat_next[t] from fatalities at t-11..t  ->  lags 0..11 of sb_fatalities
for k in range(12):
    df[f"L{k}"] = df.groupby("grid_id")["sb_fatalities"].shift(k)
lagcols = [f"L{k}" for k in range(12)]
df = df.dropna(subset=lagcols).reset_index(drop=True)

# ---- 3. competition windows (s=1): train <= 2017-12, test 2018-01..2023-12 ----
tr = df[df.year_month <= "2017-12"]
te = df[(df.year_month >= "2018-01") & (df.year_month <= "2023-12")]
ytr = tr.sb_fat_next.to_numpy(float)
yte = te.sb_fat_next.to_numpy(float)
print(f"train {len(tr)}  test {len(te)}  test positive rate {(yte>0).mean():.4f}")

# ---- 4. embedding -> NB(mu, r). mean via XGBoost Poisson; global dispersion via MoM ----
m = xgb.train(
    {"objective": "count:poisson", "max_depth": 6, "eta": 0.1, "subsample": 0.8,
     "colsample_bytree": 0.8, "seed": SEED},
    xgb.DMatrix(tr[ec].to_numpy(), label=ytr), num_boost_round=300)
mu = np.clip(m.predict(xgb.DMatrix(te[ec].to_numpy())), 1e-6, None)
mb, vb = ytr.mean(), ytr.var()
r = max(mb * mb / max(vb - mb, 1e-6), 1e-3)             # NB size param (Var = mu + mu^2/r)
print(f"xgb mean(mu)={mu.mean():.4f}  NB r={r:.5f}  (train mean={mb:.4f} var={vb:.2f})")

# ---- 5. scoring rules ----
def crps_ensemble(draws, y):
    """Energy-form CRPS for an (n, S) draw matrix vs y (n,). Returns per-obs CRPS."""
    n, Sx = draws.shape
    t1 = np.abs(draws - y[:, None]).mean(1)
    xs = np.sort(draws, 1)
    k = np.arange(1, Sx + 1)
    EdXX = (2.0 / Sx**2) * ((2 * k - Sx - 1) * xs).sum(1)   # E|X-X'|
    return t1 - 0.5 * EdXX

def mis90(draws, y):
    """Mean Interval Score for the central 90% interval (alpha=0.10)."""
    lo = np.quantile(draws, 0.05, axis=1); hi = np.quantile(draws, 0.95, axis=1)
    a = 0.10
    return (hi - lo) + (2/a)*(lo - y)*(y < lo) + (2/a)*(y - hi)*(y > hi)

# ---- 6. emb-NB (chunked sampling) ----
cr_nb = np.empty(len(te)); mi_nb = np.empty(len(te))
for i in range(0, len(te), CHUNK):
    mui = mu[i:i+CHUNK]; p = r / (r + mui)
    d = stats.nbinom.rvs(r, p[:, None], size=(len(mui), S), random_state=SEED).astype(float)
    cr_nb[i:i+CHUNK] = crps_ensemble(d, yte[i:i+CHUNK])
    mi_nb[i:i+CHUNK] = mis90(d, yte[i:i+CHUNK])

# ---- 7. benchmarks: conflictology own-12 (12-member empirical) and no-change (L0 point) ----
cf = te[lagcols].to_numpy(float)
cr_cf = crps_ensemble(cf, yte); mi_cf = mis90(cf, yte)
cr_nc = np.abs(te["L0"].to_numpy(float) - yte)          # no-change CRPS == MAE

# ---- 8. report (overall and on the truth-positive subset, where the signal lives) ----
nz = yte > 0
def line(name, cr, mi=None):
    s = f"{name:22s}  CRPS {cr.mean():.4f}   CRPS|y>0 {cr[nz].mean():.3f}"
    if mi is not None: s += f"   MIS {mi.mean():.3f}"
    print(s)
print("="*70)
print(f"pgm Africa+ME  s=1  test 2018-2023  n={len(te)}  positives={int(nz.sum())}")
print("="*70)
line("emb_NB (xgboost)", cr_nb, mi_nb)
line("conflictology_own12", cr_cf, mi_cf)
line("no_change", cr_nc)
print("="*70)
out = pd.DataFrame({
    "model": ["emb_NB", "conflictology_own12", "no_change"],
    "crps": [cr_nb.mean(), cr_cf.mean(), cr_nc.mean()],
    "crps_pos": [cr_nb[nz].mean(), cr_cf[nz].mean(), cr_nc[nz].mean()],
    "mis": [mi_nb.mean(), mi_cf.mean(), np.nan],
})
out.to_csv(f"{R}/outputs/crps_v1_pgm_africame.csv", index=False)
print("saved outputs/crps_v1_pgm_africame.csv")
