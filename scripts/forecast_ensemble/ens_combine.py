"""ens_combine.py -- forecasting ensemble, stage 3: combine the fine-tuned encoders for one target and score them (CPU).
(Forecasting with a pretrained encoding; not the LORO transfer design.)

Inputs (ENS_KIND=mt, default, the final four-head design): outputs/ens/preds_mt_{target}_L_contrastive_e_s{seed}.npz and the
done_mt_* markers for EXACTLY the seeds in ENS_SEEDS (default all five all-region encoders 123,456,789,101112,131415),
plus preds_baseline_mt_{target}.npz and done_baseline_mt.json. ENS_KIND=ft reads the single-target files instead.
Combiners (fitted on the stack window 2021-07..2021-12 only, scored on the test window >= 2022-01):
  ens_mean   mean of the encoders' forecasts (nothing fitted)
  ens_stack  logistic regression on the encoders' logit forecasts, fitted on the stack window (L2, C=1)
Scored on the test window for every single encoder, both combiners, persistence and base: PR-AUC overall and by region,
and hits when each model flags as many cell-months as there are events (k = events; for onset, persistence is
constant 0, so its hits depend only on row order and mean nothing).
Diversity: mean pairwise Spearman correlation of the encoders' test forecasts (NaN with one encoder).
Checks (Codex 12): the exact roster of encoder files must exist (no more, no fewer); every file scores the same rows in
the same order with the same labels; outside SMOKE the encoder and baseline scoring populations must be identical
sets of (grid_id, year_month); window labels must match the calendar; every file must carry the same training-sample
hash, sampling settings and data version.
Outputs (never overwritten). A roster other than the full five gets its own name, so pilots never take production
names: scores_{mt_}{target}{_pilot_s123-456-789}.csv and preds_ens_{mt_}{target}{...}.npz. The forecasts are published first and
the scores CSV last; the scores CSV is the completion marker (an orphan forecasts file from an interrupted run is replaced).
"""
import os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

R = os.environ.get("ENS_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
E = os.environ.get
TARGET = E("ENS_TARGET", "incidence"); SMOKE = E("SMOKE", "0") == "1"; SUF = "_smoke" if SMOKE else ""
KIND = E("ENS_KIND", "mt")      # mt = four-head model (final design, ens_multitask.py); ft = single-target ens_finetune.py
assert KIND in ("mt", "ft"), KIND
KP = "mt_" if KIND == "mt" else ""
FULL = [123, 456, 789, 101112, 131415]
SEEDS = [int(x) for x in E("ENS_SEEDS", ",".join(map(str, FULL))).split(",")]
assert len(SEEDS) == len(set(SEEDS)) and set(SEEDS) <= set(FULL), f"bad ENS_SEEDS {SEEDS}"
RTAG = "" if sorted(SEEDS) == FULL else "_pilot_s" + "-".join(map(str, sorted(SEEDS)))
OD = f"{R}/outputs/ens"
OUT_S, OUT_P = f"{OD}/scores_{KP}{TARGET}{RTAG}{SUF}.csv", f"{OD}/preds_ens_{KP}{TARGET}{RTAG}{SUF}.npz"
import fcntl
def _job_lock(path):
    """Exclusive per-job lock (Codex 12c): taken before the completion-marker check and held until exit, so an
    overlapping duplicate job fails fast instead of racing through publication."""
    fh = open(path, "a")
    try: fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit(f"[lock] {path} is held by another running job; exiting without writing")
    return fh
_LOCK = _job_lock(f"{OD}/lock_combine_{KP}{TARGET}{RTAG}{SUF}.lock")
assert not os.path.exists(OUT_S), f"refusing to overwrite {OUT_S}"
if os.path.exists(OUT_P): print(f"[recover] {OUT_P} exists without {OUT_S}; it will be replaced", flush=True)

# ---- exact roster ----
files = [f"{OD}/preds_{KIND}_{TARGET}_L_contrastive_e_s{sd}{SUF}.npz" for sd in sorted(SEEDS)]
markers = ([f"{OD}/done_mt_L_contrastive_e_s{sd}{SUF}.json" for sd in sorted(SEEDS)] + [f"{OD}/done_baseline_mt{SUF}.json"]
           if KIND == "mt" else [])
missing = [f for f in files + markers if not os.path.exists(f)]
assert not missing, f"missing encoder files: {missing}"
F = [np.load(f) for f in files]
names = [f"enc_s{sd}" for sd in sorted(SEEDS)]
ref = F[0]
for f, z in zip(files[1:], F[1:]):
    for k in ("grid_id", "year_month", "window", "y"):
        assert np.array_equal(ref[k], z[k]), f"{f}: {k} differs from {files[0]}"

# ---- provenance: same training sample, settings and data version everywhere ----
bl = np.load(f"{OD}/preds_baseline_{KP}{TARGET}{SUF}.npz")
prov = lambda z: (str(z["sample_hash"]), str(z["data_version"]), int(z["pos_cap"]), int(z["neg_per_pos"]))
for f, z in zip(files + ["baseline"], F + [bl]):
    assert prov(z) == prov(ref), f"{f}: training-sample provenance differs {prov(z)} vs {prov(ref)}"

# ---- windows against the calendar ----
ym = ref["year_month"].astype(str); win = ref["window"].astype(str)
assert ((win == "test") == (ym >= "2022-01")).all() and ((win == "stack") == ((ym >= "2021-07") & (ym <= "2021-12"))).all(), \
    "window labels do not match the calendar"

# ---- scoring populations ----
key = pd.DataFrame({"grid_id": ref["grid_id"], "year_month": ym})
bdf = pd.DataFrame({"grid_id": bl["grid_id"], "year_month": bl["year_month"].astype(str), "y_b": bl["y"],
                    "p_persistence": bl["p_persistence"], "p_base": bl["p_base"], "region": bl["region"]})
assert not bdf.duplicated(["grid_id", "year_month"]).any() and not key.duplicated().any()
if not SMOKE:
    assert len(bdf) == len(key), f"scoring populations differ: {len(key):,} encoder rows vs {len(bdf):,} baseline rows"
m = key.merge(bdf, on=["grid_id", "year_month"], how="left", validate="one_to_one")
assert m["y_b"].notna().all(), f"{int(m['y_b'].isna().sum())} encoder rows missing from the baseline file"
y = ref["y"].astype(int); assert np.array_equal(y, m["y_b"].to_numpy().astype(int)), "labels differ between encoder and baseline files"
st, te = win == "stack", win == "test"
assert st.sum() > 0 and te.sum() > 0 and 0 < y[st].sum() < st.sum() and y[te].sum() > 0
region = m["region"].to_numpy()

# ---- combiners ----
P = np.column_stack([z["p"].astype(float) for z in F]); assert np.isfinite(P).all()
eps = 1e-6
L = np.log(np.clip(P, eps, 1 - eps) / (1 - np.clip(P, eps, 1 - eps)))
lr = LogisticRegression(C=1.0, max_iter=2000).fit(L[st], y[st])
models = {n: P[:, j] for j, n in enumerate(names)}
models["ens_mean"] = P.mean(1)
models["ens_stack"] = lr.predict_proba(L)[:, 1]
models["persistence"] = m["p_persistence"].to_numpy().astype(float)
models["base"] = m["p_base"].to_numpy().astype(float)

def hits(yt, p):
    k = int(yt.sum()); top = np.argsort(-p, kind="stable")[:k]
    return int(yt[top].sum())

rows = []
for nm, p in models.items():
    rows.append(dict(target=TARGET, model=nm, region="ALL", n=int(te.sum()), events=int(y[te].sum()),
                     pr_auc=average_precision_score(y[te], p[te]), hits_at_k=hits(y[te], p[te])))
    for rg in sorted(set(region[te])):
        s = te & (region == rg)
        if y[s].sum() == 0: continue
        rows.append(dict(target=TARGET, model=nm, region=rg, n=int(s.sum()), events=int(y[s].sum()),
                         pr_auc=average_precision_score(y[s], p[s]), hits_at_k=hits(y[s], p[s])))
out = pd.DataFrame(rows)
def mean_pairwise_spearman(X):
    """Mean pairwise Spearman over encoders with non-constant forecasts (Codex 12c); NaN if fewer than 2 usable."""
    keep = [j for j in range(X.shape[1]) if np.nanstd(X[:, j]) > 0]
    if len(keep) < X.shape[1]: print(f"[diversity] constant forecasts dropped: {[names[j] for j in range(X.shape[1]) if j not in keep]}")
    if len(keep) < 2: return np.nan, len(keep)
    R_ = pd.DataFrame(X[:, keep]).rank().to_numpy()
    C = np.corrcoef(R_, rowvar=False)
    return float(np.mean(C[np.triu_indices(len(keep), 1)])), len(keep)
div, n_div = mean_pairwise_spearman(P[te])
out["mean_pairwise_spearman_encoders"] = div; out["n_encoders_in_diversity"] = n_div
out["roster"] = ",".join(map(str, sorted(SEEDS))); out["sample_hash"] = prov(ref)[0]

# ---- publish: forecasts first, scores (completion marker) last ----
TMP_S, TMP_P = OUT_S + ".part", OUT_P + ".part.npz"
out.to_csv(TMP_S, index=False)
np.savez_compressed(TMP_P, grid_id=ref["grid_id"], year_month=ym.astype("U7"), window=win.astype("U5"), y=y.astype(np.int8),
                    region=region.astype("U12"), stack_coef=lr.coef_.ravel(), stack_intercept=lr.intercept_,
                    roster=np.array(sorted(SEEDS)), **{f"p_{k}": v.astype(np.float32) for k, v in models.items()})
chk = np.load(TMP_P); assert chk["y"].shape == (len(y),)
os.replace(TMP_P, OUT_P); os.replace(TMP_S, OUT_S)
a = out[out.region == "ALL"].set_index("model")[["pr_auc", "hits_at_k"]]
print(f"[{TARGET}{RTAG}] test n={int(te.sum()):,} events={int(y[te].sum()):,}; encoder diversity (mean Spearman) {div:.3f} over {n_div} encoders")
print(a.sort_values("pr_auc", ascending=False).round(4).to_string())
print(f"[done] {OUT_S}")
