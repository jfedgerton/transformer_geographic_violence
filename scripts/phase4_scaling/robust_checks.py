"""robust_checks.py -- appendix sensitivity checks for the nested test: embedding permutation and reporting-bias
measurement error. One job = one held-out region RG x one encoder seed SEED. GPU (encoder forward passes only).

Testbed. The nested arm of the share grid at the 10 percent share (share_grid_conflict.py, Table 2 last column):
  same five stratified training draws (draw function copied verbatim, depends only on region/share/draw), same
  test rows (RG, Jan 2022 .. Apr 2025 forecast months), same XGBoost settings with seed 123, same frozen LORO encoder
  and vocabulary remap. The job first reproduces the saved share-grid forecasts (base exactly, emb_base within a
  small tolerance for GPU numerics) and stops if it cannot. Every check changes ONE thing relative to emb_base.

Check 1, embedding permutation (model "emb_perm").
  Within each forecast month, the embedding vectors are shuffled across the cell-months of that month (training
  draw rows and test rows shuffled separately). Each row keeps its own hand-built features and label; only the link
  between a cell-month and its own event history is broken. If the nested gain comes from what the encoder learned
  about each cell's history, emb_perm should lose it (emb_perm - base near zero or below). Permutation seed
  [123, draw]; one permutation per draw, so five per region-seed.

Check 2, reporting-bias measurement error (models "emb_me_{proxy}_{p}").
  Events are deleted from the event sequences of under-reported cells; the hand-built features are NOT changed
  (author decision: degrade sequences only). The degraded sequences go through the same frozen encoder, and
  emb_base is refit on the degraded embeddings. Training and test rows are degraded alike.
  proxies   remote = top third of the region's event cells by PRIO-GRID travel time to the nearest city
                     (priogrid_static ttime_mean; ties broken by grid_id)
            press  = bottom third of the region's event cells by mean ACLED source_count per event, 1990-2020
                     (ties broken by fewer ACLED events, then grid_id). UCDP rows are excluded from the proxy because
                     UCDP number_of_sources is counted differently; UCDP events in flagged cells are still deleted.
            "event cells" = cells of the region with at least one ACLED or UCDP event in 1990-2020. Thirds are
            computed among event cells, so both proxies flag the same number of cells.
  levels    p = 0.10, 0.25, 0.50 of each flagged cell's events deleted.
  history   (Codex 11, R01) Stored sequences keep only the last 200 events of the 12-month window, so they cannot
            be degraded directly. For every flagged cell the script rebuilds a per-month event catalogue: the events
            of absolute month m are read from every stored row of the cell whose window covers m, and the longest
            list is kept; every other list must be a suffix of it (front truncation only drops the oldest events).
            Each degraded row is rebuilt from the catalogue as its uncapped 12-month history, and the rebuilt history
            cut to its last 200 events must equal the stored sequence exactly for EVERY rebuilt row (zero tolerance).
  identity  An event is deleted from EVERY history it appears in (an unreported event is missing from all later
            histories). Event identity = (cell, absolute month, position in that month's catalogue list). The
            deletion draw is u = blake2b(seed 123, identity) / 2^64 and an event is deleted when u < p, so the levels
            are nested (every event deleted at 10 percent is also deleted at 25 and 50 percent).
  order     deletion is applied to the rebuilt uncapped history, then the last 200 events are kept (as in the main
            pipeline), so older events move into the window when newer ones are deleted.
Seeds. Measurement-error draws and permutations use seed 123 only (author decision); encoder seeds 123/456/789 are
  separate jobs as everywhere else.

Outputs (outputs/robust/, never overwritten; .part until the job finishes)
  metrics_robust_{RG}_s{SEED}.csv  one row per model x draw: PR-AUC, ROC-AUC, Brier, n_train, plus flagged-cell and
                                   deleted-event counts for the measurement-error rows
  preds_robust_{RG}_s{SEED}.npz     test forecasts for every row of the metrics file plus test-row identity
  flags_robust_{RG}_s{SEED}.csv     one file per seed job (Codex 11, R03), compared with any other seed's file that
                                   already exists; its md5 is recorded in the metrics rows: grid_id, ttime_mean,
                                   acled_mean_sources, acled_events, flag_remote, flag_press
Region cell pool (Codex 11, R02): cells with a region-labelled panel row in 1990-2020 (not later rows).
Env: EVAL_REGION, EVAL_SEED, VOCAB_REMAP (must be 1, main specification), ROBUST_ROOT and ROBUST_DEBUG_DIR (synthetic test only).
"""
import os, sys, json, time, hashlib
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

R = os.environ.get("ROBUST_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
E = os.environ.get
RG = E("EVAL_REGION"); SEED = int(E("EVAL_SEED", "123"))
REGION_ORDER = ["africa", "mena", "europe", "americas", "south_asia", "asia_pacific"]
assert RG in REGION_ORDER, f"bad EVAL_REGION {RG}"
RIDX = REGION_ORDER.index(RG)
TAG = f"L_contrastive_e_loro_{RG}_s{SEED}"
CKPT = f"{R}/data_model/repr_model_{TAG}.pt"
SHARE = 0.10
DRAWS = [0, 1, 2, 3, 4]
LEVELS = [0.10, 0.25, 0.50]
PROXIES = []                       # reporting-bias check dropped (Oct 16 2026); permutation only
CHECK_SEED = 123                     # permutation and deletion draws (author decision: seed 123 only)
DRAW_BASE = 123; XGB_SEED = 123; VAL_MIN_POS = 10; MIN_TRAIN_POS = 5
TOL_EMB_PR = 2e-3                    # reproduction tolerance for emb_base PR-AUC (GPU numerics, batch composition)
EMBED_DIM, N_LAYERS, N_HEADS = 256, 6, 8
MAX_SEQ_LEN, BATCH = 200, 512
TRAIN_START, TRAIN_END, TEST_START = "1990-01", "2020-12", "2022-01"
SUPER_FIELDS = ["event_type", "sub_event_type", "actor1", "actor2", "actor1_type", "actor2_type", "month_offset",
                "fatalities", "deaths_civilians"]
SEL = [0, 1, 2, 3, 4, 5, 6]; FIELDS = [SUPER_FIELDS[i] for i in SEL]; NUM_FIELDS = len(SEL); USE_PRIO = True
MO = SUPER_FIELDS.index("month_offset")
DEBUG_DIR = os.environ.get("ROBUST_DEBUG_DIR")   # synthetic test only: dumps some degraded histories
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vocab_remap
vocab_remap.R = R                    # same root as everything else (differs only in the synthetic test)
from vocab_remap import remap_enabled, remap_tables
from regions import REGION_OF, REGIONS
assert set(REGION_ORDER) == set(REGIONS), "REGION_ORDER out of sync with regions.py"
REGION_COUNTRIES = set(c for c, r in REGION_OF.items() if r == RG)
VOCAB_REMAP = remap_enabled()
assert VOCAB_REMAP, "robust checks run on the main specification only (VOCAB_REMAP=1)"

OD = f"{R}/outputs/robust"; os.makedirs(OD, exist_ok=True)
JOB = f"{RG}_s{SEED}"
OUT_M = f"{OD}/metrics_robust_{JOB}.csv"; OUT_P = f"{OD}/preds_robust_{JOB}.npz"; OUT_F = f"{OD}/flags_robust_{RG}_s{SEED}.csv"
assert not os.path.exists(OUT_M), f"refusing to overwrite completed job {OUT_M}"
TMP_M, TMP_P = OUT_M + ".part", OUT_P + ".part.npz"
assert os.path.exists(CKPT), f"missing encoder {CKPT}"
SG_P = f"{R}/outputs/share_grid/preds_nested_{JOB}.npz"; SG_M = f"{R}/outputs/share_grid/metrics_nested_{JOB}.csv"
assert os.path.exists(SG_P) and os.path.exists(SG_M), f"missing share-grid outputs for {JOB}"
torch.manual_seed(SEED); np.random.seed(SEED)
print(f"[cfg] JOB={JOB} share={SHARE} draws={DRAWS} levels={LEVELS} proxies={PROXIES} dev={DEVICE}", flush=True)

# ---- vocab, PRIO lookup, Encoder: copied from share_grid_conflict.py ----
vocabs = json.load(open(f"{R}/data_intermediate/vocab_super.json"))
vsm = {k: max(v.values()) + 1 + 1 for k, v in vocabs.items()}
LUT = remap_tables(TAG, FIELDS, vsm, VOCAB_REMAP)
ps = pd.read_parquet(f"{R}/data_intermediate/priogrid_static.parquet")
assert ps["grid_id"].is_unique, "priogrid_static has duplicate grid_id"
static_cols = [c for c in ps.columns if c != "grid_id"]; STATIC_DIM = len(static_cols)
assert "ttime_mean" in static_cols, "ttime_mean missing from priogrid_static"
prio_lookup = {int(r.grid_id): np.asarray([getattr(r, c) for c in static_cols], np.float32) for r in ps.itertuples(index=False)}
# month_offset token id -> months before the row month (raw value); vocab keys must be integers
inv_mo = {int(i): int(k) for k, i in vocabs["month_offset"].items() if k.isdigit()}   # '<PAD>' skipped
assert sorted(inv_mo.values()) == list(range(1, 13)), f"unexpected month_offset vocabulary {vocabs['month_offset']}"
tok_of_mo = {v: k for k, v in inv_mo.items()}                                         # months before -> token id


class Encoder(nn.Module):
    def __init__(self, vsm, d, n_heads, n_layers, fields, use_prio, static_dim, dropout=0.1):
        super().__init__()
        self.fields, self.use_prio = fields, use_prio
        self.embs = nn.ModuleDict({f: nn.Embedding(vsm[f], d, padding_idx=0) for f in fields})
        self.field_proj = nn.Linear(len(fields) * d, d); self.proj_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        if use_prio: self.prio_proj = nn.Linear(static_dim, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d * 4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)

    def cls_vec(self, seq, mask, static):
        B = seq.size(0)
        e = torch.cat([self.embs[f](seq[:, :, i]) for i, f in enumerate(self.fields)], dim=-1)
        e = self.proj_dropout(self.field_proj(e))
        cls = self.cls_token.expand(B, -1, -1)
        if self.use_prio: cls = cls + self.prio_proj(static).unsqueeze(1)
        e = torch.cat([cls, e], dim=1)
        mwc = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        z = self.ln(self.transformer(e, src_key_padding_mask=~mwc)[:, 0, :])
        pad = (mask.sum(1) == 0).unsqueeze(1)
        return torch.where(pad, torch.zeros_like(z), z)


def full_seq(s):
    """Stored event_sequence -> full list of events (no truncation)."""
    if isinstance(s, str): return json.loads(s)
    if isinstance(s, (list, tuple, np.ndarray)): return [list(ev) for ev in s]
    if s is None or (isinstance(s, float) and np.isnan(s)): return []
    raise TypeError(f"unsupported event_sequence type {type(s)}")


class SeqDS(Dataset):
    """Takes already-parsed, already-truncated event lists."""
    def __init__(self, seqs, gids):
        self.seqs = seqs; self.gids = gids

    def __len__(self): return len(self.seqs)

    def __getitem__(self, i):
        seq = self.seqs[i]
        assert len(seq) <= MAX_SEQ_LEN
        t = torch.zeros(MAX_SEQ_LEN, NUM_FIELDS, dtype=torch.long); m = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        for j, ev in enumerate(seq):
            for f in range(NUM_FIELDS): t[j, f] = LUT[f][ev[SEL[f]]]
            m[j] = True
        st = torch.from_numpy(prio_lookup.get(int(self.gids[i]), np.zeros(STATIC_DIM, np.float32)))
        return t, m, st


enc = Encoder(vsm, EMBED_DIM, N_HEADS, N_LAYERS, FIELDS, USE_PRIO, STATIC_DIM).to(DEVICE)
_sd = torch.load(CKPT, map_location=DEVICE, weights_only=True)
_miss, _unexp = enc.load_state_dict(_sd, strict=False)
assert len(_miss) == 0, f"encoder weights missing from checkpoint: {_miss[:5]}"
enc.eval()


def embed(seq_lists, gid_arr):
    Z = []
    with torch.no_grad():
        for t, m, st in DataLoader(SeqDS(seq_lists, gid_arr), batch_size=BATCH, shuffle=False, num_workers=0):
            Z.append(enc.cls_vec(t.to(DEVICE), m.to(DEVICE), st.to(DEVICE)).cpu().numpy())
    Z = np.concatenate(Z) if Z else np.zeros((0, EMBED_DIM), np.float32)
    assert Z.shape == (len(seq_lists), EMBED_DIM) and np.isfinite(Z).all()
    return Z


# ---- data: same panel, merge, filter and split as share_grid_conflict.py ----
base_cols = ["violence_lag1", "violence_lag3", "violence_lag6", "spatial_lag_violence",
             "violent_count_roll1", "violent_count_roll3", "all_count_roll1", "any_event_lag1",
             "log_event_density_lag1", "months_since_last_event"]
panel = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet",
                        columns=["grid_id", "year_month", "event_sequence", "violence_next_month"] + base_cols)
cm = pd.read_parquet(f"{R}/data_intermediate/baseline_features.parquet", columns=["grid_id", "year_month", "country"])
panel = panel.merge(cm, on=["grid_id", "year_month"], how="left")
d = panel[panel["country"].isin(REGION_COUNTRIES) & panel["violence_next_month"].notna()].sort_values("year_month").reset_index(drop=True)
del panel, cm
n0 = len(d)
d = d.merge(ps, on="grid_id", how="left"); assert len(d) == n0, "prio merge changed row count"
FEATS = base_cols + static_cols
d[FEATS] = d[FEATS].fillna(0.0)
X = d[FEATS].to_numpy(np.float32)
ym = d["year_month"].astype(str).to_numpy()
trm = (ym >= TRAIN_START) & (ym <= TRAIN_END); tem = ym >= TEST_START
y = d["violence_next_month"].to_numpy().astype(int)
seqs = d["event_sequence"].values; gids = d["grid_id"].values.astype(np.int64)
te_idx = np.where(tem)[0]; yte = y[te_idx]
assert trm.sum() > 1000 and len(te_idx) > 1000 and 0 < yte.sum() < len(yte)
per = pd.PeriodIndex(ym, freq="M"); mi = (per.year * 12 + per.month).to_numpy().astype(np.int64)
print(f"[region {RG}] rows={len(d):,} train={trm.sum():,} test={len(te_idx):,} test_pos={yte.mean():.4f}", flush=True)
TEST_ID = dict(__grid_id=gids[te_idx], __year_month=ym[te_idx].astype("U7"), __y=yte.astype(np.int8))

# test rows must be the share grid's test rows, in the same order
sg = np.load(SG_P)
assert np.array_equal(sg["__grid_id"].astype(np.int64), TEST_ID["__grid_id"]), "test rows differ from share grid (grid_id)"
assert np.array_equal(sg["__year_month"].astype(str), TEST_ID["__year_month"].astype(str)), "test rows differ from share grid (month)"
assert np.array_equal(sg["__y"].astype(int), yte), "test labels differ from share grid"


def draw_rows(mask, share, k, purpose):
    """Copied verbatim from share_grid_conflict.py (purpose 0 = training draw)."""
    rng = np.random.default_rng([DRAW_BASE, RIDX, int(round(share * 1e5)), k, purpose])
    pool = np.where(mask)[0]; pos = pool[y[pool] == 1]; neg = pool[y[pool] == 0]
    npos = int(round(len(pos) * share)); nneg = int(round(len(neg) * share))
    if purpose == 1:
        if len(pos) < VAL_MIN_POS: return None
        if npos < VAL_MIN_POS: npos = VAL_MIN_POS; nneg = min(len(neg), int(round(npos * len(neg) / len(pos))))
    elif npos < MIN_TRAIN_POS: return None
    return np.sort(np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, nneg, replace=False)]))


draws = {k: draw_rows(trm, SHARE, k, 0) for k in DRAWS}
assert all(v is not None for v in draws.values()), "a 10 percent draw has too few positives"
need = np.unique(np.concatenate(list(draws.values()) + [te_idx]))
posn = -np.ones(len(d), np.int64); posn[need] = np.arange(len(need))

# ---- clean embeddings ----
t0 = time.time()
clean_lists = [full_seq(seqs[i])[-MAX_SEQ_LEN:] for i in need]
Z0 = embed(clean_lists, gids[need])
len_need = np.array([len(x) for x in clean_lists]); del clean_lists          # Codex 11: release parsed lists
print(f"[emb clean] {Z0.shape} in {time.time() - t0:.0f}s", flush=True)


def xgb_p(Xtr, ytr, Xte):
    b = xgb.train({"objective": "binary:logistic", "max_depth": 6, "eta": 0.1, "subsample": 0.8,
                   "colsample_bytree": 0.8, "seed": XGB_SEED}, xgb.DMatrix(Xtr, label=ytr), 200)
    return b.predict(xgb.DMatrix(Xte))


def scores(p):
    return dict(pr=float(average_precision_score(yte, p)), roc=float(roc_auc_score(yte, p)),
                brier=float(np.mean((np.clip(p, 0, 1) - yte) ** 2)))


rows, preds = [], {}
def emit(model, k, idx, p, **extra):
    assert len(p) == len(yte) and np.isfinite(p).all(), (model, k)
    r = dict(region=RG, enc_seed=SEED, model=model, share=SHARE, draw=k, n_train=int(len(idx)),
             pos_train=int(y[idx].sum()), n_test=int(len(yte)), test_pos=float(yte.mean()), **scores(p), **extra)
    rows.append(r); preds[f"{model}|{k}"] = p.astype(np.float32)
    pd.DataFrame(rows).to_csv(TMP_M, index=False)
    print(f"  [{model}] draw={k} PR={r['pr']:.4f} ROC={r['roc']:.4f}", flush=True)


def fit_emb(Z, idx):
    """emb_base forecasts using embedding matrix Z (rows aligned with `need`)."""
    return xgb_p(np.concatenate([Z[posn[idx]], X[idx]], 1), y[idx], np.concatenate([Z[posn[te_idx]], X[te_idx]], 1))


# ---- step 0: reproduce the share grid at 10 percent ----
sgm = pd.read_csv(SG_M)
for k, idx in draws.items():
    pb = xgb_p(X[idx], y[idx], X[te_idx]); pe = fit_emb(Z0, idx)
    sb, se = sg[f"base|{SHARE:g}|{k}"].astype(np.float64), sg[f"emb_base|{SHARE:g}|{k}"].astype(np.float64)
    assert abs(average_precision_score(yte, pb) - average_precision_score(yte, sb)) < 1e-6 and \
        np.max(np.abs(pb.astype(np.float32) - sb)) < 1e-4, f"draw {k}: base does not reproduce the share grid"
    pr_sg = sgm[(sgm.model == "emb_base") & np.isclose(sgm.share, SHARE) & (sgm.draw == k)]["pr"]
    assert len(pr_sg) == 1
    d_pr = abs(average_precision_score(yte, pe) - pr_sg.iloc[0])
    assert d_pr < TOL_EMB_PR, f"draw {k}: emb_base PR differs from the share grid by {d_pr:.5f}"
    emit("base", k, idx, pb); emit("emb_base", k, idx, pe, repro_pr_diff=float(d_pr))

# ---- check 1: permutation within forecast month (training rows and test rows separately) ----
for k, idx in draws.items():
    rng = np.random.default_rng([CHECK_SEED, k])
    Zp = Z0.copy()
    for part in (idx, te_idx):
        for m in np.unique(mi[part]):
            rws = part[mi[part] == m]
            src = posn[rws]
            Zp[src] = Z0[src[rng.permutation(len(src))]]
    chg = np.any(Zp[posn[te_idx]] != Z0[posn[te_idx]], axis=1)
    moved = float(np.mean(chg))
    ne = len_need[posn[te_idx]] > 0
    _, inv, cnt = np.unique(mi[te_idx], return_inverse=True, return_counts=True)
    emit("emb_perm", k, idx, fit_emb(Zp, idx), share_test_rows_changed=moved,
         share_test_rows_changed_nonempty=float(np.mean(chg[ne])) if ne.any() else np.nan,
         share_test_rows_singleton_month=float(np.mean(cnt[inv] == 1)))
    del Zp

# ---- check 2 (reporting-bias measurement error) removed Oct 16 2026: Jared chose the permutation check only ----

# ---- publish ----
M = pd.DataFrame(rows)
exp_models = {"base", "emb_base", "emb_perm"} | {f"emb_me_{q}_{p:g}" for q in PROXIES for p in LEVELS}
assert set(M.model) == exp_models and (M.groupby("model").size() == len(DRAWS)).all(), "incomplete results"
np.savez_compressed(TMP_P, **TEST_ID, **preds)
chk = np.load(TMP_P); assert all(k in chk.files for k in preds)
os.replace(TMP_P, OUT_P); os.replace(TMP_M, OUT_M)
s = M.groupby("model")["pr"].mean(); base = s["base"]
print("\n[summary] mean PR-AUC over draws, gain over base\n" + (s - base).round(4).to_string(), flush=True)
print(f"[done] {OUT_M} ({len(M)} rows), {OUT_P}", flush=True)
