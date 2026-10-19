"""ens_multitask.py -- forecasting ensemble, FINAL design: one pretrained encoder fine-tuned once with four heads.
(Forecasting with a pretrained encoding, GAPR memo / appendix; not the LORO transfer design.)

One job = one ALL-REGION pretrained encoder (ENS_ENC_SEED in 123, 456, 789, 101112, 131415;
repr_model_L_contrastive_e_s{seed}.pt), original vocabulary. Shared encoder + four linear heads, trained jointly:
  incidence  any violent event in t+1
  civilian   any violence-against-civilians event in t+1
  onset      any violent event in t+1; defined only on rows with has_violence == 0 in month t (loss and scores masked)
  fatal25    25 or more fatalities in t+1
Loss: mean over heads of the head's mean BCE over its unmasked rows in the batch (equal weights).
Windows (forecast months): train 1990-01..2020-12; stop 2021-01..06; stack 2021-07..12; test 2022-01 on.
Shared training sample (depends only on the data and the sampling settings, never on the encoder; reproduced by
ens_baseline_mt.py and checked by hash):
  positives = every train-window row positive for civilian, onset or fatal25, plus incidence-only positives drawn at
  random up to POS_CAP positives in total (rare heads are never subsampled); negatives = rows negative on every target,
  NEG_PER_POS per positive.
Stop sample: every stop-window row positive on any target plus STOP_SHARE of the all-negative rows. Learning rate and
epoch chosen by the mean over heads of stop-sample PR-AUC (onset on its unmasked rows).
Outputs (never overwritten; the marker is written last and is what ens_combine.py requires):
  data_model/ens/mt_{TAG}.pt, outputs/ens/log_mt_{TAG}.csv,
  outputs/ens/preds_mt_{target}_{TAG}.npz for each target (stack + test rows; onset rows have has_violence == 0),
  outputs/ens/done_mt_{TAG}.json (completion marker).
ENS_PREFLIGHT=1 builds the sample and sequences, prints counts and peak memory, and exits.
Guard: ENS_PRETRAIN_CUTOFF_OK=1 required (pretraining data end 2020-12; verified 2026-10-17).
"""
import os, sys, json, time, hashlib
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score

R = os.environ.get("ENS_ROOT", "/storage/group/LiberalArts/default/jfe4_collab/tensor_geographic_violence/conflict_forecast")
E = os.environ.get
ENC_SEED = int(E("ENS_ENC_SEED", "123"))
SMOKE = E("SMOKE", "0") == "1"; PREFLIGHT = E("ENS_PREFLIGHT", "0") == "1"
REGION_ORDER = ["africa", "mena", "europe", "americas", "south_asia", "asia_pacific"]
TARGETS = ["incidence", "civilian", "onset", "fatal25"]
assert ENC_SEED in (123, 456, 789, 101112, 131415), f"bad ENS_ENC_SEED {ENC_SEED}"
assert E("ENS_PRETRAIN_CUTOFF_OK", "0") == "1", "set ENS_PRETRAIN_CUTOFF_OK=1 only after verifying pretraining data end <= 2020-12"
TAG = f"L_contrastive_e_s{ENC_SEED}"
CKPT = f"{R}/data_model/repr_model_{TAG}.pt"
assert os.path.exists(CKPT), f"missing encoder {CKPT}"

TRAIN_START, TRAIN_END = "1990-01", "2020-12"
STOP_START, STOP_END = "2021-01", "2021-06"
STACK_START, STACK_END = "2021-07", "2021-12"
TEST_START = "2022-01"
POS_CAP = int(E("ENS_POS_CAP", "150000")); NEG_PER_POS = int(E("ENS_NEG_PER_POS", "4"))
STOP_SHARE = 0.25
LRS = [float(x) for x in E("ENS_LRS", "1e-4,3e-4").split(",")]; WD = 0.01
MAX_EP = int(E("ENS_MAX_EP", "15")); PATIENCE = int(E("ENS_PATIENCE", "3")); MIN_DELTA = 1e-4
SAMPLE_SEED = 123
MAX_SEQ_LEN, BATCH, FT_BATCH = 200, 512, 128
WORKERS = int(E("ENS_WORKERS", "4"))
if SMOKE:
    POS_CAP = int(E("ENS_POS_CAP", "300")); LRS = LRS if "ENS_LRS" in os.environ else [1e-4]
    MAX_EP = int(E("ENS_MAX_EP", "2")); PATIENCE = 1
SUF = "_smoke" if SMOKE else ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vocab_remap import remap_tables
from regions import REGION_OF, REGIONS
assert set(REGION_ORDER) == set(REGIONS), "REGION_ORDER out of sync with regions.py"

JOB = f"{TAG}{SUF}"
MD, OD = f"{R}/data_model/ens", f"{R}/outputs/ens"
os.makedirs(MD, exist_ok=True); os.makedirs(OD, exist_ok=True)
OUT_W, OUT_L, OUT_DONE = f"{MD}/mt_{JOB}.pt", f"{OD}/log_mt_{JOB}.csv", f"{OD}/done_mt_{JOB}.json"
OUT_P = {t: f"{OD}/preds_mt_{t}_{JOB}.npz" for t in TARGETS}
import fcntl
def _job_lock(path):
    """Exclusive per-job lock (Codex 12c): taken before the completion-marker check and held until exit, so an
    overlapping duplicate job fails fast instead of racing through publication."""
    fh = open(path, "a")
    try: fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit(f"[lock] {path} is held by another running job; exiting without writing")
    return fh
if not PREFLIGHT:
    _LOCK = _job_lock(f"{OD}/lock_mt_{JOB}.lock")
    assert not os.path.exists(OUT_DONE), f"refusing to overwrite completed job {OUT_DONE}"
    for t, f in OUT_P.items():
        if os.path.exists(f): print(f"[recover] {f} exists without {OUT_DONE}; it will be replaced", flush=True)
TMP_W, TMP_L = OUT_W + ".part", OUT_L + ".part"

CAPACITY = {"S": (64, 2, 4), "M": (128, 4, 8), "L": (256, 6, 8), "XL": (512, 8, 16)}
EMBED_DIM, N_LAYERS, N_HEADS = CAPACITY[TAG.split("_")[0]]
SUPER_FIELDS = ["event_type","sub_event_type","actor1","actor2","actor1_type","actor2_type","month_offset","fatalities","deaths_civilians"]
SEL = [0,1,2,3,4,5,6]; FIELDS = [SUPER_FIELDS[i] for i in SEL]; NUM_FIELDS = len(SEL)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(ENC_SEED); np.random.seed(ENC_SEED)
print(f"[cfg] JOB=mt_{JOB} lrs={LRS} wd={WD} max_ep={MAX_EP} patience={PATIENCE} pos_cap={POS_CAP} "
      f"neg_per_pos={NEG_PER_POS} dev={DEVICE}", flush=True)

# ---- vocab, PRIO lookup, Encoder: copied from share_grid_conflict.py ----
vocabs = json.load(open(f"{R}/data_intermediate/vocab_super.json"))
vsm = {k: max(v.values()) + 1 + 1 for k, v in vocabs.items()}
LUT = remap_tables(TAG, FIELDS, vsm, False)   # identity tables: original vocabulary
ps = pd.read_parquet(f"{R}/data_intermediate/priogrid_static.parquet")
assert ps["grid_id"].is_unique, "priogrid_static has duplicate grid_id"
static_cols = [c for c in ps.columns if c != "grid_id"]; STATIC_DIM = len(static_cols)
prio_lookup = {int(r.grid_id): np.asarray([getattr(r, c) for c in static_cols], np.float32) for r in ps.itertuples(index=False)}

class Encoder(nn.Module):
    def __init__(self, vsm, d, n_heads, n_layers, fields, static_dim, dropout=0.1):
        super().__init__()
        self.fields = fields
        self.embs = nn.ModuleDict({f: nn.Embedding(vsm[f], d, padding_idx=0) for f in fields})
        self.field_proj = nn.Linear(len(fields) * d, d); self.proj_dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.prio_proj = nn.Linear(static_dim, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
    def cls_vec(self, seq, mask, static):
        B = seq.size(0)
        e = torch.cat([self.embs[f](seq[:, :, i]) for i, f in enumerate(self.fields)], dim=-1)
        e = self.proj_dropout(self.field_proj(e))
        cls = self.cls_token.expand(B, -1, -1) + self.prio_proj(static).unsqueeze(1)
        e = torch.cat([cls, e], dim=1)
        mwc = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        z = self.ln(self.transformer(e, src_key_padding_mask=~mwc)[:, 0, :])
        pad = (mask.sum(1) == 0).unsqueeze(1)
        return torch.where(pad, torch.zeros_like(z), z)

class Clf4(nn.Module):
    """shared encoder, one linear head per target (columns in TARGETS order)"""
    def __init__(self, enc):
        super().__init__(); self.enc = enc; self.head = nn.Linear(enc.ln.normalized_shape[0], len(TARGETS))
    def forward(self, seq, mask, static): return self.head(self.enc.cls_vec(seq, mask, static))

def parse_seq(s):
    if isinstance(s, str): seq = json.loads(s)
    elif isinstance(s, (list, tuple, np.ndarray)): seq = [list(ev) for ev in s]
    elif s is None or (isinstance(s, float) and np.isnan(s)): seq = []
    else: raise TypeError(f"unsupported event_sequence type {type(s)}")
    return seq[-MAX_SEQ_LEN:]

LUT_NP = [np.asarray(l, np.int64) for l in LUT]
assert max(int(l.max()) for l in LUT_NP) < 32767, "token ids do not fit int16"
def compact_seq(s):
    """raw sequence -> (L, NUM_FIELDS) int16 array of encoder token ids (LUT applied), L <= MAX_SEQ_LEN (Codex 12, memory)."""
    seq = parse_seq(s)
    if not seq: return np.zeros((0, NUM_FIELDS), np.int16)
    a = np.asarray(seq, np.int64)[:, SEL]
    return np.stack([LUT_NP[f][a[:, f]] for f in range(NUM_FIELDS)], 1).astype(np.int16)

class SeqDS(Dataset):
    def __init__(self, compact, gids, labels=None):
        self.seqs = compact                      # object array of int16 arrays, shared with the caller (no copy)
        self.gids = gids; self.labels = None if labels is None else np.asarray(labels, np.float32)
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        a = self.seqs[i]
        t = torch.zeros(MAX_SEQ_LEN, NUM_FIELDS, dtype=torch.long); m = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        if len(a): t[:len(a)] = torch.from_numpy(a.astype(np.int64)); m[:len(a)] = True
        st = torch.from_numpy(prio_lookup.get(int(self.gids[i]), np.zeros(STATIC_DIM, np.float32)))
        y = 0.0 if self.labels is None else self.labels[i]
        return t, m, st, torch.tensor(y)

# ---- panel and the four targets (labels built on a grid-sorted copy, written back by index) ----
# PANEL-BLOCK-START (identical in ens_baseline_mt.py)
panel = pd.read_parquet(f"{R}/data_intermediate/baseline_features_super.parquet",
                        columns=["grid_id", "year_month", "violence_next_month", "has_violence",
                                 "count_fatalities", "etype_violence_against_civilians"])
assert not panel.duplicated(["grid_id", "year_month"]).any(), "duplicate grid-months"
srt = panel[["grid_id", "year_month", "has_violence", "count_fatalities", "etype_violence_against_civilians"]].sort_values(
    ["grid_id", "year_month"], kind="mergesort")
g_ = srt["grid_id"].to_numpy(); per = pd.PeriodIndex(srt["year_month"].astype(str), freq="M"); mi = (per.year * 12 + per.month).to_numpy()
assert ((g_[1:] != g_[:-1]) | (mi[1:] - mi[:-1] == 1)).all(), "month gap within a cell; next-month labels would misalign"
nxt = srt.groupby("grid_id")[["has_violence", "count_fatalities", "etype_violence_against_civilians"]].shift(-1)
srt["has_next"] = nxt["has_violence"].notna()
srt["fat_next"] = nxt["count_fatalities"]; srt["civ_next"] = nxt["etype_violence_against_civilians"].fillna(0.0)
srt["hv_next"] = nxt["has_violence"]
for c in ("has_next", "fat_next", "civ_next", "hv_next"): panel[c] = srt[c]
del srt, nxt, g_, per, mi
ok = panel["has_next"] & panel["violence_next_month"].notna()
assert (panel.loc[ok, "hv_next"].astype(int) == panel.loc[ok, "violence_next_month"].astype(int)).all(), "has_violence(t+1) != violence_next_month(t)"
cm = pd.read_parquet(f"{R}/data_intermediate/baseline_features.parquet", columns=["grid_id", "year_month", "country"])
panel = panel.merge(cm, on=["grid_id", "year_month"], how="left"); del cm
panel["region"] = panel["country"].map(REGION_OF).fillna("other")
d = panel[panel["region"].isin(REGION_ORDER) & panel["violence_next_month"].notna() & panel["has_next"]].reset_index(drop=True)
del panel
assert d["fat_next"].notna().all(), "missing next-month fatalities"
quiet = d["has_violence"].fillna(0).astype(int).to_numpy() == 0
Y = np.full((len(d), 4), np.nan, np.float32)
Y[:, 0] = d["violence_next_month"].to_numpy().astype(int)
Y[:, 1] = (d["civ_next"].to_numpy() > 0).astype(int)
Y[quiet, 2] = Y[quiet, 0]                                   # onset: defined only where month t is quiet
Y[:, 3] = (d["fat_next"].to_numpy() >= 25).astype(int)
ym = d["year_month"].astype(str).to_numpy(); gids = d["grid_id"].to_numpy().astype(np.int64); reg = d["region"].to_numpy()
w_tr = (ym >= TRAIN_START) & (ym <= TRAIN_END); w_stop = (ym >= STOP_START) & (ym <= STOP_END)
w_stack = (ym >= STACK_START) & (ym <= STACK_END); w_te = ym >= TEST_START
assert not (w_tr & w_stop).any() and not (w_stop & w_stack).any() and not (w_stack & w_te).any()
for k, t in enumerate(TARGETS):
    for nm, w in (("train", w_tr), ("stop", w_stop), ("stack", w_stack), ("test", w_te)):
        yy = Y[w, k]; yy = yy[~np.isnan(yy)]
        assert len(yy) and yy.sum() > 0, f"{t} {nm}: no positives"
        print(f"[split] {t:9s} {nm:5s} rows={len(yy):,} positives={int(yy.sum()):,} rate={yy.mean():.4f}", flush=True)
# ---- shared training sample: rare heads never subsampled ----
rng = np.random.default_rng([SAMPLE_SEED, 99])
rare_pos = w_tr & ((Y[:, 1] == 1) | (Y[:, 2] == 1) | (Y[:, 3] == 1))
inc_only = w_tr & (Y[:, 0] == 1) & ~rare_pos
all_neg = w_tr & (Y[:, 0] == 0) & (Y[:, 1] == 0) & (Y[:, 3] == 0)          # onset is 0 or masked on these rows
p_rare = np.where(rare_pos)[0]; p_inc = np.where(inc_only)[0]
n_inc = max(0, min(len(p_inc), POS_CAP - len(p_rare)))
pos = np.concatenate([p_rare, rng.choice(p_inc, n_inc, replace=False)])
neg = rng.choice(np.where(all_neg)[0], min(int(all_neg.sum()), NEG_PER_POS * len(pos)), replace=False)
tr_idx = np.sort(np.concatenate([pos, neg]))
any_pos_stop = w_stop & ((Y[:, 0] == 1) | (Y[:, 1] == 1) | (Y[:, 3] == 1))
neg_stop = np.where(w_stop & ~any_pos_stop)[0]
st_idx = np.sort(np.concatenate([np.where(any_pos_stop)[0], rng.choice(neg_stop, int(round(STOP_SHARE * len(neg_stop))), replace=False)]))
assert len(np.intersect1d(tr_idx, st_idx)) == 0
def data_version():
    out = []
    for f in ("baseline_features_super.parquet", "baseline_features.parquet"):
        st_ = os.stat(f"{R}/data_intermediate/{f}"); out.append(f"{f}:{st_.st_size}:{int(st_.st_mtime)}")
    return "|".join(out)
h_ = hashlib.sha256(); h_.update(np.ascontiguousarray(gids[tr_idx]).tobytes()); h_.update("|".join(ym[tr_idx]).encode())
h_.update(f"multitask|{POS_CAP}|{NEG_PER_POS}|{SAMPLE_SEED}|{TRAIN_START}|{TRAIN_END}|{data_version()}".encode())
SAMPLE_HASH = h_.hexdigest()
# PANEL-BLOCK-END
for k, t in enumerate(TARGETS):
    yy = Y[tr_idx, k]; yy = yy[~np.isnan(yy)]
    assert yy.sum() > 0 and (yy == 0).sum() > 0, f"{t}: training sample lacks a class"
    print(f"[sample] {t:9s} train rows={len(yy):,} positives={int(yy.sum()):,}  stop positives={int(np.nansum(Y[st_idx, k])):,}", flush=True)
print(f"[provenance] train-sample hash {SAMPLE_HASH[:16]} ({data_version()}); train={len(tr_idx):,} stop={len(st_idx):,}", flush=True)

# ---- rows scored per target: stack + test windows (onset: quiet rows only) ----
out_all = np.where(w_stack | w_te)[0]
if SMOKE: out_all = np.sort(np.random.default_rng(0).choice(out_all, min(3000, len(out_all)), replace=False))

# ---- event sequences, streamed for the needed rows only, stored compactly ----
import pyarrow.parquet as pq
def ym_int(a): return np.char.replace(np.asarray(a).astype("U7"), "-", "").astype(np.int64)
need = np.unique(np.concatenate([tr_idx, st_idx, out_all]))
need_key = gids[need] * 1000000 + ym_int(ym[need])
assert len(np.unique(need_key)) == len(need), "duplicate (grid, month) keys among needed rows"
pos_of = pd.Series(need, index=need_key)
seqs = np.empty(len(d), dtype=object); filled = np.zeros(len(d), bool)
for bt in pq.ParquetFile(f"{R}/data_intermediate/baseline_features_super.parquet").iter_batches(
        batch_size=200000, columns=["grid_id", "year_month", "event_sequence"]):
    bk = bt.column("grid_id").to_numpy().astype(np.int64) * 1000000 + ym_int(bt.column("year_month").to_numpy(zero_copy_only=False))
    hit = np.isin(bk, need_key)
    if not hit.any(): continue
    rows = pos_of.loc[bk[hit]].to_numpy(); es = bt.column("event_sequence").to_numpy(zero_copy_only=False)[hit]
    for r_, e_ in zip(rows, es): seqs[r_] = compact_seq(e_)
    filled[rows] = True
assert filled[need].all(), f"{int((~filled[need]).sum())} needed rows have no event sequence"
if PREFLIGHT:
    import resource
    lens = np.array([len(seqs[i]) for i in need])
    print(f"[preflight] rows needed={len(need):,} mean events={lens.mean():.1f} max={lens.max()} "
          f"peak RSS={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2:.1f} GB", flush=True)
    sys.exit(0)

def new_clf():
    enc = Encoder(vsm, EMBED_DIM, N_HEADS, N_LAYERS, FIELDS, STATIC_DIM).to(DEVICE)
    sd = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    miss, _ = enc.load_state_dict(sd, strict=False)
    assert len(miss) == 0, f"encoder weights missing from checkpoint: {miss[:5]}"
    return Clf4(enc).to(DEVICE)

def predict(clf, ds):
    clf.eval(); out = []
    with torch.no_grad():
        for t, m, st, _ in DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=WORKERS):
            out.append(torch.sigmoid(clf(t.to(DEVICE), m.to(DEVICE), st.to(DEVICE))).float().cpu().numpy())
    return np.concatenate(out)

def masked_loss(logits, yy):
    """mean over heads (present in the batch) of each head's mean BCE over its unmasked rows"""
    ok = ~torch.isnan(yy); losses = []
    for k in range(yy.shape[1]):
        if ok[:, k].any():
            losses.append(nn.functional.binary_cross_entropy_with_logits(logits[ok[:, k], k], yy[ok[:, k], k]))
    return torch.stack(losses).mean()

def stop_score(P):
    prs = []
    for k in range(4):
        yk = Y[st_idx, k]; s = ~np.isnan(yk)
        prs.append(float(average_precision_score(yk[s], P[s, k])))
    return float(np.mean(prs)), prs

ds_tr = SeqDS(seqs[tr_idx], gids[tr_idx], Y[tr_idx]); ds_st = SeqDS(seqs[st_idx], gids[st_idx])
log_rows, best = [], None           # best = (mean stop PR, lr, epoch, state_dict, per-head PRs)
for lr in LRS:
    torch.manual_seed(ENC_SEED * 1000 + LRS.index(lr))
    clf = new_clf(); opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=WD)
    ld = DataLoader(ds_tr, batch_size=FT_BATCH, shuffle=True, num_workers=WORKERS,
                    generator=torch.Generator().manual_seed(ENC_SEED * 1000 + LRS.index(lr)))
    run_best, bad = -1.0, 0
    for ep in range(1, MAX_EP + 1):
        clf.train(); t0 = time.time(); tot, n = 0.0, 0; diverged = False
        for t, m, st, yy in ld:
            loss = masked_loss(clf(t.to(DEVICE), m.to(DEVICE), st.to(DEVICE)), yy.to(DEVICE))
            if not torch.isfinite(loss): diverged = True; break
            opt.zero_grad(); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            if not torch.isfinite(gn): diverged = True; break
            opt.step(); tot += float(loss) * len(yy); n += len(yy)
        if diverged:
            log_rows.append(dict(job=JOB, lr=lr, epoch=ep, train_loss=np.nan, stop_pr_mean=np.nan, diverged=True))
            pd.DataFrame(log_rows).to_csv(TMP_L, index=False); print(f"  lr={lr:g} DIVERGED at epoch {ep}", flush=True); break
        P_st = predict(clf, ds_st); assert np.isfinite(P_st).all(), "non-finite stop forecasts"
        v, prs = stop_score(P_st)
        log_rows.append(dict(job=JOB, lr=lr, epoch=ep, train_loss=tot / n, stop_pr_mean=v, diverged=False,
                             secs=round(time.time() - t0), **{f"stop_pr_{t}": p for t, p in zip(TARGETS, prs)}))
        pd.DataFrame(log_rows).to_csv(TMP_L, index=False)
        print(f"  lr={lr:g} ep={ep} loss={tot / n:.4f} stopPR mean={v:.4f} " +
              " ".join(f"{t}={p:.3f}" for t, p in zip(TARGETS, prs)) + f" {time.time() - t0:.0f}s", flush=True)
        if best is None or v > best[0]:
            best = (v, lr, ep, {k_: t_.detach().cpu().clone() for k_, t_ in clf.state_dict().items()}, prs)
        bad = 0 if v > run_best + MIN_DELTA else bad + 1
        run_best = max(run_best, v)
        if bad >= PATIENCE: break
    del clf, opt
    if torch.cuda.is_available(): torch.cuda.empty_cache()
assert best is not None, "every learning rate diverged before its first epoch finished"
print(f"[best] lr={best[1]:g} epoch={best[2]} stopPR mean={best[0]:.4f}", flush=True)

clf = new_clf(); clf.load_state_dict(best[3])
P_out = predict(clf, SeqDS(seqs[out_all], gids[out_all]))
assert P_out.shape == (len(out_all), 4) and np.isfinite(P_out).all()

# ---- publish: weights, log, per-target forecasts, then the completion marker ----
pd.DataFrame(log_rows).to_csv(TMP_L, index=False)
torch.save(dict(state_dict=best[3], lr=best[1], epoch=best[2], stop_pr_mean=best[0], tag=TAG, targets=TARGETS), TMP_W)
tmp_p = {}
for k, t in enumerate(TARGETS):
    yk = Y[out_all, k]; s = ~np.isnan(yk); rows = out_all[s]
    win = np.where(w_te[rows], "test", "stack")
    for w in ("stack", "test"):
        q = win == w
        print(f"[score] {t:9s} {w}: PR-AUC={average_precision_score(yk[s][q], P_out[s, k][q]):.4f} n={q.sum():,} pos={int(yk[s][q].sum()):,}", flush=True)
    tmp_p[t] = OUT_P[t] + ".part.npz"
    np.savez_compressed(tmp_p[t], grid_id=gids[rows], year_month=ym[rows].astype("U7"), region=reg[rows].astype("U12"),
                        window=win.astype("U5"), y=yk[s].astype(np.int8), p=P_out[s, k].astype(np.float32),
                        train_rows=tr_idx.size, stop_rows=st_idx.size, sample_hash=np.array(SAMPLE_HASH),
                        data_version=np.array(data_version()), pos_cap=POS_CAP, neg_per_pos=NEG_PER_POS)
    chk = np.load(tmp_p[t]); assert chk["p"].shape == chk["y"].shape == chk["grid_id"].shape
os.replace(TMP_W, OUT_W); os.replace(TMP_L, OUT_L)
for t in TARGETS: os.replace(tmp_p[t], OUT_P[t])
json.dump(dict(tag=TAG, sample_hash=SAMPLE_HASH, lr=best[1], epoch=best[2], stop_pr_mean=best[0],
               stop_pr=dict(zip(TARGETS, best[4])), files=OUT_P), open(OUT_DONE + ".part", "w"), indent=1)
os.replace(OUT_DONE + ".part", OUT_DONE)
print(f"[done] {OUT_DONE}", flush=True)
